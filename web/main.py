import os
import json
import html
import aiosqlite
from fastapi import FastAPI, Request, Form, UploadFile, File, BackgroundTasks
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
from dotenv import load_dotenv
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
)
from aiortc.contrib.media import MediaBlackhole, MediaRelay
from silero_vad import load_silero_vad, get_speech_timestamps
from utils import (
    extract_text,
    load_job_requirements_many,
    parse_job_paths_env,
    list_requirement_files,
)
from agents.resume_analyzer import ResumeAnalyzerAgent
import torch
from db import create_candidate_and_analysis, save_interview, init_db

# Load .env so env vars are available when running via uvicorn
load_dotenv()

DB_PATH = os.getenv("DB_PATH", "./hr_bot.sqlite3")
REQ_DIR = os.getenv("JOB_REQUIREMENTS_DIR", "./requirements")
os.makedirs(REQ_DIR, exist_ok=True)
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
FILES_DIR = os.getenv("FILES_DIR", "./storage")
os.makedirs(FILES_DIR, exist_ok=True)
RESUME_THRASH = os.getenv("RESUME_THRASH")
RESUME_THRESHOLD = (
    int(RESUME_THRASH)
    if RESUME_THRASH is not None
    else int(os.getenv("RESUME_THRESHOLD", "75"))
)
JOB_DESCRIPTION_PATHS = os.getenv("JOB_DESCRIPTION_PATHS", "")
JOB_DESCRIPTION_PATH_LEGACY = os.getenv("JOB_DESCRIPTION_PATH", "")
CALL_BASE_URL = os.getenv("CALL_BASE_URL", "http://localhost:8000")
STT_DEVICE_PREF = (os.getenv("STT_DEVICE", "auto") or "auto").lower()  # auto|cuda|cpu
DEBUG_VAD = (os.getenv("DEBUG_VAD", "0") or "0").lower() in ("1", "true", "yes", "on")
RMS_GATE = float(os.getenv("RMS_GATE", "0.03"))  # fallback голосовой порог по RMS

app = FastAPI(title="HR Resume Bot — Admin")
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
    name="static",
)
templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)
agent = (
    ResumeAnalyzerAgent(api_base_url=API_BASE_URL, model_name=MODEL_NAME)
    if (API_BASE_URL and MODEL_NAME)
    else None
)

# Ensure DB schema (including new tables) exists on app start
@app.on_event("startup")
async def _startup_db():
    try:
        await init_db()
    except Exception:
        pass

    # init VAD (Silero) once
    global vad_model, _VAD_DEVICE, SAMPLERATE, CHUNK_SEC, PAUSE_CHUNKS
    try:
        _VAD_DEVICE = "cpu"  # silero VAD works great on CPU
        vad_model = load_silero_vad(onnx=False)
    except Exception:
        vad_model = None
    # align constants with CLI version
    SAMPLERATE = 16000
    CHUNK_SEC = 1
    PAUSE_CHUNKS = 2

# --- WebRTC globals ---
peers: dict[str, RTCPeerConnection] = {}
# map candidate id -> active peer connections (to force-close on finish)
peers_by_cid: dict[int, set] = {}
relay = MediaRelay()
RTC_CONF = RTCConfiguration(iceServers=[RTCIceServer("stun:stun.l.google.com:19302")])


# interview sessions per candidate id
class _InterviewSession:
    def __init__(self, cid: int):
        self.cid = cid
        self.interviewer = None
        self.websockets: set[WebSocket] = set()
        self.audio_buffer = []  # list of float32 numpy arrays (mono)
        self.silent_chunks = 0
        self.log: list[dict] = []
        self.tasks: set = set()
        self.ended: bool = False


sessions: dict[int, _InterviewSession] = {}


async def _get_or_create_session(cid: int) -> _InterviewSession:
    sess = sessions.get(cid)
    if not sess:
        sess = _InterviewSession(cid)
        # build interviewer from last analysis
        try:
            import json as _json

            async with aiosqlite.connect(DB_PATH) as db:
                db.row_factory = aiosqlite.Row
                cur = await db.execute(
                    "SELECT raw_json FROM analyses WHERE candidate_id=? ORDER BY id DESC LIMIT 1",
                    (cid,),
                )
                row = await cur.fetchone()
            analysis = {}
            questions = []
            if row and row["raw_json"]:
                rep = (
                    _json.loads(row["raw_json"])
                    if isinstance(row["raw_json"], str)
                    else row["raw_json"]
                )
                if isinstance(rep, dict):
                    anal = rep.get("Анализ") or {}
                    if isinstance(anal, dict):
                        analysis = anal
                    qs = rep.get("Вопросы") or []
                    if isinstance(qs, list):
                        questions = qs
        except Exception:
            analysis, questions = {}, []
        # lazy import to avoid cost at module import
        try:
            from agents.interviewer import HRInterviewerAgent
            # resolve device
            try:
                has_cuda = torch.cuda.is_available()
            except Exception:
                has_cuda = False
            if (os.getenv("STT_DEVICE", "auto").lower() == "cuda") and not has_cuda:
                print("[Interview] CUDA requested but not available. Falling back to CPU.")
            stt_device = (
                "cuda"
                if ((os.getenv("STT_DEVICE", "auto").lower() in ("auto", "cuda")) and has_cuda)
                else "cpu"
            )
            if stt_device == "cuda":
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass
            sess.interviewer = HRInterviewerAgent(
                api_base_url=API_BASE_URL or "",
                model_name=MODEL_NAME or "",
                analysis=analysis,
                questions=questions,
                device=stt_device,
            )
        except Exception:
            sess.interviewer = None
        sessions[cid] = sess
        # Console: devices & start banner
        try:
            dev = getattr(sess.interviewer, 'device', 'cpu') if sess.interviewer else 'n/a'
            print(f"🖥️ Whisper: {str(dev).upper()} | VAD: {_VAD_DEVICE} | CID={cid}")
            print("🎤 Собеседование началось. Говорите… (скажите 'стоп' чтобы завершить)")
        except Exception:
            pass
    return sess


async def _ws_broadcast(sess: _InterviewSession, payload: dict):
    dead = []
    for ws in list(sess.websockets):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        try:
            sess.websockets.remove(ws)
        except Exception:
            pass


def _resample_to_16k(x: "np.ndarray", orig_sr: int) -> "np.ndarray":
    import numpy as _np

    if orig_sr == 16000:
        return x.astype("float32")
    # simple linear resample
    duration = x.shape[0] / float(orig_sr)
    new_len = int(duration * 16000)
    if new_len <= 0:
        return _np.zeros(0, dtype="float32")
    xp = _np.linspace(0, 1, x.shape[0])
    fp = x.astype("float32")
    x_new = _np.linspace(0, 1, new_len)
    return _np.interp(x_new, xp, fp).astype("float32")


async def _transcribe_and_reply(sess: _InterviewSession, audio_f32_16k: "np.ndarray"):
    import numpy as _np

    if sess.ended or sess.interviewer is None or getattr(sess.interviewer, "stt_model", None) is None:
        return
    try:
        # normalize
        if audio_f32_16k.size == 0:
            return
        m = _np.max(_np.abs(audio_f32_16k))
        if m > 0:
            audio_f32_16k = audio_f32_16k / m
        result = sess.interviewer.stt_model.transcribe(
            audio_f32_16k, language="ru", fp16=False
        )
        user_text = (result.get("text") or "").strip()
        if not user_text:
            return
        await _ws_broadcast(sess, {"role": "user", "text": user_text})
        try:
            sess.log.append({"role": "user", "text": user_text})
        except Exception:
            pass
        reply = sess.interviewer.reply(user_text)
        await _ws_broadcast(sess, {"role": "assistant", "text": reply})
        try:
            sess.log.append({"role": "assistant", "text": reply})
        except Exception:
            pass
        # optional: speak locally on server, not sent back yet
        try:
            sess.interviewer.tts.speak(reply)
        except Exception:
            pass
    except Exception:
        pass


@app.websocket("/call/ws/{cid}")
async def call_ws(ws: WebSocket, cid: int):
    await ws.accept()
    sess = await _get_or_create_session(cid)
    sess.websockets.add(ws)
    await _ws_broadcast(sess, {"role": "system", "text": "Подключено к сессии"})
    try:
        while True:
            # keep alive; client doesn't send messages for now
            await ws.receive_text()
    except WebSocketDisconnect:
        try:
            sess.websockets.remove(ws)
        except Exception:
            pass


@app.post("/call/finish/{cid}")
async def call_finish(cid: int):
    sess = await _get_or_create_session(cid)
    sess.ended = True
    # cancel background tasks if any
    try:
        for t in list(sess.tasks):
            try:
                t.cancel()
            except Exception:
                pass
        sess.tasks.clear()
    except Exception:
        pass
    report = None
    try:
        if sess.interviewer and not sess.interviewer.finished:
            report = sess.interviewer.finish_and_report()
        if sess.interviewer:
            sess.interviewer.finished = True
    except Exception:
        report = None
    try:
        await save_interview(cid, sess.log, report)
    except Exception:
        pass
    try:
        await _ws_broadcast(sess, {"role": "system", "text": "Интервью завершено"})
    except Exception:
        pass
    # force-close peer connections for this candidate
    try:
        conns = list(peers_by_cid.get(cid, set()))
        for pc in conns:
            try:
                await pc.close()
            except Exception:
                pass
        peers_by_cid.pop(cid, None)
    except Exception:
        pass
    return {"ok": True}


STATUS_CHOICES = [
    ("screen_pending", "Ожидает скрининг"),
    ("screen_failed", "Не прошёл скрининг"),
    ("screen_passed", "Прошёл скрининг"),
    ("interview_failed", "Собеседование не успешно"),
    ("interview_passed", "Собеседование успешно"),
]


def status_label(code: str) -> str:
    for k, v in STATUS_CHOICES:
        if k == code:
            return v
    return code or "—"


def _parse_line_ranges_spec(spec) -> list[int]:
    # Accept int, "1-3,5,7-8", list[int]
    out: list[int] = []
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, list):
        for x in spec:
            if isinstance(x, int):
                out.append(x)
            elif isinstance(x, str) and x.isdigit():
                out.append(int(x))
        return out
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        for p in parts:
            if "-" in p:
                try:
                    s, e = p.split("-", 1)
                    s = int(s.strip())
                    e = int(e.strip())
                    if s <= e:
                        out.extend(list(range(s, e + 1)))
                except Exception:
                    continue
            elif p.isdigit():
                out.append(int(p))
    return out


def _build_highlighted_resume(text: str, annotations: list[dict] | None) -> str:
    if not text:
        return '<div class="muted">Текст резюме отсутствует</div>'
    if not annotations:
        # Вернём построчно с номерами для визуальной синхронизации
        lines = text.splitlines()
        blocks = []
        for i, line_text in enumerate(lines, start=1):
            blocks.append(
                f'<div class="line" data-line="{i}">{html.escape(line_text)}</div>'
            )
        return f"<div class=\"resume-annotated\">{''.join(blocks)}</div>"

    # Prefer line-based annotations, with support for ranges and full-line highlights
    has_line = any(
        isinstance(a, dict)
        and (
            "line" in a
            or "строка" in a
            or "line_from" in a
            or "line_to" in a
            or "lines" in a
        )
        for a in annotations
    )

    if has_line:
        lines = text.splitlines()
        per_line = {i + 1: [] for i in range(len(lines))}

        for a in annotations:
            try:
                tone = (a.get("тон") or "").lower()
                if tone not in ("good", "bad", "warn"):
                    continue

                # parse line indices
                indices: list[int] = []
                if "line" in a:
                    indices = _parse_line_ranges_spec(a.get("line"))
                elif "строка" in a:
                    indices = _parse_line_ranges_spec(a.get("строка"))
                elif "line_from" in a or "line_to" in a:
                    try:
                        s = int(a.get("line_from"))
                        e = int(a.get("line_to", s))
                        indices = list(range(s, e + 1)) if s <= e else []
                    except Exception:
                        indices = []
                elif "lines" in a:
                    indices = _parse_line_ranges_spec(a.get("lines"))

                if not indices:
                    continue
                # keep only valid
                indices = [i for i in indices if 1 <= i <= len(lines)]
                if not indices:
                    continue

                q = a.get("цитата") or ""
                s = a.get("start")
                e = a.get("end")
                have_fragment = (s is not None and e is not None) or (
                    q if isinstance(q, str) else ""
                )
                comment = str(a.get("комментарий") or "").strip()
                req = str(a.get("требование") or "").strip()

                for li in indices:
                    line_text = lines[li - 1]
                    ls, le = 0, len(line_text)
                    if have_fragment:
                        ss = ee = None
                        if s is not None and e is not None:
                            try:
                                ss = int(s)
                                ee = int(e)
                            except Exception:
                                ss = ee = None
                        if not (
                            ss is not None
                            and ee is not None
                            and 0 <= ss < ee <= len(line_text)
                        ):
                            if q and isinstance(q, str) and q:
                                idx = line_text.find(q)
                                if idx != -1:
                                    ss, ee = idx, idx + len(q)
                        if (
                            ss is not None
                            and ee is not None
                            and 0 <= ss < ee <= len(line_text)
                        ):
                            ls, le = ss, ee
                    per_line[li].append(
                        {
                            "s": ls,
                            "e": le,
                            "tone": tone,
                            "comment": comment,
                            "req": req,
                        }
                    )
            except Exception:
                continue

        # Render per-line, removing overlaps
        blocks = []
        for i, line_text in enumerate(lines, start=1):
            spans = sorted(per_line.get(i) or [], key=lambda x: (x["s"], x["e"]))
            cleaned = []
            last_e = -1
            for sp in spans:
                if sp["s"] < last_e:
                    continue
                cleaned.append(sp)
                last_e = sp["e"]
            pos = 0
            parts = []
            for sp in cleaned:
                if pos < sp["s"]:
                    parts.append(html.escape(line_text[pos : sp["s"]]))
                tone_cls = {"good": "hl-good", "bad": "hl-bad", "warn": "hl-warn"}[
                    sp["tone"]
                ]
                data_comment = html.escape(sp["comment"]) if sp["comment"] else ""
                data_req = html.escape(sp["req"]) if sp["req"] else ""
                data_tone = html.escape(sp["tone"]) if sp["tone"] else ""
                content = html.escape(line_text[sp["s"] : sp["e"]])
                parts.append(
                    f'<span class="hl {tone_cls}" data-comment="{data_comment}" data-req="{data_req}" data-tone="{data_tone}">{content}</span>'
                )
                pos = sp["e"]
            if pos < len(line_text):
                parts.append(html.escape(line_text[pos:]))
            blocks.append(
                f"<div class=\"line\" data-line=\"{i}\">{''.join(parts)}</div>"
            )
        return f"<div class=\"resume-annotated\">{''.join(blocks)}</div>"

    # Fallback: old whole-text spans
    L = len(text)
    spans = []
    for a in annotations:
        try:
            s = int(a.get("start", -1))
            e = int(a.get("end", -1))
            tone = (a.get("тон") or "").lower()
            comment = str(a.get("комментарий") or "").strip()
            req = str(a.get("требование") or "").strip()
            quote = a.get("цитата")
            if 0 <= s < e <= L and tone in ("good", "bad", "warn"):
                spans.append(
                    {
                        "s": s,
                        "e": e,
                        "tone": tone,
                        "comment": comment,
                        "req": req,
                        "quote": quote or "",
                    }
                )
                continue
            if quote and isinstance(quote, str) and tone in ("good", "bad", "warn"):
                idx = text.find(quote)
                if idx != -1:
                    spans.append(
                        {
                            "s": idx,
                            "e": idx + len(quote),
                            "tone": tone,
                            "comment": comment,
                            "req": req,
                            "quote": quote,
                        }
                    )
        except Exception:
            continue
    spans.sort(key=lambda x: (x["s"], x["e"]))
    cleaned = []
    last_e = -1
    for sp in spans:
        if sp["s"] < last_e:
            continue
        cleaned.append(sp)
        last_e = sp["e"]
    out = []
    pos = 0
    for sp in cleaned:
        if pos < sp["s"]:
            out.append(html.escape(text[pos : sp["s"]]))
        tone_cls = {"good": "hl-good", "bad": "hl-bad", "warn": "hl-warn"}[sp["tone"]]
        content = html.escape(text[sp["s"] : sp["e"]])
        data_comment = html.escape(sp["comment"]) if sp["comment"] else ""
        data_req = html.escape(sp["req"]) if sp["req"] else ""
        data_tone = html.escape(sp["tone"]) if sp["tone"] else ""
        out.append(
            f'<span class="hl {tone_cls}" data-comment="{data_comment}" data-req="{data_req}" data-tone="{data_tone}">{content}</span>'
        )
        pos = sp["e"]
    if pos < L:
        out.append(html.escape(text[pos:]))
    return f"<div class=\"resume-annotated\"><pre>{''.join(out)}</pre></div>"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, q: str | None = None, status: str | None = None):
    sql = "SELECT id, full_name, phone, candidate_status, created_at, (SELECT avg_score FROM analyses WHERE candidate_id=candidates.id ORDER BY id DESC LIMIT 1) AS avg_score FROM candidates"
    params = []
    where = []
    if q:
        where.append("(LOWER(full_name) LIKE ? OR phone LIKE ?)")
        params.extend([f"%{q.lower()}%", f"%{q}%"])
    if status:
        where.append("candidate_status = ?")
        params.append(status)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT 200"

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(sql, params)
        rows = await cur.fetchall()

    rows = [dict(r) for r in rows]
    for r in rows:
        r["status_label"] = status_label(r.get("candidate_status"))

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "rows": rows,
            "STATUS_CHOICES": STATUS_CHOICES,
            "q": q or "",
            "filter_status": status or "",
            "upload_enabled": bool(API_BASE_URL and MODEL_NAME),
        },
    )


@app.get("/candidates/{cid}", response_class=HTMLResponse)
async def candidate_detail(request: Request, cid: int):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        c = await db.execute("SELECT * FROM candidates WHERE id=?", (cid,))
        candidate = await c.fetchone()
        if not candidate:
            return HTMLResponse("Not found", status_code=404)

        a = await db.execute(
            "SELECT id, avg_score, verdict, summary, created_at, raw_json, is_resume FROM analyses WHERE candidate_id=? ORDER BY id DESC",
            (cid,),
        )
        analyses = await a.fetchall()
        iv = await db.execute(
            "SELECT id, created_at, log_json, report_json FROM interviews WHERE candidate_id=? ORDER BY id DESC LIMIT 1",
            (cid,),
        )
        last_iv = await iv.fetchone()

    candidate = dict(candidate)
    analyses = [dict(x) for x in analyses]
    # decode JSON reports for template rendering
    for it in analyses:
        raw = it.get("raw_json")
        try:
            it["report"] = json.loads(raw) if raw else None
        except Exception:
            it["report"] = None
        # build highlighted resume from annotations if any
        ann = []
        if it.get("report") and isinstance(it["report"], dict):
            anal = it["report"].get("Анализ") or {}
            # 1) Новый формат: внутри "Оценка требований" у каждого требования есть массив "подсветка"
            if isinstance(anal, dict) and isinstance(
                anal.get("Оценка требований"), dict
            ):
                grades = anal.get("Оценка требований") or {}
                for req_text, obj in grades.items():
                    if not isinstance(obj, dict):
                        continue
                    hls = obj.get("подсветка") or []
                    if not isinstance(hls, list):
                        continue
                    # score-based tone mapping for all highlights of this requirement
                    score = None
                    try:
                        score = int(obj.get("балл"))
                    except Exception:
                        score = None
                    tone = None
                    if isinstance(score, int):
                        if score > 60:
                            tone = "good"
                        elif score >= 30:
                            tone = "warn"
                        else:
                            tone = "bad"
                    req_comment = str(obj.get("комментарий") or "").strip()
                    for h in hls:
                        if not isinstance(h, dict):
                            continue
                        # enrich highlight with requirement context
                        hh = dict(h)
                        hh["требование"] = str(req_text)
                        if tone:
                            hh["тон"] = tone
                        if req_comment:
                            hh["комментарий"] = req_comment
                        ann.append(hh)
            # 2) Backward-compat: прежнее поле "Аннотации" (список)
            if isinstance(anal, dict) and isinstance(anal.get("Аннотации"), list):
                ann.extend(anal.get("Аннотации"))
        if not ann:
            ann = None
        it["resume_html"] = _build_highlighted_resume(
            candidate.get("raw_resume_text") or "", ann
        )
    # prepare last interview content
    last_interview = None
    if last_iv:
        try:
            lj = last_iv["log_json"]
            rj = last_iv["report_json"]
            last_interview = {
                "id": last_iv["id"],
                "created_at": last_iv["created_at"],
                "log": json.loads(lj) if lj else None,
                "report": json.loads(rj) if rj else None,
            }
        except Exception:
            last_interview = {"id": last_iv["id"], "created_at": last_iv["created_at"], "log": None, "report": None}

    return templates.TemplateResponse(
        "candidate.html",
        {
            "request": request,
            "c": candidate,
            "analyses": analyses,
            "last_interview": last_interview,
            "STATUS_CHOICES": STATUS_CHOICES,
            "status_label": status_label(candidate.get("candidate_status")),
            "call_base_url": CALL_BASE_URL,
        },
    )


@app.post("/candidates/{cid}/status")
async def update_status(cid: int, new_status: str = Form(...)):
    if new_status not in [k for k, _ in STATUS_CHOICES]:
        return HTMLResponse("Bad status", status_code=400)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE candidates SET candidate_status=? WHERE id=?", (new_status, cid)
        )
        await db.commit()
    return RedirectResponse(url=f"/candidates/{cid}", status_code=303)


@app.post("/candidates/{cid}/delete")
async def delete_candidate(cid: int):
    async with aiosqlite.connect(DB_PATH) as db:
        # delete analyses first (no ON DELETE CASCADE in schema)
        await db.execute("DELETE FROM analyses WHERE candidate_id=?", (cid,))
        # delete candidate
        await db.execute("DELETE FROM candidates WHERE id=?", (cid,))
        await db.commit()
    return RedirectResponse(url="/", status_code=303)


@app.get("/stats", response_class=HTMLResponse)
async def stats(request: Request):
    import json as _json
    import re as _re

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        # распределение статусов
        cur = await db.execute(
            "SELECT candidate_status AS status, COUNT(*) AS cnt FROM candidates GROUP BY candidate_status"
        )
        by_status_rows = [dict(r) for r in await cur.fetchall()]

        # последние анализы по каждому кандидату с данными кандидата
        cur2 = await db.execute(
            """
            SELECT c.id as cid, c.full_name, c.gender, c.address, c.age,
                   a.avg_score, a.summary, a.raw_json
            FROM candidates c
            JOIN (
              SELECT candidate_id, MAX(id) AS last_id
              FROM analyses GROUP BY candidate_id
            ) last ON last.candidate_id = c.id
            JOIN analyses a ON a.id = last.last_id
            """
        )
        last_rows = [dict(r) for r in await cur2.fetchall()]

    # avg score
    scores = [r["avg_score"] for r in last_rows if r.get("avg_score") is not None]
    avg_all = (sum(scores) / len(scores)) if scores else None

    # by status
    total = sum(r["cnt"] for r in by_status_rows) or 0
    by_status = [
        {
            "label": status_label(r["status"]),
            "code": r["status"],
            "cnt": r["cnt"],
            "pct": (r["cnt"] / total * 100 if total else 0),
        }
        for r in by_status_rows
    ]

    # vacancy, gender, city aggregates
    vacancy_counts: dict[str, int] = {}
    gender_counts: dict[str, int] = {}
    city_counts: dict[str, int] = {}
    age_by_vacancy: dict[str, list[int]] = {}

    def city_from_address(addr: str | None) -> str:
        if not addr:
            return "—"
        s = str(addr).strip()
        for sep in [",", "—", "-", ";", "/"]:
            if sep in s:
                s = s.split(sep, 1)[0]
                break
        return s.strip() or "—"

    for r in last_rows:
        # gender
        g = (r.get("gender") or "не указан").strip().lower()
        gender_counts[g] = gender_counts.get(g, 0) + 1
        # city
        city = city_from_address(r.get("address"))
        city_counts[city] = city_counts.get(city, 0) + 1

        # chosen vacancy from raw_json
        chosen = None
        try:
            rep = _json.loads(r.get("raw_json") or "")
            if isinstance(rep, dict):
                anal = rep.get("Анализ") or {}
                if isinstance(anal, dict):
                    v = anal.get("Выбранная вакансия")
                    if isinstance(v, str) and v.strip():
                        chosen = v.strip()
        except Exception:
            pass
        if not chosen:
            s = (r.get("summary") or "").strip()
            m = _re.search(
                r"ваканси[ие]\s+['\"]([^'\"]+)['\"]", s, flags=_re.IGNORECASE
            )
            if m:
                chosen = m.group(1).strip()
        chosen = chosen or "—"
        vacancy_counts[chosen] = vacancy_counts.get(chosen, 0) + 1

        # age per vacancy
        try:
            age = int(r["age"]) if r.get("age") is not None else None
        except Exception:
            age = None
        if age is not None:
            age_by_vacancy.setdefault(chosen, []).append(age)

    vac_labels = list(vacancy_counts.keys())
    vac_values = [vacancy_counts[k] for k in vac_labels]
    gen_labels = list(gender_counts.keys())
    gen_values = [gender_counts[k] for k in gen_labels]
    top_cities = sorted(city_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    city_labels = [c for c, _ in top_cities]
    city_values = [n for _, n in top_cities]
    age_labels = []
    age_values = []
    for k, arr in age_by_vacancy.items():
        if arr:
            age_labels.append(k)
            age_values.append(sum(arr) / len(arr))

    return templates.TemplateResponse(
        "stats.html",
        {
            "request": request,
            "by_status": by_status,
            "avg_all": avg_all,
            "total": total,
            "vac_labels": _json.dumps(vac_labels, ensure_ascii=False),
            "vac_values": _json.dumps(vac_values, ensure_ascii=False),
            "gen_labels": _json.dumps(gen_labels, ensure_ascii=False),
            "gen_values": _json.dumps(gen_values, ensure_ascii=False),
            "city_labels": _json.dumps(city_labels, ensure_ascii=False),
            "city_values": _json.dumps(city_values, ensure_ascii=False),
            "age_labels": _json.dumps(age_labels, ensure_ascii=False),
            "age_values": _json.dumps(age_values, ensure_ascii=False),
        },
    )


@app.get("/logs", response_class=HTMLResponse)
async def logs(request: Request, limit: int = 200):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM tg_logs ORDER BY id DESC LIMIT ?", (min(limit, 1000),)
        )
        rows = [dict(r) for r in await cur.fetchall()]
    return templates.TemplateResponse("logs.html", {"request": request, "rows": rows})


# --------- Requirements management ---------
SUPPORTED_REQ_EXTS = {".pdf", ".docx", ".txt", ".doc", ".rtf", ".odt"}


def _list_requirements():
    files = []
    try:
        for name in os.listdir(REQ_DIR):
            full = os.path.join(REQ_DIR, name)
            if os.path.isfile(full):
                files.append(name)
    except Exception:
        pass
    files.sort()
    return files


@app.get("/requirements", response_class=HTMLResponse)
async def requirements_page(request: Request):
    files = _list_requirements()
    return templates.TemplateResponse(
        "requirements.html",
        {
            "request": request,
            "files": files,
            "req_dir": REQ_DIR,
            "exts": ", ".join(sorted(SUPPORTED_REQ_EXTS)),
        },
    )


@app.post("/requirements/upload")
async def requirements_upload(files: List[UploadFile] = File(...)):
    for uf in files:
        name = os.path.basename(uf.filename or "")
        if not name:
            continue
        _, ext = os.path.splitext(name)
        if ext.lower() not in SUPPORTED_REQ_EXTS:
            continue
        dest = os.path.join(REQ_DIR, name)
        # overwrite allowed
        content = await uf.read()
        with open(dest, "wb") as f:
            f.write(content)
    return RedirectResponse(url="/requirements", status_code=303)


@app.post("/requirements/delete")
async def requirements_delete(filename: str = Form(...)):
    # prevent path traversal
    name = os.path.basename(filename)
    target = os.path.join(REQ_DIR, name)
    if os.path.isfile(target):
        try:
            os.remove(target)
        except Exception:
            pass
    return RedirectResponse(url="/requirements", status_code=303)


@app.post("/candidates/upload")
async def candidates_upload(
    background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)
):
    job_sets = _load_all_job_sets()
    if not job_sets:
        return RedirectResponse(url="/requirements", status_code=303)

    saved_any = False
    for uf in files:
        name = os.path.basename(uf.filename or "").strip()
        if not name:
            continue
        # save file
        os.makedirs(FILES_DIR, exist_ok=True)
        dest = os.path.join(FILES_DIR, name)
        data = await uf.read()
        with open(dest, "wb") as f:
            f.write(data)

        # extract text
        resume_text = extract_text(dest)
        if not (resume_text or "").strip():
            continue

        # create pending candidate and enqueue background analysis
        try:
            from db import create_candidate_pending, save_analysis_for_candidate

            candidate_id = await create_candidate_pending(
                resume_path=dest, raw_resume_text=resume_text
            )
        except Exception:
            continue

        async def _bg_analyze_and_store(cid: int, text: str, sets: list[dict]):
            try:
                import asyncio as _asyncio
                from db import (
                    acquire_agent_lock,
                    release_agent_lock,
                    save_analysis_for_candidate,
                )

                # очередь: один агент обрабатывает по очереди
                locked = await acquire_agent_lock(
                    "screening_agent", timeout_sec=1800, poll_sec=0.5
                )
                if not locked:
                    return
                loop = _asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, lambda: agent.analyze_and_questions(text, sets)
                )
                # parse
                avg_score: int | None = None
                summary = ""
                try:
                    if isinstance(result, dict) and isinstance(
                        result.get("Анализ"), dict
                    ):
                        avg_score = int(result["Анализ"].get("Средняя оценка"))
                        summary = str(result["Анализ"].get("Общий вывод", ""))
                except Exception:
                    pass
                status = (
                    (result.get("Статус") or {}) if isinstance(result, dict) else {}
                )
                is_resume = bool(status.get("Это резюме", True))
                await save_analysis_for_candidate(
                    candidate_id=cid,
                    avg_score=avg_score if isinstance(avg_score, int) else -1,
                    verdict=(
                        "positive"
                        if (
                            isinstance(avg_score, int) and avg_score >= RESUME_THRESHOLD
                        )
                        else "negative"
                    ),
                    summary=summary or "—",
                    raw_json=result,
                    is_resume=is_resume,
                    full_name=(
                        (result.get("Кандидат", {}) or {}).get("ФИО")
                        if isinstance(result, dict)
                        else None
                    ),
                    age=(
                        (result.get("Кандидат", {}) or {}).get("Возраст")
                        if isinstance(result, dict)
                        else None
                    ),
                    gender=(
                        (result.get("Кандидат", {}) or {}).get("Пол")
                        if isinstance(result, dict)
                        else None
                    ),
                    phone=(
                        (result.get("Кандидат", {}) or {}).get("Телефон")
                        if isinstance(result, dict)
                        else None
                    ),
                    address=(
                        (result.get("Кандидат", {}) or {}).get("Адрес")
                        if isinstance(result, dict)
                        else None
                    ),
                )
            except Exception:
                pass
            finally:
                try:
                    await release_agent_lock("screening_agent")
                except Exception:
                    pass

        background_tasks.add_task(
            _bg_analyze_and_store, candidate_id, resume_text, job_sets
        )
        saved_any = True

    return RedirectResponse(url="/", status_code=303)


def _load_all_job_sets():
    paths = set(parse_job_paths_env(JOB_DESCRIPTION_PATHS))
    for p in parse_job_paths_env(JOB_DESCRIPTION_PATH_LEGACY):
        paths.add(p)
    for p in list_requirement_files(REQ_DIR):
        paths.add(p)
    return load_job_requirements_many(sorted(paths))


# --------- WebRTC: call page + offer endpoint ---------
@app.get("/call/{cid}", response_class=HTMLResponse)
async def call_page(request: Request, cid: int):
    return templates.TemplateResponse("call.html", {"request": request, "cid": cid})


@app.post("/webrtc/offer/{cid}")
async def webrtc_offer(cid: int, payload: dict):
    import uuid, asyncio

    offer = RTCSessionDescription(sdp=payload.get("sdp"), type=payload.get("type"))
    pc = RTCPeerConnection(RTC_CONF)
    pid = f"{cid}-{uuid.uuid4()}"
    peers[pid] = pc
    peers_by_cid.setdefault(cid, set()).add(pc)
    recorder = MediaBlackhole()

    @pc.on("track")
    async def on_track(track):
        # Attach handlers for inbound audio/video to verify we receive media
        sess = await _get_or_create_session(cid)
        if track.kind == "audio":
            local_audio = relay.subscribe(track)

            async def read_audio():
                while True:
                    try:
                        frame = await local_audio.recv()
                        # convert to mono float32 numpy
                        import numpy as _np

                        samples = frame.to_ndarray()
                        # shape could be (channels, samples) or (samples,)
                        if samples.ndim == 2:
                            # average channels
                            mono = samples.mean(axis=0)
                        else:
                            mono = samples
                        # normalize to [-1,1] float32 if int16
                        if mono.dtype != _np.float32:
                            # assume int16
                            mono = mono.astype(_np.float32) / 32768.0
                        # resample to 16k from frame.sample_rate
                        sr = getattr(frame, "sample_rate", 48000) or 48000
                        audio16k = _resample_to_16k(mono, sr)
                        # accumulate ~1s window like CLI before VAD
                        if 'sec_buf' not in read_audio.__dict__:
                            read_audio.sec_buf = []
                            read_audio.sec_len = 0
                        read_audio.sec_buf.append(audio16k)
                        read_audio.sec_len += audio16k.size
                        if read_audio.sec_len < SAMPLERATE * CHUNK_SEC:
                            continue
                        # build 1-second chunk
                        sec_chunk = _np.concatenate(read_audio.sec_buf)
                        read_audio.sec_buf = []
                        read_audio.sec_len = 0

                        use_silero = (vad_model is not None)
                        # базовый RMS как резерв — более строгий, чем раньше (по умолчанию 0.01)
                        rms_val = float(_np.sqrt(_np.mean(sec_chunk * sec_chunk)))
                        has_speech_rms = rms_val > RMS_GATE
                        has_speech_silero = False
                        if use_silero:
                            try:
                                x = torch.tensor(sec_chunk, dtype=torch.float32, device=_VAD_DEVICE)
                                ts = get_speech_timestamps(x, vad_model, sampling_rate=SAMPLERATE)
                                has_speech_silero = len(ts) > 0
                            except Exception:
                                has_speech_silero = False
                        # комбинированное решение: Silero ИЛИ RMS-порог
                        has_speech = has_speech_silero or has_speech_rms

                        if DEBUG_VAD:
                            try:
                                await _ws_broadcast(sess, {"role": "system", "text": f"VAD 1s: rms={rms_val:.4f} has_speech={has_speech} (silero={has_speech_silero}, rms_gate={has_speech_rms})"})
                            except Exception:
                                pass

                        if has_speech:
                            sess.audio_buffer.append(sec_chunk)
                            sess.silent_chunks = 0
                        else:
                            sess.silent_chunks += 1
                        # when we had speech and now enough silence, process
                        if sess.silent_chunks >= PAUSE_CHUNKS and sess.audio_buffer:
                            chunk = _np.concatenate(sess.audio_buffer)
                            sess.audio_buffer.clear()
                            sess.silent_chunks = 0
                            # skip too-short chunks (<1s) to improve ASR quality
                            if chunk.size < SAMPLERATE:
                                continue
                            if DEBUG_VAD:
                                try:
                                    rms_chunk = float(_np.sqrt(_np.mean(chunk * chunk)))
                                    dur = chunk.size / SAMPLERATE
                                    await _ws_broadcast(sess, {"role": "system", "text": f"🎙️ сегмент {dur:.2f}s, rms={rms_chunk:.4f} → ASR"})
                                except Exception:
                                    pass
                            await _transcribe_and_reply(sess, chunk)
                    except Exception:
                        # MediaStreamError is raised when stream ends; exit loop quietly
                        break

            audio_task = asyncio.create_task(read_audio())
            sess.tasks.add(audio_task)
        elif track.kind == "video":
            local_video = relay.subscribe(track)

            async def read_video():
                while True:
                    try:
                        frame = await local_video.recv()
                        # TODO: plug video processing here later
                    except Exception:
                        break

            video_task = asyncio.create_task(read_video())
            sess.tasks.add(video_task)
        await recorder.start()

        @track.on("ended")
        async def on_ended():
            await recorder.stop()
            # cancel reader tasks if any
            try:
                for t in list(sess.tasks):
                    try:
                        t.cancel()
                    except Exception:
                        pass
                sess.tasks.clear()
            except Exception:
                pass

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

