import os
import json
import html
import aiosqlite
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List

DB_PATH = os.getenv("DB_PATH", "./hr_bot.sqlite3")
REQ_DIR = os.getenv("JOB_REQUIREMENTS_DIR", "./requirements")
os.makedirs(REQ_DIR, exist_ok=True)

app = FastAPI(title="HR Resume Bot — Admin")
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
    name="static",
)
templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)

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
                title = sp["comment"]
                if sp["req"]:
                    title = (sp["req"] + (" — " if sp["comment"] else "")) + sp[
                        "comment"
                    ]
                data_comment = html.escape(sp["comment"]) if sp["comment"] else ""
                data_req = html.escape(sp["req"]) if sp["req"] else ""
                data_tone = html.escape(sp["tone"]) if sp["tone"] else ""
                content = html.escape(line_text[sp["s"] : sp["e"]])
                parts.append(
                    f'<span class="hl {tone_cls}" title="{html.escape(title)}" data-comment="{data_comment}" data-req="{data_req}" data-tone="{data_tone}">{content}</span>'
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
        title = sp["comment"]
        if sp["req"]:
            title = (sp["req"] + (" — " if sp["comment"] else "")) + sp["comment"]
        content = html.escape(text[sp["s"] : sp["e"]])
        data_comment = html.escape(sp["comment"]) if sp["comment"] else ""
        data_req = html.escape(sp["req"]) if sp["req"] else ""
        data_tone = html.escape(sp["tone"]) if sp["tone"] else ""
        out.append(
            f'<span class="hl {tone_cls}" title="{html.escape(title)}" data-comment="{data_comment}" data-req="{data_req}" data-tone="{data_tone}">{content}</span>'
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
    return templates.TemplateResponse(
        "candidate.html",
        {
            "request": request,
            "c": candidate,
            "analyses": analyses,
            "STATUS_CHOICES": STATUS_CHOICES,
            "status_label": status_label(candidate.get("candidate_status")),
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


@app.get("/stats", response_class=HTMLResponse)
async def stats(request: Request):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        # распределение статусов
        cur = await db.execute(
            "SELECT candidate_status AS status, COUNT(*) AS cnt FROM candidates GROUP BY candidate_status"
        )
        by_status = [dict(r) for r in await cur.fetchall()]

        # средний балл по последним анализам
        cur2 = await db.execute(
            """
            SELECT AVG(t.avg_score) AS avg_score
            FROM (
              SELECT candidate_id, MAX(id) AS last_id
              FROM analyses GROUP BY candidate_id
            ) last
            JOIN analyses t ON t.id = last.last_id
            """
        )
        avg_row = await cur2.fetchone()
        avg_all = (
            avg_row["avg_score"]
            if avg_row and avg_row["avg_score"] is not None
            else None
        )

    # подготовка для шаблона
    total = sum(r["cnt"] for r in by_status) or 0
    by_status = [
        {
            "label": status_label(r["status"]),
            "code": r["status"],
            "cnt": r["cnt"],
            "pct": (r["cnt"] / total * 100 if total else 0),
        }
        for r in by_status
    ]
    return templates.TemplateResponse(
        "stats.html",
        {
            "request": request,
            "by_status": by_status,
            "avg_all": avg_all,
            "total": total,
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
