import os
import json as _json
import re as _re
import html
import aiosqlite
from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from utils import SUPPORTED_REQ_EXTS, build_highlighted_resume, city_from_address


load_dotenv()

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
agent = None

STATUS_CHOICES = [
    ("screen_pending", "Ожидает скрининг"),
    ("screen_failed", "Не прошёл скрининг"),
    ("screen_passed", "Прошёл скрининг"),
]


def status_label(code: str) -> str:
    for k, v in STATUS_CHOICES:
        if k == code:
            return v
    return code or "—"




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
        it["resume_html"] = build_highlighted_resume(
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
