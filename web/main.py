import os
import aiosqlite
from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

DB_PATH = os.getenv("DB_PATH", "./hr_bot.sqlite3")

app = FastAPI(title="HR Resume Bot — Admin")
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

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

    return templates.TemplateResponse("index.html", {
        "request": request,
        "rows": rows,
        "STATUS_CHOICES": STATUS_CHOICES,
        "q": q or "",
        "filter_status": status or "",
    })

@app.get("/candidates/{cid}", response_class=HTMLResponse)
async def candidate_detail(request: Request, cid: int):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        c = await db.execute("SELECT * FROM candidates WHERE id=?", (cid,))
        candidate = await c.fetchone()
        if not candidate:
            return HTMLResponse("Not found", status_code=404)

        a = await db.execute(
            "SELECT id, avg_score, verdict, summary, created_at FROM analyses WHERE candidate_id=? ORDER BY id DESC",
            (cid,)
        )
        analyses = await a.fetchall()

    candidate = dict(candidate)
    analyses = [dict(x) for x in analyses]
    return templates.TemplateResponse("candidate.html", {
        "request": request,
        "c": candidate,
        "analyses": analyses,
        "STATUS_CHOICES": STATUS_CHOICES,
        "status_label": status_label(candidate.get("candidate_status")),
    })

@app.post("/candidates/{cid}/status")
async def update_status(cid: int, new_status: str = Form(...)):
    if new_status not in [k for k, _ in STATUS_CHOICES]:
        return HTMLResponse("Bad status", status_code=400)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE candidates SET candidate_status=? WHERE id=?", (new_status, cid))
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
        avg_all = (avg_row["avg_score"] if avg_row and avg_row["avg_score"] is not None else None)

    # подготовка для шаблона
    total = sum(r["cnt"] for r in by_status) or 0
    by_status = [
        {"label": status_label(r["status"]), "code": r["status"], "cnt": r["cnt"], "pct": (r["cnt"] / total * 100 if total else 0)}
        for r in by_status
    ]
    return templates.TemplateResponse("stats.html", {
        "request": request,
        "by_status": by_status,
        "avg_all": avg_all,
        "total": total,
    })

@app.get("/logs", response_class=HTMLResponse)
async def logs(request: Request, limit: int = 200):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM tg_logs ORDER BY id DESC LIMIT ?", (min(limit, 1000),))
        rows = [dict(r) for r in await cur.fetchall()]
    return templates.TemplateResponse("logs.html", {"request": request, "rows": rows})
