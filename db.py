# db.py
import os
import json
import time
import asyncio
import aiosqlite
from utils import city_from_address

DB_PATH = os.getenv("DB_PATH", "./hr_bot.sqlite3")

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS candidates (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  full_name TEXT,
  age INTEGER,
  gender TEXT,
  phone TEXT,
  address TEXT,
  city TEXT,
  resume_path TEXT,
  raw_resume_text TEXT,
  candidate_status TEXT DEFAULT 'screen_pending', -- screen_pending|screen_failed|screen_passed
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS analyses (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  candidate_id INTEGER,
  avg_score INTEGER,
  verdict TEXT,
  summary TEXT,
  raw_json TEXT,
  is_resume INTEGER,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(candidate_id) REFERENCES candidates(id)
);

-- Простая таблица для блокировок агента (mutex через уникальный ключ)
CREATE TABLE IF NOT EXISTS agent_locks (
  name TEXT PRIMARY KEY
);
"""


async def init_db(db_path: str | None = None):
    path = db_path or DB_PATH
    async with aiosqlite.connect(path) as db:
        await db.executescript(SCHEMA)
        # Lightweight migration: add missing columns without dropping data
        # 1) add city column if absent
        try:
            cur = await db.execute("PRAGMA table_info(candidates)")
            cols = [r[1] for r in await cur.fetchall()]
            if "city" not in cols:
                await db.execute("ALTER TABLE candidates ADD COLUMN city TEXT")
                await db.commit()
        except Exception:
            pass
        await db.commit()


async def create_candidate_and_analysis(
    *,
    full_name: str | None,
    age: int | None,
    gender: str | None,
    phone: str | None,
    address: str | None,
    resume_path: str | None,
    raw_resume_text: str | None,
    avg_score: int,
    verdict: str,
    summary: str,
    raw_json: str | dict,
    is_resume: bool,
    recruiter_id: int | None = None,
) -> tuple[int, int]:
    """Создаёт кандидата и запись анализа. Возвращает (candidate_id, analysis_id)."""
    if isinstance(raw_json, (dict, list)):
        raw_json = json.dumps(raw_json, ensure_ascii=False)

    thr = int(os.getenv("RESUME_THRASH", "75"))
    auto_status = (
        "screen_passed"
        if (is_resume and isinstance(avg_score, int) and avg_score >= thr)
        else "screen_failed"
    )

    async with aiosqlite.connect(DB_PATH) as db:
        # candidate
        cur = await db.execute(
            "INSERT INTO candidates(full_name,age,gender,phone,address,resume_path,raw_resume_text,candidate_status) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                full_name,
                age,
                gender,
                phone,
                address,
                resume_path,
                (raw_resume_text or "")[:50000],
                auto_status,
            ),
        )
        await db.commit()
        candidate_id = cur.lastrowid

        # analysis
        cur2 = await db.execute(
            "INSERT INTO analyses(candidate_id,avg_score,verdict,summary,raw_json,is_resume) "
            "VALUES (?,?,?,?,?,?)",
            (
                candidate_id,
                int(avg_score) if isinstance(avg_score, int) else -1,
                verdict,
                summary[:2000],
                raw_json,
                1 if is_resume else 0,
            ),
        )
        await db.commit()
        analysis_id = cur2.lastrowid

        return candidate_id, analysis_id


async def create_candidate_pending(
    *,
    resume_path: str | None,
    raw_resume_text: str | None,
) -> int:
    """Создаёт кандидата со статусом screen_pending и возвращает candidate_id."""
    async with aiosqlite.connect(DB_PATH) as db:
        # Do not store PII; keep only technical fields
        cur = await db.execute(
            "INSERT INTO candidates(full_name,age,gender,phone,address,city,resume_path,raw_resume_text,candidate_status) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                None,  # full_name (deprecated)
                None,  # age will be set after analysis
                None,  # gender will be set after analysis
                None,  # phone (deprecated)
                None,  # address (deprecated)
                None,  # city will be set after analysis
                resume_path,
                "",    # raw_resume_text not stored for privacy
                "screen_pending",
            ),
        )
        await db.commit()
        return int(cur.lastrowid)


async def update_candidate_file(candidate_id: int, resume_path: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE candidates SET resume_path=? WHERE id=?",
            (resume_path, candidate_id),
        )
        await db.commit()


async def save_analysis_for_candidate(
    *,
    candidate_id: int,
    avg_score: int,
    verdict: str,
    summary: str,
    raw_json: str | dict,
    is_resume: bool,
    age: int | None = None,
    gender: str | None = None,
    city: str | None = None,
    **_: dict,
) -> int:
    """Обновляет карточку кандидата полями из анализа и создаёт запись анализа. Возвращает analysis_id."""
    if isinstance(raw_json, (dict, list)):
        raw_json = json.dumps(raw_json, ensure_ascii=False)

    thr = int(os.getenv("RESUME_THRASH", "75"))
    new_status = (
        "screen_passed"
        if (is_resume and isinstance(avg_score, int) and avg_score >= thr)
        else "screen_failed"
    )

    # city is expected to be provided directly (no PII stored)

    async with aiosqlite.connect(DB_PATH) as db:
        # update candidate fields (no PII)
        await db.execute(
            "UPDATE candidates SET age=COALESCE(?,age), gender=COALESCE(?,gender), city=COALESCE(?,city), candidate_status=? WHERE id=?",
            (
                age,
                gender,
                city,
                new_status,
                candidate_id,
            ),
        )
        await db.commit()

        # insert analysis
        cur2 = await db.execute(
            "INSERT INTO analyses(candidate_id,avg_score,verdict,summary,raw_json,is_resume) VALUES (?,?,?,?,?,?)",
            (
                candidate_id,
                int(avg_score) if isinstance(avg_score, int) else -1,
                verdict,
                (summary or "")[:2000],
                raw_json,
                1 if is_resume else 0,
            ),
        )
        await db.commit()
        return int(cur2.lastrowid)


async def acquire_agent_lock(
    name: str = "global", timeout_sec: int = 600, poll_sec: float = 0.5
) -> bool:
    """Пытается захватить глобальную блокировку агента через SQLite. Возвращает True/False."""
    deadline = time.monotonic() + max(1, timeout_sec)
    while True:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "CREATE TABLE IF NOT EXISTS agent_locks(name TEXT PRIMARY KEY)"
            )
            cur = await db.execute(
                "INSERT OR IGNORE INTO agent_locks(name) VALUES (?)", (name,)
            )
            await db.commit()
            if cur.rowcount == 1:
                return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(poll_sec)


async def release_agent_lock(name: str = "global") -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM agent_locks WHERE name=?", (name,))
        await db.commit()


async def log_event(
    chat_id: int | None,
    user_id: int | None,
    username: str | None,
    event: str,
    payload: str | dict | None = None,
):
    if isinstance(payload, (dict, list)):
        payload = json.dumps(payload, ensure_ascii=False)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO tg_logs(chat_id,user_id,username,event,payload) VALUES (?,?,?,?,?)",
            (chat_id, user_id, username, event, payload),
        )
        await db.commit()
