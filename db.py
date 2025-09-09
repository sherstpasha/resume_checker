# db.py
import os
import json
import aiosqlite

DB_PATH = os.getenv("DB_PATH", "./hr_bot.sqlite3")

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS recruiters (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tg_user_id INTEGER UNIQUE,
  tg_chat_id INTEGER,
  name TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS candidates (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  full_name TEXT,
  age INTEGER,
  gender TEXT,
  phone TEXT,
  address TEXT,
  resume_path TEXT,
  raw_resume_text TEXT,
  candidate_status TEXT DEFAULT 'screen_pending', -- screen_pending|screen_failed|screen_passed|interview_passed|interview_failed
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

CREATE TABLE IF NOT EXISTS tg_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id INTEGER,
  user_id INTEGER,
  username TEXT,
  event TEXT,
  payload TEXT,
  created_at TEXT DEFAULT (datetime('now'))
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
        await db.commit()


async def upsert_recruiter(
    tg_user_id: int, tg_chat_id: int, name: str | None = None
) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO recruiters(tg_user_id,tg_chat_id,name) VALUES(?,?,?) "
            "ON CONFLICT(tg_user_id) DO UPDATE SET tg_chat_id=excluded.tg_chat_id, name=COALESCE(excluded.name, recruiters.name)",
            (tg_user_id, tg_chat_id, name),
        )
        await db.commit()
        cur = await db.execute(
            "SELECT id FROM recruiters WHERE tg_user_id=?", (tg_user_id,)
        )
        row = await cur.fetchone()
        return int(row[0])


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
        cur = await db.execute(
            "INSERT INTO candidates(full_name,age,gender,phone,address,resume_path,raw_resume_text,candidate_status) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                None,
                None,
                None,
                None,
                None,
                resume_path,
                (raw_resume_text or "")[:50000],
                "screen_pending",
            ),
        )
        await db.commit()
        return int(cur.lastrowid)


async def save_analysis_for_candidate(
    *,
    candidate_id: int,
    avg_score: int,
    verdict: str,
    summary: str,
    raw_json: str | dict,
    is_resume: bool,
    full_name: str | None = None,
    age: int | None = None,
    gender: str | None = None,
    phone: str | None = None,
    address: str | None = None,
    raw_resume_text: str | None = None,
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

    async with aiosqlite.connect(DB_PATH) as db:
        # update candidate fields
        await db.execute(
            "UPDATE candidates SET full_name=COALESCE(?,full_name), age=COALESCE(?,age), gender=COALESCE(?,gender), "
            "phone=COALESCE(?,phone), address=COALESCE(?,address), raw_resume_text=COALESCE(?,raw_resume_text), candidate_status=? "
            "WHERE id=?",
            (
                full_name,
                age,
                gender,
                phone,
                address,
                (raw_resume_text or None),
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


async def acquire_agent_lock(name: str = "global", timeout_sec: int = 600, poll_sec: float = 0.5) -> bool:
    """Пытается захватить глобальную блокировку агента через SQLite. Возвращает True/False."""
    import time
    deadline = time.monotonic() + max(1, timeout_sec)
    while True:
        async with aiosqlite.connect(DB_PATH) as db:
            # на всякий случай обеспечим таблицу
            await db.execute("CREATE TABLE IF NOT EXISTS agent_locks(name TEXT PRIMARY KEY)")
            cur = await db.execute("INSERT OR IGNORE INTO agent_locks(name) VALUES (?)", (name,))
            await db.commit()
            if cur.rowcount == 1:
                return True
        if time.monotonic() >= deadline:
            return False
        # ждём и повторяем
        import asyncio
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
