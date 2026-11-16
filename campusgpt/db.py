"""SQLite store for chat logs and feedback (stdlib only). Never store real student records here."""

import json
import os
import sqlite3
import time
import uuid
from contextlib import closing
from urllib.parse import urlparse

from .config import DATA

DB_PATH = os.getenv("DB_PATH", str(DATA / "campusgpt.db"))
SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_log(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, username TEXT, role TEXT,
    question TEXT, answer TEXT, abstained INTEGER, tool TEXT, sources TEXT, latency REAL, conversation_id TEXT);
CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, username TEXT, chat_id INTEGER,
  rating INTEGER, comment TEXT);
CREATE TABLE IF NOT EXISTS events(url TEXT PRIMARY KEY, title TEXT NOT NULL, date TEXT NOT NULL, category TEXT NOT NULL,
    summary TEXT NOT NULL, text TEXT NOT NULL, fetched_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, name TEXT NOT NULL, role TEXT NOT NULL,
    password_hash TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS faculty_items(id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT NOT NULL,
    course TEXT NOT NULL, title TEXT NOT NULL, task TEXT NOT NULL, content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft', created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_faculty_items_owner ON faculty_items(owner, status, updated_at);
"""


def _conn():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    columns = {r[1] for r in c.execute("PRAGMA table_info(chat_log)")}
    if "conversation_id" not in columns:
        c.execute("ALTER TABLE chat_log ADD COLUMN conversation_id TEXT")
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_user_conversation ON chat_log(username, conversation_id, id)"
    )
    return c


def log_chat(username, role, question, result, conversation_id=None) -> int:
    conversation_id = conversation_id or str(uuid.uuid4())
    with closing(_conn()) as c, c:
        cur = c.execute(
            "INSERT INTO chat_log(ts,username,role,question,answer,abstained,tool,sources,latency,conversation_id) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                time.time(),
                username,
                role,
                question,
                result.get("answer", ""),
                int(bool(result.get("abstained"))),
                result.get("tool"),
                json.dumps(result.get("sources", [])),
                result.get("latency_s", 0.0),
                conversation_id,
            ),
        )
        return cur.lastrowid


def upsert_events(items):
    rows = []
    for item in items:
        url = str(item.get("url", "")).strip()
        if urlparse(url).hostname != "christuniversity.in":
            continue
        if not item.get("title") or not item.get("text"):
            continue
        rows.append(
            (
                url,
                str(item["title"])[:300],
                str(item.get("date", "")),
                str(item.get("category", "events")),
                str(item.get("summary", ""))[:1000],
                str(item["text"])[:20000],
                str(item.get("fetched_at", "")),
            )
        )
    with closing(_conn()) as c, c:
        c.executemany(
            """INSERT INTO events(url,title,date,category,summary,text,fetched_at) VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(url) DO UPDATE SET title=excluded.title,date=excluded.date,category=excluded.category,
            summary=excluded.summary,text=excluded.text,fetched_at=excluded.fetched_at""",
            rows,
        )
    return len(rows)


def refresh_events(events_dir=None):
    from .config import DATA

    events_dir = events_dir or DATA / "web" / "events"
    items = []
    for path in sorted(events_dir.glob("*.json")) if events_dir.exists() else []:
        try:
            items.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return upsert_events(items)


def list_events(category="", upcoming=False, query="", today=None):
    clauses, values = [], []
    if category:
        clauses.append("category = ?")
        values.append(category)
    if upcoming:
        from datetime import date

        clauses.append("date >= ? AND date != ''")
        values.append(today or date.today().isoformat())
    if query.strip():
        clauses.append("(title LIKE ? OR summary LIKE ? OR text LIKE ?)")
        term = f"%{query.strip()}%"
        values.extend((term, term, term))
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with closing(_conn()) as c:
        rows = c.execute(
            "SELECT title,date,url,category,summary,text,fetched_at FROM events"
            + where
            + " ORDER BY CASE WHEN date = '' THEN 1 ELSE 0 END, date, title",
            values,
        ).fetchall()
        return [dict(row) for row in rows]


def create_user(username, name, role, password_hash):
    if role not in {"student", "faculty", "admin"}:
        raise ValueError("role must be student, faculty, or admin")
    username = username.strip().lower()
    with closing(_conn()) as c, c:
        c.execute(
            "INSERT INTO users(username,name,role,password_hash,created_at) VALUES(?,?,?,?,?)",
            (username, name.strip(), role, password_hash, time.time()),
        )


def get_user(username):
    with closing(_conn()) as c:
        row = c.execute(
            "SELECT username,name,role,password_hash FROM users WHERE username=?",
            (username.strip().lower(),),
        ).fetchone()
        return dict(row) if row else None


def list_conversations(username, limit=50):
    with closing(_conn()) as c:
        rows = c.execute(
            """SELECT conversation_id, MAX(ts) AS updated_at,
            (SELECT question FROM chat_log l2 WHERE l2.username=chat_log.username
             AND l2.conversation_id=chat_log.conversation_id ORDER BY id LIMIT 1) AS title,
            COUNT(*) AS messages FROM chat_log WHERE username=? AND conversation_id IS NOT NULL
            GROUP BY conversation_id ORDER BY updated_at DESC LIMIT ?""",
            (username, limit),
        ).fetchall()
        return [dict(row) for row in rows]


def conversation_history(username, conversation_id):
    with closing(_conn()) as c:
        rows = c.execute(
            """SELECT id,ts,question,answer,abstained,tool,sources,latency
            FROM chat_log WHERE username=? AND conversation_id=? ORDER BY id""",
            (username, conversation_id),
        ).fetchall()
        return [dict(row) for row in rows]


def create_faculty_item(owner, course, title, task, content):
    now = time.time()
    with closing(_conn()) as c, c:
        cursor = c.execute(
            """INSERT INTO faculty_items(owner,course,title,task,content,status,created_at,updated_at)
               VALUES(?,?,?,?,?,'draft',?,?)""",
            (owner, course[:160], title[:200], task[:160], content[:30000], now, now),
        )
        return cursor.lastrowid


def list_faculty_items(owner, status=""):
    with closing(_conn()) as c:
        if status:
            rows = c.execute(
                "SELECT * FROM faculty_items WHERE owner=? AND status=? ORDER BY updated_at DESC",
                (owner, status),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM faculty_items WHERE owner=? ORDER BY updated_at DESC",
                (owner,),
            ).fetchall()
        return [dict(row) for row in rows]


def set_faculty_item_status(item_id, owner, status):
    if status not in {"approved", "rejected"}:
        raise ValueError("status must be approved or rejected")
    with closing(_conn()) as c, c:
        cursor = c.execute(
            "UPDATE faculty_items SET status=?,updated_at=? WHERE id=? AND owner=? AND status='draft'",
            (status, time.time(), item_id, owner),
        )
        return cursor.rowcount == 1


def list_faculty_items_for_admin():
    with closing(_conn()) as c:
        rows = c.execute(
            "SELECT id,status,created_at,updated_at FROM faculty_items ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def add_feedback(username, chat_id, rating, comment="") -> int:
    if rating not in (-1, 1):
        raise ValueError("rating must be 1 (helpful) or -1 (not helpful)")
    with closing(_conn()) as c, c:
        return c.execute(
            "INSERT INTO feedback(ts,username,chat_id,rating,comment) VALUES(?,?,?,?,?)",
            (time.time(), username, chat_id, rating, comment[:1000]),
        ).lastrowid


def recent_logs(limit=100):
    with closing(_conn()) as c:
        rows = c.execute(
            """SELECT l.id, l.ts, l.username, l.role, l.question, l.answer, l.abstained, l.tool,
                            (SELECT rating FROM feedback f WHERE f.chat_id = l.id ORDER BY f.id DESC LIMIT 1) AS rating
                            FROM chat_log l ORDER BY l.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def recent_feedback(limit=500):
    with closing(_conn()) as c:
        rows = c.execute(
            """SELECT id,ts,username,chat_id,rating,comment FROM feedback
                            ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def stats():
    with closing(_conn()) as c:
        q = lambda s: c.execute(s).fetchone()[0] or 0
        return dict(
            questions=q("SELECT COUNT(*) FROM chat_log"),
            abstained=q("SELECT COUNT(*) FROM chat_log WHERE abstained=1"),
            attendance_tool=q("SELECT COUNT(*) FROM chat_log WHERE tool='attendance'"),
            helpful=q("SELECT COUNT(*) FROM feedback WHERE rating=1"),
            not_helpful=q("SELECT COUNT(*) FROM feedback WHERE rating=-1"),
            avg_latency_s=round(q("SELECT AVG(latency) FROM chat_log"), 3),
        )


def unanswered(limit=50):
    """Questions the bot abstained on or users rated unhelpful: the to-do list for adding documents."""
    with closing(_conn()) as c:
        rows = c.execute(
            """SELECT DISTINCT l.question FROM chat_log l LEFT JOIN feedback f ON f.chat_id = l.id
                            WHERE l.abstained = 1 OR f.rating = -1 ORDER BY l.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [r[0] for r in rows]
