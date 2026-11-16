"""FastAPI backend. Run: uvicorn campusgpt.api:app --reload."""

import asyncio
import csv
import io
import json
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import auth, db, documents, studio, tools
from .config import CORS_ORIGINS, CUR, INDEX, PROC, RAW, ROOT, validate_runtime_config
from .rag import CampusGPT

_refresh_lock = threading.Lock()
_refresh_state_lock = threading.Lock()
_login_attempts = defaultdict(deque)
_login_lock = threading.Lock()
refresh_state = {
    "state": "idle",
    "last_started": None,
    "last_finished": None,
    "events": 0,
    "error": None,
}


def _merge_calendar():
    path = CUR / "calendar.json"
    try:
        calendar = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except (OSError, json.JSONDecodeError):
        calendar = []
    calendar = [entry for entry in calendar if entry.get("source") != "CHRIST website"]
    for entry in calendar:
        entry.setdefault("source", "Academic calendar PDF")
    for event in db.list_events():
        if not event.get("date"):
            continue
        calendar.append(
            {
                "start": event["date"],
                "end": None,
                "title": event["title"],
                "type": event["category"],
                "source": "CHRIST website",
                "url": event["url"],
            }
        )
    calendar.sort(
        key=lambda entry: (entry.get("start", "9999"), entry.get("title", ""))
    )
    CUR.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(calendar, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return calendar


def _refresh_events(reindex=True):
    if not _refresh_lock.acquire(blocking=False):
        return {"state": "already_running"}
    with _refresh_state_lock:
        refresh_state.update(state="running", last_started=time.time(), error=None)
    try:
        from . import web_ingest

        web_ingest.main()
        count = db.refresh_events()
        _merge_calendar()
        if reindex:
            _reindex(dense=True)
        with _refresh_state_lock:
            refresh_state.update(
                state="complete", last_finished=time.time(), events=count, error=None
            )
        return dict(refresh_state)
    except Exception as exc:
        with _refresh_state_lock:
            refresh_state.update(
                state="failed", last_finished=time.time(), error=type(exc).__name__
            )
        raise
    finally:
        _refresh_lock.release()


def _nightly_refresh():
    _refresh_events(reindex=True)


@asynccontextmanager
async def lifespan(application):
    validate_runtime_config()
    auth.seed_development_users()
    db.refresh_events()
    _merge_calendar()
    from apscheduler.schedulers.background import BackgroundScheduler

    scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
    scheduler.add_job(
        _nightly_refresh,
        "cron",
        hour=2,
        minute=0,
        id="nightly-events",
        replace_existing=True,
    )
    scheduler.start()
    application.state.scheduler = scheduler
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)


app = FastAPI(title="CampusGPT API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)

ROSTER = [
    ("Student A", 42, 50),
    ("Student B", 36, 48),
    ("Student C", 47, 50),
    ("Student D", 33, 46),
    ("Student E", 40, 44),
    ("Student F", 29, 40),
    ("Student G", 45, 50),
    ("Student H", 38, 50),
]  # demo data only


@lru_cache
def bot() -> CampusGPT:
    return CampusGPT()


class Login(BaseModel):
    username: str
    password: str


class Chat(BaseModel):
    message: str
    conversation_id: str | None = None
    history: list = Field(default_factory=list)


class Studio(BaseModel):
    task: str
    text: str


class Calc(BaseModel):
    attended: int
    held: int
    target: float = 85


class FacultyDraft(BaseModel):
    course: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=200)
    task: str
    text: str = Field(min_length=1, max_length=20000)
    generate: bool = True


class FacultyDecision(BaseModel):
    status: str


@app.get("/health")
def health():
    chunk_file = PROC / "chunks.jsonl"
    chunks = (
        sum(1 for _ in chunk_file.open(encoding="utf-8")) if chunk_file.exists() else 0
    )
    return {
        "ok": True,
        "chunks": chunks,
        "dense": INDEX.exists() and any(INDEX.iterdir()),
    }


def _allow_login(client_ip):
    now = time.monotonic()
    with _login_lock:
        attempts = _login_attempts[client_ip]
        while attempts and now - attempts[0] > 60:
            attempts.popleft()
        if len(attempts) >= 5:
            return False
        attempts.append(now)
        return True


@app.post("/auth/login")
def login(b: Login, request: Request):
    if not _allow_login(request.client.host if request.client else "unknown"):
        raise HTTPException(429, "Too many login attempts. Try again in one minute.")
    user = auth.verify(b.username, b.password)
    if not user:
        raise HTTPException(401, "Incorrect ID or password")
    return {
        "token": auth.create_token(user["username"]),
        "role": user["role"],
        "name": user["name"],
    }


def _conversation_context(username, conversation_id):
    rows = db.conversation_history(username, conversation_id)
    messages = []
    for row in rows:
        messages.extend(
            (
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            )
        )
    return messages[-6:]


def _save_chat(result, message, user, conversation_id):
    result.pop("hits", None)
    result["chat_id"] = db.log_chat(
        user["sub"], user["role"], message, result, conversation_id
    )
    result["conversation_id"] = conversation_id
    return result


def _sse_chat(message, user, conversation_id):
    try:
        stream = bot().answer_stream(
            message,
            role=user["role"],
            history=_conversation_context(user["sub"], conversation_id),
        )
        while True:
            try:
                text = next(stream)
            except StopIteration as stop:
                result = _save_chat(stop.value, message, user, conversation_id)
                break
            yield (
                "event: token\ndata: "
                + json.dumps({"text": text}, ensure_ascii=False)
                + "\n\n"
            )
        yield "event: done\ndata: " + json.dumps(result, ensure_ascii=False) + "\n\n"
    except Exception as exc:
        yield (
            "event: error\ndata: "
            + json.dumps({"detail": f"Chat failed ({type(exc).__name__})."})
            + "\n\n"
        )


@app.post("/chat")
def chat(b: Chat, request: Request, user=Depends(auth.current_user)):
    conversation_id = b.conversation_id or str(uuid.uuid4())
    history = _conversation_context(user["sub"], conversation_id)
    if "text/event-stream" in request.headers.get("accept", ""):
        return StreamingResponse(
            _sse_chat(b.message, user, conversation_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    try:
        r = bot().answer(b.message, role=user["role"], history=history or b.history)
    except Exception as e:
        print("CHAT ERROR:", type(e).__name__)
        raise HTTPException(
            502,
            f"The language model call failed ({type(e).__name__}). Check backend configuration.",
        )
    return _save_chat(r, b.message, user, conversation_id)


@app.get("/history")
def history(conversation_id: str | None = None, user=Depends(auth.current_user)):
    if conversation_id:
        return {
            "conversation_id": conversation_id,
            "messages": db.conversation_history(user["sub"], conversation_id),
        }
    return {"conversations": db.list_conversations(user["sub"])}


@app.get("/events")
def events(
    category: str = "",
    upcoming: bool = False,
    q: str = "",
    user=Depends(auth.current_user),
):
    return db.list_events(category=category, upcoming=upcoming, query=q)


def _official_documents():
    documents = {}
    calendars = sorted(RAW.glob("ACADEMIC_CALENDAR*.pdf"))
    handbooks = sorted(
        path for path in RAW.glob("handbook_*.pdf") if "22_23" not in path.name.lower()
    )
    archived = sorted(
        path for path in RAW.glob("handbook_*.pdf") if "22_23" in path.name.lower()
    )
    if calendars:
        path = calendars[-1]
        documents["calendar"] = {
            "title": "Academic Calendar 2026–27",
            "filename": path.name,
            "path": path,
        }
    if handbooks:
        path = handbooks[-1]
        documents["handbook"] = {
            "title": "Student Handbook",
            "filename": path.name,
            "path": path,
        }
    if archived:
        path = archived[-1]
        documents["archived-handbook"] = {
            "title": "Archived Student Handbook 2022–23",
            "filename": path.name,
            "path": path,
        }
    for key, pattern, title in (
        ("internship", "*Internship*.pdf", "Internship Policy and Details"),
        ("cia-repeat", "*CIA*REPEAT*.pdf", "CIA Repeat Application Form"),
    ):
        matches = sorted(RAW.glob(pattern))
        if matches:
            path = matches[-1]
            documents[key] = {"title": title, "filename": path.name, "path": path}
    return documents


@app.get("/public-documents")
def public_documents():
    return [
        {
            "key": key,
            "title": item["title"],
            "filename": item["filename"],
            "url": f"/downloads/{key}",
        }
        for key, item in _official_documents().items()
    ]


@app.get("/downloads/{document_key}")
def download_public_document(document_key: str):
    item = _official_documents().get(document_key)
    if not item or not item["path"].is_file():
        raise HTTPException(404, "Official document not found")
    return FileResponse(
        item["path"], media_type="application/pdf", filename=item["filename"]
    )


class Feedback(BaseModel):
    chat_id: int
    rating: int  # 1 helpful, -1 not helpful
    comment: str = ""


@app.post("/feedback")
def feedback(b: Feedback, user=Depends(auth.current_user)):
    try:
        return {"id": db.add_feedback(user["sub"], b.chat_id, b.rating, b.comment)}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/documents/ask")
async def document_ask(
    question: str = Form(...),
    file: UploadFile = File(...),
    user=Depends(auth.current_user),
):
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large (10 MB max)")
    try:
        text = documents.extract_text(file.filename, data)
        return documents.ask(question, text)
    except ValueError as e:
        raise HTTPException(400, str(e))


def _reindex(dense: bool):
    from . import ingest

    ingest.main()
    if dense:
        try:
            from . import index

            index.main()
        except Exception as e:
            print("Dense re-index skipped:", e)
    bot.cache_clear()


@app.post("/admin/upload")
async def admin_upload(
    tasks: BackgroundTasks,
    file: UploadFile = File(...),
    dense: bool = Form(False),
    user=Depends(auth.require_role("admin")),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files can be added to the knowledge base")
    name = "".join(ch for ch in file.filename if ch.isalnum() or ch in "._-")
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / name).write_bytes(await file.read())
    tasks.add_task(_reindex, dense)
    return {"saved": name, "reindexing": True}


@app.get("/admin/documents")
def admin_documents(user=Depends(auth.require_role("admin"))):
    return sorted(p.name for p in RAW.glob("*.pdf"))


@app.get("/admin/stats")
def admin_stats(user=Depends(auth.require_role("admin"))):
    return {**db.stats(), "unanswered": db.unanswered()}


@app.get("/admin/logs")
def admin_logs(limit: int = 100, user=Depends(auth.require_role("admin"))):
    return db.recent_logs(min(limit, 500))


@app.get("/admin/feedback")
def admin_feedback(limit: int = 500, user=Depends(auth.require_role("admin"))):
    return db.recent_feedback(min(limit, 1000))


@app.get("/admin/events-status")
def admin_events_status(user=Depends(auth.require_role("admin"))):
    with _refresh_state_lock:
        return dict(refresh_state)


@app.get("/admin/system")
def admin_system(user=Depends(auth.require_role("admin"))):
    chunk_file = PROC / "chunks.jsonl"
    chunk_count = (
        sum(1 for _ in chunk_file.open(encoding="utf-8")) if chunk_file.exists() else 0
    )
    scheduler = getattr(app.state, "scheduler", None)
    job = scheduler.get_job("nightly-events") if scheduler else None
    return {
        "api": "ok",
        "chunks": chunk_count,
        "dense_index_present": INDEX.exists() and any(INDEX.iterdir()),
        "source_pdfs": len(list(RAW.glob("*.pdf"))),
        "events": len(db.list_events()),
        "faculty_drafts": len(db.list_faculty_items_for_admin()),
        "refresh": dict(refresh_state),
        "nightly_refresh_next": job.next_run_time.isoformat()
        if job and job.next_run_time
        else None,
    }


@app.post("/admin/refresh-events")
def admin_refresh_events(
    tasks: BackgroundTasks,
    reindex: bool = True,
    user=Depends(auth.require_role("admin")),
):
    with _refresh_state_lock:
        if refresh_state["state"] in ("queued", "running"):
            raise HTTPException(409, "An events refresh is already running")
        refresh_state.update(state="queued", error=None)
    tasks.add_task(_refresh_events, reindex)
    return {"state": "queued", "reindex": reindex}


@app.post("/studio")
def studio_run(b: Studio, user=Depends(auth.current_user)):
    if b.task not in studio.TASKS.get(user["role"], []):
        raise HTTPException(400, "Task not available for your role")
    try:
        return {"output": studio.run(b.task, b.text, user["role"], bot().retriever)}
    except ValueError as e:
        raise HTTPException(503, str(e))


@app.get("/studio/tasks")
def studio_tasks(user=Depends(auth.current_user)):
    return studio.TASKS.get(user["role"], [])


@app.get("/faculty/dashboard")
def faculty_dashboard(user=Depends(auth.require_role("faculty"))):
    today = time.strftime("%Y-%m-%d")
    calendar = _merge_calendar()
    deadlines = [
        item
        for item in calendar
        if (item.get("end") or item.get("start", "")) >= today
        and any(
            word in item.get("title", "").lower()
            for word in ("cia", "mse", "mid semester", "exam")
        )
    ][:12]
    items = db.list_faculty_items(user["sub"])
    return {
        "deadlines": deadlines,
        "drafts": sum(item["status"] == "draft" for item in items),
        "approved": sum(item["status"] == "approved" for item in items),
        "rejected": sum(item["status"] == "rejected" for item in items),
        "tasks": studio.TASKS["faculty"],
    }


@app.get("/faculty/assessments")
def faculty_assessments(status: str = "", user=Depends(auth.require_role("faculty"))):
    if status and status not in {"draft", "approved", "rejected"}:
        raise HTTPException(400, "status must be draft, approved, or rejected")
    return db.list_faculty_items(user["sub"], status)


@app.post("/faculty/assessments")
def faculty_create_assessment(
    b: FacultyDraft, user=Depends(auth.require_role("faculty"))
):
    if b.task not in studio.TASKS["faculty"]:
        raise HTTPException(400, "Choose a supported faculty assessment task")
    if b.generate:
        try:
            content = studio.run(b.task, b.text, "faculty", bot().retriever)
        except ValueError as exc:
            raise HTTPException(503, str(exc))
    else:
        content = (
            f"{b.task}\nCourse: {b.course}\n\nSource material for faculty drafting:\n{b.text}\n\n"
            "Offline draft scaffold. No questions were generated. Needs faculty review and approval before use."
        )
    item_id = db.create_faculty_item(user["sub"], b.course, b.title, b.task, content)
    return {"id": item_id, "status": "draft", "content": content}


@app.patch("/faculty/assessments/{item_id}")
def faculty_decide_assessment(
    item_id: int, b: FacultyDecision, user=Depends(auth.require_role("faculty"))
):
    try:
        updated = db.set_faculty_item_status(item_id, user["sub"], b.status)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if not updated:
        raise HTTPException(
            404, "Draft not found, not owned by you, or already reviewed"
        )
    return {"id": item_id, "status": b.status}


@app.get("/faculty/assessments/export")
def faculty_export_assessments(user=Depends(auth.require_role("faculty"))):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["course", "title", "task", "content", "status"])
    for item in db.list_faculty_items(user["sub"], "approved"):
        writer.writerow(
            [
                item["course"],
                item["title"],
                item["task"],
                item["content"],
                item["status"],
            ]
        )
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=approved-assessments.csv"
        },
    )


@app.post("/attendance/calc")
def calc(b: Calc, user=Depends(auth.current_user)):
    return {
        "percent": round(b.attended / b.held * 100, 1),
        "can_miss": tools.classes_can_miss(b.attended, b.held, b.target),
        "need": tools.classes_needed(b.attended, b.held, b.target),
    }


@app.get("/class-attendance")
def class_attendance(target: float = 85, user=Depends(auth.require_role("faculty"))):
    return [
        dict(
            student=n,
            attended=a,
            held=t,
            percent=round(a / t * 100, 1),
            need=tools.classes_needed(a, t, target),
        )
        for n, a, t in ROSTER
    ]


@app.get("/calendar")
def calendar(user=Depends(auth.current_user)):
    return _merge_calendar()


@app.get("/policies")
def policies(q: str = "", limit: int = 40, user=Depends(auth.current_user)):
    ch = bot().retriever.chunks
    if q.strip():
        ws = [w for w in q.lower().split() if len(w) > 2]
        ch = [
            c
            for c in ch
            if all(w in (c["title"] + " " + c["text"]).lower() for w in ws)
        ]
    return [
        dict(
            id=c["id"],
            title=c["title"],
            source=c["source"],
            page=c["page"],
            kind=c["kind"],
            text=c["text"][:700],
        )
        for c in ch[: min(limit, 100)]
    ]


@app.get("/sources")
def sources(user=Depends(auth.current_user)):
    out = {}
    for c in bot().retriever.chunks:
        d = out.setdefault(
            c["source"],
            dict(source=c["source"], kind=c["kind"], year=c["year"], chunks=0),
        )
        d["chunks"] += 1
    return sorted(out.values(), key=lambda d: -d["chunks"])


@app.get("/results")
def results(user=Depends(auth.current_user)):
    import csv
    from .config import DATA, EVAL

    real = (EVAL / "results_summary.csv").exists()
    f = (EVAL if real else DATA / "mock") / "results_summary.csv"
    return {
        "mock": not real,
        "rows": list(csv.DictReader(f.open(encoding="utf-8"))) if f.exists() else [],
    }


@app.get("/", include_in_schema=False)
def root():
    from fastapi.responses import RedirectResponse

    return RedirectResponse("/demo/")


app.mount("/demo", StaticFiles(directory=ROOT / "frontend", html=True), name="demo")
