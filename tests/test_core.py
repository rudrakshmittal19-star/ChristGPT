import os

import pytest

os.environ.setdefault("LLM_BACKEND", "extractive")
from campusgpt import auth, tools
from campusgpt.config import PROC


def test_attendance_math():
    assert tools.classes_needed(39, 50, 85) == 24
    assert tools.classes_needed(45, 50, 85) == 0
    assert tools.classes_can_miss(45, 50, 75) == 10
    assert tools.parse("my attendance is 39/50, target 85%") == (39, 50, 85)
    assert "24 more" in tools.attendance_answer("I attended 39 of 50 classes, need 85%")


def test_auth():
    assert auth.verify("student", "christ123")["role"] == "student"
    assert auth.verify("student", "wrong") is None


@pytest.mark.skipif(not (PROC / "chunks.jsonl").exists(), reason="run python -m campusgpt.ingest first")
def test_pipeline_and_api():
    from fastapi.testclient import TestClient
    from campusgpt.api import app
    c = TestClient(app)
    tok = c.post("/auth/login", json={"username": "faculty", "password": "christ123"}).json()["token"]
    h = {"Authorization": f"Bearer {tok}"}
    r = c.post("/chat", json={"message": "What attendance is needed to write the end semester exam?"}, headers=h).json()
    assert "85" in r["answer"] and r["sources"]
    assert c.post("/chat", json={"message": "who won the cricket world cup"}, headers=h).json()["abstained"]
    assert c.get("/class-attendance", headers=h).status_code == 200
    s = c.post("/auth/login", json={"username": "student", "password": "christ123"}).json()["token"]
    assert c.get("/class-attendance", headers={"Authorization": f"Bearer {s}"}).status_code == 403


def test_db_feedback_and_stats(tmp_path, monkeypatch):
    from campusgpt import db
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    cid = db.log_chat("student", "student", "what is the ESE rule?", dict(answer="85%", abstained=False, tool=None, sources=[], latency_s=0.1))
    db.log_chat("student", "student", "who won the world cup", dict(answer="no", abstained=True, tool=None, sources=[], latency_s=0.1))
    db.add_feedback("student", cid, -1, "wrong")
    s = db.stats()
    assert s["questions"] == 2 and s["abstained"] == 1 and s["not_helpful"] == 1
    assert "who won the world cup" in db.unanswered() and "what is the ESE rule?" in db.unanswered()
    with pytest.raises(ValueError):
        db.add_feedback("student", cid, 5)


def test_document_qa():
    from campusgpt import documents
    text = "Attendance policy\nStudents need 85% attendance for the end semester exam.\n\nLibrary\nBooks are issued for 14 days."
    r = documents.ask("what attendance is needed for the exam", text, backend="extractive")
    assert "85%" in r["answer"] and not r["abstained"]
    assert documents.ask("quantum chromodynamics", text, backend="extractive")["abstained"]
    with pytest.raises(ValueError):
        documents.extract_text("x.exe", b"")
    assert "85%" in documents.extract_text("a.txt", text.encode())
