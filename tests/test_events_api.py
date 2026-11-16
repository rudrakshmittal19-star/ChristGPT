from fastapi.testclient import TestClient


def login(client, username):
    response = client.post(
        "/auth/login", json={"username": username, "password": "christ123"}
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['token']}"}


def test_events_filter_and_permissions(client):
    from campusgpt import db

    db.upsert_events(
        [
            {
                "title": "Open Day",
                "date": "2026-12-01",
                "url": "https://christuniversity.in/events/open-day",
                "category": "events",
                "summary": "Campus open day",
                "text": "Campus open day",
                "fetched_at": "now",
            },
            {
                "title": "Exam notice",
                "date": "2026-11-01",
                "url": "https://christuniversity.in/notices/exam",
                "category": "exam",
                "summary": "Exam timetable",
                "text": "Exam timetable",
                "fetched_at": "now",
            },
            {
                "title": "Rejected external",
                "date": "2026-12-02",
                "url": "https://example.com/event",
                "category": "events",
                "summary": "Not CHRIST",
                "text": "Not CHRIST",
                "fetched_at": "now",
            },
        ]
    )
    for role in ("student", "faculty", "admin"):
        headers = login(client, role)
        response = client.get("/events?upcoming=true&q=Open", headers=headers)
        assert response.status_code == 200
        assert [item["title"] for item in response.json()] == ["Open Day"]
        assert client.get("/history", headers=headers).status_code == 200
    assert client.get("/events").status_code == 401


def test_sse_chat_is_saved_per_user(client, monkeypatch):
    from campusgpt import api

    class FakeBot:
        def answer_stream(self, *args, **kwargs):
            def response():
                yield "Streamed "
                yield "answer [S1]"
                return {
                    "answer": "Streamed answer [S1]",
                    "sources": [],
                    "abstained": False,
                    "tool": None,
                    "hits": [],
                    "latency_s": 0.1,
                }

            return response()

    monkeypatch.setattr(api, "bot", lambda: FakeBot())
    student = login(client, "student")
    response = client.post(
        "/chat",
        json={"message": "Hello"},
        headers={**student, "Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    assert "event: token" in response.text and "event: done" in response.text
    result_line = next(
        line[6:]
        for line in response.text.splitlines()
        if line.startswith("data: {") and "conversation_id" in line
    )
    import json

    conversation_id = json.loads(result_line)["conversation_id"]
    history = client.get(
        f"/history?conversation_id={conversation_id}", headers=student
    ).json()
    assert history["messages"][0]["answer"] == "Streamed answer [S1]"
    faculty = login(client, "faculty")
    assert (
        client.get(
            f"/history?conversation_id={conversation_id}", headers=faculty
        ).json()["messages"]
        == []
    )


def test_admin_routes_are_role_protected(client, monkeypatch):
    from campusgpt import api

    monkeypatch.setattr(
        api, "_refresh_events", lambda reindex=True: {"state": "complete"}
    )
    admin = login(client, "admin")
    for role in ("student", "faculty"):
        headers = login(client, role)
        assert client.get("/admin/stats", headers=headers).status_code == 403
        assert client.get("/admin/logs", headers=headers).status_code == 403
        assert client.get("/admin/documents", headers=headers).status_code == 403
        assert client.get("/admin/feedback", headers=headers).status_code == 403
        assert client.get("/admin/events-status", headers=headers).status_code == 403
        assert client.post("/admin/refresh-events", headers=headers).status_code == 403
        assert client.post("/admin/upload", headers=headers).status_code == 403
    assert client.get("/admin/stats", headers=admin).status_code == 200
    assert client.get("/admin/logs", headers=admin).status_code == 200
    assert client.get("/admin/documents", headers=admin).status_code == 200
    assert client.get("/admin/feedback", headers=admin).status_code == 200
    assert client.get("/admin/events-status", headers=admin).status_code == 200
    assert client.post("/admin/refresh-events", headers=admin).status_code == 200


def test_official_pdf_downloads_are_allowlisted(client):
    catalog = client.get("/public-documents")
    assert catalog.status_code == 200
    calendar = next(item for item in catalog.json() if item["key"] == "calendar")
    response = client.get(calendar["url"])
    assert response.status_code == 200 and response.content.startswith(b"%PDF")
    assert client.get("/downloads/not-allowed").status_code == 404


def test_login_rate_limit(client):
    for _ in range(5):
        assert (
            client.post(
                "/auth/login", json={"username": "missing", "password": "wrong"}
            ).status_code
            == 401
        )
    assert (
        client.post(
            "/auth/login", json={"username": "student", "password": "christ123"}
        ).status_code
        == 429
    )
