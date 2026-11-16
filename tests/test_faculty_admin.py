from fastapi.testclient import TestClient


def sign_in(client, username):
    response = client.post(
        "/auth/login", json={"username": username, "password": "christ123"}
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['token']}"}


def test_faculty_assessment_review_and_export(client, monkeypatch):
    from campusgpt import api

    monkeypatch.setattr(api, "bot", lambda: type("Bot", (), {"retriever": object()})())
    monkeypatch.setattr(
        api.studio,
        "run",
        lambda task, text, role, retriever: (
            f"Draft for {task}\n\nAI draft: needs faculty review and approval before use."
        ),
    )
    faculty = sign_in(client, "faculty")
    draft = client.post(
        "/faculty/assessments",
        headers=faculty,
        json={
            "course": "Synthetic Data Literacy",
            "title": "CIA practice",
            "task": api.studio.TASKS["faculty"][0],
            "text": "Synthetic course outcomes: interpret charts and explain sampling bias.",
        },
    )
    assert draft.status_code == 200 and draft.json()["status"] == "draft"
    item_id = draft.json()["id"]
    assert "needs faculty review" in draft.json()["content"]
    assert (
        client.get("/faculty/assessments", headers=faculty).json()[0]["status"]
        == "draft"
    )
    dashboard = client.get("/faculty/dashboard", headers=faculty)
    assert dashboard.status_code == 200 and dashboard.json()["drafts"] == 1
    assert (
        client.patch(
            f"/faculty/assessments/{item_id}",
            headers=faculty,
            json={"status": "approved"},
        ).status_code
        == 200
    )
    exported = client.get("/faculty/assessments/export", headers=faculty)
    assert exported.status_code == 200 and "Synthetic Data Literacy" in exported.text
    assert (
        client.patch(
            f"/faculty/assessments/{item_id}",
            headers=faculty,
            json={"status": "rejected"},
        ).status_code
        == 404
    )
    offline = client.post(
        "/faculty/assessments",
        headers=faculty,
        json={
            "course": "Synthetic Data Literacy",
            "title": "Offline scaffold",
            "task": api.studio.TASKS["faculty"][0],
            "text": "Synthetic learning outcomes only.",
            "generate": False,
        },
    )
    assert (
        offline.status_code == 200
        and "No questions were generated" in offline.json()["content"]
    )
    student = sign_in(client, "student")
    assert client.get("/faculty/dashboard", headers=student).status_code == 403
    payload = {
        "course": "Synthetic Data Literacy",
        "title": "CIA practice",
        "task": api.studio.TASKS["faculty"][0],
        "text": "Synthetic course outcomes.",
    }
    assert (
        client.post("/faculty/assessments", headers=student, json=payload).status_code
        == 403
    )


def test_admin_system_health_is_admin_only(client):
    admin = sign_in(client, "admin")
    response = client.get("/admin/system", headers=admin)
    assert response.status_code == 200
    assert (
        "source_pdfs" in response.json() and "nightly_refresh_next" in response.json()
    )
    assert "GROQ_API_KEY" not in response.text and "JWT_SECRET" not in response.text
    assert (
        client.get("/admin/system", headers=sign_in(client, "faculty")).status_code
        == 403
    )
