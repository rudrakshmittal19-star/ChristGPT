import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def isolated_auth_database(tmp_path, monkeypatch):
    from campusgpt import api, auth, config, db, retrieve

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "campusgpt-test.db"))
    monkeypatch.setattr(config, "ENV", "dev")
    monkeypatch.setattr(config, "LLM_BACKEND", "extractive")
    monkeypatch.setattr(config, "MIN_DENSE", 0.62)
    monkeypatch.setattr(retrieve, "MIN_DENSE", 0.62)
    monkeypatch.setattr(auth, "ENV", "dev")
    monkeypatch.setattr(
        config, "JWT_SECRET", "test-secret-with-more-than-thirty-two-characters"
    )
    monkeypatch.setattr(auth, "JWT_SECRET", config.JWT_SECRET)
    api._login_attempts.clear()
    auth.seed_development_users()


@pytest.fixture
def client(tmp_path, monkeypatch):
    from campusgpt import api

    monkeypatch.setattr(api, "CUR", tmp_path / "curated")
    with TestClient(api.app) as test_client:
        yield test_client
