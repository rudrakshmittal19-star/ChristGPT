import pytest


def test_production_requires_strong_jwt_secret(monkeypatch):
    from campusgpt import config

    monkeypatch.setattr(config, "ENV", "production")
    monkeypatch.setattr(config, "JWT_SECRET", "change-me")
    with pytest.raises(RuntimeError):
        config.validate_runtime_config()
    monkeypatch.setattr(
        config, "JWT_SECRET", "a-random-production-secret-long-enough-123456"
    )
    config.validate_runtime_config()


def test_dense_threshold_is_not_overridden_by_bm25():
    from campusgpt.retrieve import Retriever

    retriever = Retriever.__new__(Retriever)
    retriever.col = object()
    assert not retriever.confident([{"dense": 0.61, "bm25": 50.0}])
    assert retriever.confident([{"dense": 0.63, "bm25": 0.0}])
    retriever.col = None
    assert retriever.confident([{"dense": None, "bm25": 10.1}])
    assert not retriever.confident([{"dense": None, "bm25": 9.9}])
