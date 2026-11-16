"""Central configuration. Override anything with environment variables or a .env file."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW, PROC, CUR, QA, EVAL = (
    DATA / d for d in ("raw", "processed", "curated", "qa", "eval")
)
INDEX = ROOT / "index"

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER = os.getenv(
    "ADAPTER_DIR", str(ROOT / "training" / "outputs" / "campusgpt-lora")
)

# extractive (no model, retrieval only) | anthropic | openai (Ollama/vLLM) | hf | hf-lora
LLM_BACKEND = os.getenv("LLM_BACKEND", "extractive")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "qwen2.5:1.5b")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
TOP_K = int(os.getenv("TOP_K", "4"))
MIN_DENSE = float(
    os.getenv("MIN_DENSE", "0.62")
)  # tuned on 10 held-out and 10 off-topic questions
MIN_BM25 = float(
    os.getenv("MIN_BM25", "10.0")
)  # BM25 score needed when dense is unavailable
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
ENV = os.getenv("ENV", "production").strip().lower()
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:8000").split(",")
    if origin.strip()
]


def validate_runtime_config():
    if ENV != "dev" and (
        JWT_SECRET.strip().lower()
        in {"change-me", "change-me-in-production", "secret", ""}
        or len(JWT_SECRET) < 32
    ):
        raise RuntimeError(
            "Set a random JWT_SECRET of at least 32 characters before starting outside ENV=dev"
        )
