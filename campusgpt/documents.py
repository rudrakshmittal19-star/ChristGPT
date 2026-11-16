"""Document Q&A: a user pastes/uploads one document (circular, syllabus, notes) and asks questions about it.
The document is held in memory for the request only; it is never added to the shared index or training data."""
import re

from . import config as C
from .llm import generate

STOP = set("a an the is are was were of to in on for and or with what how do does can i my me we it be if at by as from that this when which who about should will have has not no".split())
MAX_CHARS = 200_000


def tok(s):
    return [w for w in re.findall(r"[a-z0-9%]+", s.lower()) if w not in STOP]


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        with fitz.open(stream=data, filetype="pdf") as d:
            return "\n".join(p.get_text() for p in d)[:MAX_CHARS]
    if name.endswith((".txt", ".md", ".csv")):
        return data.decode("utf-8", errors="ignore")[:MAX_CHARS]
    raise ValueError("Unsupported file type. Upload a PDF, TXT, MD or CSV file.")


def chunk(text: str, size=900, overlap=120):
    text = re.sub(r"[ \t]+", " ", text)
    out, cur = [], ""
    for para in [p.strip() for p in text.splitlines() if p.strip()]:
        if len(cur) + len(para) > size and cur:
            out.append(cur.strip())
            cur = cur[-overlap:]
        cur += para + "\n"
    if cur.strip():
        out.append(cur.strip())
    return out


def top_chunks(question: str, chunks, k=3):
    q = set(tok(question))
    if not q or not chunks:
        return []
    if len(chunks) >= 4:
        try:
            from rank_bm25 import BM25Okapi
            scores = list(BM25Okapi([tok(c) for c in chunks]).get_scores(list(q)))
        except ImportError:
            scores = [len(q & set(tok(c))) for c in chunks]
    else:  # BM25 idf is meaningless on tiny documents
        scores = [len(q & set(tok(c))) for c in chunks]
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [(i, chunks[i]) for i in order[:k] if scores[i] > 0]


def ask(question: str, text: str, backend=None):
    chunks = chunk(text)
    hits = top_chunks(question, chunks)
    if not hits:
        return dict(answer="I could not find anything about that in the uploaded document.", passages=[], abstained=True)
    backend = backend or C.LLM_BACKEND
    passages = [dict(chunk=i + 1, text=t) for i, t in hits]
    if backend == "extractive":
        return dict(answer=hits[0][1][:900], passages=passages, abstained=False)
    ctx = "\n\n".join(f"[D{i + 1}] {t}" for i, t in hits)
    msgs = [{"role": "system", "content": "Answer ONLY from the uploaded document excerpts and cite them like [D1]. "
                                          "If the answer is not there, say so. Be concise."},
            {"role": "user", "content": f"Document excerpts:\n{ctx}\n\nQuestion: {question}"}]
    return dict(answer=generate(msgs, backend=backend), passages=passages, abstained=False)
