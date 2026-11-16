"""RAG pipeline: tool routing -> retrieve -> grounded prompt -> answer with citations / abstention."""

import re
import time

from . import config as C
from .llm import generate, generate_stream
from .retrieve import Retriever
from .tools import attendance_answer

SYSTEM_RAG = (
    "You are CampusGPT, the academic assistant for CHRIST (Deemed to be University), Bengaluru. "
    "Answer CHRIST-specific rules, dates, fees and procedures ONLY from the numbered context excerpts and cite them "
    "like [S1]. If the excerpts do not contain the answer, say you could not find it in the loaded documents and "
    "suggest KP (kp.christuniversity.in) or the Office of Examinations. For general academic questions you may use "
    "general knowledge, starting that part with 'General guidance:'. Be concise."
)
SYSTEM_BASE = "You are CampusGPT, the academic assistant for CHRIST (Deemed to be University), Bengaluru. Answer concisely."
ABSTAIN = (
    "I could not find this in the loaded CHRIST documents, so I won't guess. "
    "Please check KP (kp.christuniversity.in) or the Office of Examinations."
)


SMALLTALK = re.compile(
    r"^\s*(hi+|hello+|hey+|thanks?( you)?|thank you|good (morning|afternoon|evening)|ok(ay)?)\W*$",
    re.I,
)
GREETING = (
    "Hello! I can answer CHRIST questions about attendance, CIA, exams, the academic calendar and regulations, "
    "with sources. Try: 'What attendance is needed for the end semester exam?'"
)


class CampusGPT:
    def __init__(self, backend=None, use_rag=True):
        self.backend, self.use_rag = backend or C.LLM_BACKEND, use_rag
        self.retriever = Retriever() if use_rag else None

    def answer(self, question, role="student", history=None):
        t0 = time.time()
        if SMALLTALK.match(question):
            return dict(
                answer=GREETING,
                sources=[],
                abstained=False,
                tool=None,
                hits=[],
                latency_s=time.time() - t0,
            )
        tool = attendance_answer(question)
        if tool:
            return dict(
                answer=tool,
                sources=[],
                abstained=False,
                tool="attendance",
                hits=[],
                latency_s=time.time() - t0,
            )
        hits = []
        if self.use_rag:
            hits = self.retriever.search(question)
            if not self.retriever.confident(hits):
                return dict(
                    answer=ABSTAIN,
                    sources=[],
                    abstained=True,
                    tool=None,
                    hits=hits,
                    latency_s=time.time() - t0,
                )
        if self.backend == "extractive":
            top = hits[0]
            text = f"{top['title']}: {top['text'][:900]} [S1]"
            cited = [top]
        else:
            ctx = "\n\n".join(
                f"[S{i + 1}] {h['title']} ({h['source']}, p.{h['page']}): {h['text']}"
                for i, h in enumerate(hits)
            )
            msgs = [
                {
                    "role": "system",
                    "content": (SYSTEM_RAG if self.use_rag else SYSTEM_BASE)
                    + f" The user is a {role}.",
                }
            ]
            msgs += list(history or [])[-6:]
            msgs.append(
                {
                    "role": "user",
                    "content": (f"Context:\n{ctx}\n\n" if ctx else "")
                    + f"Question: {question}",
                }
            )
            text = generate(msgs, backend=self.backend)
            ids = {int(x) for x in re.findall(r"\[S(\d+)\]", text)}
            cited = [h for i, h in enumerate(hits, 1) if i in ids] or hits
        sources = [
            dict(id=h["id"], source=h["source"], page=h["page"], title=h["title"])
            for h in cited
        ]
        return dict(
            answer=text,
            sources=sources,
            abstained=False,
            tool=None,
            hits=hits,
            latency_s=time.time() - t0,
        )

    def answer_stream(self, question, role="student", history=None):
        t0 = time.time()
        if SMALLTALK.match(question):
            text, hits, cited, abstained, tool = GREETING, [], [], False, None
        else:
            tool_text = attendance_answer(question)
            if tool_text:
                text, hits, cited, abstained, tool = (
                    tool_text,
                    [],
                    [],
                    False,
                    "attendance",
                )
            else:
                hits = self.retriever.search(question) if self.use_rag else []
                if self.use_rag and not self.retriever.confident(hits):
                    text, cited, abstained, tool = ABSTAIN, [], True, None
                elif self.backend == "extractive":
                    top = hits[0]
                    text = f"{top['title']}: {top['text'][:900]} [S1]"
                    cited, abstained, tool = [top], False, None
                else:
                    ctx = "\n\n".join(
                        f"[S{i + 1}] {h['title']} ({h['source']}, p.{h['page']}): {h['text']}"
                        for i, h in enumerate(hits)
                    )
                    msgs = [
                        {
                            "role": "system",
                            "content": (SYSTEM_RAG if self.use_rag else SYSTEM_BASE)
                            + f" The user is a {role}.",
                        }
                    ]
                    msgs += list(history or [])[-6:]
                    msgs.append(
                        {
                            "role": "user",
                            "content": (f"Context:\n{ctx}\n\n" if ctx else "")
                            + f"Question: {question}",
                        }
                    )
                    parts = []
                    for part in generate_stream(msgs, backend=self.backend):
                        parts.append(part)
                        yield part
                    text = "".join(parts)
                    ids = {int(x) for x in re.findall(r"\[S(\d+)\]", text)}
                    cited = [h for i, h in enumerate(hits, 1) if i in ids] or hits
                    abstained, tool = False, None
                    sources = [
                        dict(
                            id=h["id"],
                            source=h["source"],
                            page=h["page"],
                            title=h["title"],
                        )
                        for h in cited
                    ]
                    return dict(
                        answer=text,
                        sources=sources,
                        abstained=abstained,
                        tool=tool,
                        hits=hits,
                        latency_s=time.time() - t0,
                    )

        for offset in range(0, len(text), 32):
            yield text[offset : offset + 32]
        sources = [
            dict(id=h["id"], source=h["source"], page=h["page"], title=h["title"])
            for h in cited
        ]
        return dict(
            answer=text,
            sources=sources,
            abstained=abstained,
            tool=tool,
            hits=hits,
            latency_s=time.time() - t0,
        )
