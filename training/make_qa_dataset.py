"""Generate instruction-tuning data from the indexed documents with an LLM.

Every 5th passage is held out as TEST (split by passage, so no leakage).
Train rows come in two formats so the model learns both domain facts and grounded answering:
  closed-book:  question -> answer
  RAG-style:    context + question -> answer with a [S1] citation

Run:  python -m training.make_qa_dataset --n 60 --backend anthropic
"""
import argparse
import json
import re

from campusgpt.config import QA
from campusgpt.llm import generate
from campusgpt.rag import SYSTEM_BASE, SYSTEM_RAG
from campusgpt.retrieve import load_chunks

PROMPT = ('Write {k} realistic questions a CHRIST student or faculty member would ask about the passage below, each with a '
          'complete, faithful answer using ONLY the passage. Reply with only a JSON array of {{"q": "...", "a": "..."}}.\n\n'
          'Passage ({src}, p.{page}): {title}\n{text}')


def parse(raw: str):
    m = re.search(r"\[.*\]", raw, re.S)
    try:
        return [x for x in json.loads(m.group(0)) if x.get("q") and x.get("a")] if m else []
    except json.JSONDecodeError:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="number of passages to use (curated first)")
    ap.add_argument("--per-chunk", type=int, default=3)
    ap.add_argument("--backend", default=None, help="defaults to LLM_BACKEND in .env (groq)")
    a = ap.parse_args()
    chunks = sorted(load_chunks(), key=lambda c: (not c["curated"], c["weight"] < 1))[: a.n]
    QA.mkdir(parents=True, exist_ok=True)
    train, test, raw = [], [], []
    for idx, c in enumerate(chunks):
        out = parse(generate([{"role": "user", "content": PROMPT.format(k=a.per_chunk, src=c["source"], page=c["page"], title=c["title"], text=c["text"])}],
                             backend=a.backend, max_tokens=1200, temperature=0.4))
        for qa in out:
            raw.append(dict(question=qa["q"], answer=qa["a"], chunk=c["id"], split="test" if idx % 5 == 0 else "train"))
            if idx % 5 == 0:
                continue
            train.append({"messages": [{"role": "system", "content": SYSTEM_BASE}, {"role": "user", "content": qa["q"]}, {"role": "assistant", "content": qa["a"]}]})
            ctx = f"[S1] {c['title']} ({c['source']}, p.{c['page']}): {c['text']}"
            train.append({"messages": [{"role": "system", "content": SYSTEM_RAG}, {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {qa['q']}"},
                                       {"role": "assistant", "content": qa["a"] + " [S1]"}]})
        print(f"{idx+1}/{len(chunks)} passages, {len(raw)} QA pairs")
    for name, rows in (("train.jsonl", train), ("qa_raw.jsonl", raw)):
        (QA / name).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    (QA / "test.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in raw if r["split"] == "test"), encoding="utf-8")
    print(f"train rows: {len(train)} | test QA: {sum(r['split']=='test' for r in raw)}. Spot-check ~50 pairs in {QA / 'qa_raw.jsonl'} before training!")


if __name__ == "__main__":
    main()
