"""Compare Base, Base+RAG, Fine-tuned, Fine-tuned+RAG on held-out questions.

Metrics: accuracy, faithfulness (RAG only), hallucination rate, retrieval hit@k, latency.
Accuracy/faithfulness/hallucination use a Claude judge (--judge); without it accuracy falls back to keyword match.

Run:  python -m evaluation.evaluate --configs base,base+rag,ft,ft+rag --judge
Use --gen-backend anthropic|openai to test the pipeline without a GPU (then ft rows are skipped).
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from campusgpt.config import ADAPTER, EVAL, QA, ROOT
from campusgpt.llm import generate
from campusgpt.rag import CampusGPT

CONFIGS = {"base": ("hf", False), "base+rag": ("hf", True), "ft": ("hf-lora", False), "ft+rag": ("hf-lora", True)}


def load_questions(limit=None):
    qs = json.loads((EVAL / "test_questions.json").read_text(encoding="utf-8"))
    extra = QA / "test.jsonl"
    if extra.exists():
        qs += [dict(question=r["question"], truth=r["answer"], keyword=None)
               for r in map(json.loads, filter(None, extra.read_text(encoding="utf-8").splitlines()))]
    return qs[:limit] if limit else qs


def judge(q, truth, answer, context):
    p = (f"Question: {q}\nGround truth: {truth}\nAnswer: {answer}\n"
         + (f"Context given to the assistant:\n{context}\n" if context else "")
         + 'Reply ONLY with JSON {"correct": bool, "hallucinated": bool, "faithful": bool}. correct = consistent with the ground truth; '
           "hallucinated = states specific CHRIST facts that contradict the ground truth; faithful = every claim is supported by the context (true if no context).")
    m = re.search(r"\{.*\}", generate([{"role": "user", "content": p}], backend=None, max_tokens=100, temperature=0), re.S)
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="base,base+rag,ft,ft+rag")
    ap.add_argument("--gen-backend", default=None, help="override the generator for non-fine-tuned rows")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    qs, rows = load_questions(a.limit), []
    for name in a.configs.split(","):
        backend, rag = CONFIGS[name]
        if backend == "hf-lora" and not Path(ADAPTER).exists():
            print(f"skip {name}: no adapter at {ADAPTER}")
            continue
        bot = CampusGPT(backend=a.gen_backend if (a.gen_backend and backend == "hf") else backend, use_rag=rag)
        for i, q in enumerate(qs):
            r = bot.answer(q["question"])
            ctx = "\n".join(h["text"] for h in r["hits"])
            kw = q.get("keyword")
            hit = bool(rag and kw and any(kw.lower() in h["text"].lower() for h in r["hits"]))
            row = dict(config=name, question=q["question"], answer=r["answer"], latency_s=r["latency_s"], hit_at_k=hit if rag and kw else None,
                       correct=(kw.lower() in r["answer"].lower()) if kw else None)
            if a.judge:
                j = judge(q["question"], q["truth"], r["answer"], ctx if rag else "")
                row.update(correct=j.get("correct"), hallucinated=j.get("hallucinated"), faithful=j.get("faithful") if rag else None)
            rows.append(row)
            print(f"{name} {i+1}/{len(qs)}")
    df = pd.DataFrame(rows)
    EVAL.mkdir(parents=True, exist_ok=True)
    df.to_csv(EVAL / "results_detail.csv", index=False)
    cols = {"accuracy": "correct", "hallucination": "hallucinated", "faithfulness": "faithful", "retrieval_hit_at_k": "hit_at_k", "latency_s": "latency_s"}
    summ = pd.DataFrame({k: df.groupby("config", sort=False)[v].apply(lambda x: pd.to_numeric(x, errors="coerce").mean()) for k, v in cols.items() if v in df}).round(3)
    summ.to_csv(EVAL / "results_summary.csv")
    (EVAL / "results_summary.md").write_text(summ.to_markdown(), encoding="utf-8")
    print(summ.to_markdown())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        summ["accuracy"].astype(float).plot.bar(color="#24398f", rot=0)
        plt.ylabel("accuracy"); plt.ylim(0, 1); plt.title("CampusGPT: accuracy by setup"); plt.tight_layout()
        plt.savefig(EVAL / "accuracy.png", dpi=150)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
