"""Hybrid retrieval: BM25 + dense (bge + ChromaDB), fused with reciprocal-rank fusion.
Works with BM25 only if the vector index has not been built yet."""

import json
import re

import numpy as np
from rank_bm25 import BM25Okapi

from .config import EMBED_MODEL, INDEX, MIN_BM25, MIN_DENSE, PROC, TOP_K

STOP = set(
    "a an the is are was were of to in on for and or with what how do does can i my me we it be if at by as from that this when which who about should will have has not no".split()
)
QPREFIX = "Represent this sentence for searching relevant passages: "


def tok(s: str):
    return [w for w in re.findall(r"[a-z0-9%]+", s.lower()) if w not in STOP]


def load_chunks():
    with open(PROC / "chunks.jsonl", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


class Retriever:
    def __init__(self, dense: bool = True):
        self.chunks = load_chunks()
        self.by_id = {c["id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([tok(c["title"] + " " + c["text"]) for c in self.chunks])
        self.col = self.model = None
        if dense and INDEX.exists() and any(INDEX.iterdir()):
            try:
                import chromadb
                from sentence_transformers import SentenceTransformer

                self.col = chromadb.PersistentClient(path=str(INDEX)).get_collection(
                    "campusgpt"
                )
                self.model = SentenceTransformer(EMBED_MODEL)
            except Exception as e:  # fall back to BM25
                print("Dense retrieval unavailable, using BM25 only:", e)

    def search(self, q: str, k: int = TOP_K):
        sc = self.bm25.get_scores(tok(q))
        bm = {
            self.chunks[i]["id"]: (r, float(sc[i]))
            for r, i in enumerate(np.argsort(sc)[::-1][:20])
            if sc[i] > 0
        }
        dn = {}
        if self.col is not None:
            emb = self.model.encode(QPREFIX + q, normalize_embeddings=True).tolist()
            res = self.col.query(query_embeddings=[emb], n_results=20)
            dn = {
                i: (r, 1 - d)
                for r, (i, d) in enumerate(zip(res["ids"][0], res["distances"][0]))
            }
        fused = {}
        for i in set(bm) | set(dn):
            s = (1 / (60 + bm[i][0]) if i in bm else 0) + (
                1 / (60 + dn[i][0]) if i in dn else 0
            )
            fused[i] = s * self.by_id[i].get("weight", 1.0)
        out = []
        for i in sorted(fused, key=fused.get, reverse=True)[:k]:
            c = dict(self.by_id[i])
            c.update(
                score=fused[i],
                bm25=bm.get(i, (0, 0.0))[1],
                dense=dn[i][1] if i in dn else None,
            )
            out.append(c)
        return out

    def confident(self, hits) -> bool:
        if self.col is not None:
            return any(
                h["dense"] is not None and h["dense"] >= MIN_DENSE for h in hits[:3]
            )
        return any(h["bm25"] >= MIN_BM25 for h in hits[:3])
