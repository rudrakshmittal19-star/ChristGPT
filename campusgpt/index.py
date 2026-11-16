"""Embed all chunks and store them in ChromaDB.   Run: python -m campusgpt.index"""
import chromadb
from sentence_transformers import SentenceTransformer

from .config import EMBED_MODEL, INDEX
from .retrieve import load_chunks


def main():
    chunks = load_chunks()
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode([c["title"] + "\n" + c["text"] for c in chunks], normalize_embeddings=True,
                       batch_size=32, show_progress_bar=True)
    client = chromadb.PersistentClient(path=str(INDEX))
    try:
        client.delete_collection("campusgpt")
    except Exception:
        pass
    col = client.create_collection("campusgpt", metadata={"hnsw:space": "cosine"})
    for i in range(0, len(chunks), 500):
        part = chunks[i:i + 500]
        col.add(ids=[c["id"] for c in part], embeddings=emb[i:i + 500].tolist(),
                documents=[c["text"] for c in part],
                metadatas=[{"source": c["source"], "page": c["page"], "year": c["year"]} for c in part])
    print(f"Indexed {len(chunks)} chunks with {EMBED_MODEL} -> {INDEX}")


if __name__ == "__main__":
    main()
