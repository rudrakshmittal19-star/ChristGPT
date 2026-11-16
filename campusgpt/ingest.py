"""PDFs in data/raw  ->  cleaned, chunked passages in data/processed/chunks.jsonl.

Run:  python -m campusgpt.ingest
Scanned PDFs are OCR'd if pytesseract + the tesseract binary are installed.
"""

import json
import re
from pathlib import Path

try:
    import pymupdf as fitz  # PyMuPDF (new name)
except ImportError:
    import fitz

from .config import CUR, DATA, PROC, RAW

CHUNK, OVERLAP = 1200, 150


def ocr_page(page) -> str:
    try:
        import io
        import pytesseract
        from PIL import Image

        return pytesseract.image_to_string(
            Image.open(io.BytesIO(page.get_pixmap(dpi=200).tobytes("png")))
        )
    except Exception:
        return ""


def clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    lines = [
        l.strip() for l in text.splitlines() if l.strip() and l.count(".") < 20
    ]  # drop TOC dot leaders
    return "\n".join(lines)


def split(text: str):
    out, cur = [], ""
    for para in text.split("\n"):
        if len(cur) + len(para) > CHUNK and cur:
            out.append(cur.strip())
            cur = cur[-OVERLAP:]
        cur += para + "\n"
    if cur.strip():
        out.append(cur.strip())
    return out


def doc_meta(path: Path):
    stem = path.stem.lower()
    y = re.search(r"(20\d\d)[-_]?(\d\d)?", stem)
    kind = (
        "handbook"
        if "handbook" in stem
        else "calendar"
        if "calendar" in stem
        else "cia_repeat"
        if "cia" in stem
        else "internship"
        if "internship" in stem
        else "document"
    )
    return kind, int(y.group(1)) if y else 0


def main():
    files = sorted(RAW.glob("*.pdf"))
    meta = {f: doc_meta(f) for f in files}
    newest = {}
    for k, y in meta.values():
        newest[k] = max(newest.get(k, 0), y)
    rows = []
    for f in files:
        kind, year = meta[f]
        weight = (
            0.6 if year and year < newest[kind] else 1.0
        )  # older versions rank lower
        n = 0
        for i, page in enumerate(fitz.open(f)):
            text = page.get_text() or ""
            if len(text.strip()) < 50:
                text = ocr_page(page)
            text = clean(text)
            if len(text) < 120:
                continue
            m = re.search(r"Handbook 20\d\d-\d\d\s+(\d+)\s*$", text, re.M)
            printed = m.group(1) if m else str(i + 1)
            title = next((l for l in text.split("\n") if 4 < len(l) < 90), f.stem)
            for j, part in enumerate(split(text)):
                rows.append(
                    dict(
                        id=f"{f.stem}-p{i + 1}-{j}",
                        source=f.stem,
                        kind=kind,
                        year=year,
                        page=printed,
                        title=title,
                        text=part,
                        weight=weight,
                        curated=False,
                    )
                )
                n += 1
        print(f"{f.name}: {n} chunks")
    for wf in sorted((DATA / "web").glob("*.json")) if (DATA / "web").exists() else []:
        w = json.loads(wf.read_text(encoding="utf-8"))
        for j, part in enumerate(split(clean(w["text"]))):
            rows.append(
                dict(
                    id=f"web-{wf.stem}-{j}",
                    source=w["url"],
                    kind="web",
                    year=2026,
                    page="web",
                    title=w["title"],
                    text=part,
                    weight=1.1,
                    curated=False,
                )
            )
        print(f"{w['url']}: web page indexed")
    event_dir = DATA / "web" / "events"
    for event_file in sorted(event_dir.glob("*.json")) if event_dir.exists() else []:
        event = json.loads(event_file.read_text(encoding="utf-8"))
        event_text = f"{event['title']}\nDate: {event.get('date', 'Date not specified')}\n{event.get('text', '')}"
        for j, part in enumerate(split(clean(event_text))):
            rows.append(
                dict(
                    id=f"event-{event_file.stem}-{j}",
                    source=event["url"],
                    kind="event",
                    year=int(event.get("date", "")[:4] or 0),
                    page=event.get("date") or "web",
                    title=event["title"],
                    text=part,
                    weight=1.15,
                    curated=False,
                )
            )
        print(f"{event['url']}: event page indexed")
    cur = CUR / "curated_passages.json"
    if cur.exists():
        for k, p in enumerate(json.loads(cur.read_text(encoding="utf-8"))):
            rows.append(
                dict(
                    id=f"curated-{k}",
                    source=p["source"],
                    kind="curated",
                    year=2026,
                    page=str(p["page"]),
                    title=p["title"],
                    text=p["text"],
                    weight=1.25,
                    curated=True,
                )
            )
    PROC.mkdir(parents=True, exist_ok=True)
    with open(PROC / "chunks.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} chunks -> {PROC / 'chunks.jsonl'}")


if __name__ == "__main__":
    main()
