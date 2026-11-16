# 🎓 CampusGPT: Domain-Adapted LLM + RAG for University Policy Assistance
CHRIST (Deemed to be University) academic assistant: hybrid RAG with citations, deterministic attendance tool,
student/faculty roles, QLoRA fine-tuning, and a 4-way evaluation (Base / Base+RAG / Fine-tuned / Fine-tuned+RAG).

## Layout
```
campusgpt/          python package
  ingest.py         PDFs -> cleaned chunks (OCR fallback)        data/processed/chunks.jsonl
  index.py          bge embeddings -> ChromaDB                   index/
  retrieve.py       BM25 + dense, reciprocal-rank fusion, abstain threshold
  rag.py            tool routing -> retrieval -> grounded prompt -> citations / abstention
  llm.py            backends: anthropic | openai (Ollama/vLLM) | hf | hf-lora
  tools.py          attendance maths (no LLM guessing)
  studio.py         study / assessment drafting tasks (faculty drafts are labelled "needs approval")
  auth.py, api.py   JWT login (student / faculty / admin) + FastAPI endpoints
  db.py             SQLite chat log + feedback + unanswered-question list
  documents.py      Ask questions about one uploaded document (not stored)
frontend/           streamlit_app.py (demo UI) and standalone_demo.html (offline, retrieval-only)
training/           make_qa_dataset.py, train_qlora.py, merge_adapter.py, colab_train.ipynb
evaluation/         evaluate.py (accuracy, faithfulness, hallucination, hit@k, latency)
data/raw            the 6 CHRIST PDFs     data/curated  hand-verified passages + calendar
data/eval           test_questions.json   data/qa       generated train/test sets
tests/              pytest
```

## Quick start (VS Code, 10 minutes, no GPU, no API key)
1. Open the folder in VS Code. Terminal → `python -m venv .venv` and activate it.
2. `pip install -r requirements.txt` (Tasks: *1. Setup venv + install*).
3. `copy .env.example .env` (Windows) or `cp .env.example .env`.
4. `python -m campusgpt.ingest` (chunks are already included; re-run after adding PDFs to `data/raw`).
5. `python -m campusgpt.index` (downloads `bge-small`, builds the dense index; skip it to use BM25 only).
6. `streamlit run frontend/streamlit_app.py` or `uvicorn campusgpt.api:app --reload` (API docs at `/docs`).
Local `ENV=dev` seeds demo logins (`student`, `faculty`, `admin`; password `christ123`). They are disabled outside dev mode.
Create non-demo accounts with `python -m campusgpt.manage_users create <username> --name "Full Name" --role student|faculty|admin`;
the CLI prompts for a password and stores an Argon2 hash in SQLite.

With `LLM_BACKEND=extractive` the assistant returns the best matching passage with its source. For generated answers set
`LLM_BACKEND=anthropic` (+ `ANTHROPIC_API_KEY`), `openai` (Ollama: `ollama run qwen2.5:1.5b`), or `hf` / `hf-lora` (local GPU).

## The LLM work (training and evaluation)
1. **Dataset**: `python -m training.make_qa_dataset --n 80`. Claude writes QA pairs per passage. Every 5th passage is held out as the
   test set (no leakage). Train rows are closed-book and RAG-style (context + `[S1]` citation). **Spot-check ~50 pairs by hand.**
2. **Fine-tune**: `training/colab_train.ipynb` on a free T4, or `python -m training.train_qlora --epochs 2`.
   Qwen2.5-1.5B-Instruct, 4-bit NF4, LoRA r=16 on all linear layers, fp16, paged AdamW 8-bit. Saves the adapter and `loss_curve.png`.
3. **Evaluate**: `python -m evaluation.evaluate --configs base,base+rag,ft,ft+rag --judge` writes
   `data/eval/results_summary.md|csv`, `results_detail.csv`, `accuracy.png`.
   Without a GPU: `--configs base,base+rag --gen-backend anthropic` to test the harness.
4. Replace/extend `data/eval/test_questions.json` with your own **30** held-out questions before reporting numbers.
   Report results honestly: RAG usually gives most of the gain and fine-tuning mainly improves format, tone and abstention.
5. Optional: `python -m training.merge_adapter` to merge LoRA for Ollama/vLLM.

## Website (recommended UI)
`uvicorn campusgpt.api:app --reload` then open http://localhost:8000 (blue/cyan CHRIST theme, logo, light/dark, mobile nav).
Roles: student, faculty, admin. Features include cited streamed chat with per-user history, Campus Pulse events, CIA/MSE deadlines,
direct official PDF downloads, a synthetic practice sheet, studio, attendance, class roster, merged calendar, document Q&A,
policy library, sources, evaluation, admin event refresh and feedback CSV export. Streamlit remains as a second UI.
To add the live CHRIST website pages: `python -m campusgpt.web_ingest && python -m campusgpt.ingest && python -m campusgpt.index` (needs internet).

## Vercel frontend-only demo
The repository root is configured to serve `frontend/standalone_demo.html` through `vercel.json`. This mode needs no Python API,
database, Groq key, or environment variables. It includes local retrieval over bundled public CHRIST passages, deterministic attendance
math, calendar, demo student/faculty modes, eSPRO/KP links, and direct calendar/handbook PDF downloads. AI generation is unavailable
unless a supported browser integration is provided; this static deployment does not call the FastAPI backend. Demo credentials are
visible in the client and are not authentication; do not use this as a secure production service.

To deploy: push the repository to GitHub, sign in at Vercel, choose **Add New → Project**, import `rudrakshmittal19-star/ChristGPT`,
leave the framework preset as **Other**, leave build and output commands blank, and click **Deploy**. The root rewrite serves the standalone
demo. Do not configure `GROQ_API_KEY` or `JWT_SECRET` for this static-only phase.

## Demo data (for the presentation)
`python -m scripts.seed_mock_data --reset` fills the admin dashboard with fake usage, writes illustrative results/plots to `data/mock/`
and a fictional `sample_circular.txt` for the "Ask a document" tab. Mock numbers are labelled MOCK in the UI; the Results tab switches to
real numbers automatically once `evaluation/evaluate.py` writes `data/eval/results_summary.csv`. Say it is illustrative if you show it.

## Extra features (CPU-only, tested)
- 👍/👎 feedback on every answer (`POST /feedback`), chat logging, admin dashboard with usage stats and the list of unanswered questions.
- "Ask a document" tab / `POST /documents/ask`: upload a PDF/TXT and ask about it; nothing is stored or indexed.
- Admin PDF upload and re-index (`POST /admin/upload`, or the Admin tab). Re-run `python -m campusgpt.index` for the dense index.
- **`CLAUDE.md` is the build plan for Copilot**: it lists everything still to do (training, tuning, evaluation, reranking, approvals, Docker, CI).

## Retrieval threshold
The current dense threshold is `MIN_DENSE=0.62`. On the local 2026–27 index, all 10 questions in
`data/eval/test_questions.json` were confident (lowest top-three dense score 0.664); 10 fixed off-topic probes were all
below threshold (highest 0.615). With a dense index present, confidence uses dense similarity only; BM25's `MIN_BM25`
is the fallback when dense retrieval is unavailable. This is a small smoke set, not a general quality guarantee; rerun the
same comparison after changing source documents or the embedding model.

## What was verified when this zip was built
Verified: PDF ingestion incl. OCR of the scanned internship policy (742 chunks), BM25 retrieval, abstention on off-topic questions,
attendance maths, auth and role checks, the API pipeline ; the feedback/logging and document Q&A tests also pass. The FastAPI endpoint tests could not be re-run in the final build environment (no packages), so run `pytest -q tests` once yourself.
**Not run here (no GPU, no model downloads, no API key):** dense retrieval, QA generation, QLoRA training, the evaluation harness.
Expect to fix small version-specific issues (TRL / transformers argument names) on first run; `train_qlora.py` already handles the
`max_length` vs `max_seq_length` rename.

## Notes
- `data/curated/` holds hand-checked passages (CIA repeat regulation and grievance regulation were read from public CHRIST pages). Handbook 2022-23 is indexed with a 0.6 weight so the current handbook wins.
- Never put real student records into training data or the vector store. Marks and attendance must come through authorised APIs.
- The attendance rules (75% per course for MSE, 85% aggregate for ESE) come from the 2026-27 handbook; always confirm on KP.
