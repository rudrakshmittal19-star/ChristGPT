# CampusGPT: spec for the AI coding assistant (Copilot / Claude Code)

Read this first. It says what is DONE, what is UNVERIFIED, and what you must BUILD next. Work through the TODO list in order,
run `pytest -q tests` after every step, and tick items off in this file.

## Project
CampusGPT: domain-adapted LLM + hybrid RAG for CHRIST (Deemed to be University) policy questions. Python 3.10+, FastAPI backend,
Streamlit UI, ChromaDB + BM25, Qwen2.5-1.5B-Instruct with QLoRA. Must run on a laptop without a GPU (extractive backend) and train on a free Colab T4.

## Hard rules
1. Attendance and any arithmetic stay in `campusgpt/tools.py` (plain Python). The LLM never computes numbers.
2. Answer CHRIST rules only from retrieved context, with `[S1]` citations; abstain when retrieval is weak.
3. Real student data must never enter training data, the vector store or logs. Marks/attendance come only via authorised APIs.
4. Faculty-generated content always carries the "needs faculty review" label.
5. Never evaluate on training data: the test split is every 5th passage (see `training/make_qa_dataset.py`).
6. Report evaluation numbers exactly as measured. Do not tune the test set to look good.
7. Keep demo credentials and `JWT_SECRET` out of production; replace `auth.py` USERS with a DB/SSO.

## Status
DONE and tested (CPU only): PDF ingestion + OCR (742 chunks), BM25 retrieval, abstention, attendance maths and routing, JWT roles
(student/faculty/admin), chat logging, feedback (thumbs), document Q&A, admin PDF upload + re-index, FastAPI endpoints, Streamlit UI.
WRITTEN BUT NEVER RUN (expect small fixes): dense retrieval (`index.py`, `retrieve.py`), `training/make_qa_dataset.py`,
`training/train_qlora.py`, `training/colab_train.ipynb`, `evaluation/evaluate.py`, LLM backends in `llm.py`.
The API tests (`test_pipeline_and_api`) need `pip install -r requirements.txt` (fastapi, httpx, python-multipart).

## Groq backend (do this first)
Groq is OpenAI-compatible (base URL https://api.groq.com/openai/v1). Add a `groq` backend:
- `config.py`: `GROQ_API_KEY`, `GROQ_MODEL` (default `llama-3.3-70b-versatile`; check console.groq.com/docs/models for current names), `GROQ_BASE_URL`.
- `llm.py`: `backend == "groq"` -> `OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)`; retry on HTTP 429 with exponential backoff (free tier is rate limited).
- `.env.example`: `LLM_BACKEND=groq`, `GROQ_API_KEY=`, `GROQ_MODEL=`.
- Make `evaluation/evaluate.py` `judge()` and `training/make_qa_dataset.py --backend` default to `C.LLM_BACKEND` instead of hard-coded `anthropic`.
- `requirements.txt` already has `openai`. Never commit `.env`. Add a test that mocks the client and checks the groq branch.

## Data + training plan (Groq generates, Colab trains)
1. `python -m campusgpt.web_ingest` (public christuniversity.in pages -> `data/web/*.json`), review the files, delete junk pages, then `python -m campusgpt.ingest` and `python -m campusgpt.index`.
2. `python -m training.make_qa_dataset --n 80` uses Groq (LLM_BACKEND) to write QA pairs per passage. Review 50 by hand. Test split stays disjoint.
3. Train on Colab with `training/colab_train.ipynb`, download the adapter to `training/outputs/campusgpt-lora`, set `LLM_BACKEND=hf-lora` to compare.
Website UI: `frontend/index.html` (served at `/`). Keep API paths stable; if you add an endpoint, add a view in `frontend/index.html`.

## TODO (in order)
- [ ] 1. `pip install -r requirements.txt`, `pytest -q tests` must be green. Add API tests for `/feedback`, `/documents/ask`, `/admin/*` (admin gets 200, student/faculty get 403).
- [ ] 2. `python -m campusgpt.index`. Then tune `MIN_DENSE` in `.env` with `data/eval/test_questions.json` plus 10 off-topic questions: no off-topic answers, no in-scope abstentions. Record the chosen value and the reasoning in README.
- [ ] 3. (Groq backend already added in llm.py; just smoke-test it.) Choose an LLM backend and smoke-test `CampusGPT().answer(...)` with it (Ollama `qwen2.5:1.5b` via `openai`, or `anthropic`). Fix prompt issues so every non-abstained answer has an `[S#]` citation.
- [ ] 4. Write 30 held-out questions in `data/eval/test_questions.json` (fields: question, truth, keyword) from the real PDFs, not from generated QA.
- [ ] 5. `python -m training.make_qa_dataset --n 80` (needs an LLM backend). Manually review 50 rows in `data/qa/train.jsonl`, delete wrong ones, keep the log of what was removed.
- [ ] 6. Train: `training/colab_train.ipynb` on a T4 (or `python -m training.train_qlora --epochs 2`). If TRL/transformers argument names changed, fix them. Save `loss_curve.png`. Fallback if OOM: Qwen2.5-0.5B-Instruct, `max_seq_len` 768, batch 1.
- [ ] 7. Evaluate: `python -m evaluation.evaluate --configs base,base+rag,ft,ft+rag --judge`. Produce `results_summary.md`, `accuracy.png`. Add bootstrap confidence intervals and McNemar test between configs. Add a short error analysis (10 worst failures, categorised).
- [ ] 8. Retrieval quality: add a cross-encoder reranker (`BAAI/bge-reranker-base`, optional flag `RERANK=1`), measure hit@1/hit@3/MRR before and after on the 30 questions, keep it only if it helps.
- [ ] 9. Streaming answers in `/chat` (SSE) and in Streamlit (`st.write_stream`).
- [ ] 10. Faculty features: MCQ / CIA question generation from an uploaded syllabus with an approval workflow (draft -> approved/rejected, stored in SQLite, only approved items are exportable). Export approved items to DOCX/PDF.
- [ ] 11. Replace demo `USERS` with a DB table, bcrypt/argon2 hashes, rate limiting on `/auth/login`, JWT secret required from env (refuse to start with the default in non-dev).
- [ ] 12. Docker: `Dockerfile` + `docker-compose.yml` (api, streamlit, optional ollama). `GET /health` as the healthcheck.
- [ ] 13. CI: GitHub Actions running `pytest` with the extractive backend.
- [ ] 14. Multilingual (Hindi/Kannada) question handling via query translation, and optional voice input. Only after 1-9 are done.
- [ ] 15. Pilot tooling: feedback export to CSV from `/admin/logs`, weekly summary of unanswered questions (`db.unanswered()`), and a short user survey page.

## Definition of done
`pytest` green; `streamlit run frontend/streamlit_app.py` works for student, faculty and admin; `results_summary.md` contains real numbers for all four configurations;
README documents the final `MIN_DENSE`, the adapter location and how to reproduce every number.

## Useful commands
`python -m campusgpt.ingest` rebuild chunks | `python -m campusgpt.index` rebuild dense index | `uvicorn campusgpt.api:app --reload`
`streamlit run frontend/streamlit_app.py` | `python -m training.make_qa_dataset --n 80` | `python -m training.train_qlora --epochs 2`
`python -m evaluation.evaluate --configs base,base+rag,ft,ft+rag --judge`
