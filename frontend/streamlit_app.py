"""Streamlit UI for the 2-day demo.   Run: streamlit run frontend/streamlit_app.py"""
import json

import pandas as pd
import streamlit as st

from campusgpt import auth, db, documents, studio, tools
from campusgpt.config import CUR
from campusgpt.rag import CampusGPT

st.set_page_config(page_title="CampusGPT", page_icon="🎓", layout="wide")


@st.cache_resource
def get_bot():
    return CampusGPT()


if "user" not in st.session_state:
    st.title("🎓 CampusGPT · CHRIST (Deemed to be University)")
    with st.form("login"):
        st.caption("Demo accounts: student / christ123 or faculty / christ123")
        u, p = st.text_input("User ID"), st.text_input("Password", type="password")
        if st.form_submit_button("Sign in"):
            user = auth.verify(u, p)
            if user:
                st.session_state.user, st.session_state.msgs, st.session_state.username = user, [], u.strip().lower()
                st.rerun()
            st.error("Incorrect ID or password")
    st.stop()

user, bot = st.session_state.user, get_bot()
role = user["role"]
st.sidebar.title("🎓 CampusGPT")
st.sidebar.write(f"**{user['name']}** · {role.title()}")
st.sidebar.link_button("Open KP", "https://kp.christuniversity.in")
if st.sidebar.button("Log out"):
    del st.session_state["user"]
    st.rerun()

if role == "admin":
    tab = st.tabs(["🛠 Admin"])
    with tab[0]:
        st.subheader("Usage")
        st.json(db.stats())
        st.subheader("Questions the bot could not answer (add documents for these)")
        st.write(db.unanswered() or "None yet")
        st.subheader("Recent chats")
        st.dataframe(pd.DataFrame(db.recent_logs(100)), use_container_width=True, hide_index=True)
        st.subheader("Add a PDF to the knowledge base")
        up = st.file_uploader("PDF", type=["pdf"])
        if up and st.button("Save and re-index"):
            from campusgpt import ingest
            from campusgpt.config import RAW
            (RAW / up.name).write_bytes(up.getvalue())
            with st.spinner("Re-indexing..."):
                ingest.main()
            st.cache_resource.clear()
            st.success("Added. Run `python -m campusgpt.index` to refresh the dense index.")
    st.stop()

tabs = ["💬 Assistant", "✨ Studio", "📅 Calendar"] + (["📊 Attendance"] if role == "student" else ["👥 Class attendance"]) + ["📄 Ask a document", "📈 Results"]
tab = st.tabs(tabs)

with tab[0]:
    for m in st.session_state.msgs:
        st.chat_message(m["role"]).write(m["content"])
    if q := st.chat_input("Ask about attendance, CIA, exams, calendar, grievances…"):
        st.session_state.msgs.append({"role": "user", "content": q})
        st.chat_message("user").write(q)
        r = bot.answer(q, role=role, history=[{"role": m["role"], "content": m["content"]} for m in st.session_state.msgs[:-1]])
        st.session_state.last_chat = db.log_chat(st.session_state.username, role, q, r)
        with st.chat_message("assistant"):
            st.write(r["answer"])
            for src in r["sources"]:
                st.caption(f"📄 {src['source']} · p.{src['page']}")
        st.session_state.msgs.append({"role": "assistant", "content": r["answer"]})
    if st.session_state.get("last_chat"):
        c1, c2, _ = st.columns([1, 1, 8])
        for col, icon, val in ((c1, "👍", 1), (c2, "👎", -1)):
            if col.button(icon, key=f"fb{val}"):
                db.add_feedback(st.session_state.username, st.session_state.last_chat, val)
                st.toast("Thanks for the feedback")

with tab[1]:
    task = st.selectbox("Task", studio.TASKS[role])
    text = st.text_area("Paste notes, a syllabus, a circular or a topic", height=160)
    if st.button("Generate") and text.strip():
        try:
            st.write(studio.run(task, text, role, bot.retriever))
        except ValueError as e:
            st.warning(f"{e}. Set LLM_BACKEND in .env.")

with tab[2]:
    ev = pd.DataFrame(json.loads((CUR / "calendar.json").read_text(encoding="utf-8")))
    kind = st.multiselect("Type", sorted(ev["type"].unique()), default=sorted(ev["type"].unique()))
    st.dataframe(ev[ev["type"].isin(kind)], use_container_width=True, hide_index=True)

with tab[3]:
    target = st.radio("Target", [85, 75], horizontal=True, format_func=lambda x: f"{x}% ({'ESE' if x == 85 else 'MSE'})")
    if role == "student":
        df = st.data_editor(pd.DataFrame([["Machine Learning", 39, 50], ["Deep Learning", 44, 50]], columns=["Subject", "Attended", "Held"]), num_rows="dynamic")
    else:
        st.caption("Demo roster, not real student data.")
        df = pd.DataFrame([("Student A", 42, 50), ("Student B", 36, 48), ("Student C", 33, 46), ("Student D", 29, 40)], columns=["Subject", "Attended", "Held"])
    df = df.dropna()
    df["Now %"] = (df["Attended"] / df["Held"] * 100).round(1)
    df["Can miss"] = [tools.classes_can_miss(a, h, target) for a, h in zip(df["Attended"], df["Held"])]
    df["Need in a row"] = [tools.classes_needed(a, h, target) for a, h in zip(df["Attended"], df["Held"])]
    st.dataframe(df, use_container_width=True, hide_index=True)

with tab[4]:
    st.caption("Upload a circular, syllabus or notes and ask questions about it. The file is not stored or added to the shared index.")
    f = st.file_uploader("Document", type=["pdf", "txt", "md"], key="doc")
    dq = st.text_input("Your question about this document")
    if f and dq and st.button("Ask"):
        try:
            res = documents.ask(dq, documents.extract_text(f.name, f.getvalue()))
            st.write(res["answer"])
            for p_ in res["passages"]:
                with st.expander(f"Passage {p_['chunk']}"):
                    st.write(p_["text"])
        except ValueError as e:
            st.warning(str(e))

with tab[5]:
    from campusgpt.config import DATA, EVAL
    real = (EVAL / "results_summary.csv").exists()
    base = EVAL if real else DATA / "mock"
    if real:
        st.success("Measured results from evaluation/evaluate.py")
    else:
        st.warning("MOCK / ILLUSTRATIVE numbers, not measured. Run evaluation/evaluate.py to replace them.")
    st.dataframe(pd.read_csv(base / "results_summary.csv"), use_container_width=True, hide_index=True)
    for img in (("accuracy.png", "loss_curve.png") if real else ("accuracy_mock.png", "loss_curve_mock.png")):
        if (base / img).exists():
            st.image(str(base / img))
