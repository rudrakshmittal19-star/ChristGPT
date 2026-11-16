"""Study Studio (students) and Assessment Studio (faculty): LLM drafting tasks."""
from .llm import generate

TASKS = {
    "student": ["MCQ practice test: 5 questions with answer key and one-line explanations", "Flashcards: 10 question/answer cards",
                "Short revision notes in bullets", "Study plan: day-by-day revision schedule",
                "Explain this simply with an example", "Summarise this circular or notice and list important dates"],
    "faculty": ["CIA question paper with marks and course-outcome (CO) mapping", "MCQ bank: 10 questions with answer key and difficulty level",
                "Marking scheme and rubric", "Viva questions with expected answers", "Lesson plan for one session",
                "Draft announcement to students", "Summarise this circular and list deadlines"],
}
REVIEW = "\n\nAI draft: needs faculty review and approval before use."


def run(task: str, text: str, role: str, retriever=None, backend=None) -> str:
    ctx = ""
    if retriever:
        hits = [h for h in retriever.search(text, 2) if retriever.confident([h])]
        ctx = "\n".join(f"{h['title']} ({h['source']}): {h['text']}" for h in hits)
    prompt = (f"Role: {role}. Task: {task}.\nUser input:\n{text}\n"
              + (f"Official CHRIST excerpts (use if relevant, never contradict):\n{ctx}\n" if ctx else "")
              + "Be accurate and well structured in plain text; do not invent CHRIST policy.")
    out = generate([{"role": "system", "content": "You are CampusGPT for CHRIST (Deemed to be University), Bengaluru."},
                    {"role": "user", "content": prompt}], backend=backend, max_tokens=900, temperature=0.3)
    return out + (REVIEW if role == "faculty" else "")
