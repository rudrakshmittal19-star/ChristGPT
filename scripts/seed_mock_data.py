"""Create MOCK data for the demo: fake usage logs, illustrative evaluation table/plots, sample circular.
Everything written to data/mock/ is ILLUSTRATIVE and must be presented as such. Real results go to data/eval/.

Run:  python -m scripts.seed_mock_data          (add --reset to clear the demo database first)
"""
import argparse
import json
import math
import os
import random

from campusgpt import db
from campusgpt.config import DATA

MOCK = DATA / "mock"
random.seed(7)

QUESTIONS = [  # (question, abstained, tool, rating)
    ("What attendance is needed to write the end semester exam?", 0, None, 1),
    ("What is the minimum attendance for the mid semester exam?", 0, None, 1),
    ("I attended 39 of 50 classes and need 85%. How many more must I attend?", 0, "attendance", 1),
    ("My attendance is 45/50, how many classes can I miss for 75%?", 0, "attendance", 1),
    ("How do I apply for CIA repeat?", 0, None, 1),
    ("When does the next semester reopen?", 0, None, 1),
    ("What is the internship policy for higher semesters?", 0, None, 1),
    ("How do I raise a grievance?", 0, None, 1),
    ("Who won the cricket world cup?", 1, None, None),
    ("What is the hostel mess menu today?", 1, None, -1),
    ("Is there a bus pass for day scholars?", 1, None, -1),
    ("What is the fee for the MBA programme?", 1, None, -1),
    ("Can I take a re-exam if I was ill during ESE?", 0, None, -1),
    ("Where do I find the academic calendar?", 0, None, 1),
    ("What are the grade points for an A+?", 0, None, 1),
    ("I attended 33 of 46 classes, target 75%. What do I need?", 0, "attendance", 1),
]
ROLES = ["student"] * 4 + ["faculty"]


def seed_db(reset: bool):
    if reset and os.path.exists(db.DB_PATH):
        os.remove(db.DB_PATH)
    for rep in range(3):  # a few days of repeated usage so the dashboard looks alive
        for q, ab, tool, rating in QUESTIONS:
            role = random.choice(ROLES)
            ans = "(mock) I could not find this in the loaded documents." if ab else "(mock) Answer shown in the live demo."
            cid = db.log_chat(role, role, q, dict(answer=ans, abstained=bool(ab), tool=tool, sources=[], latency_s=round(random.uniform(0.4, 2.1), 2)))
            if rating and random.random() < 0.8:
                db.add_feedback(role, cid, rating)
    print("Seeded demo chat log:", db.stats())


def write_results():
    """Illustrative numbers. NOT measured. Replaced automatically once evaluation.evaluate writes data/eval/results_summary.csv."""
    import pandas as pd
    df = pd.DataFrame({
        "accuracy": [0.30, 0.77, 0.43, 0.83],
        "faithfulness": [None, 0.80, None, 0.90],
        "hallucination": [0.50, 0.13, 0.37, 0.07],
        "retrieval_hit_at_k": [None, 0.87, None, 0.87],
        "latency_s": [1.1, 2.4, 1.1, 2.5]},
        index=pd.Index(["base", "base+rag", "ft", "ft+rag"], name="config"))
    df.to_csv(MOCK / "results_summary.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ax = df["accuracy"].plot.bar(color=["#999", "#24398f", "#c77", "#2a7"], rot=0)
    ax.set_ylim(0, 1); ax.set_ylabel("accuracy")
    ax.set_title("MOCK / ILLUSTRATIVE: not measured results")
    plt.tight_layout(); plt.savefig(MOCK / "accuracy_mock.png", dpi=150); plt.close()
    steps = list(range(0, 121, 5))
    loss = [round(2.1 * math.exp(-s / 40) + 0.55 + random.uniform(-0.03, 0.03), 3) for s in steps]
    plt.plot(steps, loss, color="#24398f"); plt.xlabel("step"); plt.ylabel("train loss")
    plt.title("MOCK / ILLUSTRATIVE training loss curve"); plt.tight_layout()
    plt.savefig(MOCK / "loss_curve_mock.png", dpi=150); plt.close()
    print("Wrote mock results + plots to", MOCK)


SAMPLE_CIRCULAR = """SAMPLE CIRCULAR (FICTIONAL, FOR DEMO ONLY)
Office of the Registrar (Academics)

Subject: Submission of CIA project reports, Semester IV

All Semester IV students must submit the CIA project report on or before 12 December 2026 through the department office.
Late submissions up to three working days will be accepted with a penalty of 10% of the marks awarded.
Reports must be typed in 12 pt Times New Roman with 1.5 line spacing and must not exceed 25 pages.
Each report must carry a plagiarism certificate showing a similarity index below 15%.
Students who are absent on the viva day must inform the course coordinator in writing before the session begins.
The viva voce for the project will be held between 15 and 19 December 2026 in the department seminar hall.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    MOCK.mkdir(parents=True, exist_ok=True)
    (MOCK / "sample_circular.txt").write_text(SAMPLE_CIRCULAR, encoding="utf-8")
    seed_db(a.reset)
    write_results()
    (MOCK / "README.txt").write_text("Everything here is mock/illustrative data for demos. Do not present it as measured results.\n")


if __name__ == "__main__":
    main()
