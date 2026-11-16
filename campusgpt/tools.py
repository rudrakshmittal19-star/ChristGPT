"""Deterministic tools: the LLM never does attendance arithmetic."""
import math
import re


def classes_can_miss(attended: int, held: int, target: float) -> int:
    return max(0, math.floor(attended / (target / 100) - held + 1e-9))


def classes_needed(attended: int, held: int, target: float):
    p = target / 100
    if attended / held >= p:
        return 0
    return None if p >= 1 else math.ceil((p * held - attended) / (1 - p) - 1e-9)


def parse(q: str):
    m = re.search(r"(\d+)\s*(?:/|out of|of)\s*(\d+)", q)
    if not m or not re.search(r"attend|miss|class|percent|%", q, re.I):
        return None
    a, t = int(m[1]), int(m[2])
    pm = re.search(r"(\d{2,3})\s*%", q)
    p = int(pm[1]) if pm else 85
    return (a, t, p) if 0 < t and a <= t and p < 100 else None


def attendance_answer(q: str):
    args = parse(q)
    if not args:
        return None
    a, t, p = args
    need = classes_needed(a, t, p)
    head = f"Attendance {a}/{t} = {a / t * 100:.1f}% (target {p}%). "
    body = (f"You need to attend {need} more consecutive classes to reach {p}%." if need
            else f"You can still miss {classes_can_miss(a, t, p)} class(es) and stay at or above {p}%.")
    return head + body + " ESE needs 85% aggregate; MSE needs 75% per course (Handbook 2026-27)."
