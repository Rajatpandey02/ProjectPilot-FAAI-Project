"""Scheduling tools: capacity maps, EDF allocation, session splitting, conflict detection.

All scheduling works in 0.5-hour quanta so plans stay readable.
"""
from __future__ import annotations

import math

import pandas as pd

from projectpilot.tools.registry import tool

QUANTUM = 0.5
SEVERITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}
CONFLICT_COLS = ["severity", "type", "task_id", "date", "message"]


def to_quantum_up(hours: float) -> float:
    """Round effort up to the next 0.5h (never under-plan)."""
    return math.ceil(round(float(hours) / QUANTUM, 6)) * QUANTUM


@tool("build_capacity_map", "Map each day from start to end to usable study hours (zero-hour days omitted).")
def build_capacity_map(availability: dict, start, end) -> dict:
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    capacity = {}
    for day in pd.date_range(start, end):
        hours = math.floor(float(availability.get(day.day_name(), 0.0)) / QUANTUM) * QUANTUM
        if hours > 0:
            capacity[day] = hours
    return capacity


@tool(
    "allocate_hours",
    "Fill days in order: earlier items in the list get each day's hours first. "
    "Returns allocations and any hours that did not fit.",
)
def allocate_hours(items: list[dict], capacity: dict) -> dict:
    remaining = {it["task_id"]: float(it["hours"]) for it in items}
    allocations = []
    for day, cap in capacity.items():
        free = float(cap)
        for it in items:
            if free <= 1e-9:
                break
            tid = it["task_id"]
            if remaining[tid] <= 1e-9:
                continue
            give = min(remaining[tid], free)
            allocations.append({"task_id": tid, "date": day, "hours": give})
            remaining[tid] -= give
            free -= give
        if all(v <= 1e-9 for v in remaining.values()):
            break
    unscheduled = {t: round(v, 2) for t, v in remaining.items() if v > 1e-9}
    return {"allocations": allocations, "unscheduled": unscheduled}


@tool("finish_dates", "Last day on which each task has scheduled work.")
def finish_dates(allocations: list[dict]) -> dict:
    out: dict = {}
    for a in allocations:
        if a["task_id"] not in out or a["date"] > out[a["task_id"]]:
            out[a["task_id"]] = a["date"]
    return out


@tool("split_sessions", "Split one day's hours for a task into study sessions of at most max_session hours.")
def split_sessions(hours: float, max_session: float = 2.0) -> list[float]:
    max_session = max(QUANTUM, float(max_session))
    sessions, left = [], float(hours)
    while left > 1e-9:
        chunk = min(max_session, left)
        sessions.append(round(chunk, 2))
        left -= chunk
    return sessions


@tool(
    "detect_conflicts",
    "Validate a plan: overbooked days, unscheduled work, deadline misses, overdue catch-ups, "
    "zero-buffer tasks and deadline clusters.",
)
def detect_conflicts(schedule: pd.DataFrame, task_plan: pd.DataFrame, availability: dict, today) -> pd.DataFrame:
    rows: list[dict] = []

    def add(severity, ctype, task_id, day, message):
        rows.append({"severity": severity, "type": ctype, "task_id": task_id, "date": day, "message": message})

    # 1) Day-level: never schedule more than the student can study
    if not schedule.empty:
        for day, hours in schedule.groupby("date")["hours"].sum().items():
            cap = float(availability.get(pd.Timestamp(day).day_name(), 0.0))
            if hours > cap + 1e-9:
                add("High", "OVERBOOKED_DAY", "", day, f"{hours:.1f}h scheduled but only {cap:.1f}h available")

    # 2) Task-level
    if not task_plan.empty:
        for r in task_plan.itertuples():
            unsched = r.plan_hours - r.scheduled_hours
            if unsched > 1e-9:
                add("High", "UNSCHEDULED", r.task_id, r.deadline,
                    f"'{r.title}': {unsched:.1f}h cannot fit in the planning horizon")
            elif r.is_overdue:
                add("High", "OVERDUE", r.task_id, r.deadline,
                    f"'{r.title}' is already overdue; catch-up plan finishes {r.finish_date:%d %b}")
            elif pd.notna(r.finish_date) and r.finish_date > r.deadline:
                late = (r.finish_date - r.deadline).days
                add("High", "DEADLINE_MISS", r.task_id, r.deadline,
                    f"'{r.title}' finishes {late} day(s) after its {r.deadline:%d %b} deadline")
            elif r.buffer_days == 0:
                add("Low", "NO_BUFFER", r.task_id, r.deadline,
                    f"'{r.title}' finishes on its deadline day: no slack if anything slips")

        # 3) Deadline clusters: 3+ deadlines inside any 3-day window
        live = task_plan[~task_plan["is_overdue"].astype(bool)].sort_values("deadline")
        reported: list[set] = []
        for d in pd.to_datetime(live["deadline"].unique()):
            window = live[(live["deadline"] >= d) & (live["deadline"] <= d + pd.Timedelta(days=2))]
            ids = set(window["task_id"])
            if len(ids) >= 3 and not any(ids <= prev for prev in reported):
                reported.append(ids)
                add("Medium", "DEADLINE_CLUSTER", "", d,
                    f"{len(ids)} deadlines within 3 days ({d:%d %b} to {window['deadline'].max():%d %b}): "
                    + ", ".join(window["title"]))

    out = pd.DataFrame(rows, columns=CONFLICT_COLS)
    if out.empty:
        return out
    out["_sev"] = out["severity"].map(SEVERITY_ORDER)
    return out.sort_values(["_sev", "date"]).drop(columns="_sev").reset_index(drop=True)