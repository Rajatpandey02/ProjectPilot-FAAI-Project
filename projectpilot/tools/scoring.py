"""Priority tools. Each is a small pure function so it can be tested alone.

Score = urgency (0-40) + workload pressure (0-30) + stakes (0-15) + status (0-15) = 0-100
"""
from __future__ import annotations

import pandas as pd

from projectpilot.availability import capacity_between
from projectpilot.tools.registry import tool

TIERS = ["Low", "Medium", "High", "Critical"]
TYPE_STAKES = {"Project": 15.0, "Quiz": 12.0, "Practical": 10.0, "Assignment": 8.0}
MAX_PRESSURE_RATIO = 5.0


@tool("urgency_score", "Deadline proximity score 0-40 (40 = due today or overdue).")
def urgency_score(days_left: int) -> float:
    if days_left <= 0:
        return 40.0
    return round(max(2.0, 40.0 * (1 - days_left / 14)), 1)


@tool("capacity_until", "Study hours available from today through the deadline (inclusive).")
def capacity_until(availability: dict, today, deadline) -> float:
    return capacity_between(availability, today, deadline)


@tool("pressure_ratio", "Remaining effort / available hours before the deadline (capped at 5).")
def pressure_ratio(remaining_hours: float, capacity_hours: float) -> float:
    if remaining_hours <= 0:
        return 0.0
    if capacity_hours <= 0:
        return MAX_PRESSURE_RATIO
    return round(min(remaining_hours / capacity_hours, MAX_PRESSURE_RATIO), 2)


@tool("pressure_score", "Workload pressure score 0-30 (30 = needs all available hours or more).")
def pressure_score(ratio: float) -> float:
    return round(min(30.0, 30.0 * ratio), 1)


@tool("stakes_score", "Score 0-15 by task type: Project 15, Quiz 12, Practical 10, Assignment 8.")
def stakes_score(task_type: str) -> float:
    return TYPE_STAKES.get(task_type, 8.0)


@tool("status_score", "Status adjustment 0-15: Delayed 15; Not Started 5 if due within 3 days else 2.")
def status_score(status: str, days_left: int) -> float:
    if status == "Delayed":
        return 15.0
    if status == "Not Started":
        return 5.0 if days_left <= 3 else 2.0
    return 0.0


@tool("classify_tier", "Map a 0-100 score to a tier: >=70 Critical, >=50 High, >=30 Medium, else Low.")
def classify_tier(score: float) -> str:
    if score >= 70:
        return "Critical"
    if score >= 50:
        return "High"
    if score >= 30:
        return "Medium"
    return "Low"


@tool("escalate_tier", "Move a tier up by one level (Critical stays Critical).")
def escalate_tier(tier: str) -> str:
    return TIERS[min(TIERS.index(tier) + 1, len(TIERS) - 1)]


@tool(
    "cumulative_overload",
    "Earliest-deadline-first check: hours by which the running workload exceeds capacity at each deadline.",
)
def cumulative_overload(tasks: pd.DataFrame, availability: dict, today) -> dict[str, float]:
    """`tasks` must be enriched (has remaining_hours). Returns {task_id: overload_hours}."""
    active = tasks[tasks["status"] != "Completed"].sort_values("deadline")
    running = 0.0
    result: dict[str, float] = {}
    for row in active.itertuples():
        running += row.remaining_hours
        capacity = capacity_between(availability, today, row.deadline)
        result[row.task_id] = round(max(0.0, running - capacity), 1)
    return result