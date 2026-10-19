"""Priority Agent: scores tasks via tools, then applies conditional rules."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from projectpilot.agents.base import BaseAgent, TraceStep
from projectpilot.task_store import enrich

TIER_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
NEW_COLS = ["capacity_hours", "pressure_ratio", "priority_score", "priority_tier",
            "at_risk", "reasons", "priority_rank"]


@dataclass
class PriorityResult:
    table: pd.DataFrame
    trace: list[TraceStep]
    tool_calls: dict[str, int]


class PriorityAgent(BaseAgent):
    name = "Priority Agent"

    def run(self, tasks: pd.DataFrame, availability: dict, today=None) -> PriorityResult:
        self.reset()
        today = pd.Timestamp(today or date.today()).normalize()
        data = enrich(tasks, today)
        active = data[data["status"] != "Completed"].copy()
        self.log("observe", f"{len(data)} tasks loaded: {len(active)} active, "
                            f"{len(data) - len(active)} completed (skipped).")

        if active.empty:
            for col in NEW_COLS:
                active[col] = pd.Series(dtype=object)
            return PriorityResult(active, self.trace, dict(self.tool_calls))

        overload = self.use_tool("cumulative_overload", tasks=active,
                                 availability=availability, today=today)
        crowded = sum(1 for v in overload.values() if v > 0)
        self.log("observe", f"Cumulative-load check: {crowded} task(s) cannot be finished on time "
                            "if everything due earlier is done first.")

        records = [self._assess(row, availability, today, overload[row.task_id])
                   for row in active.itertuples()]
        active = active.join(pd.DataFrame(records, index=active.index))

        active["_order"] = active["priority_tier"].map(TIER_ORDER)
        active = (active.sort_values(["_order", "priority_score", "deadline"],
                                     ascending=[True, False, True])
                  .drop(columns="_order").reset_index(drop=True))
        active["priority_rank"] = active.index + 1

        counts = active["priority_tier"].value_counts().to_dict()
        self.log("result", f"Tier counts: {counts}. At-risk tasks: {int(active['at_risk'].sum())}.")
        return PriorityResult(active, self.trace, dict(self.tool_calls))

    def _assess(self, row, availability, today, overload_hours: float) -> dict:
        days_left = int(row.days_left)
        capacity = self.use_tool("capacity_until", availability=availability,
                                 today=today, deadline=row.deadline)
        ratio = self.use_tool("pressure_ratio", remaining_hours=row.remaining_hours,
                              capacity_hours=capacity)
        parts = {
            "urgency": self.use_tool("urgency_score", days_left=days_left),
            "pressure": self.use_tool("pressure_score", ratio=ratio),
            "stakes": self.use_tool("stakes_score", task_type=row.task_type),
            "status": self.use_tool("status_score", status=row.status, days_left=days_left),
        }
        score = round(sum(parts.values()), 1)
        tier = self.use_tool("classify_tier", score=score)
        base_tier = tier
        driver = max(parts, key=parts.get)
        reasons = [f"main driver: {driver} ({parts[driver]})"]

        infeasible = ratio > 1.0
        crowded = overload_hours > 0
        at_risk = bool(row.is_overdue or infeasible or crowded)

        # ---- conditional decision rules (first match wins) ----
        if row.remaining_hours <= 0:
            tier, at_risk = "Low", False
            reasons = ["all estimated effort is logged: mark it Completed if it is done"]
        elif row.is_overdue:
            tier = "Critical"
            reasons.append(f"overdue by {abs(days_left)} day(s)")
        elif days_left <= 1:
            tier = "Critical"
            reasons.append("due within 24 hours with work remaining")
        elif infeasible or crowded:
            tier = self.use_tool("escalate_tier", tier=tier)
            if infeasible:
                reasons.append(f"needs {row.remaining_hours:.1f}h but only {capacity:.1f}h available before deadline")
            if crowded:
                reasons.append(f"earlier deadlines leave {overload_hours:.1f}h of overload by this date")

        if tier != base_tier:
            self.log("decision", f"{row.task_id} '{row.title}': {base_tier} → {tier} ({'; '.join(reasons[1:])})")
        self.log("task", f"{row.task_id} score {score} = "
                         + ", ".join(f"{k} {v}" for k, v in parts.items()) + f" → {tier}")

        return {
            "capacity_hours": capacity,
            "pressure_ratio": ratio,
            "priority_score": score,
            "priority_tier": tier,
            "at_risk": at_risk,
            "reasons": "; ".join(reasons),
        }