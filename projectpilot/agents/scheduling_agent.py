"""Scheduling Agent: builds a conflict-checked, day-by-day plan from ranked tasks."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from projectpilot.agents.base import BaseAgent, TraceStep
from projectpilot.tools.scheduling import CONFLICT_COLS, to_quantum_up

SCHEDULE_COLS = ["date", "weekday", "slot", "task_id", "title", "course", "task_type",
                 "session", "hours", "deadline", "priority_tier", "plan_status"]
PLAN_COLS = ["task_id", "title", "course", "task_type", "priority_tier", "priority_rank",
             "deadline", "is_overdue", "plan_hours", "scheduled_hours", "start_date",
             "finish_date", "buffer_days", "deferred", "plan_status"]
DAILY_COLS = ["date", "capacity", "scheduled", "utilization"]


@dataclass
class ScheduleResult:
    schedule: pd.DataFrame
    task_plan: pd.DataFrame
    daily_load: pd.DataFrame
    conflicts: pd.DataFrame
    trace: list[TraceStep]
    tool_calls: dict[str, int]


class SchedulingAgent(BaseAgent):
    name = "Scheduling Agent"

    def run(self, priority_table: pd.DataFrame, availability: dict, today=None,
            max_session_hours: float = 2.0, buffer_days: int = 14) -> ScheduleResult:
        self.reset()
        today = pd.Timestamp(today or date.today()).normalize()
        tasks = priority_table.copy()

        if not tasks.empty:
            done = tasks["remaining_hours"] <= 0
            if done.any():
                self.log("observe", f"Skipping {int(done.sum())} task(s) with no remaining effort: "
                                    + ", ".join(tasks.loc[done, "title"]))
            tasks = tasks[~done].copy()
        if tasks.empty:
            self.log("result", "Nothing to schedule.")
            return self._result(pd.DataFrame(columns=SCHEDULE_COLS), pd.DataFrame(columns=PLAN_COLS),
                                pd.DataFrame(columns=DAILY_COLS), pd.DataFrame(columns=CONFLICT_COLS))

        tasks["plan_hours"] = tasks["remaining_hours"].map(to_quantum_up)
        info = tasks.set_index("task_id")

        # ---- observe: how much work vs how much time
        horizon_end = max(tasks["deadline"].max(), today) + pd.Timedelta(days=buffer_days)
        capacity = self.use_tool("build_capacity_map", availability=availability, start=today, end=horizon_end)
        self.log("observe", f"{len(tasks)} task(s) need {tasks['plan_hours'].sum():.1f}h; "
                            f"{sum(capacity.values()):.1f}h of study time between "
                            f"{today:%d %b} and {horizon_end:%d %b}.")
        if not capacity:
            self.log("decision", "No study hours are available: nothing can be scheduled. Raise your availability.")

        overdue = tasks.loc[tasks["is_overdue"]].sort_values("priority_rank")["task_id"].tolist()
        on_time = tasks.loc[~tasks["is_overdue"]].sort_values(["deadline", "priority_rank"])["task_id"].tolist()
        deferred: list[str] = []
        if overdue:
            self.log("decision", "Overdue tasks scheduled first as catch-up: "
                                 + ", ".join(info.at[t, "title"] for t in overdue))

        # ---- decide: simulate EDF; while a deadline would be missed, defer the lowest-priority competitor
        for _ in range(len(tasks) + 1):
            sim = self.use_tool("allocate_hours", items=self._items(info, overdue, on_time), capacity=capacity)
            finish = self.use_tool("finish_dates", allocations=sim["allocations"])
            missed = [t for t in on_time
                      if t in sim["unscheduled"] or finish[t] > info.at[t, "deadline"]]
            if not missed:
                break
            first = missed[0]
            cutoff = info.at[first, "deadline"]
            candidates = [t for t in on_time if info.at[t, "deadline"] <= cutoff]
            victim = max(candidates, key=lambda t: info.at[t, "priority_rank"])
            on_time.remove(victim)
            deferred.append(victim)
            self.log("decision",
                     f"{first} '{info.at[first, 'title']}' would miss its {cutoff:%d %b} deadline. "
                     f"Deferring lowest-priority competitor {victim} '{info.at[victim, 'title']}' "
                     f"({info.at[victim, 'priority_tier']}, rank {int(info.at[victim, 'priority_rank'])}) "
                     f"to free {info.at[victim, 'plan_hours']:.1f}h.")

        # ---- act: final allocation (deferred tasks use leftover hours only)
        final_items = self._items(info, overdue, on_time, deferred)
        sim = self.use_tool("allocate_hours", items=final_items, capacity=capacity)
        finish = self.use_tool("finish_dates", allocations=sim["allocations"])
        alloc = pd.DataFrame(sim["allocations"], columns=["task_id", "date", "hours"])

        sched_hours = alloc.groupby("task_id")["hours"].sum()
        first_day = alloc.groupby("task_id")["date"].min()
        overdue_set, deferred_set = set(overdue), set(deferred)

        plan_rows = []
        for tid in tasks.sort_values("priority_rank")["task_id"]:
            scheduled = float(sched_hours.get(tid, 0.0))
            finish_day = finish.get(tid, pd.NaT)
            deadline = info.at[tid, "deadline"]
            if scheduled + 1e-9 < info.at[tid, "plan_hours"]:
                status = "Unscheduled"
            elif tid in overdue_set:
                status = "Overdue (catch-up)"
            elif finish_day > deadline:
                status = "Late"
            else:
                status = "On track"
            plan_rows.append({
                "task_id": tid, "title": info.at[tid, "title"], "course": info.at[tid, "course"],
                "task_type": info.at[tid, "task_type"], "priority_tier": info.at[tid, "priority_tier"],
                "priority_rank": int(info.at[tid, "priority_rank"]), "deadline": deadline,
                "is_overdue": bool(info.at[tid, "is_overdue"]),
                "plan_hours": float(info.at[tid, "plan_hours"]), "scheduled_hours": scheduled,
                "start_date": first_day.get(tid, pd.NaT), "finish_date": finish_day,
                "buffer_days": (deadline - finish_day).days if pd.notna(finish_day) else float("nan"),
                "deferred": tid in deferred_set, "plan_status": status,
            })
        task_plan = pd.DataFrame(plan_rows, columns=PLAN_COLS)
        for col in ("deadline", "start_date", "finish_date"):
            task_plan[col] = pd.to_datetime(task_plan[col])

        # ---- session-level schedule
        status_of = dict(zip(task_plan["task_id"], task_plan["plan_status"]))
        order_index = {it["task_id"]: n for n, it in enumerate(final_items)}
        alloc["_order"] = alloc["task_id"].map(order_index)
        rows = []
        for day, day_rows in alloc.sort_values(["date", "_order"]).groupby("date", sort=True):
            slot = 0
            for r in day_rows.itertuples():
                sessions = self.use_tool("split_sessions", hours=r.hours, max_session=max_session_hours)
                for k, hrs in enumerate(sessions, 1):
                    slot += 1
                    rows.append({
                        "date": day, "weekday": day.day_name(), "slot": slot, "task_id": r.task_id,
                        "title": info.at[r.task_id, "title"], "course": info.at[r.task_id, "course"],
                        "task_type": info.at[r.task_id, "task_type"], "session": f"{k}/{len(sessions)}",
                        "hours": hrs, "deadline": info.at[r.task_id, "deadline"],
                        "priority_tier": info.at[r.task_id, "priority_tier"],
                        "plan_status": status_of[r.task_id],
                    })
        schedule = pd.DataFrame(rows, columns=SCHEDULE_COLS)

        # ---- daily load (for charts)
        load = alloc.groupby("date")["hours"].sum()
        display_end = max([tasks["deadline"].max()] + ([alloc["date"].max()] if not alloc.empty else []))
        daily = pd.DataFrame(
            [{"date": d, "capacity": c, "scheduled": float(load.get(d, 0.0))}
             for d, c in capacity.items() if d <= display_end],
            columns=DAILY_COLS,
        )
        daily["utilization"] = (daily["scheduled"] / daily["capacity"]).round(2) if not daily.empty else []

        # ---- validate
        conflicts = self.use_tool("detect_conflicts", schedule=schedule, task_plan=task_plan,
                                  availability=availability, today=today)
        for r in task_plan[task_plan["plan_status"] == "Late"].itertuples():
            self.log("decision", f"{r.task_id} '{r.title}' will finish {-int(r.buffer_days)} day(s) late: "
                                 "flagged for the Recovery Agent.")
        by_sev = conflicts["severity"].value_counts().to_dict() if not conflicts.empty else {}
        self.log("result", f"Scheduled {alloc['hours'].sum():.1f}h over {alloc['date'].nunique()} day(s). "
                           f"Status: {task_plan['plan_status'].value_counts().to_dict()}. "
                           f"Conflicts: {by_sev or 'none'}.")
        return self._result(schedule, task_plan, daily, conflicts)

    @staticmethod
    def _items(info, overdue, on_time, deferred=()):
        ordered = list(overdue) + list(on_time) + sorted(deferred, key=lambda t: info.at[t, "priority_rank"])
        return [{"task_id": t, "hours": float(info.at[t, "plan_hours"])} for t in ordered]

    def _result(self, schedule, task_plan, daily, conflicts) -> ScheduleResult:
        return ScheduleResult(schedule, task_plan, daily, conflicts, self.trace, dict(self.tool_calls))