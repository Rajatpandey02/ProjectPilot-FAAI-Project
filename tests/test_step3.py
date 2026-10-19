import pandas as pd

from projectpilot import task_store as ts
from projectpilot.agents import PriorityAgent, SchedulingAgent
from projectpilot.agents.scheduling_agent import PLAN_COLS
from projectpilot.config import DEFAULT_AVAILABILITY
from projectpilot.synthetic import generate_synthetic_tasks
from projectpilot.tools import default_registry

TODAY = pd.Timestamp("2026-10-12")  # a Monday
AVAIL = dict(DEFAULT_AVAILABILITY)
DAY = pd.Timedelta(days=1)


def pipeline(df=None, avail=None, **kw):
    df = generate_synthetic_tasks(TODAY) if df is None else df
    avail = avail or AVAIL
    prio = PriorityAgent().run(df, avail, TODAY)
    return SchedulingAgent().run(prio.table, avail, TODAY, **kw)


def tasks_from(rows):
    base = {"course": "X", "task_type": "Assignment", "estimated_hours": 1}
    clean, errors = ts.validate_tasks(pd.DataFrame([{**base, **r} for r in rows]))
    assert not errors
    return clean


def types(plan):
    return set(plan.conflicts["type"])


# ---------------------------------------------------------------- tools
def test_capacity_map_omits_zero_days():
    cap = default_registry.call("build_capacity_map", availability=AVAIL, start=TODAY, end=TODAY + 6 * DAY)
    assert len(cap) == 7 and sum(cap.values()) == 16.5
    no_monday = {**AVAIL, "Monday": 0.0}
    cap = default_registry.call("build_capacity_map", availability=no_monday, start=TODAY, end=TODAY + 6 * DAY)
    assert len(cap) == 6


def test_allocate_hours_is_earliest_first_and_reports_overflow():
    cap = {TODAY: 3.0, TODAY + DAY: 3.0}
    out = default_registry.call("allocate_hours", capacity=cap,
                                items=[{"task_id": "A", "hours": 4.0}, {"task_id": "B", "hours": 2.0}])
    got = {(a["task_id"], a["date"].day, a["hours"]) for a in out["allocations"]}
    assert got == {("A", 12, 3.0), ("A", 13, 1.0), ("B", 13, 2.0)} and out["unscheduled"] == {}
    out = default_registry.call("allocate_hours", capacity=cap, items=[{"task_id": "A", "hours": 10.0}])
    assert out["unscheduled"] == {"A": 4.0}


def test_split_sessions_sums_back():
    sessions = default_registry.call("split_sessions", hours=5.0, max_session=2.0)
    assert sessions == [2.0, 2.0, 1.0]


def test_detect_conflicts_flags_overbooked_day():
    schedule = pd.DataFrame([{"date": TODAY, "hours": 10.0}])
    out = default_registry.call("detect_conflicts", schedule=schedule,
                                task_plan=pd.DataFrame(columns=PLAN_COLS), availability=AVAIL, today=TODAY)
    assert list(out["type"]) == ["OVERBOOKED_DAY"]


# ---------------------------------------------------------------- agent
def test_schedule_never_exceeds_daily_availability_and_conserves_hours():
    plan = pipeline()
    for day, hours in plan.schedule.groupby("date")["hours"].sum().items():
        assert hours <= AVAIL[day.day_name()] + 1e-9
    assert "OVERBOOKED_DAY" not in types(plan)
    assert abs(plan.schedule["hours"].sum() - plan.task_plan["scheduled_hours"].sum()) < 1e-9
    assert (plan.task_plan["scheduled_hours"] <= plan.task_plan["plan_hours"] + 1e-9).all()


def test_overdue_task_is_scheduled_first_and_reported():
    plan = pipeline()
    first = plan.schedule.sort_values(["date", "slot"]).iloc[0]
    assert first["title"] == "DS Assignment: Graph traversal" and first["date"] == TODAY
    assert "OVERDUE" in types(plan)


def test_infeasible_plan_defers_lowest_priority_task_and_flags_it():
    rows = [
        {"title": "Quiz A", "task_type": "Quiz", "deadline": TODAY + 2 * DAY, "estimated_hours": 5},
        {"title": "Assign B", "task_type": "Assignment", "deadline": TODAY + 2 * DAY, "estimated_hours": 5},
    ]
    plan = pipeline(tasks_from(rows), avail={d: 2.0 for d in AVAIL})
    status = plan.task_plan.set_index("title")["plan_status"]
    assert status["Quiz A"] == "On track" and status["Assign B"] == "Late"
    assert any(s.kind == "decision" and "deferring" in s.message.lower() for s in plan.trace)
    miss = plan.conflicts[plan.conflicts["type"] == "DEADLINE_MISS"]
    assert list(miss["task_id"]) == [plan.task_plan.set_index("title").at["Assign B", "task_id"]]


def test_demo_data_produces_misses_and_cluster_warning():
    plan = pipeline()
    assert {"DEADLINE_MISS", "DEADLINE_CLUSTER"} <= types(plan)
    assert plan.tool_calls["allocate_hours"] >= 2


def test_more_study_hours_remove_deadline_misses():
    plan = pipeline(avail={d: 10.0 for d in AVAIL})
    assert not ({"DEADLINE_MISS", "UNSCHEDULED"} & types(plan))


def test_zero_hour_days_are_never_used_and_overflow_is_reported():
    plan = pipeline(avail={**AVAIL, "Saturday": 0.0, "Sunday": 0.0})
    assert not set(plan.schedule["weekday"]) & {"Saturday", "Sunday"}
    assert "UNSCHEDULED" in types(plan)


def test_empty_and_fully_logged_inputs_do_not_crash():
    empty = pipeline(ts.normalize(pd.DataFrame()))
    assert empty.schedule.empty and empty.conflicts.empty
    done = pipeline(tasks_from([{"title": "Done", "deadline": TODAY + 5 * DAY, "estimated_hours": 1,
                                 "hours_completed": 1, "status": "In Progress"}]))
    assert done.schedule.empty