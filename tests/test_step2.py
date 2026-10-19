import pandas as pd
import pytest

from projectpilot import task_store as ts
from projectpilot.agents import PriorityAgent
from projectpilot.config import DEFAULT_AVAILABILITY
from projectpilot.synthetic import generate_synthetic_tasks
from projectpilot.tools import default_registry

TODAY = pd.Timestamp("2026-10-12")  # a Monday
AVAIL = dict(DEFAULT_AVAILABILITY)


def run(df=None, avail=None):
    df = generate_synthetic_tasks(TODAY) if df is None else df
    return PriorityAgent().run(df, avail or AVAIL, TODAY)


def one_task(**overrides):
    row = {"title": "T", "course": "X", "task_type": "Assignment",
           "deadline": TODAY + pd.Timedelta(days=30), "estimated_hours": 1}
    row.update(overrides)
    clean, _ = ts.validate_tasks(pd.DataFrame([row]))
    return clean


def test_all_tools_registered():
    for name in ["urgency_score", "capacity_until", "pressure_ratio", "pressure_score",
                 "stakes_score", "status_score", "classify_tier", "escalate_tier",
                 "cumulative_overload"]:
        assert name in default_registry.names()


def test_unknown_tool_raises():
    with pytest.raises(KeyError):
        default_registry.call("does_not_exist")


def test_urgency_is_monotonic_and_tiers_map_correctly():
    scores = [default_registry.call("urgency_score", days_left=d) for d in (0, 1, 3, 7, 14, 30)]
    assert scores == sorted(scores, reverse=True)
    assert default_registry.call("classify_tier", score=70) == "Critical"
    assert default_registry.call("classify_tier", score=49.9) == "Medium"
    assert default_registry.call("escalate_tier", tier="Low") == "Medium"
    assert default_registry.call("escalate_tier", tier="Critical") == "Critical"


def test_completed_tasks_are_skipped_and_every_tool_call_is_counted():
    result = run()
    assert "OS Assignment: CPU scheduling" not in set(result.table["title"])
    assert result.tool_calls["urgency_score"] == len(result.table)
    assert result.trace


def test_overdue_task_is_critical_and_at_risk():
    row = run().table.query("title == 'DS Assignment: Graph traversal'").iloc[0]
    assert row["priority_tier"] == "Critical" and row["at_risk"]


def test_ranks_are_unique_and_tier_ordered():
    table = run().table
    assert list(table["priority_rank"]) == list(range(1, len(table) + 1))
    order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    tiers = [order[t] for t in table["priority_tier"]]
    assert tiers == sorted(tiers)


def test_infeasible_task_is_flagged_at_risk():
    row = run(one_task(task_type="Project", estimated_hours=40,
                       deadline=TODAY + pd.Timedelta(days=5))).table.iloc[0]
    assert row["at_risk"] and row["priority_tier"] in ("High", "Critical")


def test_light_far_task_stays_low():
    row = run(one_task()).table.iloc[0]
    assert row["priority_tier"] == "Low" and not row["at_risk"]


def test_logged_complete_but_not_marked_is_flagged():
    row = run(one_task(hours_completed=1, status="In Progress")).table.iloc[0]
    assert row["priority_tier"] == "Low" and "Completed" in row["reasons"]


def test_cumulative_overload_detects_deadline_cluster():
    active = ts.enrich(generate_synthetic_tasks(TODAY), TODAY)
    active = active[active["status"] != "Completed"]
    overload = default_registry.call("cumulative_overload", tasks=active, availability=AVAIL, today=TODAY)
    assert max(overload.values()) > 0


def test_more_study_hours_reduce_risk():
    generous = {day: 10.0 for day in AVAIL}
    assert run(avail=generous).table["at_risk"].sum() < run().table["at_risk"].sum()