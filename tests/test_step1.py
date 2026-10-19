import pandas as pd

from projectpilot import task_store as ts
from projectpilot.availability import capacity_between
from projectpilot.config import DEFAULT_AVAILABILITY
from projectpilot.synthetic import generate_synthetic_tasks

TODAY = pd.Timestamp("2026-10-12")  # a Monday


def test_synthetic_data_is_valid_and_ids_unique():
    df = generate_synthetic_tasks(TODAY)
    clean, errors = ts.validate_tasks(df)
    assert errors == []
    assert len(clean) == len(df)
    assert clean["task_id"].is_unique


def test_validation_drops_bad_rows_and_normalises_case():
    raw = pd.DataFrame([
        {"title": "Good", "course": "ML", "task_type": "quiz", "deadline": "2026-10-20", "estimated_hours": 2},
        {"title": "", "course": "ML", "task_type": "Quiz", "deadline": "not a date", "estimated_hours": -1},
    ])
    clean, errors = ts.validate_tasks(raw)
    assert len(clean) == 1
    assert clean.iloc[0]["task_type"] == "Quiz"
    assert len(errors) == 1


def test_add_task_generates_new_unique_id():
    df = generate_synthetic_tasks(TODAY)
    new_df, errors = ts.add_task(df, "Extra", "ML", "Quiz", "2026-11-01", 2)
    assert errors == []
    assert len(new_df) == len(df) + 1
    assert new_df["task_id"].is_unique


def test_add_task_rejects_invalid_input():
    df = generate_synthetic_tasks(TODAY)
    same, errors = ts.add_task(df, "", "ML", "Quiz", "2026-11-01", 0)
    assert errors and len(same) == len(df)


def test_enrich_flags_overdue_and_zeroes_completed():
    out = ts.enrich(generate_synthetic_tasks(TODAY), TODAY)
    overdue = out[out["is_overdue"]]
    assert len(overdue) == 1 and overdue.iloc[0]["status"] == "Delayed"
    assert (out.loc[out["status"] == "Completed", "remaining_hours"] == 0).all()


def test_capacity_for_one_full_week():
    avail = dict(DEFAULT_AVAILABILITY)
    assert capacity_between(avail, TODAY, TODAY + pd.Timedelta(days=6)) == sum(avail.values())
    assert capacity_between(avail, TODAY, TODAY - pd.Timedelta(days=1)) == 0.0