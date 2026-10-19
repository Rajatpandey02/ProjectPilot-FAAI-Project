from __future__ import annotations

from datetime import date

import pandas as pd

from projectpilot.config import (
    COLUMNS,
    DATA_DIR,
    STATUSES,
    TASK_TYPES,
    TASKS_FILE,
)

_TEXT_COLS = ["task_id", "title", "course", "task_type", "status", "notes"]
_DEFAULTS = {"task_id": "", "status": "Not Started", "hours_completed": 0.0, "notes": ""}


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with all expected columns and consistent dtypes."""
    df = df.copy()
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = _DEFAULTS.get(col, pd.NA)
    df = df[COLUMNS]

    for col in _TEXT_COLS:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df["task_type"] = df["task_type"].str.title()
    df["status"] = df["status"].str.title()
    df.loc[df["status"] == "", "status"] = "Not Started"

    df["deadline"] = pd.to_datetime(df["deadline"], errors="coerce")
    df["estimated_hours"] = pd.to_numeric(df["estimated_hours"], errors="coerce")
    df["hours_completed"] = pd.to_numeric(df["hours_completed"], errors="coerce").fillna(0.0)
    return df.reset_index(drop=True)


def assign_missing_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Give every task without an ID a unique one like T007."""
    df = df.copy()
    used = set(df.loc[df["task_id"] != "", "task_id"])
    counter = 1
    for idx in df.index[df["task_id"] == ""]:
        while f"T{counter:03d}" in used:
            counter += 1
        new_id = f"T{counter:03d}"
        df.at[idx, "task_id"] = new_id
        used.add(new_id)
    return df


def validate_tasks(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (valid_rows, error_messages). Invalid rows are dropped and reported."""
    df = normalize(df)
    errors: list[str] = []
    keep = pd.Series(True, index=df.index)

    for i, row in df.iterrows():
        problems = []
        if not row["title"]:
            problems.append("missing title")
        if not row["course"]:
            problems.append("missing course")
        if row["task_type"] not in TASK_TYPES:
            problems.append(f"task_type '{row['task_type']}' not in {TASK_TYPES}")
        if row["status"] not in STATUSES:
            problems.append(f"status '{row['status']}' not in {STATUSES}")
        if pd.isna(row["deadline"]):
            problems.append("invalid or missing deadline")
        if pd.isna(row["estimated_hours"]) or row["estimated_hours"] <= 0:
            problems.append("estimated_hours must be a number > 0")
        if row["hours_completed"] < 0:
            problems.append("hours_completed cannot be negative")
        if problems:
            keep[i] = False
            errors.append(f"Row {i + 1} ('{row['title'] or 'untitled'}'): " + "; ".join(problems))

    clean = assign_missing_ids(df[keep])
    dupes = clean["task_id"].duplicated(keep="first")
    if dupes.any():
        errors.append(f"Dropped duplicate task IDs: {', '.join(clean.loc[dupes, 'task_id'])}")
        clean = clean[~dupes]
    return clean.reset_index(drop=True), errors


def load_tasks() -> pd.DataFrame:
    if TASKS_FILE.exists():
        clean, _ = validate_tasks(pd.read_csv(TASKS_FILE))
        return clean
    return normalize(pd.DataFrame(columns=COLUMNS))


def save_tasks(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["deadline"] = out["deadline"].dt.strftime("%Y-%m-%d")
    out.to_csv(TASKS_FILE, index=False)


def tasks_to_csv_bytes(df: pd.DataFrame) -> bytes:
    out = df.copy()
    out["deadline"] = out["deadline"].dt.strftime("%Y-%m-%d")
    return out.to_csv(index=False).encode("utf-8")


def add_task(
    df: pd.DataFrame,
    title: str,
    course: str,
    task_type: str,
    deadline,
    estimated_hours: float,
    hours_completed: float = 0.0,
    status: str = "Not Started",
    notes: str = "",
) -> tuple[pd.DataFrame, list[str]]:
    """Append one task. On validation failure the original df is returned with errors."""
    row = pd.DataFrame([{
        "task_id": "", "title": title, "course": course, "task_type": task_type,
        "deadline": deadline, "estimated_hours": estimated_hours,
        "hours_completed": hours_completed, "status": status, "notes": notes,
    }])
    combined = row if df.empty else pd.concat([df, row], ignore_index=True)
    clean, errors = validate_tasks(combined)
    return (df, errors) if errors else (clean, [])


def enrich(df: pd.DataFrame, today=None) -> pd.DataFrame:
    """Add derived columns the agents will rely on later."""
    today = pd.Timestamp(today or date.today()).normalize()
    out = df.copy()
    out["remaining_hours"] = (out["estimated_hours"] - out["hours_completed"]).clip(lower=0)
    out.loc[out["status"] == "Completed", "remaining_hours"] = 0.0
    out["days_left"] = (out["deadline"] - today).dt.days
    out["is_overdue"] = (out["days_left"] < 0) & (out["status"] != "Completed")
    return out