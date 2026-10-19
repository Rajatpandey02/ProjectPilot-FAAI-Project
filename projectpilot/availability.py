from __future__ import annotations

import json

import pandas as pd

from projectpilot.config import AVAILABILITY_FILE, DATA_DIR, DEFAULT_AVAILABILITY


def load_availability() -> dict[str, float]:
    avail = dict(DEFAULT_AVAILABILITY)
    if AVAILABILITY_FILE.exists():
        try:
            stored = json.loads(AVAILABILITY_FILE.read_text())
            for day in avail:
                if day in stored:
                    avail[day] = max(0.0, float(stored[day]))
        except (ValueError, OSError):
            pass  # fall back to defaults on a corrupt file
    return avail


def save_availability(avail: dict[str, float]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    AVAILABILITY_FILE.write_text(json.dumps(avail, indent=2))


def capacity_between(avail: dict[str, float], start, end) -> float:
    """Total study hours from start to end, both inclusive."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if end < start:
        return 0.0
    return float(sum(avail.get(d.day_name(), 0.0) for d in pd.date_range(start, end)))