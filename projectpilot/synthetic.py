from __future__ import annotations

from datetime import date

import pandas as pd

from projectpilot.task_store import validate_tasks

# (course, type, title, days_from_today, estimated_h, done_h, status)
_TEMPLATES = [
    ("Machine Learning", "Assignment", "ML Assignment 2: Regression models", 3, 6, 2, "In Progress"),
    ("Machine Learning", "Quiz", "ML Quiz 3 (Unit 3)", 2, 3, 0, "Not Started"),
    ("Machine Learning", "Project", "Mini project: Churn prediction", 12, 20, 5, "In Progress"),
    ("Data Structures", "Practical", "DS Lab: Binary trees", 3, 4, 1, "In Progress"),
    ("Data Structures", "Assignment", "DS Assignment: Graph traversal", -1, 5, 3, "Delayed"),
    ("Database Systems", "Practical", "DBMS Lab: Joins and subqueries", 5, 3, 0, "Not Started"),
    ("Database Systems", "Quiz", "DBMS Quiz: Normalization", 9, 2, 0, "Not Started"),
    ("Database Systems", "Project", "DBMS Mini project: Library system", 18, 15, 2, "In Progress"),
    ("Operating Systems", "Assignment", "OS Assignment: CPU scheduling", 6, 4, 4, "Completed"),
    ("Operating Systems", "Practical", "OS Lab: Process synchronization", 3, 3, 0, "Not Started"),
    ("Software Engineering", "Assignment", "SE: SRS document", 7, 8, 3, "In Progress"),
    ("Computer Networks", "Quiz", "CN Quiz: Subnetting", 1, 2, 1, "In Progress"),
    ("Computer Networks", "Assignment", "CN Assignment: Routing protocols", 10, 5, 0, "Not Started"),
]


def generate_synthetic_tasks(today=None) -> pd.DataFrame:
    today = pd.Timestamp(today or date.today()).normalize()
    rows = [
        {
            "title": title, "course": course, "task_type": ttype,
            "deadline": today + pd.Timedelta(days=offset),
            "estimated_hours": est, "hours_completed": done, "status": status,
            "notes": "",
        }
        for course, ttype, title, offset, est, done, status in _TEMPLATES
    ]
    clean, _ = validate_tasks(pd.DataFrame(rows))
    return clean