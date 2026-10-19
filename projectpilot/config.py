from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TASKS_FILE = DATA_DIR / "tasks.csv"
AVAILABILITY_FILE = DATA_DIR / "availability.json"

TASK_TYPES = ["Assignment", "Practical", "Project", "Quiz"]
STATUSES = ["Not Started", "In Progress", "Completed", "Delayed"]

COLUMNS = [
    "task_id",
    "title",
    "course",
    "task_type",
    "deadline",
    "estimated_hours",
    "hours_completed",
    "status",
    "notes",
]
REQUIRED_IMPORT_COLUMNS = ["title", "course", "task_type", "deadline", "estimated_hours"]

# Hours a student can realistically study each weekday.
DEFAULT_AVAILABILITY = {
    "Monday": 2.0,
    "Tuesday": 2.0,
    "Wednesday": 2.0,
    "Thursday": 2.0,
    "Friday": 1.5,
    "Saturday": 4.0,
    "Sunday": 3.0,
}
WEEKDAYS = list(DEFAULT_AVAILABILITY)