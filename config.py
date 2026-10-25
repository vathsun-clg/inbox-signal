from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "saved_models"
DB_PATH = PROJECT_ROOT / "data" / "email_triage.sqlite3"
SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic_emails.csv"

PRIORITIES = ["Critical", "High", "Medium", "Low"]
INTENTS = ["Request", "Complaint", "Information", "Approval", "Meeting", "Notification", "Other"]
REVIEW_STATUSES = ["All", "Pending review", "Reviewed", "Not required"]
ACTIVE_MODEL_OPTIONS = ["logistic_regression", "multinomial_nb"]
DEFAULT_ACTIVE_MODEL = "logistic_regression"
DEFAULT_CONFIDENCE_THRESHOLD = 0.62
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
REQUIRED_CSV_COLUMNS = ["email_id", "subject", "body", "received_at"]

PRIORITY_COLORS = {
    "Critical": "#D9485F",
    "High": "#F08C46",
    "Medium": "#C89B3C",
    "Low": "#4E9F79",
}

INTENT_COLORS = {
    "Request": "#4F46E5",
    "Complaint": "#D9485F",
    "Information": "#2F80ED",
    "Approval": "#8E5CC7",
    "Meeting": "#3A9D8F",
    "Notification": "#6B7280",
    "Other": "#9A6A3A",
}

for folder in (DATA_DIR, MODEL_DIR):
    folder.mkdir(parents=True, exist_ok=True)
