from pathlib import Path

APP_NAME = "AI Trading Research"
MODEL_VERSION = "logreg-v1"
FEATURE_VERSION = "causal-features-v1"
DEFAULT_DB_PATH = Path.home() / ".ai_trading" / "journal.db"
SAMPLE_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "demo_daily.csv"
MAX_UPLOAD_BYTES = 2_000_000
CSV_EXPORT_PREFIXES = ("=", "+", "-", "@")
