"""Project constants and configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHARTS_DIR = PROJECT_ROOT / "charts"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATASET_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

settings = {
    "cols_to_unify": [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ],
    "binary_cols": [
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaperlessBilling",
    ],
    "one_hot_encoding_cols": ["PaymentMethod", "InternetService"],
    "useless_cols_to_drop": [
        "StreamingTV",
        "PhoneService",
        "MultipleLines",
        "StreamingMovies",
        "DeviceProtection",
        "Partner",
        "gender",
    ],
}
