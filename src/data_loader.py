"""Data loading utilities."""

from pathlib import Path

import kaggle
import pandas as pd
from dotenv import load_dotenv

# Project paths (for data)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Load .env (Kaggle credentials) before any Kaggle API calls
load_dotenv(PROJECT_ROOT / ".env")


def download_dataset() -> None:
    """Download the Telco Customer Churn dataset from Kaggle to the data folder."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    kaggle.api.dataset_download_files(
        "blastchar/telco-customer-churn", path=str(DATA_DIR), unzip=True
    )


def load_dataset() -> pd.DataFrame:
    """Load the Telco Customer Churn dataset from the data folder."""
    return pd.read_csv(DATA_DIR / DATASET_FILENAME)


def load_data() -> pd.DataFrame:
    """Download dataset from Kaggle (if needed) and load it into a DataFrame."""
    download_dataset()
    return load_dataset()
