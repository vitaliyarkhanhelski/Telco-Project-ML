"""Data loading utilities."""

import kaggle
import pandas as pd
from dotenv import load_dotenv

from src.settings import DATA_DIR, DATASET_FILENAME, PROJECT_ROOT

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
