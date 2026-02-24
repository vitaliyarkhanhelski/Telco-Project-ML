"""Utility functions for the Telco Churn project."""

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATASET_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Load .env (Kaggle credentials) before any Kaggle API calls
load_dotenv(PROJECT_ROOT / ".env")


def download_dataset() -> None:
    """Download the Telco Customer Churn dataset from Kaggle to the data folder."""
    import kaggle

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    kaggle.api.dataset_download_files(
        "blastchar/telco-customer-churn", path=str(DATA_DIR), unzip=True
    )


def load_dataset() -> pd.DataFrame:
    """Load the Telco Customer Churn dataset from the data folder."""
    return pd.read_csv(DATA_DIR / DATASET_FILENAME)


def preprocess_TotalCharges(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges to numeric and fill missing values with 0. Modifies df in place.

    Missing TotalCharges occur when tenure=0 (new customers before first billing cycle).
    """
    # convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # check out the rows where TotalCharges is NaN and select tenure column and TotalCharges column, 
    # Missing TotalCharges occur only when tenure=0, which makes sense.
    print(df[df['TotalCharges'].isna()][['tenure', 'TotalCharges']])
    # fill missing values with 0
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    return df


def generate_report(df: pd.DataFrame, report_name: str = "Telco_Customer_Churn_report.html") -> None:
    """Generate a ydata-profiling HTML report for the dataset. Saves to reports/ folder."""
    from ydata_profiling import ProfileReport

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / report_name

    report = ProfileReport(
        df,
        title="Telco Customer Churn Data",
        correlations={
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
            "kendall": {"calculate": True},
        },
    )
    report.to_file(str(output_path))
