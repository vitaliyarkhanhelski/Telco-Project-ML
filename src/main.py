"""
Telco Customer Churn - Machine Learning Analysis
Main entry point for the project.
"""
#internal
import sys
from pathlib import Path

#external
import pandas as pd

# Project root on path (so "from src.utils" works for both run modes)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import download_dataset, generate_report, load_dataset, preprocess_TotalCharges  # noqa: E402


def main() -> None:
    """Main function - project entry point."""
    # download_dataset()
    df = load_dataset()
    # generate_report(df)

    # check out the data
    print(df.head())

    # check out nan values in dataset
    print(df.isna().sum())

    # check out the data types
    print(df.info())

    df = preprocess_TotalCharges(df)



if __name__ == "__main__":
    main()
