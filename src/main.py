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

from src.utils import (  # noqa: E402
    dataset_overview,
    display_unique_values_for_object_columns,
    download_dataset,
    encode_target,
    generate_report,
    load_dataset,
    preprocess_TotalCharges,
    unify_redundant_categories,
)


def main() -> None:
    """Main function - project entry point."""
    # download_dataset()
    df = load_dataset()
    # generate_report(df)
    
    #data analysis
    dataset_overview(df)

    #data preprocessing
    df = preprocess_TotalCharges(df)
    df = encode_target(df)
    df = df.drop(columns=["customerID"])
    
    display_unique_values_for_object_columns(df)
    df = unify_redundant_categories(df)

    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    df.head()


if __name__ == "__main__":
    main()
