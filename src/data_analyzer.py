"""Data analysis utilities."""

import pandas as pd
from ydata_profiling import ProfileReport

from src.settings import REPORT_FILENAME, REPORTS_DIR


def display_unique_values_for_object_columns(df: pd.DataFrame) -> None:
    """Print unique values for each column of object (string) type."""
    for col in df.select_dtypes(include="object").columns:
        print(f"{col}: {df[col].unique()}")


def dataset_overview(df: pd.DataFrame) -> None:
    """Print initial dataset exploration: describe, head, missing values, dtypes, and unique values for object columns."""
    print(df.head())
    print(df.describe())
    print(df.describe(include="object"))

    print("\nMissing values per column:")
    print(df.isna().sum())
    print('')

    print(df.info())
    print('')

    print("\nChurn value counts:")
    print(df.Churn.value_counts())
    print("\nChurn (%):")
    print((df.Churn.value_counts(normalize=True) * 100).round(1).astype(str) + "%")
    print('')

    display_unique_values_for_object_columns(df)
    print('')


def generate_report(
        df: pd.DataFrame, report_name: str = REPORT_FILENAME
) -> None:
    """Generate a ydata-profiling HTML report for the dataset. Saves to reports/ folder."""
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
    print("\n'Telco Customer Churn' report was generated and saved to:", output_path)


def run_initial_analysis(df: pd.DataFrame) -> None:
    """Run initial data analysis: generate report and dataset overview."""
    dataset_overview(df)
    generate_report(df)

