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
    download_dataset,
    encode_target_variable,
    generate_report,
    load_dataset,
    preprocess_TotalCharges,
    print_dataset_overview,
)


def main() -> None:
    """Main function - project entry point."""
    # download_dataset()
    df = load_dataset()
    # generate_report(df)
    
    print_dataset_overview(df)

    df = preprocess_TotalCharges(df)
    df = encode_target_variable(df)


if __name__ == "__main__":
    main()
