"""
Telco Customer Churn - Machine Learning Analysis
Main entry point for the project.
ML Pipeline.
"""
#internal
import sys
from pathlib import Path

#external
import pandas as pd
pd.set_option("display.max_columns", None)

# Project root on path (so "from src.*" imports work for both run modes)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src.data_analyzer as data_analyzer  # noqa: E402
import src.data_loader as data_loader  # noqa: E402
import src.data_preprocessing as data_preprocessing  # noqa: E402
import src.visualization as visualization  # noqa: E402
import src.model_trainer as model_trainer  # noqa: E402


def main() -> None:
    """Main function - project entry point."""
    df = data_loader.load_data()
    data_analyzer.run_initial_analysis(df)
    df = data_preprocessing.clean_and_encode_data(df)
    visualization.visualize_feature_relationships(df)
    
    # Feature Selection - remove noisy features
    df = data_preprocessing.drop_useless_columns(df)

    # X = df.drop('Churn', axis=1)
    # y = df['Churn']

    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # print(f"Training set: {X_train.shape[0]} rows")
    # print(f"Test set: {X_test.shape[0]} rows")

    # Model Training & Evaluation
    X_train, X_test, y_train, y_test = model_trainer.split_data(df)
    results_df = model_trainer.train_and_compare_models(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
