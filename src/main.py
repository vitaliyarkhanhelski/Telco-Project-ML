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
_PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src.data_analyzer as data_analyzer
import src.data_loader as data_loader
import src.utils as utils
import src.data_preprocessing as data_preprocessing
import src.visualization as visualization
import src.model_trainer as model_trainer


def main() -> None:
    """Main function - project entry point."""
    # Download dataset
    df = data_loader.load_data()
    # Dataset initial analysis
    data_analyzer.run_initial_analysis(df)

    # Business visualization - Generate charts visualizing churn dependency on different features
    visualization.plot_business_insights(df)

    # Data preprocessing
    df = data_preprocessing.clean_and_encode_data(df)
    # Visualize correlation between features and target variable
    visualization.visualize_feature_relationships(df)

    # Feature Selection - remove noisy features after charts analysis
    df = data_preprocessing.drop_useless_columns(df)

    # Model Training & Evaluation
    # Train-test split
    X_train, X_test, y_train, y_test = model_trainer.split_data(df)
    # Train and compare models
    results_df = model_trainer.train_and_compare_models(X_train, X_test, y_train, y_test)
    utils.save_to_csv(results_df, "initial_model_results.csv")

    # Confusion Matrix for Logistic Regression
    y_pred_log_reg = model_trainer.plot_logistic_regression_confusion_matrix(X_train, X_test, y_train, y_test)

    # Tune hyperparameters for all models
    tuned_results_df = model_trainer.tune_hyperparameters(X_train, X_test, y_train, y_test)
    utils.save_to_csv(tuned_results_df, "tuned_model_results.csv")

    # Confusion Matrix for tuned XGBoost (winner after tuning)
    y_pred_xgb_tuned = model_trainer.plot_tuned_xgboost_confusion_matrix(X_train, X_test, y_train, y_test)

    model_trainer.print_business_impact_simulation(y_test, y_pred_log_reg, y_pred_xgb_tuned)


if __name__ == "__main__":
    main()
