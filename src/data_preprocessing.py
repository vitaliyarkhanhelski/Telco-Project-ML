"""Data preprocessing utilities."""

import pandas as pd

from src.settings import settings


def clean_and_encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run full cleaning and encoding pipeline: TotalCharges, target, drop customerID, unify, encode, one-hot."""
    df = preprocess_TotalCharges_column(df)
    df = encode_target_column(df)
    df = drop_column(df, "customerID")
    df = unify_redundant_categories(df)
    df = encode_features(df)
    df = apply_one_hot_encoding(df)
    return df


def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop columns from DataFrame. Returns new DataFrame."""
    return df.drop(columns=columns)


def drop_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Drop a single column from DataFrame. Calls drop_columns internally."""
    return drop_columns(df, [column])


def unify_redundant_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Unify redundant service categories: 'No internet service' -> 'No', 'No phone service' -> 'No'."""
    cols_to_unify = settings.get("cols_to_unify", [])
    for col in cols_to_unify:
        df[col] = df[col].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    return df


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns rated as useless by Mutual Information. Uses drop_columns internally."""
    cols_to_drop = settings.get("useless_cols_to_drop", [])
    return drop_columns(df, cols_to_drop)


def preprocess_TotalCharges_column(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges to numeric and fill missing values with 0.

    Missing TotalCharges occur when tenure=0 (new customers before first billing cycle).
    """
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    print(df[df["TotalCharges"].isna()][["tenure", "TotalCharges"]])
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df


def encode_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the binary target column 'Churn' from string labels to integers."""
    print(df.Churn.value_counts())
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def encode_gender_column(df: pd.DataFrame) -> pd.DataFrame:
    """Encode gender column: Male -> 1, Female -> 0."""
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    return df


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Yes/No columns to 1/0. Uses binary_cols from settings."""
    binary_cols = settings.get("binary_cols", [])
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    return df


def encode_contract_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal encoding for Contract: Month-to-month -> 0, One year -> 1, Two year -> 2."""
    df["Contract"] = df["Contract"].map(
        {"Month-to-month": 0, "One year": 1, "Two year": 2}
    )
    return df


def apply_one_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Apply one-hot encoding to columns from one_hot_encoding_cols in settings."""
    cols = settings.get("one_hot_encoding_cols", [])
    return pd.get_dummies(df, columns=cols, drop_first=True, dtype=int)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns: gender, binary cols, contract."""
    df = encode_gender_column(df)
    df = encode_binary_columns(df)
    df = encode_contract_column(df)
    return df
