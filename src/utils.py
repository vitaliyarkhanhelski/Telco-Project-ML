"""Utility functions."""

import pandas as pd

from src.data_loader import DATA_DIR


def save_to_csv(df: pd.DataFrame, filename: str, index: bool = False) -> None:
    """Save DataFrame to CSV file in the data folder."""
    path = DATA_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
