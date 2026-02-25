"""Visualization utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHARTS_DIR = PROJECT_ROOT / "charts"


def _plot_correlation_heatmap(
    df: pd.DataFrame,
    method: str,
    title: str,
    filename: str,
) -> None:
    """Plot correlation heatmap for Churn column."""
    correlation = df.corr(method=method)
    plt.figure(figsize=(15, 10))
    sns.heatmap(
        correlation[["Churn"]].sort_values(by="Churn", ascending=False),
        annot=True,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
    )
    plt.title(title, pad=20)
    plt.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHARTS_DIR / filename, bbox_inches="tight", dpi=300)
    plt.show()


def plot_pearson_correlation(df: pd.DataFrame) -> None:
    """Plot Pearson correlation heatmap for Churn."""
    _plot_correlation_heatmap(
        df,
        method="pearson",
        title="What affects customer churn (Churn)?",
        filename="churn_correlation_pearson.png",
    )


def plot_spearman_correlation(df: pd.DataFrame) -> None:
    """Plot Spearman correlation heatmap for Churn."""
    _plot_correlation_heatmap(
        df,
        method="spearman",
        title="What affects customer churn (Spearman)?",
        filename="churn_correlation_spearman.png",
    )


def plot_kendall_correlation(df: pd.DataFrame) -> None:
    """Plot Kendall correlation heatmap for Churn."""
    _plot_correlation_heatmap(
        df,
        method="kendall",
        title="What affects customer churn (Kendall)?",
        filename="churn_correlation_kendall.png",
    )


def visualize_feature_relationships(df: pd.DataFrame) -> None:
    """Plot correlation heatmaps and Mutual Information for features vs Churn."""
    plot_correlation_heatmaps(df)
    plot_mutual_information(df)


def plot_correlation_heatmaps(df: pd.DataFrame) -> None:
    """Plot Pearson, Spearman, and Kendall correlation heatmaps for Churn."""
    plot_pearson_correlation(df)
    plot_spearman_correlation(df)
    plot_kendall_correlation(df)


def plot_mutual_information(df: pd.DataFrame) -> None:
    """Plot Mutual Information scores for features vs Churn target."""
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_scores_series = pd.Series(mi_scores, index=X.columns)
    mi_scores_series = mi_scores_series.sort_values(ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=mi_scores_series.values, y=mi_scores_series.index, palette="viridis")

    plt.title("What affects customer churn (Mutual Information)", pad=20)
    plt.xlabel("Mutual Information score (higher = more important feature)")
    plt.ylabel("Features (Columns)")

    plt.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHARTS_DIR / "mutual_information.png", bbox_inches="tight", dpi=300)
    plt.show()

    print("\nExact Mutual Information results:")
    print(mi_scores_series)
    print(
        "None of the standard correlations (Pearson, Spearman, Kendall) are ideal "
        "for a categorical target. Mutual Information is based on Entropy, so it "
        "can catch non-linear and much more complex dependencies."
    )
