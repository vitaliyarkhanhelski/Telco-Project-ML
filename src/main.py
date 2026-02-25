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

from src.settings import settings
from src.utils import (  # noqa: E402
    dataset_overview,
    display_unique_values_for_object_columns,
    download_dataset,
    encode_target_column,
    generate_report,
    load_dataset,
    preprocess_TotalCharges_column,
    unify_redundant_categories,
)


def main() -> None:
    """Main function - project entry point."""
    pd.set_option('display.max_columns', None)
    # download_dataset()
    df = load_dataset()
    # generate_report(df)
    
    #data analysis
    dataset_overview(df)

    #data preprocessing
    df = preprocess_TotalCharges_column(df)
    df = encode_target_column(df)
    df = df.drop(columns=["customerID"])
    
    display_unique_values_for_object_columns(df)
    df = unify_redundant_categories(df)

    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    binary_cols = settings.get("binary_cols", [])
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    df.head()

    # Ordinal Encoding for Contract
    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})

    cols_to_dummy = ['PaymentMethod', 'InternetService']
    df = pd.get_dummies(df, columns=cols_to_dummy, drop_first=True, dtype=int)


    # Calculate Pearson correlation (Looks only for straight lines)
    import seaborn as sns
    import matplotlib.pyplot as plt
    correlation = df.corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation[['Churn']].sort_values(by='Churn', ascending=False), 
            annot=True, 
            cmap='coolwarm', # Red = Positive, Blue = Negative
            center=0,        # Zero is white/neutral
            vmin=-1, vmax=1) # Scale from -1 to 1
    plt.title('What affects customer churn (Churn)?', pad=20)
    plt.tight_layout()
    plt.savefig('charts/churn_correlation_pearson.png', bbox_inches='tight', dpi=300)
    plt.show()


    # Spearman correlation (Looks for monotonic relationships)
    correlation_spearman = df.corr(method='spearman')

    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation_spearman[['Churn']].sort_values(by='Churn', ascending=False), 
                annot=True, 
                cmap='coolwarm', 
                center=0,        
                vmin=-1, vmax=1) 
    plt.title('What affects customer churn (Spearman)?', pad=20)
    plt.tight_layout()
    plt.savefig('charts/churn_correlation_spearman.png', bbox_inches='tight', dpi=300)
    plt.show()

    # Kendall correlation (Looks for monotonic relationships)
    correlation_kendall = df.corr(method='kendall')

    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation_kendall[['Churn']].sort_values(by='Churn', ascending=False), 
                annot=True, 
                cmap='coolwarm', 
                center=0,        
                vmin=-1, vmax=1) 
    plt.title('What affects customer churn (Kendall)?', pad=20)
    plt.tight_layout()
    plt.savefig('charts/churn_correlation_kendall.png', bbox_inches='tight', dpi=300)
    plt.show()

    # Mutual Information
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.feature_selection import mutual_info_classif

    # 1. Split data into features (X) and target variable (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # 2. Calculate Mutual Information
    # random_state=42 ensures reproducibility of the results
    mi_scores = mutual_info_classif(X, y, random_state=42)

    # 3. Combine the results with the column names and sort them
    mi_scores_series = pd.Series(mi_scores, index=X.columns)
    mi_scores_series = mi_scores_series.sort_values(ascending=False)

    # 4. Generate and save the plot
    plt.figure(figsize=(10, 8))
    # We use a bar plot, because it best shows the ranking of features
    sns.barplot(x=mi_scores_series.values, y=mi_scores_series.index, palette='viridis')

    plt.title('What affects customer churn (Mutual Information)', pad=20)
    plt.xlabel('Mutual Information score (higher = more important feature)')
    plt.ylabel('Features (Columns)')

    # Automatic fitting and saving to file
    plt.tight_layout()
    plt.savefig('charts/mutual_information.png', bbox_inches='tight', dpi=300)
    plt.show()

    # Optional: Print exact numbers to the console
    print("\nExact Mutual Information results:")
    print(mi_scores_series)
    print('''None of the standard correlations (Pearson, Spearman, Kendall) are ideal for a categorical target. 
    Mutual Information is based on Entropy, so it can catch non-linear and much more complex dependencies.''')

    # Feature Selection - remove noisy features
    # 1. Drop columns that Mutual Information rated as useless (score close to 0)
    cols_to_drop = settings.get("cols_to_drop", [])
    df = df.drop(columns=cols_to_drop)

    # 2. Separate features (X) from the target variable (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    from sklearn.model_selection import train_test_split

    # 3. Split data into training set (for training) and test set (for evaluating the model)
    # test_size=0.2 means that 20% of the data is left for testing, and 80% is used for training the model
    # random_state=42 ensures reproducibility of the results
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training set: {X_train.shape[0]} rows")
    print(f"Test set: {X_test.shape[0]} rows")

if __name__ == "__main__":
    main()
