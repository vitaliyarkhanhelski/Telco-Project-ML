import pandas as pd
from sklearn.model_selection import train_test_split

import src.visualization as visualization
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler

def split_data(df):
    """Split data into training and test sets (ONLY ONCE!)."""
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # stratify=y zapewnia zachowanie proporcji klas w obu zbiorach
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} rows")
    print(f"Test set: {X_test.shape[0]} rows")
    
    return X_train, X_test, y_train, y_test

def train_and_compare_models(X_train, X_test, y_train, y_test):
    """Train 4 different ML models and compare their results in a table."""
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define dictionary with our models
    # max_iter=1000 for Logistic Regression to avoid convergence errors
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    results = []
    
    for name, model in models.items():
        # 1. Training phase (training on historical data)
        model.fit(X_train_scaled, y_train)
        
        # 2. Testing phase (testing on hidden data)
        y_pred = model.predict(X_test_scaled)
        
        # 3. Calculating metrics
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Adding results to the list
        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Recall": round(recall, 4),
            "Precision": round(precision, 4),
            "F1-Score": round(f1, 4)
        })
        
    # Convert list to readable DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by the best Recall (catching churners)
    results_df = results_df.sort_values(by="Recall", ascending=False).reset_index(drop=True)
    
    print("\n--- MODEL COMPARISON ---")
    print(results_df.to_string())
    
    return results_df


def get_logistic_regression_predictions(X_train, X_test, y_train):
    """Train the winning model and return its predictions for the chart."""
    # Scale data also here, so the model works properly
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    y_pred = log_reg.predict(X_test_scaled)
    
    return y_pred


def plot_logistic_regression_confusion_matrix(X_train, X_test, y_train, y_test):
    """Get logistic regression predictions and plot the confusion matrix."""
    y_pred = get_logistic_regression_predictions(X_train, X_test, y_train)
    visualization.plot_confusion_matrix(y_test, y_pred)