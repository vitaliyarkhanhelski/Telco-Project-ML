"""Model training, evaluation, hyperparameter tuning, and business impact simulation."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import src.visualization as visualization
import src.utils as utils
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

def split_data(df):
    """Split data into training and test sets (ONLY ONCE!)."""
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # stratify=y ensures class proportions are preserved in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} rows")
    print(f"Test set: {X_test.shape[0]} rows")
    
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    """Fit StandardScaler on training data, apply to both sets. Returns (X_train_scaled, X_test_scaled)."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


def train_and_compare_models(X_train_scaled, X_test_scaled, y_train, y_test):
    """Train 4 different ML models and compare their results in a table."""

    # Define dictionary with our models
    # max_iter=1000 for Logistic Regression to avoid convergence errors
    # eval_metric='logloss' for XGBoost: monitors training progress using log loss (measures prediction confidence)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')  # logloss: prediction confidence metric, less the better, model more confident in its predictions
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



def plot_logistic_regression_confusion_matrix(X_train_scaled, X_test_scaled, y_train, y_test) -> np.ndarray:
    """Train Logistic Regression, plot its confusion matrix and return predictions."""
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    y_pred = log_reg.predict(X_test_scaled)

    visualization.plot_confusion_matrix(y_test, y_pred, "Confusion Matrix - Logistic Regression", "confusion_matrix_initial_logistic_regression.png")
    return y_pred


def plot_tuned_xgboost_confusion_matrix(X_train_scaled, X_test_scaled, y_train, y_test) -> np.ndarray:
    """Train tuned XGBoost (winner after tuning), plot its confusion matrix and return predictions."""
    # Load best params from tuned_model_results.csv instead of hardcoding
    best_params = utils.get_best_params("tuned_model_results.csv", "XGBoost")

    xgb_tuned = XGBClassifier(
        **best_params,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_tuned.fit(X_train_scaled, y_train)
    y_pred = xgb_tuned.predict(X_test_scaled)

    visualization.plot_confusion_matrix(y_test, y_pred, "Confusion Matrix - Tuned XGBoost", "confusion_matrix_tuned_xgboost.png")
    return y_pred


def tune_hyperparameters(X_train_scaled, X_test_scaled, y_train, y_test):
    """Use GridSearchCV to find the best hyperparameters, optimizing for maximum Recall (catching churners)."""
    print("\n⏳ Starting hyperparameter tuning (GridSearchCV)... This may take a minute!\n")


    # Define hyperparameter options for each model
    # class_weight='balanced' automatically penalizes errors on the minority class (Churn=Yes)
    param_grid_lr = {
        'C': [0.01, 0.1, 1, 10],  # Regularization: low C = simpler model (ignores noise), high C = fits training data closely (risk of overfitting)
        'class_weight': ['balanced']
    }

    param_grid_dt = {
        'max_depth': [3, 5, 10, None],    # How deep the tree can grow
        'min_samples_split': [2, 10, 50], # Minimum samples required to split a node
        'class_weight': ['balanced']
    }

    param_grid_rf = {
        'n_estimators': [50, 100, 200], # Number of trees in the forest
        'max_depth': [5, 10, 15],       # Depth of each individual tree
        'class_weight': ['balanced']
    }

    param_grid_xgb = {
        'learning_rate': [0.01, 0.1, 0.2], # How fast the model learns (smaller = slower but more accurate)
        'max_depth': [3, 5, 7], # Maximum depth of a tree, controls model complexity
        # scale_pos_weight is XGBoost's equivalent of class_weight='balanced'
        # value = count(No) / count(Yes) = 5174 / 1869 ≈ 3
        # GridSearchCV will pick the best value from [1, 3, 5]
        'scale_pos_weight': [1, 3, 5]
    }

    # Combine models with their parameter grids
    models_to_tune = {
        "Logistic Regression": (LogisticRegression(random_state=42, max_iter=1000), param_grid_lr),
        "Decision Tree": (DecisionTreeClassifier(random_state=42), param_grid_dt),
        "Random Forest": (RandomForestClassifier(random_state=42), param_grid_rf),
        "XGBoost": (XGBClassifier(random_state=42, eval_metric='logloss'), param_grid_xgb)
    }

    results = []

    # Run GridSearchCV for each model
    for name, (model, grid) in models_to_tune.items():
        print(f"Tuning model: {name}...")

        # cv=5: evaluate each combination 5 times on different data splits (Cross-Validation)
        # scoring='recall': our primary goal is catching as many churners as possible
        # n_jobs=-1: use all available CPU cores to run combinations in parallel (faster)
        grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=5, scoring='recall', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        # Get the best model with the winning hyperparameters
        best_model = grid_search.best_estimator_

        # Test the winner on the hidden test data
        y_pred = best_model.predict(X_test_scaled)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "Tuned Model": name,
            "Accuracy": round(acc, 4),
            "Recall": round(recall, 4),
            "Precision": round(precision, 4),
            "F1-Score": round(f1, 4),
            "Best Params": str(grid_search.best_params_)  # Save the winning parameters
        })

    # Convert and display results
    results_df = pd.DataFrame(results).sort_values(by="Recall", ascending=False).reset_index(drop=True)

    print("\n🏆 --- RESULTS AFTER HYPERPARAMETER TUNING (GRID SEARCH) --- 🏆")
    pd.set_option('display.max_colwidth', None)
    print(results_df.to_string())

    return results_df

def print_business_impact_simulation(y_test, y_pred_baseline, y_pred_tuned):
    """
    Computes and displays a financial simulation based on the Confusion Matrix.
    LTV = 1000 (Customer Lifetime Value)
    Discount cost = 100
    """

    cm_base = confusion_matrix(y_test, y_pred_baseline)
    cm_tuned = confusion_matrix(y_test, y_pred_tuned)

    # ravel() flattens the 2x2 confusion matrix into a flat list [TN, FP, FN, TP]
    tn_b, fp_b, fn_b, tp_b = cm_base.ravel()
    tn_t, fp_t, fn_t, tp_t = cm_tuned.ravel()

    LTV = 1000
    DISCOUNT_COST = 100
    PROFIT_FROM_RETENTION = LTV - DISCOUNT_COST  # we saved the customer but gave a discount
    COST_OF_FALSE_POSITIVE = DISCOUNT_COST        # unnecessary discount to someone who would stay
    COST_OF_MISSED_CHURNER = LTV                  # customer left, full LTV lost

    profit_base = (tp_b * PROFIT_FROM_RETENTION) - (fp_b * COST_OF_FALSE_POSITIVE) - (fn_b * COST_OF_MISSED_CHURNER)
    profit_tuned = (tp_t * PROFIT_FROM_RETENTION) - (fp_t * COST_OF_FALSE_POSITIVE) - (fn_t * COST_OF_MISSED_CHURNER)

    profit_difference = profit_tuned - profit_base
    tp_increase_percent = ((tp_t - tp_b) / tp_b) * 100 if tp_b > 0 else 0 # Calculate % improvement

    print("\n" + "=" * 50)
    print("Business simulation (model ROI)")
    print("=" * 50)
    print(f"Customer LTV: {LTV} | Campaign cost (discount): {DISCOUNT_COST}\n")

    print("1. Baseline model (Logistic Regression):")
    print(f"   - Retained customers (TP): {tp_b} -> Profit: {tp_b * PROFIT_FROM_RETENTION}")
    print(f"   - False alarms (FP):       {fp_b} -> Loss: {fp_b * COST_OF_FALSE_POSITIVE}")
    print(f"   - Missed churners (FN):    {fn_b} -> Loss: {fn_b * COST_OF_MISSED_CHURNER}")
    print(f"   - Net profit: {profit_base}\n")

    print("2. Tuned model (XGBoost):")
    print(f"   - Retained customers (TP): {tp_t} -> Profit: {tp_t * PROFIT_FROM_RETENTION}")
    print(f"   - False alarms (FP):       {fp_t} -> Loss: {fp_t * COST_OF_FALSE_POSITIVE}")
    print(f"   - Missed churners (FN):    {fn_t} -> Loss: {fn_t * COST_OF_MISSED_CHURNER}")
    print(f"   - Net profit: {profit_tuned}\n")

    print("-" * 50)
    print("Summary:")
    print(f"Tuning caught {tp_t - tp_b} more customers (+{tp_increase_percent:.1f}% effectiveness).")
    print(f"Overall, the new model would bring {profit_difference} MORE profit than the baseline model on this test sample!")
    print("=" * 50 + "\n")