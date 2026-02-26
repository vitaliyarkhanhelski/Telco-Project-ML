import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import src.visualization as visualization
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

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


def get_tuned_rf_predictions(X_train, X_test, y_train):
    """Trenuje nastrojony Random Forest i zwraca jego przewidywania do wykresu."""
    # Pamiętamy o skalowaniu!
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Wpisujemy najlepsze parametry, które wypluł nam GridSearchCV
    rf_tuned = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        class_weight='balanced',
        random_state=42
    )
    
    rf_tuned.fit(X_train_scaled, y_train)
    y_pred = rf_tuned.predict(X_test_scaled)
    
    return y_pred


def plot_logistic_regression_confusion_matrix(X_train, X_test, y_train, y_test) -> np.ndarray:
    """Get logistic regression predictions and plot the confusion matrix."""
    y_pred = get_logistic_regression_predictions(X_train, X_test, y_train)
    visualization.plot_confusion_matrix(y_test, y_pred, "Confusion Matrix - Logistic Regression", "confusion_matrix_initial_logistic_regression.png")
    return y_pred


def plot_tuned_random_forest_confusion_matrix(X_train, X_test, y_train, y_test) -> np.ndarray:
    """Get tuned random forest predictions and plot the confusion matrix."""
    y_pred = get_tuned_rf_predictions(X_train, X_test, y_train)
    visualization.plot_confusion_matrix(y_test, y_pred, "Confusion Matrix - Tuned Random Forest", "confusion_matrix_tuned_random_forest.png")
    return y_pred


def tune_hyperparameters(X_train, X_test, y_train, y_test):
    """
    Używa GridSearchCV do znalezienia najlepszych hiperparametrów,
    skupiając się na maksymalizacji wyłapywania odejść (Recall).
    """
    print("\n⏳ Rozpoczynam strojenie hiperparametrów (GridSearchCV)... To może potrwać kilkadziesiąt sekund!\n")
    
    # Skalujemy dane (jak poprzednio, to podstawa!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Definiujemy nasze "Menu" (Siatki hiperparametrów) dla każdego modelu
    # Wartość 'balanced' automatycznie nakłada kary za błędy na klasie Churn (Yes)
    
    param_grid_lr = {
        'C': [0.01, 0.1, 1, 10],  # Siła dopasowania do danych
        'penalty': ['l2'],        # Standardowa metoda upraszczania modelu
        'class_weight': ['balanced']
    }
    
    param_grid_dt = {
        'max_depth': [3, 5, 10, None],   # Jak głęboko drzewo może rosnąć
        'min_samples_split': [2, 10, 50], # Jak duże grupy może dzielić
        'class_weight': ['balanced']
    }
    
    param_grid_rf = {
        'n_estimators': [50, 100, 200], # Ile drzew w lesie
        'max_depth': [5, 10, 15],       # Głębokość pojedynczego drzewa
        'class_weight': ['balanced']
    }
    
    param_grid_xgb = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        # scale_pos_weight = około 3 (bo mamy 73% No i 27% Yes, więc Yes musi być traktowane 3x ważniej)
        'scale_pos_weight': [1, 3, 5] 
    }
    
    # Łączymy modele z ich siatkami w jeden wygodny słownik
    models_to_tune = {
        "Regresja Logistyczna": (LogisticRegression(random_state=42, max_iter=1000), param_grid_lr),
        "Drzewo Decyzyjne": (DecisionTreeClassifier(random_state=42), param_grid_dt),
        "Random Forest": (RandomForestClassifier(random_state=42), param_grid_rf),
        "XGBoost": (XGBClassifier(random_state=42, eval_metric='logloss'), param_grid_xgb)
    }
    
    results = []
    
    # 2. Pętla uruchamiająca GridSearchCV dla każdego modelu
    for name, (model, grid) in models_to_tune.items():
        print(f"Strojenie modelu: {name}...")
        
        # cv=5 oznacza, że ocenia każdy zestaw 5 razy na różnych kawałkach danych (Cross-Validation)
        # scoring='recall' to nasz główny cel!
        grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=5, scoring='recall', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        
        # Pobieramy ZWYCIĘSKI model z najlepszymi ustawieniami
        best_model = grid_search.best_estimator_
        
        # Testujemy zwycięzcę na naszych ukrytych danych testowych
        y_pred = best_model.predict(X_test_scaled)
        
        # Liczymy metryki
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
            "Best Params": str(grid_search.best_params_) # Zapisujemy, co wygrało!
        })
        
    # Konwersja i wyświetlenie wyników
    results_df = pd.DataFrame(results).sort_values(by="Recall", ascending=False).reset_index(drop=True)
    
    print("\n🏆 --- WYNIKI PO STROJENIU HIPERPARAMETRÓW (GRID SEARCH) --- 🏆")
    # Używamy to_string(), żeby wyświetlić ładnie całe "Best Params"
    pd.set_option('display.max_colwidth', None) 
    print(results_df.to_string())
    
    return results_df

def print_business_impact_simulation(y_test, y_pred_baseline, y_pred_tuned):
    """
    Computes and displays a financial simulation based on the Confusion Matrix.
    LTV = 1000 (Customer Lifetime Value)
    Discount cost = 100
    """
    from sklearn.metrics import confusion_matrix

    cm_base = confusion_matrix(y_test, y_pred_baseline)
    cm_tuned = confusion_matrix(y_test, y_pred_tuned)

    tn_b, fp_b, fn_b, tp_b = cm_base.ravel()
    tn_t, fp_t, fn_t, tp_t = cm_tuned.ravel()

    LTV = 1000
    DISCOUNT_COST = 100
    PROFIT_FROM_RETENTION = LTV - DISCOUNT_COST
    COST_OF_FALSE_POSITIVE = DISCOUNT_COST

    profit_base = (tp_b * PROFIT_FROM_RETENTION) - (fp_b * COST_OF_FALSE_POSITIVE)
    profit_tuned = (tp_t * PROFIT_FROM_RETENTION) - (fp_t * COST_OF_FALSE_POSITIVE)

    profit_difference = profit_tuned - profit_base
    tp_increase_percent = ((tp_t - tp_b) / tp_b) * 100 if tp_b > 0 else 0

    print("\n" + "=" * 50)
    print("Business simulation (model ROI)")
    print("=" * 50)
    print(f"Customer LTV: {LTV} | Campaign cost (discount): {DISCOUNT_COST}\n")

    print("1. Baseline model (Logistic Regression):")
    print(f"   - Retained customers (TP): {tp_b} -> Profit: {tp_b * PROFIT_FROM_RETENTION}")
    print(f"   - False alarms (FP): {fp_b} -> Loss: {fp_b * COST_OF_FALSE_POSITIVE}")
    print(f"   - Net profit: {profit_base}\n")

    print("2. Tuned model (Random Forest):")
    print(f"   - Retained customers (TP): {tp_t} -> Profit: {tp_t * PROFIT_FROM_RETENTION}")
    print(f"   - False alarms (FP): {fp_t} -> Loss: {fp_t * COST_OF_FALSE_POSITIVE}")
    print(f"   - Net profit: {profit_tuned}\n")

    print("-" * 50)
    print("Summary:")
    print(f"Tuning caught {tp_t - tp_b} more customers (+{tp_increase_percent:.1f}% effectiveness).")
    print(f"Overall, the new model would bring {profit_difference} MORE profit than the baseline model on this test sample!")
    print("=" * 50 + "\n")