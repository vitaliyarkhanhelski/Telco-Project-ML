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


def plot_logistic_regression_confusion_matrix(X_train, X_test, y_train, y_test):
    """Get logistic regression predictions and plot the confusion matrix."""
    y_pred = get_logistic_regression_predictions(X_train, X_test, y_train)
    visualization.plot_confusion_matrix(y_test, y_pred, "Confusion Matrix - Logistic Regression", "confusion_matrix_initial_logistic_regression.png")


def plot_tuned_random_forest_confusion_matrix(X_train, X_test, y_train, y_test):
    """Get tuned random forest predictions and plot the confusion matrix."""
    y_pred = get_tuned_rf_predictions(X_train, X_test, y_train)
    visualization.plot_confusion_matrix(y_test, y_pred, "Confusion Matrix - Tuned Random Forest", "confusion_matrix_tuned_random_forest.png")


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