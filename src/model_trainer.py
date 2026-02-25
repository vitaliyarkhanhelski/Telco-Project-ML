import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def split_data(df):
    """Dzieli dane na zbiór treningowy i testowy (TYLKO RAZ!)."""
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # stratify=y zapewnia zachowanie proporcji klas w obu zbiorach
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Zbiór treningowy: {X_train.shape[0]} wierszy")
    print(f"Zbiór testowy: {X_test.shape[0]} wierszy")
    
    return X_train, X_test, y_train, y_test

def train_and_compare_models(X_train, X_test, y_train, y_test):
    """Trenuje 4 różne modele ML i porównuje ich wyniki w tabeli."""
    
    # Definiujemy słownik z naszymi modelami
    # max_iter=1000 dla Regresji Logistycznej zapobiega błędom o braku zbieżności
    models = {
        "Regresja Logistyczna": LogisticRegression(max_iter=1000, random_state=42),
        "Drzewo Decyzyjne": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    results = [] # Pusta lista na nasze wyniki
    
    # Pętla przechodzi przez każdy model po kolei
    for name, model in models.items():
        # 1. Faza uczenia (trening na danych historycznych)
        model.fit(X_train, y_train)
        
        # 2. Faza testowania (egzamin na ukrytych danych)
        y_pred = model.predict(X_test)
        
        # 3. Obliczanie metryk
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Dodanie wyników do listy
        results.append({
            "Model": name,
            "Accuracy (Dokładność)": round(acc, 4),
            "Recall (Czułość)": round(recall, 4),
            "Precision (Precyzja)": round(precision, 4),
            "F1-Score": round(f1, 4)
        })
        
    # Konwersja listy do czytelnego DataFrame
    results_df = pd.DataFrame(results)
    
    # Sortujemy od najlepszego wyniku w Recall (wyłapywanie odchodzących)
    results_df = results_df.sort_values(by="Recall (Czułość)", ascending=False).reset_index(drop=True)
    
    print("\n--- PORÓWNANIE MODELI ---")
    print(results_df.to_string())
    
    return results_df