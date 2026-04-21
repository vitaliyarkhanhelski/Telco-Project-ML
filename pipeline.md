# Telco Customer Churn — ML Pipeline

**Problem:** Binary classification — predict `Churn = 1` (customer leaves) or `Churn = 0` (customer stays), based on 7043 customer records from a telecom company.

**Primary metric: Recall** — a missed churner (FN) costs full LTV=$1000, a false alarm (FP) costs only $100 discount. Catching churners is 10× more valuable than avoiding false alarms.

---

## Step 1 — Data Loading
- Download dataset from Kaggle API using credentials from `.env`
- Load into pandas DataFrame

## Step 2 — Initial Analysis (EDA)
- `ydata_profiling` generates a full HTML report saved to `reports/`
- Report includes distributions, correlations (Pearson/Spearman/Kendall), missing values
- Runs on **raw data** — key finding: `TotalCharges` is stored as `object` instead of `float`

## Step 3 — Business Charts (EDA)
- 5 seaborn charts saved to `charts/eda/` showing churn dependency on:
  - `Contract` — month-to-month customers churn far more
  - `InternetService` — Fiber Optic has surprisingly high churn
  - `tenure` — new customers (low tenure) leave most often
  - `MonthlyCharges` — higher bills correlate with higher churn
  - `PaymentMethod` — Electronic Check customers churn significantly more

## Step 4 — Preprocessing
At this stage we **don't yet know which columns are useful** — so we encode all of them:

| Step | What | How |
|---|---|---|
| Fix `TotalCharges` | Convert to float, fill 11 NaN with 0 | `pd.to_numeric()`, `fillna(0)` |
| Encode target | `Churn`: Yes→1, No→0 | `.map()` |
| Drop `customerID` | Not useful for modeling | `drop()` |
| Unify categories | `No internet service`→`No`, `No phone service`→`No` | `.replace()` |
| Encode `gender` | Male→1, Female→0 | `.map()` |
| Encode binary cols | Yes→1, No→0 for 11 service columns | `.map()` |
| Encode `Contract` | Ordinal: Month-to-month=0, One year=1, Two year=2 | `.map()` |
| One-hot encode | `InternetService`, `PaymentMethod` | `pd.get_dummies(drop_first=True)` |

## Step 5 — Feature Relationships
All columns are now numeric — correlations work on the full dataset:
- Pearson / Spearman / Kendall heatmaps saved to `charts/correlation/`
- Mutual Information scores (`mutual_info_classif`) saved to `charts/mutual_information.png`
- MI is the primary signal — captures non-linear dependencies, works better than correlation for binary targets

## Step 6 — Feature Selection
*Based on what we saw in Step 5* — columns with near-zero MI scores are dropped:
`StreamingTV`, `PhoneService`, `MultipleLines`, `StreamingMovies`, `DeviceProtection`, `Partner`, `gender`

## Step 7 — Train-Test Split
- 80/20 split, `stratify=y` — preserves the 27%/73% churn ratio in both sets
- Split happens **after** feature selection — test set is never seen during training

## Step 8 — Scaling
- `StandardScaler` fitted **once** on `X_train`, applied to both sets
- Ensures all features are on equal footing (mean=0, std=1)

## Step 9 — Baseline Model Training
4 models trained and compared, sorted by Recall:

| Model | Accuracy | Recall | Precision | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | **0.8027** | **0.5535** | **0.6509** | **0.5983** |
| XGBoost | 0.7828 | 0.5160 | 0.6069 | 0.5578 |
| Decision Tree | 0.7303 | 0.5080 | 0.4922 | 0.5000 |
| Random Forest | 0.7821 | 0.4893 | 0.6120 | 0.5438 |

**Logistic Regression wins baseline** — simplest model outperforms all ensemble methods.

## Step 10 — Confusion Matrix (Logistic Regression)

|  | Predicted: Stays | Predicted: Churns |
|---|---|---|
| **Actual: Stays** | TN = 926 ✅ | FP = 109 ❌ |
| **Actual: Churns** | FN = 164 ❌ | TP = 210 ✅ |

**164 missed churners (FN)** — the most costly mistake.

## Step 11 — Hyperparameter Tuning
- `GridSearchCV(cv=5, scoring='recall', n_jobs=-1)` on all 4 models
- `class_weight='balanced'` for sklearn models, `scale_pos_weight≈3` for XGBoost — address 27%/73% class imbalance
- Results saved to `data/tuned_model_results.csv`

| Model | Accuracy | Recall | Precision | F1-Score |
|---|---|---|---|---|
| **XGBoost** | 0.6274 | **0.9305** | 0.4109 | 0.5700 |
| Decision Tree | 0.7488 | 0.8048 | 0.5172 | 0.6297 |
| Random Forest | 0.7417 | 0.8048 | 0.5084 | 0.6232 |
| Logistic Regression | 0.7331 | 0.7968 | 0.4983 | 0.6132 |

**XGBoost wins on Recall (0.93)** — trade-off: lower Precision (0.41), more false alarms.

## Step 12 — Confusion Matrix (Tuned XGBoost)

|  | Predicted: Stays | Predicted: Churns |
|---|---|---|
| **Actual: Stays** | TN = 533 ✅ | FP = 502 ❌ |
| **Actual: Churns** | FN = 23 ✅ | TP = 351 ✅ |

**FN dropped from 164 → 23** — XGBoost misses almost no real churners.

## Step 13 — Business Impact Simulation

```
Net profit = TP × (LTV - discount) - FP × discount - FN × LTV
           = TP × 900              - FP × 100       - FN × 1000
```

| | Logistic Regression | Tuned XGBoost |
|---|---|---|
| Retained customers (TP) | 210 → +$189,000 | 351 → +$315,900 |
| False alarms (FP) | 109 → -$10,900 | 502 → -$50,200 |
| Missed churners (FN) | 164 → -$164,000 | 23 → -$23,000 |
| **Net profit** | **$14,100** | **$242,700** |

**Tuned XGBoost brings $228,600 more profit** on this test sample. Key driver: FN 164 → 23, each missed churner costs $1,000.

