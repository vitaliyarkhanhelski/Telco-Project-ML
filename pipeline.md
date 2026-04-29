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

## Step 8 — Scaling (Logistic Regression only)
- `StandardScaler` fitted **once** on `X_train`, applied to both sets
- Applied **only to Logistic Regression** — gradient-based, sensitive to feature scale
- Tree-based models (Decision Tree, Random Forest, XGBoost) receive **raw unscaled data** — splits on thresholds, scale is irrelevant

## Step 9 — Baseline Model Training
4 models trained and compared, sorted by Recall:
- Logistic Regression uses **Lasso (L1) regularization** (`l1_ratio=1.0, solver='saga'`) — weak features get coefficient = 0
- Tree-based models (Decision Tree, Random Forest, XGBoost) receive raw unscaled data

| Model | Accuracy | Recall | Precision | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | **0.8034** | **0.5535** | **0.6530** | **0.5991** |
| XGBoost | 0.7828 | 0.5160 | 0.6069 | 0.5578 |
| Decision Tree | 0.7303 | 0.5080 | 0.4922 | 0.5000 |
| Random Forest | 0.7821 | 0.4893 | 0.6120 | 0.5438 |

**Logistic Regression wins baseline** — simplest model outperforms all ensemble methods.

## Step 10 — Confusion Matrix (Logistic Regression)

|  | Predicted: Stays | Predicted: Churns |
|---|---|---|
| **Actual: Stays** | TN = 925 ✅ | FP = 110 ❌ |
| **Actual: Churns** | FN = 167 ❌ | TP = 207 ✅ |

**167 missed churners (FN)** — the most costly mistake.

## Step 11 — Hyperparameter Tuning
- **Optuna** (`n_trials=50`, `scoring='recall'`, TPE Bayesian sampler) replaces GridSearchCV
- Optuna learns from previous trials which parameter regions are promising — faster and searches continuous ranges
- `class_weight='balanced'` for sklearn models, `scale_pos_weight≈3` for XGBoost — address 27%/73% class imbalance
- LR tunes both `C` and `l1_ratio` (0.0=L2/Ridge vs 1.0=L1/Lasso) — Optuna picks the better regularization automatically
- Results saved to `data/tuned_model_results.csv`
- Best XGBoost model object returned directly — no need to re-load from CSV

| Model | Accuracy | Recall | Precision | F1-Score |
|---|---|---|---|---|
| **XGBoost** | 0.6274 | **0.9305** | 0.4109 | 0.5700 |
| Decision Tree | 0.6529 | 0.8797 | 0.4256 | 0.5737 |
| Random Forest | 0.7282 | 0.8155 | 0.4927 | 0.6143 |
| Logistic Regression | 0.7346 | 0.7914 | 0.5000 | 0.6128 |

**XGBoost wins on Recall (0.93)** — trade-off: lower Precision (0.41), more false alarms.

## Step 12 — Confusion Matrix (Tuned XGBoost)

|  | Predicted: Stays | Predicted: Churns |
|---|---|---|
| **Actual: Stays** | TN = 536 ✅ | FP = 499 ❌ |
| **Actual: Churns** | FN = 26 ✅ | TP = 348 ✅ |

**FN dropped from 167 → 26** — XGBoost misses almost no real churners.

## Step 13 — Business Impact Simulation

```
Net profit = TP × (LTV - discount) - FP × discount - FN × LTV
           = TP × 900              - FP × 100       - FN × 1000
```

| | Logistic Regression | Tuned XGBoost |
|---|---|---|
| Retained customers (TP) | 207 → +$186,300 | 348 → +$313,200 |
| False alarms (FP) | 110 → -$11,000 | 499 → -$49,900 |
| Missed churners (FN) | 167 → -$167,000 | 26 → -$26,000 |
| **Net profit** | **$8,300** | **$237,300** |

**Tuned XGBoost brings $229,000 more profit** on this test sample. Key driver: FN 167 → 26, each missed churner costs $1,000.

## Step 14 — SHAP Feature Importance

Using the tuned XGBoost model to explain **which features drive predictions** and why:

- `shap_summary.png` — global beeswarm: which features matter most across all test customers
- `shap_dependence_top1.png` / `shap_dependence_top2.png` — dependence plots for the 2 most important features (selected dynamically by mean SHAP value)
- `shap_force.png` — local Force Plot for the first churner in the test set: which features pushed toward churn prediction
- `shap_waterfall.png` — Waterfall breakdown for the same churner

All charts saved to `charts/shap/`.
