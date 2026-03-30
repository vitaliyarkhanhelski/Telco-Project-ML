## Telco Customer Churn — Project Overview

The dataset comes from IBM Sample Data and contains information about **7043 customers** of a fictional telecommunications company. Each row represents one customer and includes details about their **demographics** (gender, age, partners, dependents), **services** they subscribed to (phone, internet, streaming, security), **account information** (contract type, payment method, monthly and total charges), and whether they **churned** (left the company) within the last month.

### Problem
Customer churn is one of the most critical business problems in the telecom industry — acquiring a new customer costs significantly more than retaining an existing one. The goal of this project is to **predict which customers are likely to churn**, so the company can proactively take action (e.g. offer discounts, change contract terms) to retain them before they leave.

This is a **binary classification problem**: given a set of customer features, predict `Churn = Yes` or `Churn = No`.

---

### 1. Download dataset:
- download dataset from kaggle using creds in .env file to data folder
- go to /data folder and look to `WA_Fn-UseC_-Telco-Customer-Churn.csv` file for the first time, briefly evaluate columns and rows, check for null values, and check for duplicates

### 2. Dataset initial analysis:
- dataset initial overview, column types, and basic statistics
- Exploratory Data Analysis - generate report in html using ydata_profiling for visualization of the dataset and analyzing correlations between columns
- at this stage I noticed that `TotalCharges` is of type object, and it should be float, so I will have to correct it

### 3. Generate charts visualizing churn dependency on different features:
- based on previous initial analysis, we generate charts using seaborn library on `Contract`, `InternetService`, `tenure`, `MonthlyCharges` and `PaymentMethod` columns to see strong relationship between these features and churn
- open charts/eda folder and analyze charts

| What we can conclude from the charts                                                                                                                                                                                                                                                                                                                                  |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Customers on month-to-month plans leave at a drastically higher rate than those on yearly plans. Locking customers into one-year or two-year contracts significantly increases retention.                                                                                                                                                        |
| Even though Fiber Optic is usually a premium service, it has the highest churn rate compared to DSL or having no internet service at all. This suggests a potential issue with the Fiber Optic service itself (e.g., reliability, speed drops, or it might just be too expensive).                                                               |
| Median tenure for customers who churned is very low (around 10 months). Customers who stay have a much higher median tenure (almost 40 months). If you can keep a customer through their first year, they are much more likely to stay long-term.                                                                                                |
| Customers who churn tend to pay more per month (median around 80 USD) compared to those who stay (median around 65 USD). This indicates price sensitivity.                                                                                                                                                                                       |
| Customers paying by Electronic Check have a disproportionately high churn rate compared to all other methods. Customers using automatic payments (Bank transfer or Credit card) have the lowest churn, likely because the payment requires no active effort, reducing the chance of a customer evaluating whether they want to keep the service. |

### 4. Data preprocessing (clean and encode data):
- correct `TotalCharges` column type to float, and check for null values after conversion, after conversion we have 11 null values, but they occur only for customers with `tenure` equal to 0, so we can safely fill these null values with 0
- encode target variable `Churn` to binary (Yes = 1, No = 0)
- drop `customerID` column as it is not useful for modeling
- unify redundant categories ('No internet service' = 'No', 'No phone service' = 'No')
- encode categorical columns: `gender` (Male=1, Female=0) and binary columns ('Yes'=1/'No'=0) using label encoding, `Contract` column using ordinal encoding (Month-to-month=0, One year=1, Two year=2) since there is a natural order
- apply one-hot encoding to `InternetService` and `PaymentMethod` columns since they are nominal categorical variables without a natural order, using `pd.get_dummies` function with `drop_first=True` to avoid multicollinearity

### 5. Visualize correlation between features and target variable:
- generate correlation heatmaps(Pearson, Spearman, Kendall) using seaborn to visualize the correlation between features and the target variable `Churn`
- additionally we generate a Mutual Information chart, which captures non-linear dependencies between features and the target variable and works well for classification problems — it measures how much information about `y` (Churn) is contained in each feature `X`, showing which features are most predictive of churn

### 6. Drop useless features:
- analyze the heatmaps and mutual information chart to identify which features does not have any significant correlation with the target variable `Churn` and drop them, in this case we can drop `StreamingTV`, `PhoneService`, `MultipleLines`, `StreamingMovies`, `DeviceProtection`, `Partner`, and `gender` columns as they have very low correlation with churn and do not provide much predictive power for our model. These features are only providing noise

### 7. Train-test split:
- split the dataset into training and testing sets

### 8. Train and compare models:
- scale features using `StandardScaler` (mean=0, std=1) to ensure all features are on equal footing
- train 4 models: `Logistic Regression`, `Decision Tree`, `Random Forest`, `XGBoost` — all 4 are suitable for **binary classification** (predicting Yes/No). Linear Regression is not used here because it predicts continuous values (e.g. 0.73) rather than discrete classes, and has no natural decision boundary for Yes/No output. These 4 models were chosen to cover a range of complexity: from simple (`Logistic Regression`, `Decision Tree`) to ensemble methods (`Random Forest`, `XGBoost`) which are generally more powerful
- evaluate each model using 4 metrics:
  - `Accuracy` — out of all customers, how many did we predict correctly?
  - `Recall` — out of **100 real churners**, how many did the model catch? (e.g. Recall=0.8 → caught 80, missed 20)
  - `Precision` — out of **100 customers the model flagged as churners**, how many actually churned? (e.g. Precision=0.6 → 60 real churners, 40 false alarms)
  - `F1-Score` — balance between Recall and Precision
- results are sorted by `Recall` (descending) — catching as many churners as possible is the primary business goal
- save results to `data/initial_model_results.csv` and analyze:

**Logistic Regression won across all 4 metrics** — best Recall (0.55), best Accuracy (0.80), best Precision (0.65), and best F1-Score (0.60). Surprisingly the simplest model outperformed all ensemble methods.

### 9. Confusion Matrix for Logistic Regression:
- re-train Logistic Regression once again and visualize its predictions as a **Confusion Matrix** — a 2x2 table showing exactly where the model is right and where it makes mistakes:

|  | Predicted: Stays (0) | Predicted: Churns (1) |
|---|---|---|
| **Actual: Stays (0)** | TN = 926 ✅ correctly predicted stays | FP = 109 ❌ false alarms (predicted churn, actually stayed) |
| **Actual: Churns (1)** | FN = 164 ❌ missed churners (predicted stay, actually churned) | TP = 210 ✅ correctly predicted churns |

- **FN (164 missed churners)** is the most costly business mistake — these are real customers who left but we never tried to retain them

### 10. Tune hyperparameters for all the models:
- define a set of hyperparameter values to try for each model (e.g. for Random Forest: `n_estimators`: [50, 100, 200], `max_depth`: [5, 10, 15])
- use `GridSearchCV` with `cv=5` (Cross-Validation) and `scoring='recall'` to find the best hyperparameters for each model, optimizing for `Recall` to catch as many churners as possible
- save results to `data/tuned_model_results.csv` and analyze:

**XGBoost wins on Recall (0.93)** — catches 93 out of every 100 real churners. However this comes with a trade-off:
- **Accuracy = 0.62** — model is wrong on 38% of all customers
- **Precision = 0.41** — out of every 100 customers flagged as churners, only 41 actually churn → 59 false alarms

**Is it worth it?** It depends on the business cost:
- **Missed churner (FN)** → customer leaves, revenue lost permanently
- **False alarm (FP)** → we offer a discount to someone who would have stayed anyway → small unnecessary cost

In telecom, a missed churner is typically far more expensive than a false alarm — therefore **high Recall = business priority**, and XGBoost is the winner.

### 11. Confusion Matrix for Tuned XGBoost:
- re-train XGBoost with best params from `tuned_model_results.csv` and visualize its confusion matrix:

|  | Predicted: Stays (0) | Predicted: Churns (1) |
|---|---|---|
| **Actual: Stays (0)** | TN = 533 ✅ correctly predicted stays | FP = 502 ❌ false alarms (predicted churn, actually stayed) |
| **Actual: Churns (1)** | FN = 23 ✅ missed churners (predicted stay, actually churned) | TP = 351 ✅ correctly predicted churns |

Comparison with initial Logistic Regression:

| | Logistic Regression | Tuned XGBoost |
|---|---|---|
| **FN (missed churners)** | 164 ❌ | **23 ✅** |
| **FP (false alarms)** | 109 | 502 |
| **TP (caught churners)** | 210 | **351** |

**FN dropped from 164 → 23** — XGBoost misses almost no real churners. The cost is more false alarms (502 vs 109) — we send discounts to more customers who would have stayed anyway, but we retain far more real churners. In most business scenarios this trade-off is well worth it.

### 12. Business Impact Simulation:
- simulate the financial impact of using the tuned XGBoost model vs the baseline Logistic Regression, based on two assumptions:
  - **Customer LTV = 1000** — revenue lost if a churner is missed
  - **Discount cost = 100** — cost of sending a retention offer

```
Net profit = TP × (LTV - discount) - FP × discount - FN × LTV
           = TP × 900              - FP × 100       - FN × 1000
```

| | Logistic Regression | Tuned XGBoost |
|---|---|---|
| Retained customers (TP) | 210 → profit $189,000 | 351 → profit $315,900 |
| False alarms (FP) | 109 → loss $10,900 | 502 → loss $50,200 |
| Missed churners (FN) | 164 → loss $164,000 | 23 → loss $23,000 |
| **Net profit** | **$14,100** | **$242,700** |

**Tuned XGBoost brings $228,600 more profit** on this test sample (+67.1% more retained customers). The key driver is **FN dropping from 164 → 23** — each missed churner costs full LTV=$1000, so catching 141 more churners alone saves $141,000.


