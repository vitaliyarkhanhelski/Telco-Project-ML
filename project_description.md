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