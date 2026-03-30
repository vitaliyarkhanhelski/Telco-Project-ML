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
