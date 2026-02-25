# 📚 Project Glossary: Machine Learning Terms

This document contains simple, easy-to-understand definitions of the key concepts used in our Telco Customer Churn project.

## 1. Data Preprocessing & Engineering
* **Data Preprocessing:** Cleaning and preparing raw data so the computer can actually understand and use it.
* **Feature Selection:** Choosing the most important columns (features) for our model and throwing away the useless ones (the noise).
* **Encoding:** Translating text data (like "Yes" or "No") into numbers, because math algorithms can only calculate numbers.
* **Ordinal Encoding:** Changing text to numbers when there is a logical order (e.g., "Month-to-month" = 0, "One year" = 1, "Two year" = 2).
* **One-Hot Encoding:** Creating a new True/False (1 or 0) column for each category when there is no logical order (e.g., turning a "Color" column into "Is_Red" and "Is_Blue").

## 2. Statistical & Information Theory Concepts
* **Correlation (Pearson, Spearman, Kendall):** Checking if two columns move together (e.g., if one goes up, does the other go up?). Pearson looks for straight lines, while Spearman and Kendall look for general trends.
* **Entropy:** A measure of chaos or unpredictability in the data.
* **Mutual Information:** A smart score that tells us: *"If I know the value of column A, how much does it help me guess column B?"*. It easily catches hidden patterns that normal correlation misses.
* **Stratification (Stratified Split):** Making sure that when we split our data into training and testing parts, both parts have the exact same percentage of churned customers. It prevents the model from learning on bad proportions.

## 3. Machine Learning Algorithms
* **Baseline Model:** A basic, simple model used as a starting point. We use it to see if more complex models are actually doing a better job or just overcomplicating things.
* **Logistic Regression:** A simple math model that predicts the probability of a Yes/No answer (like *"Will the customer leave?"*).
* **Decision Tree:** A model that makes decisions by asking a series of Yes/No questions, looking very much like a simple flowchart .
* **Random Forest:** A "forest" made of many different Decision Trees. They all vote on the final answer, which makes the prediction much more accurate and stable.
* **XGBoost:** A very powerful, advanced algorithm where multiple trees are built one after another, and each new tree tries to fix the mistakes made by the previous one.

## 4. Evaluation Metrics
* **Accuracy:** Overall, out of all the model's guesses, what percentage was exactly right?
* **Recall (Sensitivity):** Out of all the customers who *actually* left, how many did our model successfully catch? (This is the most important score for predicting churn).
* **Precision:** When our model yells *"This customer will leave!"*, how often is it actually right?
* **F1-Score:** A combined overall score that finds a healthy balance between Recall and Precision.