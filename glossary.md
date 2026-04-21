# 📚 Project Glossary: Machine Learning Terms

This document contains simple, easy-to-understand definitions of the key concepts used in our Telco Customer Churn project.

## 1. Data Preprocessing & Engineering
* **Data Preprocessing:** Cleaning and preparing raw data so the computer can actually understand and use it.
* **Feature Selection:** Choosing the most important columns (features) for our model and throwing away the useless ones (the noise).
* **Encoding:** Translating text data (like "Yes" or "No") into numbers, because math algorithms can only calculate numbers.
* **Ordinal Encoding:** Changing text to numbers when there is a logical order (e.g., "Month-to-month" = 0, "One year" = 1, "Two year" = 2).
* **One-Hot Encoding:** Creating a new True/False (1 or 0) column for each category when there is no logical order (e.g., turning a "Color" column into "Is_Red" and "Is_Blue").

## 2. Exploratory Data Analysis (EDA)
* **EDA (Exploratory Data Analysis):** The first step after loading data — visually and statistically exploring it to understand its structure, spot patterns, and find relationships between columns before building any model.

## 3. Statistical & Information Theory Concepts
* **Correlation (Pearson, Spearman, Kendall):** Checking if two columns move together (e.g., if one goes up, does the other go up?). Pearson looks for straight lines, while Spearman and Kendall look for general trends.
* **Entropy:** A measure of chaos or unpredictability in the data. It is measured on a scale from **0 to 1** (for binary classification like Yes/No): **0** = perfectly predictable (e.g. everyone churns), **1** = maximum chaos (e.g. exactly 50% churn, 50% stay — hardest to predict). In our dataset, churn is 27% Yes / 73% No, so entropy ≈ 0.84 — fairly high, meaning it's not trivial to predict. Entropy is the foundation of Mutual Information and Decision Trees — they split data to reduce entropy as much as possible.
* **Mutual Information:** A smart score that tells us: *"If I know the value of column A, how much does it help me guess column B?"*. It easily catches hidden patterns that normal correlation misses.
* **Stratification (Stratified Split):** Making sure that when we split our data into training and testing parts, both parts have the exact same percentage of churned customers. It prevents the model from learning on bad proportions.

## 4. Machine Learning Algorithms
* **Baseline Model:** A basic, simple model used as a starting point. We use it to see if more complex models are actually doing a better job or just overcomplicating things.
* **Logistic Regression:** A simple math model that predicts the probability of a Yes/No answer (like *"Will the customer leave?"*). Internally, it assigns a **weight** to each feature (e.g., *monthly charges* gets a high weight, *gender* gets a low one), multiplies them together, and feeds the result into a **sigmoid function** — an S-shaped curve that squashes any number into a value between 0 and 1. That value is the predicted probability: e.g. 0.82 means *"82% chance this customer churns"*. If the probability is above 0.5, the model predicts "Yes" (churn); otherwise "No" (stays). The weights are learned automatically during training by minimizing prediction errors.
* **Decision Tree:** A model that makes decisions by asking a series of Yes/No questions, looking very much like a simple flowchart .
* **Random Forest:** A "forest" made of many different Decision Trees. They all vote on the final answer, which makes the prediction much more accurate and stable.
* **XGBoost:** A very powerful, advanced algorithm where multiple trees are built one after another, and each new tree tries to fix the mistakes made by the previous one.

## 5. Evaluation Metrics
* **Confusion Matrix:** A 2×2 table that shows exactly where the model is right and where it makes mistakes. Rows = what actually happened, Columns = what the model predicted. Four cells: TN (correctly predicted stays), FP (false alarms), FN (missed churners), TP (correctly predicted churns). It is the foundation for all other metrics — Accuracy, Recall, Precision, and F1 are all calculated from these 4 numbers.
* **Accuracy:** Overall, out of all the model's guesses, what percentage was exactly right? E.g. 1000 customers total: 700 stayed → model correctly predicted 650 of them; 300 churned → model correctly predicted 150 of them. Total correct: 650 + 150 = 800. Accuracy = 800 / 1000 = **80%**.
* **Recall (Sensitivity):** Out of all the customers who *actually* left, how many did our model successfully catch? (This is the most important score for predicting churn). E.g. 1000 customers actually churned → model correctly flagged 800 of them. Recall = 800 / 1000 = **80%**. The remaining 200 were missed (FN) — the most costly mistake.
* **Precision:** When our model yells *"This customer will leave!"*, how often is it actually right? E.g. 1000 customers actually churned, model flagged 800 as "will churn" → all 800 actually churned (zero false alarms). Precision = 800 / 800 = **100%** — but Recall is only 80% because 200 churners were missed. Another example: model flagged 900, but 100 of them stayed → Precision = 800 / 900 = **89%**. Low precision means wasting money on discounts for customers who weren't leaving anyway.
* **F1-Score:** A combined score that balances Recall and Precision into one number. Useful when both missing churners (low Recall) and false alarms (low Precision) are costly. Formula: `F1 = 2 × (Precision × Recall) / (Precision + Recall)`. E.g. Precision = 90%, Recall = 80% → F1 = 2 × (0.9 × 0.8) / (0.9 + 0.8) = **84.7%**. A model with Recall = 100% but Precision = 10% would have F1 = only 18% — it catches everyone but raises too many false alarms to be useful.
* **TP (True Positive):** Model predicted churn → customer actually churned ✅
* **TN (True Negative):** Model predicted stay → customer actually stayed ✅
* **FP (False Positive):** Model predicted churn → customer actually stayed ❌ (false alarm — we sent a discount unnecessarily)
* **FN (False Negative):** Model predicted stay → customer actually churned ❌ (missed churner — the most costly mistake)
