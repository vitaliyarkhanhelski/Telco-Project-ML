# Telco Customer Churn - Machine Learning Project

![Telco](images/telco.png)

A Python machine learning project for analyzing and predicting customer churn in a telecommunications company, using the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from Kaggle.

## Problem Definition

This project focuses on predicting customer churn using a real-world telecom dataset. Customer churn prediction is a critical business problem that helps companies retain customers by identifying those likely to leave.

**Objective:** Predict customer churn for a telecommunications company. Customer churn refers to customers who stop using the service, which has a direct financial impact on the business.

**Evaluation:** Due to class imbalance and business priorities, recall for churned customers is treated as a primary evaluation metric—we aim to catch as many at-risk customers as possible, <u>even at the cost of some false positives</u>(<i>we prefer to over-flag (and sometimes over-act on non-churners) rather than under-flag and miss real churners</i>).

## Dataset Overview

The dataset contains customer demographic information, account details, service subscriptions, and churn status. The **target variable** is `Churn`, which indicates whether a customer has discontinued the service.

## Business Insight

The model can help the company:

- **Identify** customers at high risk of churn
- **Take** proactive retention actions (offers, support, targeted campaigns)
- **Reduce** revenue loss by focusing on high-risk segments

## Project Structure

```
telco_project_ML/
├── src/
│   └── main.py          # Main entry point
├── requirements.txt     # Python dependencies with versions
└── README.md
```

## Setup

### 1. Create a virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Unix/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains all required libraries with their versions. It will be updated as the project evolves. Each time you add a new library, add it with its version (e.g. `library_name>=1.2.0`) so others can recreate the environment exactly.

### 3. Download the Kaggle dataset

The dataset must be in the `data/` folder for the project to work. You can obtain it in either way below.

**Note:** If you prefer not to register on Kaggle or create an API key, use **Option A**—no credentials or Kaggle API setup needed. Place the CSV in `data/` and the project will use it.

#### Option A: Manual download (no Kaggle API)

1. Go to [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Sign in to your Kaggle account
3. Click the **Download** button
4. Extract the archive and place the CSV file (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) in the `data/` folder

No `.env` or `kaggle.json` required.

#### Option B: Download via Kaggle API

Requires Kaggle API credentials. Use **either** `.env` or `kaggle.json`—not both.

**Using `.env` (recommended):**

1. Copy the template: `cp .env.example .env`
2. Edit `.env` and add your credentials:
   - **KAGGLE_USERNAME:** your Kaggle username
   - **KAGGLE_KEY:** your API key from [Kaggle → Settings → API → Create New Token](https://www.kaggle.com/settings)

When using `.env`, there is no need for `kaggle.json`.

**Using `kaggle.json` (alternative):**

1. Log in to Kaggle → **Settings** → **API** → **Create New Token** (downloads `kaggle.json`)
2. Place it at `~/.kaggle/kaggle.json` (macOS/Linux) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows)
3. On Unix: `chmod 600 ~/.kaggle/kaggle.json`

**Download the dataset:**

```bash
kaggle datasets download -d blastchar/telco-customer-churn -p data/ --unzip
```

This creates the `data/` folder and extracts the CSV there.

## Usage

Run the main script:

```bash
python src/main.py
```

Or from the project root with module execution:

```bash
python -m src.main
```

## ML Pipeline (to be documented)

Stages such as data preprocessing, feature engineering, model training, and evaluation will be described here as the project progresses. The goal is to demonstrate a complete and realistic machine learning pipeline—from data understanding to business-driven model optimization.

## Development Notes

- The project uses `.py` files only (no Jupyter notebooks)
- Run and test code via the Python console or by executing scripts
- Additional utilities will be added to `src/` as the project grows

## Dataset Reference

- **Source:** [Telco Customer Churn | Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Author:** blastchar
- **License:** Database: Open Database, Contents: Database Contents
