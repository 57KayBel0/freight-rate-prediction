# 🚚 Freight Rate Prediction using Machine Learning

## Overview

This project was developed as part of a Machine Learning Engineer technical assessment.

The objective is to build an end-to-end machine learning pipeline capable of predicting freight transportation rates using historical shipment data. The solution includes data preprocessing, exploratory data analysis, feature engineering, model development, hyperparameter tuning, prediction generation, and validation.

The final solution uses a tuned **CatBoost Regressor**, which outperformed the baseline Random Forest model and successfully passed the official assessment validation script.

---

# Project Objectives

* Predict freight rates for unseen shipment requests.
* Build a reproducible machine learning pipeline.
* Perform feature engineering to improve predictive performance.
* Compare multiple machine learning models.
* Generate production-ready prediction files.
* Validate outputs using the supplied scoring script.

---

# Dataset

The project consists of three datasets.

| Dataset                     | Description                                             |
| --------------------------- | ------------------------------------------------------- |
| `train_test.csv`            | Historical shipment data used for model development     |
| `validation.csv`            | Unseen shipments requiring freight rate predictions     |
| `december-chart-inputs.csv` | Fixed December shipment scenario used for visualization |

The target variable is:

* **posted_rate**

---

# Exploratory Data Analysis

The dataset was explored to understand feature distributions, identify missing values, and discover relationships with the target variable.

Key observations included:

* Distance is the strongest predictor of freight cost.
* Equipment type significantly affects freight pricing.
* Market Index and Quote Signal contribute additional pricing information.
* Only a small percentage of missing values were present.

Missing values were handled using median imputation computed from the training dataset.

---

# Feature Engineering

Several additional features were created to improve model performance.

### Temporal Features

* Year
* Month
* Day
* Weekday
* Week Number
* Weekend Indicator

### Route Feature

A combined route feature was created:

```
Pickup → Delivery
```

### Geographic Features

* Latitude Difference
* Longitude Difference

### Interaction Features

* Distance Squared
* Distance / Weight Ratio
* Distance × Market Index
* Distance × Quote Signal

---

# Models Evaluated

## Random Forest (Baseline)

A Random Forest Regressor was trained as the initial benchmark.

Performance:

* MAE: **123.58**
* RMSE: **584.36**
* R²: **0.8403**

---

## CatBoost

CatBoost was selected because it performs exceptionally well on structured tabular datasets and handles categorical variables efficiently.

Performance:

* MAE: **111.92**
* RMSE: **533.00**
* R²: **0.8672**

---

## Tuned CatBoost (Final Model)

Hyperparameter tuning together with early stopping produced the best-performing model.

Final Performance

| Metric |      Score |
| ------ | ---------: |
| MAE    | **103.42** |
| RMSE   | **526.10** |
| R²     | **0.8706** |

---

# Validation

The generated prediction files successfully passed the supplied validation script.

Validation Output:

```
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
```

---

# Project Structure

```
freight-rate-prediction/
│
├── data/
├── models/
├── notebooks/
├── outputs/
├── report/
├── scorer_results/
├── src/
│   ├── train.py
│   ├── train_catboost.py
│   ├── train_catboost_tuned.py
│   └── predict.py
│
├── validation_predictions.csv
├── december_predictions.csv
├── score.py
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/57KayBel0/freight-rate-prediction.git
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Training

Run the tuned CatBoost model

```bash
python src/train_catboost_tuned.py
```

---

# Generate Predictions

```bash
python src/predict.py
```

---

# Validate Submission

```bash
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* CatBoost
* Matplotlib
* Git
* GitHub

---

# Future Improvements

Potential enhancements include:

* Time-aware cross-validation
* Ensemble learning
* Automated hyperparameter optimization using Optuna
* Explainability using SHAP values
* Model deployment through a REST API

---

# Author

**Kabelo Motshabi Makgae**

Machine Learning | Data Science | Software Engineering

GitHub:
https://github.com/57KayBel0
