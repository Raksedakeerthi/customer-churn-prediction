# Customer Churn Prediction – End-to-End ML System

## 📌 Project Overview
Customer churn is a major challenge for subscription-based businesses, as losing existing customers directly impacts revenue and growth. This project implements an **end-to-end machine learning system** to predict customer churn using structured customer data. The system is designed with **production readiness**, **model explainability**, and **business relevance** in mind.

The final solution trains multiple supervised learning models, handles class imbalance, selects the optimal model based on business-driven metrics, and exposes predictions through a **FastAPI REST API** for real-time inference.

---

## 🎯 Problem Statement
Given historical customer data, predict:
- Whether a customer is likely to churn
- The probability of churn

This enables businesses to proactively apply retention strategies and reduce financial loss.

---

## 🧠 Solution Highlights
- End-to-end ML pipeline (data → model → API)
- Leakage-safe preprocessing using pipelines
- Class imbalance handling
- Multi-model comparison
- Model explainability
- Production-ready inference API

---

## 🏗️ System Architecture

Raw Data
↓
Data Ingestion
↓
Preprocessing (Imputation, Encoding, Scaling)
↓
Train/Test Split (Stratified)
↓
Imbalance Handling
↓
Model Training & Evaluation
↓
Model Selection
↓
Model Explainability
↓
Serialization
↓
FastAPI Inference API


---

## 📊 Dataset
- **Dataset**: Telco Customer Churn (IBM)
- **Type**: Structured tabular data
- **Target Variable**: `Churn` (Yes / No)

---

## 🔍 Exploratory Data Analysis (EDA)
Key insights discovered during EDA:
- Dataset is **imbalanced** (~73% non-churn, ~27% churn)
- Customers with **low tenure** churn more
- **Month-to-month contracts** have significantly higher churn
- Higher monthly charges and lack of support services increase churn risk

---

## ⚙️ Data Preprocessing
- Dropped non-informative identifier (`customerID`)
- Converted `TotalCharges` to numeric
- Missing values handled using:
  - Median imputation (numerical)
  - Most-frequent imputation (categorical)
- One-hot encoding for categorical features
- Feature scaling for numerical features
- Implemented using **scikit-learn Pipelines and ColumnTransformer**

---

## ⚖️ Handling Class Imbalance
- Class imbalance handled using:
  - `class_weight="balanced"` for Logistic Regression
  - SMOTE (applied only on training data) for tree-based models
- Primary optimization metric: **Recall (Churn class)**

---

## 🤖 Model Training & Comparison
The following models were trained and evaluated under identical conditions:

| Model | ROC-AUC | Recall (Churn) | F1-score |
|------|--------|----------------|----------|
| Logistic Regression | **0.841** | **0.78** | **0.61** |
| Random Forest | 0.818 | 0.56 | 0.57 |
| XGBoost | 0.840 | 0.62 | 0.60 |

### ✅ Final Model Selection
**Logistic Regression** was selected as the final model due to:
- Highest recall (business priority)
- Highest ROC-AUC
- Stability and interpretability
- Strong generalization performance

---

## 🔎 Model Explainability
Since Logistic Regression is inherently interpretable, **coefficient-based explainability** was used to identify key churn drivers:

**Top churn drivers:**
- Low tenure
- Month-to-month contracts
- Fiber optic internet service
- High monthly charges
- Electronic check payment method
- Lack of technical support and online security

These insights can directly inform **targeted retention strategies**.

---

## 💾 Model Serialization
- Final preprocessing + model pipeline serialized using `joblib`
- Ensures consistent and reproducible inference
- Saved as a single artifact for deployment

---

## 🚀 FastAPI Deployment
A REST API was built using **FastAPI** to serve real-time predictions.

### Endpoint
`POST /predict`

### Input
Customer attributes in JSON format

### Output
- Churn probability
- Churn prediction (Yes / No)

Example response:
```json
{
  "churn_probability": 0.89,
  "churn_prediction": "Yes"
}
