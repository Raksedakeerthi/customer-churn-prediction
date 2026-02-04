import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.data.load_data import load_raw_data
from src.data.preprocessing import preprocess_data
from xgboost import XGBClassifier


def train_and_evaluate():
    # Load and preprocess data
    df = load_raw_data()
    X, y, preprocessor = preprocess_data(df)

    # Train-test split (IMPORTANT: stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    results = {}

    # -------------------------------
    # 1. Logistic Regression
    # -------------------------------
    log_reg_pipeline = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    log_reg_pipeline.fit(X_train, y_train)
    y_pred = log_reg_pipeline.predict(X_test)
    y_prob = log_reg_pipeline.predict_proba(X_test)[:, 1]

    results["Logistic Regression"] = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }

    # -------------------------------
    # 2. Random Forest + SMOTE
    # -------------------------------
    rf_pipeline = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ])

    rf_pipeline.fit(X_train, y_train)
    y_pred = rf_pipeline.predict(X_test)
    y_prob = rf_pipeline.predict_proba(X_test)[:, 1]

    results["Random Forest"] = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }
    
        
    # -------------------------------
    # 3. XGBoost + SMOTE
    # -------------------------------
    xgb_pipeline = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            n_jobs=-1
        ))
    ])

    xgb_pipeline.fit(X_train, y_train)
    y_pred = xgb_pipeline.predict(X_test)
    y_prob = xgb_pipeline.predict_proba(X_test)[:, 1]

    results["XGBoost"] = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    # -------------------------------
    # Print summary
    # -------------------------------
    for model, metrics in results.items():
        print(f"\n===== {model} =====")
        print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
        print("Recall (Churn):",
              metrics["report"]["1"]["recall"])
        print("F1-score (Churn):",
              metrics["report"]["1"]["f1-score"])




if __name__ == "__main__":
    train_and_evaluate()
