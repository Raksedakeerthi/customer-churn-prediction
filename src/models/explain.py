import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline

from src.data.load_data import load_raw_data
from src.data.preprocessing import preprocess_data


def explain_logistic_regression():
    # Load data
    df = load_raw_data()
    X, y, preprocessor = preprocess_data(df)

    # Rebuild the final selected model
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    pipeline = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # Fit on full dataset for explanation
    pipeline.fit(X, y)

    # Get feature names after preprocessing
    feature_names = pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    # Get coefficients
    coefficients = pipeline.named_steps[
        "classifier"
    ].coef_[0]

    # Create explanation dataframe
    explanation_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients)
    })

    # Sort by absolute importance
    explanation_df = explanation_df.sort_values(
        by="abs_coefficient",
        ascending=False
    )

    print("\nTop features influencing churn:\n")
    print(explanation_df.head(15))


if __name__ == "__main__":
    explain_logistic_regression()
