import joblib

from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline

from src.data.load_data import load_raw_data
from src.data.preprocessing import preprocess_data


def train_and_serialize_model():
    # Load data
    df = load_raw_data()
    X, y, preprocessor = preprocess_data(df)

    # Final selected model
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    # Full pipeline (preprocessing + model)
    pipeline = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # Train on full dataset
    pipeline.fit(X, y)

    # Save pipeline
    model_path = "artifacts/churn_model.pkl"
    joblib.dump(pipeline, model_path)

    print(f"Model pipeline saved to: {model_path}")


if __name__ == "__main__":
    train_and_serialize_model()
