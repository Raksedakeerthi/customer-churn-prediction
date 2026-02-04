from load_data import load_raw_data
from preprocessing import preprocess_data

if __name__ == "__main__":
    df = load_raw_data()
    X, y, preprocessor = preprocess_data(df)

    X_transformed = preprocessor.fit_transform(X)

    print("Original feature shape:", X.shape)
    print("Transformed feature shape:", X_transformed.shape)
    print("Target distribution:")
    print(y.value_counts())
