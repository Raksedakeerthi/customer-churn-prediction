import pandas as pd
from pathlib import Path

def load_raw_data():
    """
    Load the raw Telco Customer Churn dataset.
    """
    data_path = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(data_path)
    return df


if __name__ == "__main__":
    df = load_raw_data()

    print("Dataset loaded successfully\n")

    print("Shape of dataset:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())
