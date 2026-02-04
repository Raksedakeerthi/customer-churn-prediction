from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load trained model pipeline
MODEL_PATH = "artifacts/churn_model.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn probability using a trained ML model",
    version="1.0"
)

# Input schema
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict")
def predict_churn(data: CustomerData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Predict
    churn_prob = model.predict_proba(input_df)[0][1]
    churn_pred = model.predict(input_df)[0]

    return {
        "churn_probability": round(float(churn_prob), 4),
        "churn_prediction": "Yes" if churn_pred == 1 else "No"
    }
