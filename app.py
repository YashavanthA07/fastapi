from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib

app = FastAPI()

# Load model safely
try:
    model = joblib.load("loan_model.joblib")
except Exception:
    model = None

class LoanRequest(BaseModel):
    Age: int = Field(..., ge=18, le=60, description="Age between 18 and 60")
    Salary: float = Field(..., gt=10000, description="Salary greater than 10000")

# Home endpoint
@app.get("/")
def home():
    return {
        "message": "FastAPI is running"
    }

@app.get("/health")
def health():
    if model is None:
        return {
            "status": "down",
            "message": "Model not loaded"
        }
    return {
        "status": "up",
        "message": "Model is loaded"
    }

@app.post("/Predict")
def predict(data: LoanRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    if data.Salary > 100000:
        raise HTTPException(
            status_code=400,
            detail="Salary must be less than 100000"
        )

    try:
        # Make prediction using the loaded model
        # Input features must be in the same order as trained: [[Age, Salary]]
        prediction = model.predict([[data.Age, data.Salary]])
        return {
            "prediction": int(prediction[0])
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )