"""
FastAPI REST API for insurance enrollment predictions.

Run with: uvicorn src.api:app --reload
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

# Load model and preprocessor
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "preprocessor.pkl")

app = FastAPI(
    title="Insurance Enrollment Prediction API",
    description="Predict whether an employee will enroll in voluntary insurance",
    version="1.0.0"
)

# Global model and preprocessor (loaded on startup)
model = None
preprocessor = None


class EmployeeInput(BaseModel):
    """Input schema for a single employee."""
    age: int = Field(..., ge=18, le=100, example=45)
    gender: str = Field(..., example="Male")
    marital_status: str = Field(..., example="Married")
    salary: float = Field(..., gt=0, example=75000)
    employment_type: str = Field(..., example="Full-time")
    region: str = Field(..., example="West")
    has_dependents: str = Field(..., example="Yes")
    tenure_years: float = Field(..., ge=0, example=5.0)


class PredictionOutput(BaseModel):
    """Output schema for prediction."""
    prediction: int
    probability: float
    label: str


class BatchInput(BaseModel):
    """Input schema for batch predictions."""
    employees: List[EmployeeInput]


class BatchOutput(BaseModel):
    """Output schema for batch predictions."""
    predictions: List[PredictionOutput]


class ModelInfo(BaseModel):
    """Model metadata."""
    model_type: str
    features: List[str]
    status: str


@app.on_event("startup")
def load_model():
    """Load model and preprocessor on startup."""
    global model, preprocessor

    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model not found at {MODEL_PATH}")
        print("Run 'python src/experiment.py' first to train and save the model.")
        return

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("Model and preprocessor loaded successfully.")


@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
def get_model_info():
    """Get model metadata."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfo(
        model_type=type(model).__name__,
        features=[
            "age", "gender", "marital_status", "salary",
            "employment_type", "region", "has_dependents", "tenure_years"
        ],
        status="ready"
    )


def preprocess_input(employee: EmployeeInput) -> np.ndarray:
    """Convert input to model-ready format."""
    df = pd.DataFrame([{
        'age': employee.age,
        'gender': employee.gender,
        'marital_status': employee.marital_status,
        'salary': employee.salary,
        'employment_type': employee.employment_type,
        'region': employee.region,
        'has_dependents': employee.has_dependents,
        'tenure_years': employee.tenure_years
    }])

    # Binary encode has_dependents
    df['has_dependents'] = (df['has_dependents'] == 'Yes').astype(int)

    return preprocessor.transform(df)


@app.post("/predict", response_model=PredictionOutput, tags=["Predictions"])
def predict_single(employee: EmployeeInput):
    """Predict enrollment for a single employee."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = preprocess_input(employee)
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])

        return PredictionOutput(
            prediction=prediction,
            probability=round(probability, 4),
            label="Enrolled" if prediction == 1 else "Not Enrolled"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=BatchOutput, tags=["Predictions"])
def predict_batch(batch: BatchInput):
    """Predict enrollment for multiple employees."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        predictions = []
        for employee in batch.employees:
            X = preprocess_input(employee)
            pred = int(model.predict(X)[0])
            prob = float(model.predict_proba(X)[0][1])

            predictions.append(PredictionOutput(
                prediction=pred,
                probability=round(prob, 4),
                label="Enrolled" if pred == 1 else "Not Enrolled"
            ))

        return BatchOutput(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
