"""FastAPI service exposing a /predict endpoint for diabetes risk.

Send patient data as JSON, get back the prediction, the probability, and the
SHAP feature contributions that explain the score.

Run:   venv\\Scripts\\python.exe -m uvicorn api:app --reload
Docs:  http://127.0.0.1:8000/docs   (interactive tester in your browser)
"""

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained model + feature order + explainer ONCE at startup,
# not on every request (loading is slow; serving must be fast).
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")
explainer = shap.TreeExplainer(model)

app = FastAPI(title="ClinicalAI Risk Platform")


class Patient(BaseModel):
    """The exact JSON shape a request must send. FastAPI validates it for us."""
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


@app.get("/")
def health():
    """Simple health check so you can confirm the service is up."""
    return {"status": "ok", "service": "ClinicalAI Risk Platform"}


@app.post("/predict")
def predict(patient: Patient):
    # Build a one-row DataFrame in the exact column order the model was trained on.
    row = pd.DataFrame([patient.model_dump()])[features]

    proba = float(model.predict_proba(row)[0, 1])   # probability of diabetic
    prediction = int(proba >= 0.5)                  # 0.5 threshold (a choice!)

    # SHAP contributions for the diabetic class, for THIS patient.
    sv = explainer(row)[..., 1]
    contributions = {
        feat: round(float(v), 3) for feat, v in zip(features, sv.values[0])
    }

    return {
        "prediction": prediction,
        "label": "diabetic" if prediction else "not diabetic",
        "probability": round(proba, 3),
        "shap_contributions": contributions,
    }
