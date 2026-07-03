"""Streamlit demo frontend for the ClinicalAI Risk Platform.

This UI does NOT load the model itself. It collects patient values and calls
the FastAPI /predict endpoint, then shows the prediction, probability, and the
SHAP feature contributions the API returns.

Start the API first (uvicorn api:app), then in a second terminal:
    streamlit run app.py
"""

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.title("ClinicalAI Risk Platform")
st.caption("Diabetes risk prediction with explainable AI")
st.info(
    "Fields are pre-filled with population averages (dataset medians). "
    "If you don't know a value - e.g. Insulin or Skin Thickness - leave it as is."
)

# Defaults below are the dataset medians, so an unknown field falls back to a
# sensible population average.
preg = st.number_input("Pregnancies", 0, 20, 3)
glucose = st.number_input("Glucose", 0, 300, 117)
bp = st.number_input("Blood Pressure", 0, 200, 72)
skin = st.number_input("Skin Thickness", 0, 100, 23)
insulin = st.number_input("Insulin", 0, 900, 31)
bmi = st.number_input("BMI", 0.0, 70.0, 32.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 5.0, 0.37)
age = st.number_input("Age", 1, 120, 29)

if st.button("Predict"):
    payload = {
        "Pregnancies": preg,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as error:
        st.error(f"Could not reach the API. Is it running on port 8000? ({error})")
    else:
        prob_pct = result["probability"] * 100
        if result["prediction"] == 1:
            st.error(f"High Risk - {result['label']} ({prob_pct:.1f}%)")
        else:
            st.success(f"Low Risk - {result['label']} ({prob_pct:.1f}%)")

        st.subheader("Why this result? (SHAP contributions)")
        contributions = result["shap_contributions"]
        for feat, val in sorted(contributions.items(), key=lambda kv: -abs(kv[1])):
            direction = "raises risk" if val > 0 else "lowers risk"
            st.write(f"**{feat}**: {val:+.3f}  ({direction})")
