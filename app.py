"""
Simple Streamlit app for the Student Performance Predictor.
Run with: streamlit run app.py
"""

import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance Predictor")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Student Performance Predictor")
st.write(
    "Enter a student's details below to predict whether they are "
    "likely to Pass or Fail."
)

study_hours = st.slider("Study hours per day", 0.0, 8.0, 4.0, 0.5)
attendance = st.slider("Attendance (%)", 40, 100, 75)
previous_score = st.slider("Previous exam score", 0, 100, 65)
extra_activities = st.selectbox("Participates in extracurricular activities?", ["Yes", "No"])

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_score": previous_score,
        "extra_activities": 1 if extra_activities == "Yes" else 0,
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # probability of "Pass"

    if prediction == 1:
        st.success(f"Predicted result: Pass ({probability:.0%} chance of passing)")
    else:
        st.error(f"Predicted result: Fail ({probability:.0%} chance of passing)")

st.divider()
st.caption(
    "Model: Logistic Regression, trained on a generated dataset of 300 students "
    "(see generate_data.py and train_model.py)."
)
