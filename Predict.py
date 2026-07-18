"""
pages/Predict.py
-----------------
Lets the user enter one student's details, runs the trained model, and
displays the predicted grade, confidence, probability breakdown, feature
importance, and AI-generated study advice.
"""

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Allow importing predict.py / ai_assistant.py / utils.py from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from predict import predict_student
from ai_assistant import generate_study_advice

st.title("🎯 Predict Student Performance")
st.caption("Enter a student's details below to get a grade prediction and personalized study advice.")

with st.form("student_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=14, max_value=20, value=17)
        gender = st.selectbox("Gender", ["Male", "Female"])
        attendance = st.slider("Attendance (%)", 0, 100, 75)

    with col2:
        study_hours = st.number_input("Weekly Study Hours", min_value=0.0, max_value=40.0, value=4.0, step=0.5)
        previous_grade = st.number_input("Previous Grade (0-100)", min_value=0, max_value=100, value=65)
        failures = st.selectbox("Number of Past Failures", [0, 1, 2, 3])

    with col3:
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])
        parent_education = st.selectbox("Parent Education", ["High School", "Bachelors", "Masters", "PhD"])
        family_support = st.selectbox("Family Support", ["Yes", "No"])

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    student_input = {
        "age": age,
        "gender": gender,
        "attendance_percentage": attendance,
        "weekly_study_hours": study_hours,
        "previous_grade": previous_grade,
        "internet_access": internet_access,
        "parent_education": parent_education,
        "family_support": family_support,
        "failures": failures,
    }

    # Basic input validation - handled gracefully instead of crashing
    if study_hours < 0 or attendance < 0:
        st.error("Please enter valid, non-negative values.")
    else:
        with st.spinner("Predicting..."):
            result = predict_student(student_input)

        st.divider()
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.metric("Predicted Grade", result["predicted_grade"])
            st.metric("Confidence", f"{result['confidence']}%")

        with res_col2:
            st.markdown("**Probability by Grade**")
            probs_df = pd.DataFrame(
                list(result["probabilities"].items()), columns=["Grade", "Probability (%)"]
            ).sort_values("Grade")
            fig, ax = plt.subplots(figsize=(5, 2.5))
            ax.bar(probs_df["Grade"], probs_df["Probability (%)"], color="#4C72B0")
            ax.set_ylabel("Probability (%)")
            st.pyplot(fig)
            plt.close(fig)

        # Feature importance (precomputed during training, saved as a plot)
        fi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plots", "feature_importance.png"))
        if os.path.exists(fi_path):
            st.markdown("**What drives predictions most (model-wide)**")
            st.image(fi_path, use_container_width=True)

        st.divider()
        st.markdown("### 🤖 AI Study Advice")
        advice_input = {**student_input, **result}
        advice_input["predicted_grade"] = result["predicted_grade"]
        advice_input["confidence"] = result["confidence"]

        with st.spinner("Generating personalized advice with a local AI model..."):
            advice = generate_study_advice(advice_input)

        st.write(advice)
