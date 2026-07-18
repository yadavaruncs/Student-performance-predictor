"""
predict.py
----------
Turns a single student's raw inputs into a prediction using the saved
model + preprocessing artifacts. Used by the Streamlit Predict page.
"""

import pandas as pd
from utils import (
    load_artifacts, engineer_features, encode_features, scale_features,
    NUMERIC_COLS
)


def predict_student(student_input: dict) -> dict:
    """
    Predict a grade for one student.

    Parameters
    ----------
    student_input : dict
        Raw values, e.g.
        {
            "age": 17, "gender": "Male", "attendance_percentage": 72,
            "weekly_study_hours": 2, "previous_grade": 78,
            "internet_access": "Yes", "parent_education": "Bachelors",
            "family_support": "Yes", "failures": 1
        }

    Returns
    -------
    dict with predicted_grade, confidence (%), and the full probability
    distribution across all grade classes.
    """
    model, encoders, scaler, feature_cols, target_encoder = load_artifacts()

    df = pd.DataFrame([student_input])

    # Apply the EXACT same preprocessing used during training. This
    # consistency is the single most important rule in production ML -
    # a mismatch here is the most common real-world bug.
    df = engineer_features(df)
    df = encode_features(df, encoders=encoders, fit=False)
    df = df[feature_cols]
    df = scale_features(df, NUMERIC_COLS + ["study_efficiency"], scaler=scaler, fit=False)

    pred_encoded = model.predict(df)[0]
    pred_grade = target_encoder.inverse_transform([pred_encoded])[0]

    probs = model.predict_proba(df)[0]
    prob_dict = {
        target_encoder.inverse_transform([i])[0]: round(float(p) * 100, 1)
        for i, p in enumerate(probs)
    }
    confidence = max(prob_dict.values())

    return {
        "predicted_grade": pred_grade,
        "confidence": confidence,
        "probabilities": prob_dict,
    }


if __name__ == "__main__":
    # Quick manual test
    sample = {
        "age": 17, "gender": "Male", "attendance_percentage": 72,
        "weekly_study_hours": 2, "previous_grade": 78,
        "internet_access": "Yes", "parent_education": "Bachelors",
        "family_support": "Yes", "failures": 1
    }
    result = predict_student(sample)
    print(result)
