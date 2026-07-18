"""
utils.py
--------
Shared helper functions used by train_model.py, predict.py, and the
Streamlit pages. Keeping these in one place means the SAME cleaning /
encoding logic is used everywhere — a rule you should always follow in
real ML projects, because "training data preprocessing" and "prediction
time preprocessing" must always match exactly.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

DATA_PATH = "data/student.csv"
MODEL_PATH = "models/best_model.pkl"
ENCODERS_PATH = "models/encoders.pkl"
SCALER_PATH = "models/scaler.pkl"
METADATA_PATH = "models/metadata.pkl"

CATEGORICAL_COLS = ["gender", "internet_access", "parent_education", "family_support"]
NUMERIC_COLS = ["age", "attendance_percentage", "weekly_study_hours",
                 "previous_grade", "failures"]
TARGET_COL = "grade"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw student dataset from CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset.

    WHAT: Removes duplicate rows, fills missing values, and drops columns
    that would leak the answer to the model.
    WHY: Real-world data is never perfect - duplicates skew training,
    missing values crash most sklearn models, and leaked columns
    (like final_score, which directly determines the grade) would let the
    model "cheat" instead of learning genuine patterns.
    """
    df = df.copy()

    # 1. Remove exact duplicate rows
    df = df.drop_duplicates()

    # 2. Handle missing values
    #    - Numeric columns -> fill with the column median (robust to outliers)
    #    - Categorical columns -> fill with the most frequent value (mode)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Drop columns that would leak the target
    #    final_score is used ONLY to construct the synthetic grade - a real
    #    prediction system would never have it available beforehand.
    if "final_score" in df.columns:
        df = df.drop(columns=["final_score"])

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a couple of simple, interpretable engineered features.

    WHAT: study_efficiency combines attendance and study hours into one
    signal; academic_risk flags students with multiple past failures.
    WHY: These are cheap to compute, easy to explain in an interview, and
    give the model useful combined signal instead of relying purely on
    raw columns.
    """
    df = df.copy()
    df["study_efficiency"] = (
        df["weekly_study_hours"] * (df["attendance_percentage"] / 100)
    ).round(2)
    df["academic_risk"] = (df["failures"] >= 2).astype(int)
    return df


def encode_features(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """
    Encode categorical columns to numbers so ML models can use them.

    WHAT IS LABEL ENCODING: converts each category (e.g. "Yes"/"No") into
    an integer (1/0). WHY WE USE IT HERE: all our categorical columns are
    binary or small/ordinal-ish (gender, internet_access, family_support
    are binary; parent_education has a natural low->high ordering), so a
    single integer column per feature keeps the model simple and
    interview-friendly. (OneHotEncoder is the alternative when categories
    have NO natural order and few unique values - mentioned in the README
    and demoed in the notebook for comparison.)
    """
    df = df.copy()
    if fit:
        encoders = {}
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        return df, encoders
    else:
        for col in CATEGORICAL_COLS:
            le = encoders[col]
            # Handle unseen categories gracefully at prediction time
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
        return df


def scale_features(df: pd.DataFrame, feature_cols, scaler: StandardScaler = None, fit: bool = True):
    """
    Scale numeric features to mean 0 / std 1.

    WHY SCALING IS NEEDED FOR SOME MODELS BUT NOT RANDOM FOREST:
    Logistic Regression uses gradient-based optimization and distance-like
    math (dot products of weights and features), so features with larger
    raw ranges (e.g. attendance_percentage: 0-100) would dominate features
    with small ranges (e.g. failures: 0-3) unless everything is scaled to
    the same range. Decision Trees and Random Forests split data based on
    threshold comparisons on ONE feature at a time (e.g. "is
    attendance_percentage > 70?"), so the scale of a feature never affects
    which splits are chosen - scaling is mathematically unnecessary for
    tree-based models, though it does no harm either.
    """
    df = df.copy()
    if fit:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        return df, scaler
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
        return df


def save_artifacts(model, encoders, scaler, feature_cols, target_encoder):
    """Persist the trained model + preprocessing objects to disk."""
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(
        {"feature_cols": feature_cols, "target_encoder": target_encoder},
        METADATA_PATH,
    )


def load_artifacts():
    """Load the trained model + preprocessing objects from disk."""
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    scaler = joblib.load(SCALER_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, encoders, scaler, metadata["feature_cols"], metadata["target_encoder"]
