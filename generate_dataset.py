"""
generate_dataset.py
--------------------
Generates data/student.csv — a synthetic dataset with the SAME columns and
realistic value ranges as the UCI/Kaggle "Student Performance" datasets.

WHY THIS FILE EXISTS:
This sandbox has no internet access, so the real Kaggle/UCI CSV could not be
downloaded here. This script builds a synthetic dataset using the same
feature set and realistic relationships (better attendance + more study
hours + fewer failures -> better grades, plus random noise) so the rest of
the pipeline (train_model.py, app.py, etc.) is fully runnable end-to-end.

TO USE THE REAL DATASET INSTEAD:
Download "Student Performance Data Set" (UCI) or "Students Performance in
Exams" (Kaggle), rename/remap columns to match the schema below, and save
it as data/student.csv. No other file needs to change.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1200  # number of synthetic students


def generate_student_data(n: int = N) -> pd.DataFrame:
    """Create a synthetic but realistic student performance dataset."""

    age = np.random.randint(15, 19, n)
    gender = np.random.choice(["Male", "Female"], n)
    attendance_percentage = np.clip(np.random.normal(75, 15, n), 30, 100).round(1)
    weekly_study_hours = np.clip(np.random.normal(4, 2.5, n), 0, 20).round(1)
    previous_grade = np.clip(np.random.normal(65, 15, n), 0, 100).round(1)
    internet_access = np.random.choice(["Yes", "No"], n, p=[0.85, 0.15])
    parent_education = np.random.choice(
        ["High School", "Bachelors", "Masters", "PhD"], n, p=[0.4, 0.35, 0.2, 0.05]
    )
    family_support = np.random.choice(["Yes", "No"], n, p=[0.6, 0.4])
    failures = np.random.choice([0, 1, 2, 3], n, p=[0.65, 0.2, 0.1, 0.05])

    # A hidden "true score" drives the final grade — this keeps the
    # relationship between features and target realistic (not pure noise),
    # so the ML models have real signal to learn from.
    education_map = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
    support_bonus = np.where(family_support == "Yes", 3, 0)
    internet_bonus = np.where(internet_access == "Yes", 2, 0)
    parent_bonus = np.array([education_map[p] for p in parent_education]) * 1.5

    true_score = (
        0.35 * attendance_percentage
        + 2.2 * weekly_study_hours
        + 0.35 * previous_grade
        - 6 * failures
        + support_bonus
        + internet_bonus
        + parent_bonus
        + np.random.normal(0, 8, n)  # random noise -> real-world unpredictability
    )

    # Scale true_score to a 0-100 "final_score" and bucket into grades
    final_score = np.clip(
        (true_score - true_score.min()) / (true_score.max() - true_score.min()) * 100,
        0, 100
    ).round(1)

    def score_to_grade(score):
        if score >= 85:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 55:
            return "C"
        elif score >= 40:
            return "D"
        else:
            return "F"

    grade = np.array([score_to_grade(s) for s in final_score])

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "attendance_percentage": attendance_percentage,
        "weekly_study_hours": weekly_study_hours,
        "previous_grade": previous_grade,
        "internet_access": internet_access,
        "parent_education": parent_education,
        "family_support": family_support,
        "failures": failures,
        "final_score": final_score,
        "grade": grade,
    })

    # Inject a small amount of realistic messiness (missing values, dupes)
    # so the "Data Cleaning" step in train_model.py has real work to do.
    for col in ["attendance_percentage", "weekly_study_hours", "parent_education"]:
        idx = np.random.choice(df.index, size=int(0.02 * n), replace=False)
        df.loc[idx, col] = np.nan

    dupes = df.sample(15, random_state=1)
    df = pd.concat([df, dupes], ignore_index=True)

    return df


if __name__ == "__main__":
    dataset = generate_student_data()
    dataset.to_csv("data/student.csv", index=False)
    print(f"Saved {len(dataset)} rows to data/student.csv")
    print(dataset["grade"].value_counts())
