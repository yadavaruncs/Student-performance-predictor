"""
Creates a fake dataset of students and saves it to data/students.csv.

I couldn't find a clean real-world dataset for this, so this script just
makes one up using some simple rules (more study hours + attendance +
good previous score = more likely to pass), plus a bit of randomness so
it isn't a perfectly straight line.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

n_students = 300

study_hours = np.random.uniform(0, 8, n_students)          # hours per day
attendance = np.random.normal(75, 15, n_students)          # percent
attendance = np.clip(attendance, 40, 100)
previous_score = np.random.normal(65, 15, n_students)      # marks out of 100
previous_score = np.clip(previous_score, 0, 100)
extra_activities = np.random.choice(["Yes", "No"], n_students, p=[0.4, 0.6])

# rough "final score" used only to decide pass/fail, plus some noise
noise = np.random.normal(0, 5, n_students)
final_score = (
    0.35 * previous_score
    + 4.5 * study_hours
    + 0.25 * attendance
    + np.where(extra_activities == "Yes", 2, 0)
    + noise
)

result = np.where(final_score >= 60, "Pass", "Fail")

df = pd.DataFrame({
    "study_hours": study_hours.round(1),
    "attendance": attendance.round(1),
    "previous_score": previous_score.round(1),
    "extra_activities": extra_activities,
    "result": result,
})

df.to_csv("data/students.csv", index=False)
print(f"Saved {len(df)} rows to data/students.csv")
print(df["result"].value_counts())
