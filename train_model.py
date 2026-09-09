"""
Trains a Logistic Regression model to predict whether a student
will Pass or Fail, and saves it to model.pkl so app.py can use it.
"""

import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/students.csv")

# turn Yes/No and Pass/Fail into numbers the model can use
df["extra_activities"] = df["extra_activities"].map({"Yes": 1, "No": 0})
df["result"] = df["result"].map({"Pass": 1, "Fail": 0})

features = ["study_hours", "attendance", "previous_score", "extra_activities"]
X = df[features]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy on test data: {accuracy:.2%}")
print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to model.pkl")
