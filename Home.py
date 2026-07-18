"""
pages/Home.py
--------------
Landing page: explains what the project does, the dataset, and the
overall ML workflow, so anyone opening the app understands it immediately.
"""

import streamlit as st

st.title("🎓 EduPredict AI")
st.subheader("Student Performance Prediction with an AI Study Assistant")

st.markdown("""
EduPredict AI predicts a student's likely academic grade (A-F) from
academic and personal factors, then uses a small local AI model to turn
that prediction into personalized, encouraging study advice.

This project was built as an end-to-end, interview-ready machine learning
application - covering the full pipeline from raw data to a deployed
interactive app.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Dataset")
    st.markdown("""
    - **Rows:** ~1,200 students
    - **Target:** Grade Category (A, B, C, D, F)
    - **Features used:**
        - Age, Gender
        - Attendance %
        - Weekly Study Hours
        - Previous Grade
        - Internet Access
        - Parent Education Level
        - Family Support
        - Number of Past Failures

    *Note: this deployment ships with a synthetic dataset that mirrors the
    structure of the UCI/Kaggle "Student Performance" datasets (generated
    because this environment has no internet access to download the real
    file). Drop a real dataset into `data/student.csv` with matching
    column names to use it instead - no other code changes needed.*
    """)

with col2:
    st.markdown("### 🧠 Machine Learning Workflow")
    st.markdown("""
    1. **Data Cleaning** - remove duplicates, handle missing values
    2. **Feature Engineering** - study efficiency, academic risk flag
    3. **Encoding** - Label Encoding for categorical columns
    4. **Scaling** - StandardScaler for numeric columns
    5. **Model Training** - Logistic Regression, Decision Tree, Random Forest
    6. **Evaluation** - Accuracy, Precision, Recall, F1, Confusion Matrix
    7. **Best Model Selection** - by macro F1 score
    8. **AI Study Advice** - local flan-t5 model generates personalized tips
    """)

st.divider()
st.markdown("### 🔁 Workflow Diagram")

st.markdown("""
```
Raw CSV Data
     │
     ▼
Data Cleaning  (duplicates, missing values, drop leaky columns)
     │
     ▼
Feature Engineering  (study_efficiency, academic_risk)
     │
     ▼
Encoding + Scaling  (LabelEncoder, StandardScaler)
     │
     ▼
Train / Test Split  (80 / 20, stratified)
     │
     ▼
Train 3 Models  →  Logistic Regression | Decision Tree | Random Forest
     │
     ▼
Evaluate & Pick Best Model  (macro F1 score)
     │
     ▼
Save Model + Preprocessing Artifacts
     │
     ▼
Streamlit App:  Predict Page  →  AI Study Assistant (flan-t5, local)
```
""")

st.info(
    "Use the sidebar to navigate to **Predict** for a live prediction, "
    "or **Analytics** to explore the dataset and model performance."
)
