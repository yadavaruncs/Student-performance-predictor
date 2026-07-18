"""
pages/Analytics.py
-------------------
Displays dataset overview stats and all the plots generated during
training (grade distribution, attendance, correlation heatmap, feature
importance, confusion matrix, model comparison).
"""

import sys
import os
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_data, clean_data

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLOTS_DIR = os.path.join(ROOT, "plots")

st.title("📊 Analytics")

df = clean_data(load_data())

st.markdown("### Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Students", len(df))
c2.metric("Features", df.shape[1] - 1)
c3.metric("Avg. Attendance", f"{df['attendance_percentage'].mean():.1f}%")
c4.metric("Avg. Study Hours", f"{df['weekly_study_hours'].mean():.1f}")

with st.expander("View sample data"):
    st.dataframe(df.head(20), use_container_width=True)

st.divider()

st.markdown("### Grade & Attendance Distribution")
col1, col2 = st.columns(2)
with col1:
    p = os.path.join(PLOTS_DIR, "grade_distribution.png")
    if os.path.exists(p):
        st.image(p, use_container_width=True)
with col2:
    p = os.path.join(PLOTS_DIR, "attendance_distribution.png")
    if os.path.exists(p):
        st.image(p, use_container_width=True)

st.markdown("### Correlation Heatmap")
p = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
if os.path.exists(p):
    st.image(p, use_container_width=True)

st.markdown("### Model Comparison & Feature Importance")
col3, col4 = st.columns(2)
with col3:
    p = os.path.join(PLOTS_DIR, "model_accuracy_comparison.png")
    if os.path.exists(p):
        st.image(p, use_container_width=True)
with col4:
    p = os.path.join(PLOTS_DIR, "feature_importance.png")
    if os.path.exists(p):
        st.image(p, use_container_width=True)

st.markdown("### Confusion Matrix (Best Model)")
p = os.path.join(PLOTS_DIR, "confusion_matrix.png")
if os.path.exists(p):
    st.image(p, use_container_width=True)

if not os.path.exists(os.path.join(PLOTS_DIR, "grade_distribution.png")):
    st.warning("No plots found yet - run `python train_model.py` first to generate them.")
