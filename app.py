"""
app.py
------
Entry point for the EduPredict AI Streamlit application.

Run with:  streamlit run app.py

Uses Streamlit's built-in multipage navigation (st.navigation / st.Page)
so each page lives in its own file under pages/ - this keeps the code
modular and matches how real Streamlit apps are structured.
"""

import streamlit as st

st.set_page_config(
    page_title="EduPredict AI",
    page_icon="🎓",
    layout="wide",
)

home_page = st.Page("pages/Home.py", title="Home", icon="🏠")
predict_page = st.Page("pages/Predict.py", title="Predict", icon="🎯")
analytics_page = st.Page("pages/Analytics.py", title="Analytics", icon="📊")

pg = st.navigation([home_page, predict_page, analytics_page])
pg.run()
