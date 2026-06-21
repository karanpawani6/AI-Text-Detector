"""
app.py — Unified entry point for the AI Text Detector demo.

Switches between the two full model GUIs (Decision Tree / BERT-MLP)
using Streamlit's top navigation. Run with:

    streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="AI Text Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

pg = st.navigation(
    [
        st.Page("decision_tree/dt_gui.py", title="Decision Tree", icon="🌳", default=True),
        st.Page("bert_mlp/ann_gui.py", title="BERT-MLP", icon="🧠"),
    ],
    position="top",
)
pg.run()
