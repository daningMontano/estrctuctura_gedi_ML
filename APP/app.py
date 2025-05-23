import streamlit as st
from exploracion import run_exploracion
from prediccion import run_prediccion

st.set_page_config(
    page_title="GEDI Structure ML",
    layout="wide",
    initial_sidebar_state="auto",
)

st.title("🌳 GEDI Structure ML App")

# Sidebar de navegación
page = st.sidebar.radio("Navegar a:", ["🔎 Exploración", "🤖 Predicción"])

if page == "🔎 Exploración":
    run_exploracion()
else:
    run_prediccion()
