import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
@st.cache_data
def load_data():
    # base_dir apunta a la carpeta raíz 'GEDI_STRUCTURE_ML'
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "ML" / "DATA"

    # si quieres DATA_GEDI..., o df_union_gedi.csv, usa el nombre correcto:
    csv_path = data_dir / "df_union_gedi.csv"
    if not csv_path.exists():
        st.error(f"No encontré el archivo: {csv_path}")
        return pd.DataFrame()  # evitar que rompa todo

    df = pd.read_csv(csv_path)
    return df

def run_exploracion():
    st.header("🔎 Análisis Exploratorio de Datos")
    df = load_data()
    
    st.markdown("""
    **Descripción:**  
    Este módulo muestra las estadísticas descriptivas y algunas visualizaciones exploratorias
    de tus datos de bandas Sentinel y GEDI.
    """)

    # Estadísticas
    st.subheader("📊 Estadísticas descriptivas")
    st.dataframe(df.describe(), use_container_width=True)

    # Matriz de correlación
    st.subheader("🔗 Matriz de correlación (Spearman)")
    corr = df.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", linewidths=0.2, ax=ax)
    st.pyplot(fig)

    # Gráficos de dispersión seleccionados
    st.subheader("📈 Dispersión vs. N_efectivo_estratos")
    target = "N_efectivo_estratos"
    predictors = ["B12", "lat", "B11", "lon", "GCI", "B4", "elevation"] 
    for col in predictors:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=col, y=target, ax=ax)
        sns.regplot(data=df, x=col, y=target, scatter=False, lowess=True, ax=ax, color="red")
        ax.set_title(f"{col} vs {target}")
        st.pyplot(fig)
