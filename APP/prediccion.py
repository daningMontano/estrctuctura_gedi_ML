# APP/prediccion.py

import streamlit as st
import joblib
import mlflow.pyfunc
from pathlib import Path
import numpy as np
import rasterio
from rasterio.io import MemoryFile
import folium
from streamlit_folium import st_folium
from branca.colormap import LinearColormap

# --------------------------------------------------
# 1) Función para cargar artefactos: scaler + modelo
# --------------------------------------------------
@st.cache_resource
def load_scaler_and_model():
    project_root = Path(__file__).resolve().parent.parent
    scaler_path  = project_root / "ML" / "scaler.pkl"
    if not scaler_path.exists():
        st.error(f"No encontré el scaler en: {scaler_path}")
        return None, None
    scaler = joblib.load(scaler_path)

    mlflow.set_tracking_uri("http://localhost:9090")
    model_uri = "models:/gedi_structure_ml/1"
    model = mlflow.pyfunc.load_model(model_uri)

    return scaler, model

# --------------------------------------------------
# 2) Lógica de la pestaña de Predicción
# --------------------------------------------------

def run_prediccion():
    st.header("🤖 Predicción de Estructura GEDI")

    scaler, model = load_scaler_and_model()
    if scaler is None or model is None:
        st.stop()

    st.markdown("Sube un archivo **GeoTIFF** con las bandas satelitales:")
    uploaded = st.file_uploader("GeoTIFF", type=["tif", "tiff"])
    if not uploaded:
        st.info("Esperando que subas un GeoTIFF…")
        return

    # Guarda los bytes en session_state la primera vez
    if "tif_bytes" not in st.session_state:
        st.session_state["tif_bytes"] = uploaded.read()

    # Botón para disparar la predicción
    if st.button("▶️ Ejecutar predicción"):
        tif_bytes = st.session_state["tif_bytes"]
        with MemoryFile(tif_bytes) as mem:
            with mem.open() as src:
                bands  = src.read()
                bounds = src.bounds
                profile = src.profile.copy()

        # Prepara datos
        n_bands, h, w = bands.shape
        data = bands.reshape(n_bands, -1).T

        # Predicción
        X_scaled = scaler.transform(data)
        preds = model.predict(X_scaled)
        pred_map = np.array(preds).reshape(h, w)

        # Guarda en session_state
        st.session_state["pred_map"] = pred_map
        st.session_state["bounds"]   = bounds
        st.session_state["profile"]  = profile

    # Si ya calculamos la predicción, la mostramos y ofrecemos descarga
    if "pred_map" in st.session_state:
        pred_map = st.session_state["pred_map"]
        bounds   = st.session_state["bounds"]
        profile  = st.session_state["profile"]

        # Visualización (misma paleta que antes)
        colormap = LinearColormap(
            colors=["#fee5d9", "#a50f15"], vmin=1, vmax=12,
            caption="Predicción GEDI (1–12)"
        )
        mid_lat = (bounds.top + bounds.bottom) / 2
        mid_lon = (bounds.left + bounds.right) / 2
        m = folium.Map(location=[mid_lat, mid_lon], zoom_start=10)
        folium.raster_layers.ImageOverlay(
            image=pred_map,
            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            opacity=0.7, name="Predicción GEDI",
            colormap=colormap
        ).add_to(m)
        folium.LayerControl().add_to(m)
        colormap.add_to(m)

        st.subheader("🌍 Mapa de Predicciones")
        st_folium(m, width=700, height=500)

        # ————————————————
        # Botón de descarga
        # ————————————————
        # Prepara el profile para un solo canal de float32
        profile.update({
            "count": 1,
            "dtype": rasterio.float32,
            "compress": "lzw"
        })
        # Escribe en buffer
        with MemoryFile() as mem_out:
            with mem_out.open(**profile) as dst:
                dst.write(pred_map.astype(np.float32), 1)
            geotiff_bytes = mem_out.read()

        st.download_button(
            label="💾 Descargar predicción (GeoTIFF)",
            data=geotiff_bytes,
            file_name="prediccion_gedi.tif",
            mime="image/tiff"
        )

        st.success("¡Predicción completada con éxito!")
