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
# 1) Carga del scaler local y del modelo MLflow
# --------------------------------------------------
@st.cache_resource
def load_scaler_and_model():
    project_root = Path(__file__).resolve().parent.parent

    # 1.1) Carga del scaler entrenado localmente
    scaler_path = project_root / "ML" / "scaler.pkl"
    if not scaler_path.exists():
        st.error(f"No encontré el scaler en: {scaler_path}")
        return None, None
    scaler = joblib.load(scaler_path)

    # 1.2) Conexión a tu MLflow Tracking Server
    mlflow.set_tracking_uri("http://localhost:9090")

    # 1.3) Carga de la versión 1 de tu modelo registrado
    #    Ajusta a la versión que corresponda si es distinta
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

    st.markdown("1️⃣ Sube un archivo **GeoTIFF** con las bandas satelitales:")
    uploaded = st.file_uploader("GeoTIFF", type=["tif", "tiff"])
    if not uploaded:
        st.info("Sube un GeoTIFF para predecir.")
        return

    # Guardamos los bytes para poder reusar el GeoTIFF
    if "tif_bytes" not in st.session_state:
        st.session_state["tif_bytes"] = uploaded.read()

    # Botón para ejecutar la predicción
    if st.button("▶️ Ejecutar predicción"):
        tif_bytes = st.session_state["tif_bytes"]
        with MemoryFile(tif_bytes) as mem:
            with mem.open() as src:
                bands   = src.read()          # (n_bands, height, width)
                bounds  = src.bounds
                profile = src.profile.copy()

        # Preparar matriz de entrada
        n_bands, h, w = bands.shape
        data = bands.reshape(n_bands, -1).T  # (n_pixels, n_bands)

        # Escalado con el scaler local
        X_scaled = scaler.transform(data)
        # Predicción con el modelo MLflow
        preds    = model.predict(X_scaled)
        pred_map = np.array(preds).reshape(h, w)

        # Guardamos en session_state para no recálculo
        st.session_state["pred_map"] = pred_map
        st.session_state["bounds"]   = bounds
        st.session_state["profile"]  = profile

    # Si ya tenemos el resultado, lo mostramos y ofrecemos descarga
    if "pred_map" in st.session_state:
        pred_map = st.session_state["pred_map"]
        bounds   = st.session_state["bounds"]
        profile  = st.session_state["profile"]

        # Paleta de rojo claro a intenso para valores 1–12
        colormap = LinearColormap(
            colors=["#fee5d9", "#a50f15"],
            vmin=1,
            vmax=12,
            caption="Predicción GEDI (1–12)"
        )
        mid_lat = (bounds.top + bounds.bottom) / 2
        mid_lon = (bounds.left + bounds.right) / 2

        m = folium.Map(location=[mid_lat, mid_lon], zoom_start=10)
        folium.raster_layers.ImageOverlay(
            image=pred_map,
            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            opacity=0.7,
            name="Predicción GEDI",
            colormap=colormap
        ).add_to(m)
        folium.LayerControl().add_to(m)
        colormap.add_to(m)

        st.subheader("🌍 Mapa de Predicción")
        st_folium(m, width=700, height=500)

        # Botón de descarga como GeoTIFF
        profile.update(count=1, dtype=rasterio.float32, compress="lzw")
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

# --------------------------------------------------
# 3) Ejecución directa para pruebas locales
# --------------------------------------------------
if __name__ == "__main__":
    run_prediccion()

