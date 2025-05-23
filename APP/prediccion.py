# APP/prediccion.py

import streamlit as st
import joblib
from pathlib import Path
import numpy as np
import rasterio
from rasterio.io import MemoryFile
import folium
from streamlit_folium import st_folium
from branca.colormap import LinearColormap

@st.cache_resource
def load_pipeline():
    project_root = Path(__file__).resolve().parent.parent
    path = project_root / "ML" / "pipeline_model.pkl"
    if not path.exists():
        st.error(f"No encontré el pipeline en: {path}")
        return None
    return joblib.load(path)

def run_prediccion():
    st.header("🤖 Predicción de Estructura GEDI")
    pipeline = load_pipeline()
    if pipeline is None:
        st.stop()

    uploaded = st.file_uploader("GeoTIFF", type=["tif","tiff"])
    if not uploaded:
        return st.info("Sube un GeoTIFF para predecir.")
    if "tif_bytes" not in st.session_state:
        st.session_state["tif_bytes"] = uploaded.read()

    if st.button("▶️ Ejecutar predicción"):
        tif = st.session_state["tif_bytes"]
        with MemoryFile(tif) as mem:
            with mem.open() as src:
                bands, bounds, profile = src.read(), src.bounds, src.profile.copy()
        n_b,h,w = bands.shape
        data = bands.reshape(n_b,-1).T
        preds = pipeline.predict(data).reshape(h,w)
        st.session_state.update(pred_map=preds, bounds=bounds, profile=profile)

    if "pred_map" in st.session_state:
        pred_map = st.session_state["pred_map"]
        bounds   = st.session_state["bounds"]
        profile  = st.session_state["profile"]

        cmap = LinearColormap(["#fee5d9","#a50f15"], vmin=1, vmax=12, caption="1–12")
        m = folium.Map(
            location=[(bounds.top+bounds.bottom)/2, (bounds.left+bounds.right)/2],
            zoom_start=10
        )
        folium.raster_layers.ImageOverlay(
            image=pred_map,
            bounds=[[bounds.bottom,bounds.left],[bounds.top,bounds.right]],
            opacity=0.7,
            name="Predicción",
            colormap=cmap
        ).add_to(m)
        folium.LayerControl().add_to(m)
        cmap.add_to(m)
        st.subheader("🌍 Mapa de Predicción")
        st_folium(m, width=700, height=500)

        # descarga GeoTIFF
        profile.update(count=1, dtype=rasterio.float32, compress="lzw")
        with MemoryFile() as out:
            with out.open(**profile) as dst:
                dst.write(pred_map.astype(np.float32),1)
            geo_bytes = out.read()
        st.download_button("💾 Descargar GeoTIFF", geo_bytes, "prediccion.tif", "image/tiff")
        st.success("¡Listo!")
