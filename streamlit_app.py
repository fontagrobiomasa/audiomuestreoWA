import streamlit as st
import os
import tempfile
import zipfile
import re
import numpy as np
import pandas as pd
from faster_whisper import WhisperModel

st.title("AudioMuestreo - Chat de WhatsApp")

st.markdown("""
Esta herramienta permite subir un `.zip` con un chat de WhatsApp y sus archivos multimedia, para transcribir los audios y asociarlos a puntos georreferenciados.
""")

uploaded_zip = st.file_uploader("Subí el archivo .zip con el chat y los archivos multimedia", type=["zip"])
lang = st.selectbox("Seleccioná el idioma del audio:", ["es", "en", "pt", "fr", "de"])

def extraer_alturas(texto):
    alturas = re.findall(r'\d+(?:[.,]\d+)?', texto)
    return [float(a.replace(",", ".")) for a in alturas]

if uploaded_zip and st.button("Procesar .zip"):
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        # Buscar el archivo de chat
        chat_txt = None
        for fname in os.listdir(tmpdir):
            if fname.lower().endswith(".txt"):
                chat_txt = os.path.join(tmpdir, fname)
                break

        if not chat_txt:
            st.error("No se encontró archivo .txt del chat en el .zip.")
        else:
            with open(chat_txt, 'r', encoding='utf-8') as f:
                chat_lines = f.readlines()

            puntos = []
            current_point = {}
            for line in chat_lines:
                if re.search(r'IMG.*\.jpg', line):
                    current_point = {
                        "foto": re.search(r'IMG.*\.jpg', line).group(),
                        "nombre": None,
                        "lat": None,
                        "lon": None,
                        "audio": None
                    }
                elif current_point.get("foto") and current_point["nombre"] is None:
                    current_point["nombre"] = line.strip()
                elif "https://maps.google.com" in line:
                    coords = re.findall(r"q=([-0-9.]+),([-0-9.]+)", line)
                    if coords:
                        lat, lon = coords[0]
                        current_point["lat"] = float(lat)
                        current_point["lon"] = float(lon)
                elif re.search(r'PTT.*\.opus', line):
