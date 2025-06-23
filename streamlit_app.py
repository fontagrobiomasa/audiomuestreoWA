import streamlit as st
import os
import tempfile
import zipfile
import re
import numpy as np
import pandas as pd
from faster_whisper import WhisperModel

st.set_page_config(layout="wide")

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
                if re.search(r'*\.jpg', line):
                    match_fh = re.match(r"(\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2})", line)
                    fecha_hora = match_fh.group(1) if match_fh else ""
                    current_point = {
                        "foto": re.search(r'*\.jpg', line).group(),
                        "nombre": None,
                        "lat": None,
                        "lon": None,
                        "audio": None,
                        "fecha_hora": fecha_hora
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
                    audio_match = re.search(r'PTT.*\.opus', line)
                    current_point["audio"] = audio_match.group()
                    puntos.append(current_point)
                    current_point = {}

            if not puntos:
                st.warning("No se detectaron puntos válidos en el chat.")
            else:
                st.info(f"Se detectaron {len(puntos)} puntos para procesar.")
                resultados = []

                try:
                    model = WhisperModel("base", compute_type="int8")

                    for punto in puntos:
                        audio_path = os.path.join(tmpdir, punto["audio"])
                        if not os.path.exists(audio_path):
                            st.warning(f"No se encontró el archivo de audio: {punto['audio']}")
                            continue

                        try:
                            segments, _ = model.transcribe(audio_path, language=lang)
                            texto = " ".join([seg.text for seg in segments])
                            alturas = extraer_alturas(texto)

                            if alturas:
                                alturas_array = np.array(alturas)
                                promedio = np.mean(alturas_array)
                                desvio = np.std(alturas_array)
                                minimo = np.min(alturas_array)
                                maximo = np.max(alturas_array)
                                mediana = np.median(alturas_array)
                                n = len(alturas_array)
                            else:
                                promedio = desvio = minimo = maximo = mediana = n = 0

                            resultados.append({
                                "Punto": punto["nombre"],
                                "Lat": punto["lat"],
                                "Lon": punto["lon"],
                                "Altura": round(mediana, 2),
                                "Mediana": round(mediana, 2),
                                "Promedio": round(promedio, 2),
                                "Desvío estándar": round(desvio, 2),
                                "Mínimo": round(minimo, 2),
                                "Máximo": round(maximo, 2),
                                "N": n,
                                "FechaHora": punto.get("fecha_hora", ""),
                                "Archivo": punto["audio"]
                            })

                        except Exception as e:
                            resultados.append({
                                "Punto": punto["nombre"],
                                "Lat": punto["lat"],
                                "Lon": punto["lon"],
                                "Altura": "-",
                                "Mediana": "-",
                                "Promedio": "-",
                                "Desvío estándar": f"{e}",
                                "Mínimo": "-",
                                "Máximo": "-",
                                "N": "Error",
                                "FechaHora": "-",
                                "Archivo": punto["audio"]                                
                            })

                except Exception as e:
                    st.error(f"Error general durante la transcripción: {e}")

                if resultados:
                    # Guardar en session_state si no está cargado
                    if "df_resultados" not in st.session_state:
                        st.session_state.df_resultados = pd.DataFrame(resultados)
                        st.session_state.df_resultados["Seleccionar"] = False

# Mostrar tabla con selección (aunque no se haya vuelto a cargar)
if "df_resultados" in st.session_state:
    st.markdown("### Resultados por punto (marcá las filas)")

    # Reordenar columnas antes de mostrar (por si se perdió el orden en una edición previa)
    df_editable = st.session_state.df_resultados.copy()
    columnas_ordenadas = ["Seleccionar", "Punto", "Lat", "Lon", "Altura", "Mediana", "Promedio", "Desvío estándar", "Mínimo", "Máximo", "N", "FechaHora", "Archivo"]
    columnas_presentes = [col for col in columnas_ordenadas if col in df_editable.columns]
    df_editable = df_editable[columnas_presentes]

    edited_df = st.data_editor(
        df_editable,
        key="data_editor",
        use_container_width=True,
        column_config={
            "Seleccionar": st.column_config.CheckboxColumn(
                "Seleccionar", help="Marcá los puntos que querés incluir"
            ),
            "Altura": st.column_config.NumberColumn(
                "Altura", help="Valor editable de la mediana (modificá si hubo error)", format="%.2f"
            )
        },
        hide_index=True
    )

    seleccionados = edited_df[edited_df["Seleccionar"] == True]

    if not seleccionados.empty:
        try:
            cadena = 'Lat,Lon,Altura;'+";".join(
                f"{row['Lat']},{row['Lon']},{row['Altura']}"
                for _, row in seleccionados.iterrows()
                if isinstance(row["Altura"], (int, float))
            )
            st.markdown("### Cadena generada (lat,lon,altura):")
            st.text_area("Cadena generada:", cadena, height=100)
        except Exception as e:
            st.error(f"No se pudo generar la cadena: {e}")
    else:
        st.info("Marcá al menos una fila para generar la cadena.")
