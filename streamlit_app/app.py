import streamlit as st
import requests
import json
import time
import os
import socket  # Para obtener hostname y dirección IP

# -----------------------------
# Configuración inicial de la app de Streamlit
# -----------------------------
st.set_page_config(
    page_title="House Price Predictor",  # Título de la pestaña
    layout="wide",                       # Página en formato ancho
    initial_sidebar_state="collapsed"    # Sidebar colapsada por defecto
)

# -----------------------------
# Encabezado y descripción
# -----------------------------
st.title("House Price Prediction")
st.markdown(
    """
    <p style="font-size: 18px; color: gray;">
        A simple MLOps demonstration project for real-time house price prediction
    </p>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Layout de dos columnas
# -----------------------------
col1, col2 = st.columns(2, gap="large")

# -----------------------------
# Columna izquierda: Formulario de entrada
# -----------------------------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Slider para "Square Footage"
    st.markdown(f"<p><strong>Square Footage:</strong> <span id='sqft-value'></span></p>", unsafe_allow_html=True)
    sqft = st.slider("", 500, 5000, 1500, 50, label_visibility="collapsed", key="sqft")
    st.markdown(f"<script>document.getElementById('sqft-value').innerText = '{sqft} sq ft';</script>", unsafe_allow_html=True)

    # Selección de dormitorios y baños en dos columnas
    bed_col, bath_col = st.columns(2)
    with bed_col:
        st.markdown("<p><strong>Bedrooms</strong></p>", unsafe_allow_html=True)
        bedrooms = st.selectbox("", options=[1, 2, 3, 4, 5, 6], index=2, label_visibility="collapsed")

    with bath_col:
        st.markdown("<p><strong>Bathrooms</strong></p>", unsafe_allow_html=True)
        bathrooms = st.selectbox("", options=[1, 1.5, 2, 2.5, 3, 3.5, 4], index=2, label_visibility="collapsed")

    # Dropdown de ubicación
    st.markdown("<p><strong>Location</strong></p>", unsafe_allow_html=True)
    location = st.selectbox("", options=["Urban", "Suburban", "Rural", "Urban", "Waterfront", "Mountain"], index=1, label_visibility="collapsed")

    # Slider para "Year Built"
    st.markdown(f"<p><strong>Year Built:</strong> <span id='year-value'></span></p>", unsafe_allow_html=True)
    year_built = st.slider("", 1900, 2025, 2000, 1, label_visibility="collapsed", key="year")
    st.markdown(f"<script>document.getElementById('year-value').innerText = '{year_built}';</script>", unsafe_allow_html=True)

    # Botón para predecir
    predict_button = st.button("Predict Price", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Columna derecha: Resultados de la predicción
# -----------------------------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h2>Prediction Results</h2>", unsafe_allow_html=True)

    # Si el usuario hace clic en "Predict"
    if predict_button:
        # Muestra spinner mientras se calcula
        with st.spinner("Calculating prediction..."):
            # Datos de entrada para la API
            api_data = {
                "sqft": sqft,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "location": location.lower(),
                "year_built": year_built,
                "condition": "Good"
            }

            try:
                # URL del backend (FastAPI), configurable vía variable de entorno
                api_endpoint = os.getenv("API_URL", "http://model:8000")
                predict_url = f"{api_endpoint.rstrip('/')}/predict"

                st.write(f"Connecting to API at: {predict_url}")

                # Llamada a la API de FastAPI
                response = requests.post(predict_url, json=api_data)
                response.raise_for_status()  # Lanza error si status code != 200
                prediction = response.json()

                # Guardar resultados en la sesión de Streamlit
                st.session_state.prediction = prediction
                st.session_state.prediction_time = time.time()
            except requests.exceptions.RequestException as e:
                # Si falla la API, se usa mock data
                st.error(f"Error connecting to API: {e}")
                st.warning("Using mock data for demonstration purposes. Please check your API connection.")
                st.session_state.prediction = {
                    "predicted_price": 467145,
                    "confidence_interval": [420430.5, 513859.5],
                    "features_importance": {
                        "sqft": 0.43,
                        "location": 0.27,
                        "bathrooms": 0.15
                    },
                    "prediction_time": "0.12 seconds"
                }
                st.session_state.prediction_time = time.time()

    # -----------------------------
    # Mostrar resultados si existen
    # -----------------------------
    if "prediction" in st.session_state:
        pred = st.session_state.prediction

        # Precio formateado con separadores
        formatted_price = "${:,.0f}".format(pred["predicted_price"])
        st.markdown(f'<div class="prediction-value">{formatted_price}</div>', unsafe_allow_html=True)

        # Información adicional en tarjetas
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<p class="info-label">Confidence Score</p>', unsafe_allow_html=True)
            st.markdown('<p class="info-value">92%</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<p class="info-label">Model Used</p>', unsafe_allow_html=True)
            st.markdown('<p class="info-value">XGBoost</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Rango de precios y tiempo de predicción
        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<p class="info-label">Price Range</p>', unsafe_allow_html=True)
            lower = "${:,.1f}".format(pred["confidence_interval"][0])
            upper = "${:,.1f}".format(pred["confidence_interval"][1])
            st.markdown(f'<p class="info-value">{lower} - {upper}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_d:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<p class="info-label">Prediction Time</p>', unsafe_allow_html=True)
            st.markdown('<p class="info-value">0.12 seconds</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Principales factores que afectan el precio
        st.markdown('<div class="top-factors">', unsafe_allow_html=True)
        st.markdown("<p><strong>Top Factors Affecting Price:</strong></p>", unsafe_allow_html=True)
        st.markdown("""
        <ul>
            <li>Square Footage</li>
            <li>Number of Bedrooms/Bathrooms</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Mensaje placeholder si aún no hay predicción
        st.markdown("""
        <div style="display: flex; height: 300px; align-items: center; justify-content: center; color: #6b7280; text-align: center;">
            Fill out the form and click "Predict Price" to see the estimated house price.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Footer con versión, hostname e IP
# -----------------------------
version = os.getenv("APP_VERSION", "1.0.0")  # Toma versión desde variable de entorno
hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)

st.markdown("<hr>", unsafe_allow_html=True)  # Línea de separación
st.markdown(
    f"""
    <div style="text-align: center; color: gray; margin-top: 20px;">
        <p><strong>Built for MLOps Bootcamp</strong></p>
        <p>by <a href="https://www.schoolofdevops.com" target="_blank">School of Devops</a></p>
        <p><strong>Version:</strong> {version}</p>
        <p><strong>Hostname:</strong> {hostname}</p>
        <p><strong>IP Address:</strong> {ip_address}</p>
    </div>
    """,
    unsafe_allow_html=True,
)
