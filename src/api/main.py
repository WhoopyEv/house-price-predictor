from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from inference import predict_price, batch_predict
from schemas import HousePredictionRequest, PredictionResponse

# ---------------------------------------------------------
# Inicialización de la aplicación FastAPI con metadatos
# ---------------------------------------------------------
app = FastAPI(
    title="House Price Prediction API",  # Título de la API
    description=(
        "An API for predicting house prices based on various features. "
        "This application is part of the MLOps Bootcamp by School of Devops. "
        "Authored by Gourav Shah."
    ),  # Descripción mostrada en la documentación
    version="1.0.0",  # Versión de la API
    contact={  # Información de contacto
        "name": "School of Devops",
        "url": "https://schoolofdevops.com",
        "email": "learn@schoolofdevops.com",
    },
    license_info={  # Licencia
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

# ---------------------------------------------------------
# Configuración de CORS (Cross-Origin Resource Sharing)
# Permite que clientes desde cualquier origen hagan peticiones
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Se permite cualquier dominio de origen
    allow_credentials=True,     # Se permiten credenciales (cookies/autenticación)
    allow_methods=["*"],        # Se permiten todos los métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],        # Se permiten todos los encabezados
)

# ---------------------------------------------------------
# Endpoint de verificación de estado (health check)
# Útil para monitoreo y asegurarse de que el servicio está activo
# ---------------------------------------------------------
@app.get("/health", response_model=dict)
async def health_check():
    return {"status": "healthy", "model_loaded": True}

# ---------------------------------------------------------
# Endpoint para predicción de un solo registro
# Recibe un objeto HousePredictionRequest y devuelve un PredictionResponse
# ---------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: HousePredictionRequest):
    return predict_price(request)

# ---------------------------------------------------------
# Endpoint para predicción en batch (varios registros a la vez)
# Recibe una lista de HousePredictionRequest y devuelve una lista de respuestas
# ---------------------------------------------------------
@app.post("/batch-predict", response_model=list)
async def batch_predict_endpoint(requests: list[HousePredictionRequest]):
    return batch_predict(requests)
