import joblib
import pandas as pd
from datetime import datetime
from schemas import HousePredictionRequest, PredictionResponse

# Definir las rutas donde se guardaron el modelo y el preprocesador
MODEL_PATH = "models/trained/house_price_model.pkl"
PREPROCESSOR_PATH = "models/trained/preprocessor.pkl"

try:
    # Cargar el modelo entrenado desde archivo
    model = joblib.load(MODEL_PATH)
    # Cargar el preprocesador (transformaciones usadas en entrenamiento)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    # Si ocurre un error en la carga, se lanza una excepción
    raise RuntimeError(f"Error loading model or preprocessor: {str(e)}")

def predict_price(request: HousePredictionRequest) -> PredictionResponse:
    """
    Predice el precio de una casa en base a las características de entrada.
    """
    # Convertir los datos de la request (dict) en un DataFrame de Pandas
    input_data = pd.DataFrame([request.dict()])

    # Crear nuevas características derivadas
    input_data['house_age'] = datetime.now().year - input_data['year_built']  # Edad de la casa
    input_data['bed_bath_ratio'] = input_data['bedrooms'] / input_data['bathrooms']  # Relación habitaciones/baños
    input_data['price_per_sqft'] = 0  # Valor ficticio para compatibilidad con el preprocesador

    # Aplicar el preprocesamiento al DataFrame
    processed_features = preprocessor.transform(input_data)

    # Generar la predicción con el modelo
    predicted_price = model.predict(processed_features)[0]

    # Convertir el resultado a float nativo de Python y redondear a 2 decimales
    predicted_price = round(float(predicted_price), 2)

    # Calcular un intervalo de confianza aproximado (±10% del valor predicho)
    confidence_interval = [predicted_price * 0.9, predicted_price * 1.1]

    # Convertir el intervalo a float y redondear a 2 decimales
    confidence_interval = [round(float(value), 2) for value in confidence_interval]

    # Devolver la respuesta estructurada como un objeto PredictionResponse
    return PredictionResponse(
        predicted_price=predicted_price,
        confidence_interval=confidence_interval,
        features_importance={},  # Aquí se podría incluir importancia de variables si se calcula
        prediction_time=datetime.now().isoformat()  # Timestamp de la predicción
    )

def batch_predict(requests: list[HousePredictionRequest]) -> list[float]:
    """
    Realiza predicciones en lote para múltiples casas.
    """
    # Convertir la lista de requests en un DataFrame
    input_data = pd.DataFrame([req.dict() for req in requests])

    # Crear las mismas características derivadas que en predicción individual
    input_data['house_age'] = datetime.now().year - input_data['year_built']
    input_data['bed_bath_ratio'] = input_data['bedrooms'] / input_data['bathrooms']
    input_data['price_per_sqft'] = 0  # Valor ficticio para compatibilidad

    # Aplicar el preprocesamiento a todos los registros
    processed_features = preprocessor.transform(input_data)

    # Generar predicciones para cada registro
    predictions = model.predict(processed_features)

    # Convertir los resultados a lista de floats de Python
    return predictions.tolist()
