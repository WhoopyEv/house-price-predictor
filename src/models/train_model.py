import argparse
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import yaml
import logging
from mlflow.tracking import MlflowClient
import platform
import sklearn

# -----------------------------
# Configure logging
# -----------------------------
# Configura el sistema de logs para mostrar mensajes de ejecución con timestamp y nivel (INFO, ERROR, etc.)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# Argument parser
# -----------------------------
def parse_args():
    # Define los argumentos que se deben pasar al script desde la terminal
    parser = argparse.ArgumentParser(description="Train and register final model from config.")
    parser.add_argument("--config", type=str, required=True, help="Path to model_config.yaml")  # Ruta del archivo de configuración del modelo
    parser.add_argument("--data", type=str, required=True, help="Path to processed CSV dataset")  # Ruta del dataset preprocesado
    parser.add_argument("--models-dir", type=str, required=True, help="Directory to save trained model")  # Carpeta donde se guardará el modelo entrenado
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="MLflow tracking URI")  # URI de MLflow para registrar experimentos
    return parser.parse_args()

# -----------------------------
# Load model from config
# -----------------------------
def get_model_instance(name, params):
    # Mapea los nombres de modelos con sus clases de sklearn/xgboost
    model_map = {
        'LinearRegression': LinearRegression,
        'RandomForest': RandomForestRegressor,
        'GradientBoosting': GradientBoostingRegressor,
        'XGBoost': xgb.XGBRegressor
    }
    # Si el modelo no está soportado, lanza un error
    if name not in model_map:
        raise ValueError(f"Unsupported model: {name}")
    # Devuelve una instancia del modelo con sus parámetros configurados
    return model_map[name](**params)

# -----------------------------
# Main logic
# -----------------------------
def main(args):
    # 1. Cargar configuración desde YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    model_cfg = config['model']

    # 2. Configurar MLflow si se especifica URI
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(model_cfg['name'])

    # 3. Cargar dataset procesado
    data = pd.read_csv(args.data)
    target = model_cfg['target_variable']

    # Dividir en features (X) y variable objetivo (y)
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Crear instancia del mejor modelo definido en el YAML
    model = get_model_instance(model_cfg['best_model'], model_cfg['parameters'])

    # 5. Iniciar un run en MLflow
    with mlflow.start_run(run_name="final_training"):
        logger.info(f"Training model: {model_cfg['best_model']}")
        model.fit(X_train, y_train)  # Entrenar modelo
        y_pred = model.predict(X_test)  # Hacer predicciones

        # Calcular métricas
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        # Registrar parámetros y métricas en MLflow
        mlflow.log_params(model_cfg['parameters'])
        mlflow.log_metrics({'mae': mae, 'r2': r2})

        # Guardar modelo en MLflow
        mlflow.sklearn.log_model(model, "tuned_model")
        model_name = model_cfg['name']
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/tuned_model"

        # 6. Registrar modelo en el Model Registry de MLflow
        logger.info("Registering model to MLflow Model Registry...")
        client = MlflowClient()
        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.RestException:
            pass  # Si ya existe, no hace nada

        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=mlflow.active_run().info.run_id
        )

        # Pasar el modelo a "Staging"
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        # 7. Agregar descripción al modelo en el registry
        description = (
            f"Model for predicting house prices.\n"
            f"Algorithm: {model_cfg['best_model']}\n"
            f"Hyperparameters: {model_cfg['parameters']}\n"
            f"Features used: All features in the dataset except the target variable\n"
            f"Target variable: {target}\n"
            f"Trained on dataset: {args.data}\n"
            f"Model saved at: {args.models_dir}/trained/{model_name}.pkl\n"
            f"Performance metrics:\n"
            f"  - MAE: {mae:.2f}\n"
            f"  - R²: {r2:.4f}"
        )
        client.update_registered_model(name=model_name, description=description)

        # 8. Agregar tags con metadatos útiles
        client.set_registered_model_tag(model_name, "algorithm", model_cfg['best_model'])
        client.set_registered_model_tag(model_name, "hyperparameters", str(model_cfg['parameters']))
        client.set_registered_model_tag(model_name, "features", "All features except target variable")
        client.set_registered_model_tag(model_name, "target_variable", target)
        client.set_registered_model_tag(model_name, "training_dataset", args.data)
        client.set_registered_model_tag(model_name, "model_path", f"{args.models_dir}/trained/{model_name}.pkl")

        # 9. Agregar tags con dependencias (para reproducibilidad)
        deps = {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "xgboost_version": xgb.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        }
        for k, v in deps.items():
            client.set_registered_model_tag(model_name, k, v)

        # 10. Guardar modelo entrenado en disco local
        save_path = f"{args.models_dir}/trained/{model_name}.pkl"
        joblib.dump(model, save_path)
        logger.info(f"Saved trained model to: {save_path}")
        logger.info(f"Final MAE: {mae:.2f}, R²: {r2:.4f}")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    # Si el script se ejecuta directamente desde la terminal
    args = parse_args()  # Lee los argumentos pasados
    main(args)           # Llama a la función principal con esos argumentos
