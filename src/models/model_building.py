import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import yaml
import logging
import os
import mlflow
import mlflow.sklearn

from xgboost import XGBClassifier

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            logger.debug('Parameters retrieved from %s', params_path)
            return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> GradientBoostingClassifier:
    """Train the Gradient Boosting model."""
    try:

        model_type = params.get('model_type', 'Gradient_boosting')

        if model_type == 'xgboost':
            clf = XGBClassifier(
                n_estimators=params['n_estimators'], 
                learning_rate=params['learning_rate'], 
                max_depth=params['max_depth'],
                use_label_encoder=False,
                eval_metric='logloss'
            )
            logger.debug('XGBoost model training completed')
        else:   

            clf = GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])
        
            logger.debug('Model training completed')
        clf.fit(X_train, y_train)
        logger.debug('Model fitting completed')
        return clf
    
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:

        all_params = load_params('params.yaml')
        model_params = all_params['model_building']
        feature_params = all_params['feature_engineering']

        # Mlflow tracking
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
        mlflow.set_experiment("emotion-detection")

        # Load the processed data
        train_data = load_data('./data/processed/train_bow.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        # start mlflow run
        with mlflow.start_run(run_name="Xgboost-100-d3"):

            # log data info
            import hashlib

            data_hash = hashlib.md5(pd.util.hash_pandas_object(train_data).values.tobytes()).hexdigest()

            mlflow.log_param("train_data_path", './data/processed/train_bow.csv')
            mlflow.log_param("train_data_hash", data_hash)
            class_counts = train_data.iloc[:, -1].value_counts().to_dict()
            mlflow.log_param("train_data_class_distribution", class_counts)

            logger.info("MLflow data parameters logged successfully")

            # log parameters
            mlflow.log_param("model_type", model_params.get('model_type', 'Gradient_boosting'))
            mlflow.log_param("n_estimators", model_params['n_estimators'])
            mlflow.log_param("learning_rate", model_params['learning_rate'])
            mlflow.log_param("feature_method", feature_params.get('method', 'bow'))
            mlflow.log_param("max_depth", model_params.get('max_depth', 'N/A'))
            mlflow.log_param("max_features", feature_params.get('max_features', 500))
            mlflow.log_param("train_shape", int(X_train.shape[0]))
            mlflow.log_param("train_features", int(X_train.shape[1]))
            
            logger.info(" MLflow parameters logged successfully")

            # log tags
            mlflow.set_tag("engineer", "Pavan Modi")
            mlflow.set_tag("model_version", "1.0")
            mlflow.set_tag("pipeline", "dvc_pipeline")
            mlflow.set_tag("stage", "model_building")

            logger.info("MLflow tags logged successfully")

            # Train Model
            clf = train_model(X_train, y_train, model_params)

            # save model

            # Save model pickle FIRST (for DVC pipeline)
            save_model(clf, 'model/model.pkl')
            logger.info("Model saved to model/model.pkl")

            # Log the model to MLflow  s3 (for MLflow tracking and registry)
            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model",
                registered_model_name="emotion_detection_model"
            )

            logger.info("Model logged to MLflow successfully")

            #log training metrics

            train_accuracy = clf.score(X_train, y_train)
            mlflow.log_metric("training_accuracy", train_accuracy)

            logger.info("MLflow training metrics logged successfully")

            

            # log run ID
            run = mlflow.active_run()
            run_id = run.info.run_id
            mlflow.set_tag("run_id", run_id)

            logger.info(f"MLflow run completed successfully with run ID: {run_id}")

            # save run_id to a file so model_evaluation can find this and read it.
            os.makedirs('reports', exist_ok=True)
            with open('reports/run_info.json', 'w') as f:
                import json
                json.dump({"run_id": run_id},f)

            logger.info("MLflow run ID saved to reports/run_info.json successfully")

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()