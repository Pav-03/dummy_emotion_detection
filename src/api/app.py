from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import numpy as np
import pandas as pd
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger
from src.api.middleware.cors import setup_cors
from src.api.middleware.logging_middleware import log_request
from src.api.middleware.auth import auth_guard, authenticate_user

logger = get_logger("api")

# create FastAPI instance
app = FastAPI(
    title = "Emotion Detection Api",
    description = "API for prediction emotion form given text",
    version = "1.0.0"
)

# Register middleware
setup_cors(app)
app.middleware("http")(log_request)
app.middleware("http")(auth_guard)

# pydantic model for request body

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    emotion: str
    confidence: float
    model_version: str = "v6.0.0"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vectorizer_loaded: bool

class ModelInfoResponse(BaseModel):
    model_type: str
    model_version: str = "v6.0.0"

class BatchPredictRequest(BaseModel):
    texts: list[str]

class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    total: int

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"



@app.post("/auth/login", response_model=LoginResponse)
def login(request: LoginRequest):
    """ login and get JWT token for authentication in other endpoints """
    token = authenticate_user(request.username, request.password)
    return LoginResponse(
        access_token=token,
        token_type="bearer"
    )

@app.get("/")
def root():
    """ root endpoint to check if the application is running and get basic info about the API """
    return {
        "service": "Emotion Detection API",
        "status": "healthy",
        "version": "1.0.0",
        "docs": "/docs"
    }

# load the model and vectorizer

MODEL  = None
VECTORIZER = None

@app.on_event("startup")
def load_model_vectorizer():
    global MODEL, VECTORIZER

    model_path = os.getenv("MODEL_PATH",'model/model.joblib')
    vectorizer_path = os.getenv("VECTORIZER_PATH",'model/vectorizer.joblib')

    try:
        MODEL = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}")
        MODEL = None
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        MODEL = None
    
    try:
        VECTORIZER = joblib.load(vectorizer_path)
        logger.info(f"Vectorizer loaded successfully from {vectorizer_path}")
    except FileNotFoundError:
        logger.error(f"Vectorizer file not found at {vectorizer_path}")
        VECTORIZER = None
    except Exception as e:
        logger.error(f"Error loading vectorizer from {vectorizer_path}: {e}")
        VECTORIZER = None
    
# Preprocesing function for input text

def preprocess_text(text: str) -> str:
    """ clean the input text exactly like training data preprocessing """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

# Api end point for prediction
@app.get("/health", response_model=HealthResponse)
def health_check():
    """ health checks- kubernetes and load balancer can use this endpoint to check the health of the application """

    status = "healthy" if MODEL is not None and VECTORIZER is not None else "unhealthy"
    model_loaded = MODEL is not None
    vectorizer_loaded = VECTORIZER is not None

    logger.debug(f"Health check performed. Status: {status}, Model loaded: {model_loaded}, Vectorizer loaded: {vectorizer_loaded}")
    return HealthResponse(
        status=status, 
        model_loaded=model_loaded, 
        vectorizer_loaded=vectorizer_loaded)

@app.get("/model-info")

def model_info():
    """ endpoint to get model information like model type and version """

    logger.debug("Model info endpoint called")
    return {
        "model_name": "emotion-detection-model",
        "model_type": type(MODEL).__name__ if MODEL else "Model not loaded",
        "features": "Bag of Words (CountVectorizer, 500 features)",
        "version": os.getenv("MODEL_VERSION", "v6.0.0"),
    }

    
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """ predict the emotion from the input text and return the emotion and confidence score """
    start_time = time.time()

    # check model if loaded

    if MODEL is None or VECTORIZER is None:
        logger.error("Model or vectorizer not loaded. Cannot perform prediction.")
        raise HTTPException(
            status_code=503, 
            detail="Model or vectorizer not loaded. Please try again later.")
    
    # check text is not empty
    if not request.text.strip():
        logger.error("Input text is empty. Cannot perform prediction.")
        raise HTTPException(
            status_code=400, 
            detail="Input text cannot be empty.")
    
    try:
        #preprocess the imput text
        processed_text = preprocess_text(request.text)
        logger.debug(f"Input text preprocessed: {processed_text}")

        # vectorise the input text
        text_vector = VECTORIZER.transform([processed_text])
        logger.debug(f"Input text vectorized: {text_vector.shape}")

        #predict the emotion
        prediction = MODEL.predict(text_vector)[0]  
        confidence = np.max(MODEL.predict_proba(text_vector))
        logger.debug(f"Prediction made: {prediction} with confidence {confidence}")

        # Formate the response
        emotion = "positive" if prediction == 1 else "negative"
        confidence = round(confidence,4)

        # log prediction with latency
        latency = time.time() - start_time
        logger.info(f"Prediction completed in {latency:.2f} seconds.")

        return PredictResponse(
            emotion=emotion, 
            confidence=confidence,
            model_version=os.getenv("MODEL_VERSION", "v6.0.0"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=500, 
            detail="An error occurred during prediction. Please try again later.")

@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    """Predict emotions for multiple texts."""
    start_time = time.time()

    if MODEL is None or VECTORIZER is None:
        logger.error("Batch prediction attempted but model/vectorizer not loaded")
        raise HTTPException(status_code=503, detail="Model or vectorizer not loaded.")

    if not request.texts:
        logger.warning("Empty batch received")
        raise HTTPException(status_code=400, detail="Texts list cannot be empty.")

    try:
        results = []
        for text in request.texts:
            processed_text = preprocess_text(text)
            features = VECTORIZER.transform([ processed_text])
            prediction = MODEL.predict(features)[0]
            probabilities = MODEL.predict_proba(features)[0]

            emotion = "positive" if prediction == 1 else "negative"
            confidence = float(np.max(probabilities))

            results.append(PredictResponse(
                emotion=emotion,
                confidence=round(confidence, 4),
                model_version=os.getenv("MODEL_VERSION", "v6")
            ))

        latency = time.time() - start_time
        logger.info(f"Batch prediction: {len(request.texts)} texts | Latency: {latency:.3f}s")

        return BatchPredictResponse(
            predictions=results,
            total=len(results)
        )

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


    
        