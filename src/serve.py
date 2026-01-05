from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import time
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter('api_requests_total', 'Total API requests')
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'API request latency')
PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions', ['prediction'])

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# Load model and scaler
try:
    with open(MODELS_DIR / "best_model.txt", 'r') as f:
        best_model_name = f.read().strip()
    
    model = joblib.load(MODELS_DIR / f"{best_model_name}.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    logger.info(f"✓ Loaded model: {best_model_name}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model, scaler = None, None

class HeartDiseaseInput(BaseModel):
    age: float = Field(..., ge=0, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: float = Field(..., ge=0, le=300)
    chol: float = Field(..., ge=0, le=600)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=0, le=250)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=4)
    thal: int = Field(..., ge=0, le=3)

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    risk_level: str
    model_used: str

@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API", "status": "running"}

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: HeartDiseaseInput):
    start_time = time.time()
    REQUEST_COUNT.inc()
    
    try:
        features = np.array([[
            input_data.age, input_data.sex, input_data.cp,
            input_data.trestbps, input_data.chol, input_data.fbs,
            input_data.restecg, input_data.thalach, input_data.exang,
            input_data.oldpeak, input_data.slope, input_data.ca,
            input_data.thal
        ]])
        
        features_scaled = scaler.transform(features)
        
        prediction = int(model.predict(features_scaled)[0])
        confidence = float(model.predict_proba(features_scaled)[0][prediction])
        
        risk_level = "High" if prediction == 1 else "Low"
        
        PREDICTION_COUNTER.labels(prediction=prediction).inc()
        REQUEST_LATENCY.observe(time.time() - start_time)
        
        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.2f}")
        
        with open(MODELS_DIR / "best_model.txt", 'r') as f:
            model_name = f.read().strip()
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "risk_level": risk_level,
            "model_used": model_name
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
