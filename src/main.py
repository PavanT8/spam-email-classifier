import os
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np

# Global model dictionary
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    classifier_path = os.path.join(models_dir, "classifier.pkl")
    vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
    
    if os.path.exists(classifier_path) and os.path.exists(vectorizer_path):
        ml_models["classifier"] = joblib.load(classifier_path)
        ml_models["vectorizer"] = joblib.load(vectorizer_path)
        print("Models loaded successfully.")
    else:
        print("Warning: Model files not found in", models_dir)
    
    yield
    
    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(title="Spam Email Classifier API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: str
    confidence: float

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    classifier = ml_models.get("classifier")
    vectorizer = ml_models.get("vectorizer")
    
    if not classifier or not vectorizer:
        raise HTTPException(status_code=503, detail="Models are not loaded")
    
    X = vectorizer.transform([req.text])
    pred = classifier.predict(X)[0]
    
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(X)[0]
        confidence = float(max(probs))
    else:
        confidence = 1.0
        
    # Standardize string output
    pred_str = str(pred).lower()
    if pred_str == "1":
        pred_str = "spam"
    elif pred_str == "0":
        pred_str = "ham"
        
    return PredictResponse(prediction=pred_str, confidence=confidence)

static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
os.makedirs(static_dir, exist_ok=True)

# We can mount static dir, but we also want to serve the frontend from `/`
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Welcome to the Spam Email Classifier API. Place frontend in static/index.html"}
