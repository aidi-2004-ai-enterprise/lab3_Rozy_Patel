"""
main.py

This FastAPI application loads a trained XGBoost model and exposes a /predict endpoint
to classify penguin species based on features. It validates input using Pydantic models,
applies one-hot encoding to categorical fields, and ensures alignment with the training pipeline.


"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import pandas as pd
import joblib
import os
import logging
from typing import Dict

# FastAPI Initialization 
app = FastAPI(title="Penguin Species Predictor API")

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model Loading
MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "model.json")
model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]
label_encoder = model_bundle["label_encoder"]
columns = model_bundle["columns"]

# Inverse transform requires the full label encoder
label_classes = label_encoder.classes_

logger.info(" Model and metadata loaded successfully.")

# Enums for Input Validation
class Island(str, Enum):
    """
    Enumeration for valid island values.
    """
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"

class Sex(str, Enum):
    """
    Enumeration for valid sex values.
    """
    Male = "male"
    Female = "female"

# Input Schema 
class PenguinFeatures(BaseModel):
    """
    Schema for penguin features input using Pydantic.

    """
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    year: int
    sex: Sex
    island: Island

# Feature Preprocessing
def preprocess_features(features: PenguinFeatures, expected_columns: list):
    """ 
    Preprocesses input features to match training format using one-hot encoding.
    Input data from the POST request.
    Processed feature vector aligned to training columns.

    """
    input_dict = features.model_dump()  # This returns a dictionary of the model's fields
    X_input = pd.DataFrame([input_dict])
    X_input = pd.get_dummies(X_input, columns=["sex", "island"]) # Ensure the same 
    expected_cols = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "sex_Female",
        "sex_Male",
        "island_Biscoe",
        "island_Dream",
        "island_Torgersen",
    ]
    X_input = X_input.reindex(columns=expected_cols, fill_value=0)
    X_input = X_input.astype(float)
    return X_input


# API Routes

@app.get("/")
async def root():
    """
    Root endpoint to confirm the API is running.

    """
    return {"message": "Hello, Hope you are doing well"}

@app.get("/health")
async def health():
    """
    Health check endpoint.

    """
    return {"status": "ok"}


@app.post("/predict")
async def predict(features: PenguinFeatures):
     logging.info("Received prediction request")
     """
     Predicts the penguin species from input features.

     """    
     try:
        X_input = preprocess_features(features,columns)
        pred = model.predict(X_input.values)
        predicted_label = label_classes[int(pred)]
        logging.info(f"Predicted: {predicted_label}")
        return {"prediction": int(pred[0])}

     except Exception as e:
         logging.error(f"Prediction failed: {e}")
         raise HTTPException(status_code=400, detail="Prediction failed due to internal error.")

