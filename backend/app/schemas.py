from pydantic import BaseModel
from typing import List

class Prediction(BaseModel):
    aircraft: str
    confidence: float

class PredictResponse(BaseModel):
    aircraft: str         
    confidence: float     
    predictions: List[Prediction]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool