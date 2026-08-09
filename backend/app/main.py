from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import PredictResponse, HealthResponse
from app import inference

app = FastAPI(title="Aircraft Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    try:
        inference.load_model()
    except FileNotFoundError:
        print("WARNING: model weights not found. Run train.py first.")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=inference.is_loaded())


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not inference.is_loaded():
        raise HTTPException(503, "Model not loaded. Run train.py first.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")

    image_bytes = await file.read()
    try:
        top3 = inference.predict(image_bytes, top_k=3)
    except Exception as e:
        raise HTTPException(400, f"Could not process image: {e}")

    return PredictResponse(
        aircraft=top3[0]["aircraft"],
        confidence=top3[0]["confidence"],
        predictions=top3,
    )