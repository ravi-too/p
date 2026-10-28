"""FastAPI inference service. Set MODEL_CHECKPOINT to a trained checkpoint."""
import base64
import io
import os
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from src.model import load_model
from src.predict import predict_image, annotate

app = FastAPI(title="Pedestrian Vision API", version="1.0.0")
MODEL = None
DEVICE = None


@app.on_event("startup")
def startup():
    global MODEL, DEVICE
    checkpoint = os.getenv("MODEL_CHECKPOINT", "checkpoints/pedestrian_maskrcnn.pth")
    # If no checkpoint exists, the app intentionally fails with a clear message rather
    # than silently presenting an untrained/random model as a trained solution.
    MODEL, DEVICE = load_model(checkpoint, pretrained_if_missing=False)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/predict")
async def predict(file: UploadFile = File(...), score_threshold: float = Query(0.5, ge=0, le=1)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file.")
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read uploaded image.") from exc
    result = predict_image(MODEL, DEVICE, image, score_threshold)
    buffer = io.BytesIO()
    annotate(image, result).save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return JSONResponse({"detections": [
        {"box_xyxy": box, "score": score, "label": "pedestrian"}
        for box, score in zip(result["boxes"], result["scores"])
    ], "annotated_image_base64": encoded, "device": str(DEVICE)})
