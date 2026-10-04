"""MalariaScan AI backend.

Run from the project root (the MalariaScanAI folder):
    uvicorn backend.main:app --reload
Then open http://127.0.0.1:8000/app/
"""
import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from backend.inference import predict_image

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"

app = FastAPI(title="MalariaScan AI")
# Lets the page work even if you open frontend/index.html directly.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])
app.mount("/app", StaticFiles(directory=FRONTEND, html=True), name="app")


@app.get("/")
def info():
    return {"name": "MalariaScan AI",
            "description": "Malaria screening from blood cell images (prototype, not a diagnosis)",
            "endpoints": {"GET /health": "health check",
                          "POST /predict": "upload an image, get the predicted class",
                          "GET /app/": "web interface"}}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):  # missing file -> automatic 422 error
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image.load()
    except Exception:
        raise HTTPException(400, "That file is not a valid image. Upload a JPG or PNG.")
    try:
        return predict_image(image)
    except FileNotFoundError:
        raise HTTPException(503, "Model files not found. Run notebook 05 to create "
                                 "models/best_model.pth and models/class_to_idx.json.")
    except Exception as e:
        raise HTTPException(500, f"The model could not make a prediction: {e}")
