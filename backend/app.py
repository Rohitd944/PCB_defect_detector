from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from pathlib import Path
from services.analysis import generate_analysis
import shutil
import os
import traceback
import uuid

from predictor import predict_image, get_model_info

app = FastAPI(
    title="AI Powered PCB Defect Detection using YOLOv8",
    description="Backend API for PCB Defect Detection",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://ai-powered-pcb-defects-detection.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results"

UPLOADS_DIR = BASE_DIR / "uploads"

# StaticFiles raises at import time if the directory is missing, so create
# both directories *before* mounting (fresh clones had no results/ folder).
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/results",
    StaticFiles(directory=RESULTS_DIR),
    name="results"
)


UPLOAD_DIR =  UPLOADS_DIR


@app.post(
    "/predict",
    tags=["Prediction"],
    summary="Predict PCB Defects",
    description="""
Upload a PCB image and detect PCB manufacturing defects using the trained YOLOv8 model.

Returns:
- Detected defect names
- Confidence scores
- Bounding box coordinates
- URL of the annotated result image
"""
)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    # FIXED: previous default 0.06 flooded every photo with junk boxes
    # (pcb1.jpg: 80 detections at 0.06 vs 9 at 0.25). 0.25 is measured
    # as the balanced operating point; see predictor.py.
    confidence: float = Form(0.25),
    iou: float = Form(0.40)
):

    # Check if a file was selected
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )

    # Check file extension
    allowed_extensions = {".jpg", ".jpeg", ".png"}

    extension = os.path.splitext(file.filename)[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are allowed."
        )

    unique_name = f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
    file_path = UPLOAD_DIR / unique_name

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    conf_val = max(0.05, min(0.95, float(confidence)))
    iou_val = max(0.10, min(0.90, float(iou)))

    try:
        result = predict_image(str(file_path), conf=conf_val, iou=iou_val)
        analysis = generate_analysis(
            result["detections"],
            file.filename
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    result_filename = Path(result["result_image"]).name

    result_image_url = f"{request.base_url}results/{result_filename}"

    return {
        "status": "success",
        "filename": file.filename,
        "total_detections": len(result["detections"]),
        "detections": result["detections"],
        "result_image_url": result_image_url,
        "analysis": analysis,
        "confidence_threshold": conf_val
    }


@app.get(
    "/",
    tags=["Home"],
    summary="Home",
    description="Welcome endpoint for the PCB Defect Detection API."
)
def home():
    return {
        "message": "PCB Defect Detection API Running"
    }


@app.get(
    "/model-info",
    tags=["Model"],
    summary="Model Information",
    description="Returns YOLO model information including available defect classes."
)
def model_info():
    return get_model_info()

@app.get(
    "/health",
    tags=["Health"],
    summary="Health Check",
    description="Check whether the backend API is running correctly."
)
def health_check():

    model_info = get_model_info()

    return {
        "status": "healthy",
        "model_loaded": True,
        "version": app.version,
        "classes": len(model_info["classes"])
    }