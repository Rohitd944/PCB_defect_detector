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
from services.compare import compare_pcb_images

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


app.mount(
    "/results",
    StaticFiles(directory=RESULTS_DIR),
    name="results"
)


UPLOAD_DIR =  UPLOADS_DIR
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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
    confidence: float = Form(0.06),
    iou: float = Form(0.45)
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


import cv2

@app.post(
    "/compare",
    tags=["Comparison"],
    summary="Reference-based PCB Defect Detection",
    description="""
Upload a known-good **reference** PCB image and a **test** PCB image.
The backend aligns the two images and highlights every pixel-level difference as a defect.
This approach is far more accurate than blind YOLO detection for real-world PCBs.

Returns:
- Detected defect regions with type, confidence, bounding box
- URL of annotated result image (defects highlighted on test PCB)
- URL of diff heat-map image (shows where differences were found)
"""
)
async def compare(
    request: Request,
    reference: UploadFile = File(..., description="Known-good reference PCB image"),
    test: UploadFile = File(..., description="PCB image to inspect for defects"),
    diff_threshold: int = Form(30, description="Pixel brightness difference threshold (0-255)"),
):
    allowed = {".jpg", ".jpeg", ".png"}

    ref_ext = os.path.splitext(reference.filename)[1].lower()
    tst_ext = os.path.splitext(test.filename)[1].lower()

    if ref_ext not in allowed or tst_ext not in allowed:
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, PNG images are allowed.")

    uid = uuid.uuid4().hex[:8]

    ref_path = UPLOAD_DIR / f"{uid}_ref{ref_ext}"
    tst_path = UPLOAD_DIR / f"{uid}_test{tst_ext}"

    with open(ref_path, "wb") as f:
        shutil.copyfileobj(reference.file, f)
    with open(tst_path, "wb") as f:
        shutil.copyfileobj(test.file, f)

    try:
        cmp = compare_pcb_images(
            reference_path=str(ref_path),
            test_path=str(tst_path),
            diff_threshold=max(10, min(100, diff_threshold)),
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

    # Save annotated image
    annotated_name = f"{uid}_annotated_result.jpg"
    annotated_path = RESULTS_DIR / annotated_name
    cv2.imwrite(str(annotated_path), cmp["annotated_image"])

    # Save diff heat-map
    diff_name = f"{uid}_diff_heatmap.jpg"
    diff_path = RESULTS_DIR / diff_name
    cv2.imwrite(str(diff_path), cmp["diff_image"])

    detections = cmp["detections"]

    analysis = generate_analysis(detections, test.filename)

    return {
        "status":             "success",
        "mode":               "reference_comparison",
        "filename":           test.filename,
        "reference_filename": reference.filename,
        "aligned":            cmp["aligned"],
        "total_detections":   len(detections),
        "detections":         detections,
        "result_image_url":   f"{request.base_url}results/{annotated_name}",
        "diff_image_url":     f"{request.base_url}results/{diff_name}",
        "analysis":           analysis,
        "diff_threshold":     diff_threshold,
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