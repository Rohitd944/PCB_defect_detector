from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from pathlib import Path
from typing import Optional
from services.analysis import generate_analysis
from services.reference_store import ReferenceStore
from services.hybrid_inspection import run_hybrid_inspection
import cv2
import numpy as np
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

REFERENCES_DIR = BASE_DIR / "references"

# StaticFiles raises at import time if the directory is missing, so create
# both directories *before* mounting (fresh clones had no results/ folder).
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
REFERENCES_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/results",
    StaticFiles(directory=RESULTS_DIR),
    name="results"
)

app.mount(
    "/references",
    StaticFiles(directory=REFERENCES_DIR),
    name="references"
)


UPLOAD_DIR =  UPLOADS_DIR

reference_store = ReferenceStore(REFERENCES_DIR)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_IMAGE_BYTES = 25 * 1024 * 1024


def _validate_image_upload(file: UploadFile):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")
    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are allowed."
        )


async def _decode_upload(file: UploadFile):
    """Read an UploadFile into a BGR numpy array without writing a temp file."""
    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 25 MB.")
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode the image.")
    return image


def _reference_public_url(request: Request, reference_id: str) -> str:
    return f"{request.base_url}references/{reference_id}.jpg"


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


# ==========================================================
# Reference-based hybrid inspection
# ==========================================================

@app.post(
    "/reference",
    tags=["Reference"],
    summary="Save a known-good reference PCB",
    description="""
Upload the known-good PCB image. The image is stored on the server and its
ORB features are cached so it can be reused by `/inspect/hybrid`.
"""
)
async def create_reference(
    request: Request,
    file: UploadFile = File(...),
    name: Optional[str] = Form(None)
):
    _validate_image_upload(file)
    image = await _decode_upload(file)

    try:
        meta = reference_store.save(image, file.filename, name=name)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Could not save the reference image: {str(e)}"
        )

    meta["image_url"] = _reference_public_url(request, meta["reference_id"])
    return {"status": "success", **meta}


@app.get(
    "/reference",
    tags=["Reference"],
    summary="List saved reference PCBs"
)
def list_references(request: Request, limit: int = 20):
    references = reference_store.list(limit=max(1, min(100, limit)))
    for meta in references:
        meta["image_url"] = _reference_public_url(request, meta["reference_id"])
    return {"status": "success", "references": references}


@app.get(
    "/reference/latest",
    tags=["Reference"],
    summary="Get the most recently saved reference PCB"
)
def latest_reference(request: Request):
    entry = reference_store.latest()
    if entry is None:
        raise HTTPException(status_code=404, detail="No reference image has been saved yet.")
    meta = dict(entry["meta"])
    meta["image_url"] = _reference_public_url(request, meta["reference_id"])
    return {"status": "success", **meta}


@app.get(
    "/reference/{reference_id}",
    tags=["Reference"],
    summary="Get a saved reference PCB"
)
def get_reference(reference_id: str, request: Request):
    entry = reference_store.get(reference_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Reference image not found.")
    meta = dict(entry["meta"])
    meta["image_url"] = _reference_public_url(request, meta["reference_id"])
    return {"status": "success", **meta}


@app.post(
    "/inspect/hybrid",
    tags=["Prediction"],
    summary="Hybrid PCB inspection (YOLO + reference comparison)",
    description="""
Inspect a testing PCB against a known-good reference image.

Combines:
- YOLOv8 defect detection (same model as `/predict`)
- ORB + RANSAC alignment
- exposure normalisation + SSIM + structural difference
- a verification engine that confirms/rejects YOLO detections

Returns a PASS / FAIL / REVIEW verdict, annotated testing image, side-by-side
comparison and a difference heatmap.
"""
)
async def inspect_hybrid(
    request: Request,
    file: UploadFile = File(...),
    reference: Optional[UploadFile] = File(None),
    reference_id: Optional[str] = Form(None),
    confidence: float = Form(0.25),
    iou: float = Form(0.40),
    diff_threshold: float = Form(18.0)
):
    _validate_image_upload(file)

    # Resolve the reference: explicit upload > stored id > most recent.
    reference_entry = None
    if reference is not None and reference.filename:
        _validate_image_upload(reference)
        reference_image = await _decode_upload(reference)
        meta = reference_store.save(reference_image, reference.filename)
        reference_entry = reference_store.get(meta["reference_id"])
    elif reference_id:
        reference_entry = reference_store.get(reference_id)
        if reference_entry is None:
            raise HTTPException(status_code=404, detail="Reference image not found.")
    else:
        reference_entry = reference_store.latest()

    if reference_entry is None:
        raise HTTPException(
            status_code=400,
            detail="A reference PCB image is required. Upload one to /reference first."
        )

    unique_name = f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
    file_path = UPLOAD_DIR / unique_name
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    conf_val = max(0.05, min(0.95, float(confidence)))
    iou_val = max(0.10, min(0.90, float(iou)))
    diff_val = max(6.0, min(60.0, float(diff_threshold)))

    try:
        result = run_hybrid_inspection(
            testing_path=str(file_path),
            reference_entry=reference_entry,
            confidence=conf_val,
            iou=iou_val,
            diff_threshold=diff_val,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Hybrid inspection failed: {str(e)}"
        )

    base = str(request.base_url).rstrip("/")
    result["images"] = {
        f"{key}_url": f"{base}/results/{Path(path).name}"
        for key, path in result.get("images", {}).items()
        if path
    }
    result["reference"]["image_url"] = _reference_public_url(
        request, result["reference"]["reference_id"]
    )
    return result


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