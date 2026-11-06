import cv2
import uuid
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Project Root 
PROJECT_ROOT = Path(__file__).resolve().parent

# Result directory 
RESULT_DIR = PROJECT_ROOT / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Model Path 
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

# Load model only once
model = YOLO(str(MODEL_PATH))

def get_model_info():
    return {
        "model_path": str(MODEL_PATH),
        "classes": model.names
    }

# ==========================================
# Post-processing: remove tiny/noise boxes
# ==========================================
def _filter_tiny_boxes(boxes_data, image_w, image_h, min_area_ratio=0.0003):
    """
    Remove bounding boxes that are too small relative to the image.
    min_area_ratio=0.0003 means box must cover at least 0.03% of image area.
    This eliminates single-pixel noise detections.
    """
    image_area = image_w * image_h
    filtered = []
    for item in boxes_data:
        x1, y1, x2, y2 = item["bbox"]
        box_area = (x2 - x1) * (y2 - y1)
        if box_area >= image_area * min_area_ratio:
            filtered.append(item)
    return filtered

# ==========================================
# Prediction Function
# ==========================================
def predict_image(image_path: str, conf: float = 0.25, iou: float = 0.30):
    """
    Run YOLOv8 inference with configurable confidence and NMS IoU thresholds.
    
    Default conf=0.25 (25%) provides a good balance for real-world PCB photos:
    - Filters out low-confidence false positives (normal through-hole pads ~5-15%)
    - Keeps genuine high-confidence defects (usually >25%)
    
    iou=0.30 aggressively removes overlapping duplicate boxes for the same defect.
    """
    results = model.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        save=False,
        verbose=False
    )
    result = results[0]

    # Read image dimensions for tiny-box filtering
    img = cv2.imread(image_path)
    if img is not None:
        image_h, image_w = img.shape[:2]
    else:
        image_h, image_w = 640, 640

    # Collect raw detections first
    raw_detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence_val = round(float(box.conf[0]), 3)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        raw_detections.append({
            "type": model.names[class_id],
            "confidence": confidence_val,
            "bbox": [x1, y1, x2, y2]
        })

    # Remove boxes that are unrealistically tiny (noise)
    detections = _filter_tiny_boxes(raw_detections, image_w, image_h)

    # Re-draw only the filtered detections on a clean copy of the image
    if img is not None and len(detections) != len(raw_detections):
        # Redraw manually so removed boxes don't appear in the result image
        annotated_image = img.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['type']} {det['confidence']:.2f}"
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(annotated_image, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 2)
    else:
        # All boxes passed filter — use YOLO's native annotated image (better visuals)
        annotated_image = result.plot()

    # Create unique output filename to avoid browser caching issues
    unique_suffix = uuid.uuid4().hex[:8]
    output_filename = f"{Path(image_path).stem}_{unique_suffix}_result.jpg"
    output_path = RESULT_DIR / output_filename

    # Save image
    cv2.imwrite(str(output_path), annotated_image)

    return {
        "detections": detections,
        "result_image": str(output_path)
    }
