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
_model = None

# Models trained from different dataset configs expose the same six classes
# under different spellings ("Open Circuit" vs "open_circuit"). Normalise to
# one display name so per-class thresholds, colours and the UI stay stable.
DISPLAY_NAMES = {
    "open_circuit": "Open Circuit",
    "short_circuit": "Short Circuit",
    "mouse_bite": "Mouse Bite",
    "spur": "Spur",
    "spurious_copper": "Spurious Copper",
    "missing_hole": "Missing Hole",
}


def _display_name(name):
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    return DISPLAY_NAMES.get(key, str(name))


def _get_model():
    global _model
    if _model is None:
        _model = YOLO(str(MODEL_PATH))
    return _model


def get_model_info():
    model = _get_model()
    return {
        "model_path": str(MODEL_PATH),
        "classes": {i: _display_name(n) for i, n in model.names.items()}
    }


# ==========================================
# Post-processing: remove tiny/noise boxes
# ==========================================
# Per-class thresholds: Mouse Bite / Spur / Short have the highest
# background FP rates (confusion matrix: 1247 / 869 / 691 background FPs).
PER_CLASS_CONF = {
    "Open Circuit": 0.30,
    "Short Circuit": 0.35,
    "Mouse Bite": 0.40,
    "Spur": 0.40,
    "Spurious Copper": 0.35,
    "Missing Hole": 0.30,
}


def _filter_tiny_boxes(boxes_data, image_w, image_h, min_area_ratio=0.00005, max_area_ratio=0.015):
    """
    Remove bounding boxes that are too small (noise) or too huge (wrong
    location). GT (dataset_merged, 2953 boxes): median area 0.04-0.10%,
    p90 < 0.21%. Boxes > 1.5% are ~7x p90 (e.g. pcb2 Short 1.73%) — wrong
    location, reject. Boxes < 0.005% are pure noise.
    """
    image_area = image_w * image_h
    filtered = []
    for item in boxes_data:
        x1, y1, x2, y2 = item["bbox"]
        bw, bh = x2 - x1, y2 - y1
        box_area = bw * bh
        if box_area < image_area * min_area_ratio or box_area > image_area * max_area_ratio:
            continue
        if bw < 4 or bh < 4:
            continue
        # Extreme sliver (aspect > 6) — a trace edge, not a defect.
        if max(bw, bh) / max(min(bw, bh), 1) > 6:
            continue
        # Per-class confidence floor (confusion-matrix driven).
        if item["confidence"] < PER_CLASS_CONF.get(item["type"], 0.25):
            continue
        filtered.append(item)
    return filtered


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms_merge(detections, iou_thresh=0.45, per_class_cap=5, total_cap=12):
    """
    Class-aware NMS + density caps.
    - Same-class duplicates (IoU > 0.5): suppress lower-confidence (one
      defect = one box).
    - Cross-class overlaps (IoU > 0.7): suppress lower-confidence (same
      region called both Open and Mouse Bite — keep winner).
    - Caps: max 5 per class, max 12 total per board. GT: mean 4.3 boxes /
      image, max 6. Caps kill systematic floods (pcb13: 47 Missing Hole
      -> 5, samples: 137 Short -> 5) which are FPs by definition.
    """
    if not detections:
        return detections
    ordered = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    counts = {}
    for cand in ordered:
        duplicate = False
        for k in kept:
            ov = _iou(cand["bbox"], k["bbox"])
            thresh = 0.5 if cand["type"] == k["type"] else 0.7
            if ov > thresh:
                duplicate = True
                break
        if duplicate:
            continue
        c = cand["type"]
        if counts.get(c, 0) >= per_class_cap:
            continue
        if len(kept) >= total_cap:
            break
        counts[c] = counts.get(c, 0) + 1
        kept.append(cand)
    kept.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
    return kept


# ==========================================
# Tiled (sliced) inference — supplementary pass
# ==========================================
def _predict_whole(img_bgr, conf, iou, imgsz=640):
    """Single full-image pass (the model's native, calibrated scale)."""
    model = _get_model()
    results = model.predict(source=img_bgr, conf=conf, iou=iou, imgsz=imgsz, save=False, verbose=False)
    raw = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        raw.append({
            "type": _display_name(model.names[class_id]),
            "confidence": round(float(box.conf[0]), 3),
            "bbox": list(map(int, box.xyxy[0])),
        })
    return raw


def _predict_tiles(img_bgr, conf, iou, tile_size=640, overlap=0.25, imgsz=640):
    """
    Overlapping-tile pass used ONLY as a supplement to the whole-image pass.

    Measurements on this model (trained at imgsz=640) showed:
      - Whole-image at 640 is the calibrated scale; imgsz 1024/1280 makes
        confidences jump around and loses detections (pcb2: 1 -> 0).
      - Tiles alone lose defects that straddle tile boundaries (pcb2's
        tall Short Circuit spans two y-tiles and disappears).
    So tiles are run at a *higher* threshold (conf + 0.10) and UNIONED with
    the whole-image boxes, then de-duplicated. Tiny defects too small for
    the whole-image pass can still surface from tiles, without tile-only
    false positives flooding photo-domain images.
    """
    model = _get_model()
    H, W = img_bgr.shape[:2]
    if max(H, W) <= tile_size * 1.1 or min(H, W) == 0:
        return []
    tile_conf = min(0.85, float(conf) + 0.10)
    stride = int(tile_size * (1 - overlap))
    detections = []
    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            x1 = min(W, x0 + tile_size)
            y1 = min(H, y0 + tile_size)
            if (x1 - x0) < tile_size * 0.5 or (y1 - y0) < tile_size * 0.5:
                continue
            tile = img_bgr[y0:y1, x0:x1]
            results = model.predict(source=tile, conf=tile_conf, iou=iou, imgsz=imgsz, save=False, verbose=False)
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                tx1, ty1, tx2, ty2 = map(int, box.xyxy[0])
                detections.append({
                    "type": _display_name(model.names[class_id]),
                    "confidence": round(float(box.conf[0]), 3),
                    "bbox": [tx1 + x0, ty1 + y0, tx2 + x0, ty2 + y0],
                })
    return detections


# ==========================================
# Prediction Function
# ==========================================
def predict_image(image_path: str, conf: float = 0.25, iou: float = 0.40, use_tiles: bool = False):
    """
    Run YOLOv8 inference (whole-image only by default for accuracy).

    Whole-image at imgsz=640 is the model's calibrated scale (trained
    50 epochs, imgsz 640, yolov8n). Tiled inference AMPLIFIED false
    positives in every test we ran (pcb10: whole=0 vs tiles=+2,
    samples: 137 -> still 100 capped) with no recall win on test_images
    (pcb2 tiles=0 vs whole=1). Tiles therefore default OFF.
    Set use_tiles=True only for tiny-defect recall experiments.

    Per-class thresholds (Mouse Bite/Spur 0.40, Short/Spurious 0.35,
    Open/Missing 0.30) + area filter (GT median 0.04-0.10%, p90 0.21%)
    + caps (5/class, 12 total) fix the accuracy drivers:
      1) normal boards flagged (4341 background FPs in confusion matrix)
      2) wrong locations (1.7% huge Short boxes, GT p90 0.2%)
      3) components/traces as defects (via = Missing Hole floods)
      4) wrong classes (Open<->Mouse Bite 78/56, Short<->Spur 38/41 —
         class-aware NMS keeps the right winner)
    """
    model = _get_model()

    img = cv2.imread(image_path)
    if img is not None:
        image_h, image_w = img.shape[:2]
    else:
        image_h, image_w = 640, 640

    if img is not None:
        raw_detections = _predict_whole(img, conf=conf, iou=iou)
        if use_tiles:
            raw_detections += _predict_tiles(img, conf=max(float(conf), 0.45), iou=iou)
    else:
        raw_detections = []

    # Remove noise/huge/wrong-class boxes, then class-aware NMS + caps.
    filtered = _filter_tiny_boxes(raw_detections, image_w, image_h)
    detections = _nms_merge(filtered, iou_thresh=max(0.30, min(0.60, float(iou) + 0.05)))

    # Draw manually (YOLO's result.plot() would show boxes we filtered out
    # and would not know about tiled offsets).
    if img is not None:
        annotated_image = img.copy()
        colors = {
            "Open Circuit": (30, 100, 255),
            "Short Circuit": (0, 0, 230),
            "Mouse Bite": (180, 50, 220),
            "Spur": (0, 200, 120),
            "Spurious Copper": (0, 215, 255),
            "Missing Hole": (255, 140, 0),
        }
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            x1 = max(0, min(x1, image_w - 1))
            y1 = max(0, min(y1, image_h - 1))
            x2 = max(0, min(x2, image_w))
            y2 = max(0, min(y2, image_h))
            color = colors.get(det["type"], (0, 165, 255))
            label = f"{det['type']} {det['confidence']:.2f}"
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ly1 = max(0, y1 - th - 6)
            cv2.rectangle(annotated_image, (x1, ly1), (min(image_w, x1 + tw + 6), y1), color, -1)
            cv2.putText(annotated_image, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        annotated_image = np.zeros((image_h, image_w, 3), dtype=np.uint8)

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
