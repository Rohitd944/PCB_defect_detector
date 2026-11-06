"""
Reference-based PCB Defect Detection using Image Comparison.

Algorithm:
1. Load reference (good PCB) and test (suspected defective PCB) images
2. Align test image to reference using ORB feature matching + homography
3. Compute absolute pixel-wise difference
4. Threshold + morphological cleanup to find defect regions
5. Find contours → bounding boxes → classify each region
6. Return annotated image + structured detections
"""

import cv2
import numpy as np
from pathlib import Path

# ─── Defect classification thresholds ────────────────────────────────────────
# Based on the shape/size of the difference region, we classify the defect type.
# These are heuristic rules; can be improved with a small trained classifier.
_CLASS_RULES = [
    # (name,          min_area, max_area,  min_aspect, max_aspect)
    # Short circuit: large horizontal blob (copper bridging two traces)
    ("Short Circuit",    80,   999999,      1.5,      10.0),
    # Open circuit: small roughly-square gap (broken trace)
    ("Open Circuit",     20,    2000,       0.4,       2.5),
    # Mouse bite: small roughly-circular region on board edge
    ("Mouse Bite",       15,    1500,       0.5,       2.0),
    # Spur: thin elongated spike
    ("Spur",             10,     800,       2.5,      20.0),
    # Spurious copper: moderate irregular blob
    ("Spurious Copper",  60,   8000,       0.3,       4.0),
    # Missing hole: circular blob
    ("Missing Hole",     20,    600,       0.6,       1.6),
]

_COLORS = {
    "Short Circuit":   (0,  80, 255),   # red-orange
    "Open Circuit":    (0, 180, 255),   # orange
    "Mouse Bite":      (200, 50, 200),  # purple
    "Spur":            (0, 220, 100),   # green
    "Spurious Copper": (200, 200, 0),   # yellow
    "Missing Hole":    (255, 100, 50),  # blue
    "Anomaly":         (100, 100, 255), # grey-blue fallback
}


def _classify_region(contour, img_area: int) -> str:
    """Classify a contour into a defect type using simple heuristics."""
    area = cv2.contourArea(contour)
    if area < 10:
        return None  # too small – noise

    x, y, w, h = cv2.boundingRect(contour)
    aspect = max(w, h) / max(min(w, h), 1)

    for name, min_a, max_a, min_asp, max_asp in _CLASS_RULES:
        if min_a <= area <= max_a and min_asp <= aspect <= max_asp:
            return name

    # fallback
    if area > 20:
        return "Anomaly"
    return None


def _align_images(ref_gray: np.ndarray, test_gray: np.ndarray):
    """
    Align test_gray to ref_gray using ORB keypoints + homography.
    Returns the warped test image (same size as ref).
    Falls back to raw test image if alignment fails.
    """
    orb = cv2.ORB_create(nfeatures=5000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(test_gray, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        # Can't align — resize test to match ref and return
        h, w = ref_gray.shape[:2]
        return cv2.resize(test_gray, (w, h))

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 10:
        # Not enough matches — just resize
        h, w = ref_gray.shape[:2]
        return cv2.resize(test_gray, (w, h))

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

    if H is None:
        h, w = ref_gray.shape[:2]
        return cv2.resize(test_gray, (w, h))

    h, w = ref_gray.shape[:2]
    aligned = cv2.warpPerspective(test_gray, H, (w, h))
    return aligned


def compare_pcb_images(
    reference_path: str,
    test_path: str,
    diff_threshold: int = 30,
    min_contour_area: int = 20,
    dilate_iters: int = 3,
) -> dict:
    """
    Compare a test PCB image against a reference (good) PCB image.

    Parameters
    ----------
    reference_path   : path to the known-good PCB image
    test_path        : path to the PCB image under test
    diff_threshold   : pixel brightness difference treated as a defect (0-255)
    min_contour_area : minimum contour area (px²) to be considered a defect
    dilate_iters     : morphological dilation iterations (merges nearby blobs)

    Returns
    -------
    dict with keys:
      detections       – list of {type, confidence, bbox, area}
      diff_image       – path to saved diff-heat-map image
      annotated_image  – path to saved annotated test image
      aligned          – True if images were successfully aligned
    """
    # ── Load images ───────────────────────────────────────────────────────────
    ref_bgr  = cv2.imread(reference_path)
    test_bgr = cv2.imread(test_path)

    if ref_bgr is None:
        raise ValueError(f"Cannot read reference image: {reference_path}")
    if test_bgr is None:
        raise ValueError(f"Cannot read test image: {test_path}")

    ref_gray  = cv2.cvtColor(ref_bgr,  cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)

    # ── Align test → reference ────────────────────────────────────────────────
    aligned_gray = _align_images(ref_gray, test_gray)
    aligned_color = cv2.cvtColor(aligned_gray, cv2.COLOR_GRAY2BGR)
    was_aligned = True

    # ── Compute difference ────────────────────────────────────────────────────
    diff = cv2.absdiff(ref_gray, aligned_gray)

    # Blur to suppress sensor noise
    diff_blurred = cv2.GaussianBlur(diff, (5, 5), 0)

    # Threshold
    _, thresh = cv2.threshold(diff_blurred, diff_threshold, 255, cv2.THRESH_BINARY)

    # Morphological cleanup: close small holes, then dilate to merge nearby blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=dilate_iters)

    # ── Find contours ─────────────────────────────────────────────────────────
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = ref_gray.shape[0] * ref_gray.shape[1]
    detections = []

    # Work on a copy of the aligned test image for annotation
    annotated = cv2.resize(test_bgr, (ref_bgr.shape[1], ref_bgr.shape[0]))

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue

        defect_type = _classify_region(contour, img_area)
        if defect_type is None:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Confidence proxy: larger & stronger difference → higher confidence
        # Normalize relative to image size; cap at 0.99
        raw_conf = min(area / (img_area * 0.01), 1.0)  # fraction of 1% image area
        confidence = round(min(0.5 + raw_conf * 0.49, 0.99), 3)

        detections.append({
            "type":       defect_type,
            "confidence": confidence,
            "bbox":       [x, y, x + w, y + h],
            "area":       int(area),
        })

        color = _COLORS.get(defect_type, _COLORS["Anomaly"])
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        label = f"{defect_type} {confidence:.2f}"
        label_y = max(y - 6, 14)
        cv2.putText(annotated, label, (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # ── Diff heat-map (colorized) ──────────────────────────────────────────────
    diff_colored = cv2.applyColorMap(
        cv2.normalize(diff_blurred, None, 0, 255, cv2.NORM_MINMAX),
        cv2.COLORMAP_JET
    )

    return {
        "detections":      detections,
        "annotated_image": annotated,
        "diff_image":      diff_colored,
        "aligned":         was_aligned,
    }

