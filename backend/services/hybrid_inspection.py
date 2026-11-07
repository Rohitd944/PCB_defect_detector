"""Hybrid PCB inspection: YOLO defects + reference comparison verification.

The YOLO model stays authoritative for *what* a defect is; the reference
comparison is used to verify *whether the testing board actually differs from
the known-good board*.  A YOLO box that does not overlap a real structural
difference is reported as rejected/unverified rather than as a defect, which
removes the model's false positives on boards that match the reference.

Verdict policy
  FAIL    at least one YOLO defect is confirmed by a structural difference
  REVIEW  structural difference(s) without a YOLO match, or unreliable alignment
  PASS    testing board matches the reference within tolerance
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uuid

from predictor import RESULT_DIR, predict_image
from services.alignment import AlignmentResult, align_images, resize_for_processing
from services.diff_detection import DiffRegion, DiffResult, compute_difference


HYBRID_VERSION = "1.0"

# A YOLO box confirms a diff region when it contains at least this fraction of
# that region (the region is normally much smaller than the box).
CONFIRM_REGION_COVERAGE = 0.35
CONFIRM_MIN_SEVERITY = 8.0

# Unmatched regions need at least this severity to trigger REVIEW.
ANOMALY_MIN_SEVERITY = 12.0
ANOMALY_BORDER_MIN_SEVERITY = 25.0
ANOMALY_MAX_AREA_RATIO = 0.05

BOX_COLORS = {
    "Open Circuit": (30, 100, 255),
    "Short Circuit": (0, 0, 230),
    "Mouse Bite": (180, 50, 220),
    "Spur": (0, 200, 120),
    "Spurious Copper": (0, 215, 255),
    "Missing Hole": (255, 140, 0),
}

_DANGER = (92, 98, 255)
_WARNING = (90, 184, 231)
_SUCCESS = (145, 211, 101)
_ACCENT = (79, 255, 215)
_MUTED = (168, 173, 167)
_REGION_COLOR = (92, 98, 255)


# ── geometry helpers ────────────────────────────────────────────────────────
def _map_bbox(bbox, homography: np.ndarray, test_scale: float, out_w: int, out_h: int):
    x1, y1, x2, y2 = [float(v) * test_scale for v in bbox]
    corners = np.array(
        [[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.float32
    )
    mapped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    mx1 = int(np.clip(np.floor(mapped[:, 0].min()), 0, out_w - 1))
    my1 = int(np.clip(np.floor(mapped[:, 1].min()), 0, out_h - 1))
    mx2 = int(np.clip(np.ceil(mapped[:, 0].max()), 1, out_w))
    my2 = int(np.clip(np.ceil(mapped[:, 1].max()), 1, out_h))
    return [mx1, my1, mx2, my2]


def _intersection(a, b) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _region_coverage(box, region) -> float:
    inter = _intersection(box, region.bbox)
    if inter <= 0:
        return 0.0
    return inter / max(region.area, 1)


def _location_pct(bbox, width, height):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return {
        "x": round(100.0 * cx / max(width, 1), 1),
        "y": round(100.0 * cy / max(height, 1), 1),
        "center": [int(round(cx)), int(round(cy))],
    }


# ── fusion ──────────────────────────────────────────────────────────────────
def _fuse(
    detections: list,
    diff: Optional[DiffResult],
    alignment: AlignmentResult,
    testing_width: int,
    testing_height: int,
):
    """Attach reference-verification fields to YOLO detections."""
    if not detections:
        return []

    homography = alignment.homography
    ref_w, ref_h = alignment.ref_size
    trusted = alignment.ok

    for index, det in enumerate(detections):
        det["original_bbox"] = list(det["bbox"])
        det["processed_bbox"] = [
            int(round(float(v) * alignment.test_scale)) for v in det["bbox"]
        ]
        det["location"] = _location_pct(det["bbox"], testing_width, testing_height)
        det["reference_difference"] = None
        det["box_mean_difference"] = None
        det["box_max_difference"] = None
        det["matched_regions"] = 0
        det["verification"] = "not_compared"

        if diff is None or homography is None:
            det["verification"] = "unverified"
            continue

        mapped = _map_bbox(det["bbox"], homography, alignment.test_scale, ref_w, ref_h)
        det["aligned_bbox"] = mapped

        x1, y1, x2, y2 = mapped
        patch = diff.diff_map[y1:y2, x1:x2] if diff.diff_map is not None else None
        if patch is not None and patch.size:
            det["box_mean_difference"] = round(float(patch.mean()), 2)
            det["box_max_difference"] = int(patch.max())

        matched = []
        for region in diff.regions:
            coverage = _region_coverage(mapped, region)
            if coverage >= CONFIRM_REGION_COVERAGE:
                matched.append(region)

        det["matched_regions"] = len(matched)
        for region in matched:
            region.matched = True
            if region.matched_detection is None:
                region.matched_detection = index

        if matched:
            best_severity = max(r.severity for r in matched)
            det["reference_difference"] = round(float(best_severity), 1)
            if best_severity >= CONFIRM_MIN_SEVERITY:
                det["verification"] = "confirmed"
                continue

        # A YOLO box with no structural support is either a model false
        # positive (reference trusted) or cannot be checked (weak alignment).
        if trusted:
            det["verification"] = "rejected"
        else:
            det["verification"] = "unverified"

    return detections


def _collect_anomalies(diff: Optional[DiffResult], reference_area: int) -> list:
    if diff is None:
        return []
    anomalies = []
    max_area = ANOMALY_MAX_AREA_RATIO * reference_area
    for region in diff.regions:
        if region.matched:
            continue
        if region.area > max_area:
            continue
        minimum = ANOMALY_BORDER_MIN_SEVERITY if region.touches_border else ANOMALY_MIN_SEVERITY
        if region.severity < minimum:
            continue
        anomalies.append(region)
    return anomalies


def _verdict(
    detections: list,
    anomalies: list,
    diff: Optional[DiffResult],
    alignment: AlignmentResult,
    reference_area: int,
):
    confirmed = [d for d in detections if d.get("verification") == "confirmed"]
    rejected = [d for d in detections if d.get("verification") == "rejected"]

    if alignment.quality == "failed":
        return (
            "REVIEW",
            "Reference comparison unavailable. " + alignment.reason,
            "low",
        )

    if alignment.quality == "weak":
        reason = alignment.reason
        if confirmed:
            reason += f" {len(confirmed)} YOLO detection(s) also differ from the reference, but alignment is not reliable enough for an automatic verdict."
        return "REVIEW", reason, "medium"

    if confirmed:
        return (
            "FAIL",
            f"{len(confirmed)} defect(s) confirmed by reference comparison.",
            "high",
        )

    # A large share of the board changed at once is almost always a lighting,
    # exposure or alignment problem, not a localised manufacturing defect.
    if diff is not None and diff.changed_pixel_pct > 2.0:
        return (
            "REVIEW",
            "Most of the board changed at once; this looks like a lighting, exposure or alignment problem rather than a localised defect.",
            "medium",
        )

    if anomalies:
        return (
            "REVIEW",
            f"{len(anomalies)} structural difference(s) found without a matching YOLO defect. Manual review required.",
            "medium",
        )

    if rejected:
        return (
            "PASS",
            f"Testing PCB matches the reference. {len(rejected)} YOLO candidate(s) were rejected because the reference shows no structural change at those locations.",
            "high",
        )

    return "PASS", "Testing PCB matches the reference within tolerance.", "high"


# ── rendering ───────────────────────────────────────────────────────────────
def _font_scale(width: int) -> float:
    return max(0.45, min(1.0, width / 1600.0))


def _draw_regions(image, regions, thickness_color=_REGION_COLOR):
    for i, region in enumerate(regions[:30], start=1):
        x1, y1, x2, y2 = region.bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), thickness_color, 2, cv2.LINE_AA)
        label = f"diff {region.severity:.0f}"
        scale = _font_scale(image.shape[1]) * 0.8
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ly1 = max(0, y1 - th - 6)
        cv2.rectangle(image, (x1, ly1), (min(image.shape[1], x1 + tw + 6), y1), thickness_color, -1)
        cv2.putText(image, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_detections(image, detections, bbox_key="aligned_bbox"):
    scale = _font_scale(image.shape[1])
    thickness = max(1, int(round(image.shape[1] / 900.0)))
    for det in detections:
        bbox = det.get(bbox_key) or det.get("original_bbox") or det["bbox"]
        x1, y1, x2, y2 = bbox
        color = BOX_COLORS.get(det["type"], (0, 165, 255))
        verified = det.get("verification")
        if verified == "rejected":
            color = _MUTED
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        label = f"{det['type']} {det['confidence'] * 100:.0f}%"
        if verified == "confirmed" and det.get("reference_difference") is not None:
            label += f" | diff {det['reference_difference']:.0f}"
        elif verified == "rejected":
            label += " | rejected"
        elif verified == "unverified":
            label += " | unverified"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        ly1 = max(0, y1 - th - 6)
        cv2.rectangle(image, (x1, ly1), (min(image.shape[1], x1 + tw + 8), y1), color, -1)
        cv2.putText(
            image, label, (x1 + 4, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA,
        )


def _verdict_color(verdict: str):
    return {"PASS": _SUCCESS, "FAIL": _DANGER, "REVIEW": _WARNING}.get(verdict, _ACCENT)


def _draw_verdict_banner(image, verdict: str, reason: str):
    w = image.shape[1]
    bar_h = max(44, int(w * 0.032))
    color = _verdict_color(verdict)
    cv2.rectangle(image, (0, 0), (w, bar_h), color, -1)

    scale = max(0.7, min(1.4, w / 1100.0))
    cv2.putText(
        image, f"HYBRID INSPECTION: {verdict}", (14, int(bar_h * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), max(2, int(scale * 2)), cv2.LINE_AA,
    )
    short = reason if len(reason) <= 78 else reason[:75] + "..."
    text_scale = max(0.4, scale * 0.55)
    (tw, _), _ = cv2.getTextSize(short, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)
    if tw < w - 28:
        cv2.putText(
            image, short, (16, bar_h - max(6, int(bar_h * 0.14))),
            cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return image


def _comparison(ref_proc: np.ndarray, aligned: np.ndarray, diff: Optional[DiffResult]) -> np.ndarray:
    h = min(ref_proc.shape[0], aligned.shape[0])
    if ref_proc.shape[0] != h:
        ref_proc = cv2.resize(ref_proc, (int(ref_proc.shape[1] * h / ref_proc.shape[0]), h))
    if aligned.shape[0] != h:
        aligned = cv2.resize(aligned, (int(aligned.shape[1] * h / aligned.shape[0]), h))

    gap = 6
    body = np.full((h, ref_proc.shape[1] + gap + aligned.shape[1], 3), 12, np.uint8)
    body[:, : ref_proc.shape[1]] = ref_proc
    body[:, ref_proc.shape[1] + gap :] = aligned

    header = 38
    footer = 30
    canvas = np.full((h + header + footer, body.shape[1], 3), 10, np.uint8)
    canvas[header : header + h] = body

    font = max(0.45, min(0.8, canvas.shape[1] / 1600.0))
    cv2.putText(canvas, "REFERENCE (known good)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, font, (207, 255, 79), 1, cv2.LINE_AA)
    x2 = ref_proc.shape[1] + gap + 10
    cv2.putText(canvas, "TESTING (aligned)", (x2, 25), cv2.FONT_HERSHEY_SIMPLEX, font, (207, 255, 79), 1, cv2.LINE_AA)

    if diff is not None:
        summary = f"SSIM {diff.ssim:.3f}  ·  difference score {diff.difference_score:.0f}/100  ·  changed {diff.changed_pixel_pct:.2f}%  ·  {len(diff.regions)} region(s)"
    else:
        summary = "Diff unavailable — reference alignment failed"
    cv2.putText(canvas, summary, (10, canvas.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, font * 0.9, (200, 210, 200), 1, cv2.LINE_AA)
    return canvas


def _unavailable(image: np.ndarray, message: str) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h // 2 - 45), (w, h // 2 + 45), (10, 12, 12), -1)
    out = cv2.addWeighted(overlay, 0.75, out, 0.25, 0)
    font = max(0.5, min(1.0, w / 900.0))
    (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, font, 2)
    cv2.putText(
        out, message, (max(10, (w - tw) // 2), h // 2 + th // 2),
        cv2.FONT_HERSHEY_SIMPLEX, font, _WARNING, 2, cv2.LINE_AA,
    )
    return out


def _save(image: np.ndarray, inspection_id: str, suffix: str) -> str:
    path = RESULT_DIR / f"hybrid_{inspection_id}_{suffix}.jpg"
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return str(path)


# ── entry point ─────────────────────────────────────────────────────────────
def run_hybrid_inspection(
    testing_path: str,
    reference_entry: dict,
    confidence: float = 0.25,
    iou: float = 0.40,
    diff_threshold: float = 18.0,
) -> dict:
    reference_image = reference_entry["image"]
    reference_meta = dict(reference_entry["meta"])

    testing_image = cv2.imread(testing_path, cv2.IMREAD_COLOR)
    if testing_image is None:
        raise ValueError("Could not read the testing image.")

    orig_h, orig_w = testing_image.shape[:2]

    # 1) YOLO defects (unchanged model and post-processing).
    yolo_result = predict_image(testing_path, conf=confidence, iou=iou)
    detections = yolo_result["detections"]

    # 2) ORB + RANSAC alignment.
    alignment = align_images(reference_image, testing_image)

    # ``compute_difference`` needs the reference in the same processing frame
    # as the aligned testing image.
    ref_proc, _ = resize_for_processing(reference_image)
    if alignment.aligned is not None and ref_proc.shape[:2] != alignment.aligned.shape[:2]:
        ref_proc = cv2.resize(
            ref_proc,
            (alignment.aligned.shape[1], alignment.aligned.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    # 3) Reference difference (only when a transform exists).
    diff: Optional[DiffResult] = None
    if alignment.aligned is not None and alignment.valid_mask is not None and alignment.quality != "failed":
        diff = compute_difference(
            reference_bgr=ref_proc,
            aligned_testing_bgr=alignment.aligned,
            valid_mask=alignment.valid_mask,
            threshold=max(6.0, float(diff_threshold)),
        )

    # 4) Verification engine.
    detections = _fuse(detections, diff, alignment, orig_w, orig_h)
    anomalies = _collect_anomalies(diff, alignment.ref_size[0] * alignment.ref_size[1])
    verdict, verdict_reason, confidence_level = _verdict(
        detections, anomalies, diff, alignment, alignment.ref_size[0] * alignment.ref_size[1]
    )

    # 5) Images.
    inspection_id = "PCB-HY-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]

    if alignment.aligned is not None and alignment.homography is not None:
        annotated = alignment.aligned.copy()
        if diff is not None:
            _draw_regions(annotated, diff.regions)
        _draw_detections(annotated, detections, bbox_key="aligned_bbox")
        _draw_verdict_banner(annotated, verdict, verdict_reason)
        comparison = _comparison(ref_proc, alignment.aligned, diff)
        if diff is not None and diff.heatmap is not None:
            heatmap = diff.heatmap.copy()
            _draw_verdict_banner(heatmap, verdict, verdict_reason)
        else:
            heatmap = _unavailable(alignment.aligned, "Difference heatmap unavailable")
        aligned_output = alignment.aligned
    else:
        test_proc, _ = resize_for_processing(testing_image)
        annotated = test_proc.copy()
        _draw_detections(annotated, detections, bbox_key="processed_bbox")
        _draw_verdict_banner(annotated, verdict, verdict_reason)
        comparison = _comparison(ref_proc, test_proc, None)
        heatmap = _unavailable(test_proc, "Reference alignment failed")
        aligned_output = test_proc

    images = {
        "annotated": _save(annotated, inspection_id, "annotated"),
        "comparison": _save(comparison, inspection_id, "comparison"),
        "heatmap": _save(heatmap, inspection_id, "heatmap"),
        "aligned": _save(aligned_output, inspection_id, "aligned"),
        "yolo_result": yolo_result.get("result_image"),
    }

    confirmed = [d for d in detections if d.get("verification") == "confirmed"]
    rejected = [d for d in detections if d.get("verification") == "rejected"]
    unverified = [d for d in detections if d.get("verification") == "unverified"]

    return {
        "status": "success",
        "hybrid_version": HYBRID_VERSION,
        "inspection_id": inspection_id,
        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "confidence_level": confidence_level,
        "reference": reference_meta,
        "testing": {
            "filename": Path(testing_path).name,
            "width": int(orig_w),
            "height": int(orig_h),
        },
        "alignment": alignment.to_dict(),
        "diff": diff.to_dict() if diff is not None else None,
        "detections": detections,
        "anomalies": [r.to_dict() for r in anomalies],
        "summary": {
            "total_detections": len(detections),
            "confirmed": len(confirmed),
            "rejected": len(rejected),
            "unverified": len(unverified),
            "anomalies": len(anomalies),
            "reference_regions": len(diff.regions) if diff is not None else 0,
        },
        "parameters": {
            "confidence": round(float(confidence), 3),
            "iou": round(float(iou), 3),
            "diff_threshold": round(float(diff_threshold), 1),
        },
        "images": images,
    }
