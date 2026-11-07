"""Reference-vs-testing difference detection.

Pipeline:
  exposure normalisation -> blur -> absolute difference -> threshold ->
  morphological cleanup -> connected components -> severity scoring.

SSIM is implemented with OpenCV Gaussian blurs (no scikit-image dependency).
The module also renders the JET heatmap and region overlay used by the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


DEFAULT_THRESHOLD = 18.0
MIN_AREA_RATIO = 0.00002  # ~52 px on a 2000x1300 frame
MAX_REGION_AREA_RATIO = 0.20  # >20% of the board = global misalignment, not a defect
MIN_ANOMALY_SEVERITY = 12.0

# Colour palette (BGR) matching the rest of CircuitSentinel.
_ACCENT = (79, 255, 215)
_DANGER = (92, 98, 255)
_WARNING = (90, 184, 231)
_REGION = (92, 98, 255)


@dataclass
class DiffRegion:
    bbox: list  # [x1, y1, x2, y2] in aligned/reference frame
    area: int
    mean_diff: float
    max_diff: int
    severity: float
    ssim: float
    centroid: list
    touches_border: bool = False
    matched: bool = False
    matched_detection: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "bbox": [int(v) for v in self.bbox],
            "area": int(self.area),
            "mean_diff": round(float(self.mean_diff), 2),
            "max_diff": int(self.max_diff),
            "severity": round(float(self.severity), 1),
            "ssim": round(float(self.ssim), 4),
            "centroid": [int(v) for v in self.centroid],
            "touches_border": bool(self.touches_border),
            "matched": bool(self.matched),
        }


@dataclass
class DiffResult:
    ssim: float = 1.0
    difference_score: float = 0.0
    changed_pixel_pct: float = 0.0
    threshold: float = DEFAULT_THRESHOLD
    regions: list = field(default_factory=list)
    diff_map: Optional[np.ndarray] = None
    diff_mask: Optional[np.ndarray] = None
    heatmap: Optional[np.ndarray] = None
    overlay: Optional[np.ndarray] = None
    valid_pixels: int = 0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "ssim": round(float(self.ssim), 4),
            "difference_score": round(float(self.difference_score), 1),
            "changed_pixel_pct": round(float(self.changed_pixel_pct), 3),
            "threshold": round(float(self.threshold), 1),
            "region_count": len(self.regions),
            "regions": [r.to_dict() for r in self.regions],
            "reason": self.reason,
        }


def compute_ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Structural similarity (Wang et al.) using OpenCV only."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

    return ((2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)) / (
        (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    )


def _normalise_exposure(gray: np.ndarray, reference_gray: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Match the testing luminance mean/std to the reference inside the overlap."""
    if not np.any(valid):
        return gray
    src_mean = float(gray[valid].mean())
    src_std = float(gray[valid].std())
    ref_mean = float(reference_gray[valid].mean())
    ref_std = float(reference_gray[valid].std())

    if src_std < 1e-6:
        return np.clip(gray.astype(np.float32) - src_mean + ref_mean, 0, 255).astype(np.uint8)

    gain = ref_std / src_std
    normalised = (gray.astype(np.float32) - src_mean) * gain + ref_mean
    return np.clip(normalised, 0, 255).astype(np.uint8)


def compute_difference(
    reference_bgr: np.ndarray,
    aligned_testing_bgr: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    blur: int = 3,
    min_area_ratio: float = MIN_AREA_RATIO,
) -> DiffResult:
    """Compare the aligned testing image against the reference."""
    result = DiffResult(threshold=float(threshold))

    if valid_mask is None or not np.any(valid_mask):
        result.reason = "No overlapping pixels between reference and testing image."
        return result

    valid = valid_mask.astype(bool)
    ref_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(aligned_testing_bgr, cv2.COLOR_BGR2GRAY)

    test_norm = _normalise_exposure(test_gray, ref_gray, valid)

    kernel = (max(1, blur), max(1, blur))
    ref_blur = cv2.GaussianBlur(ref_gray, kernel, 0)
    test_blur = cv2.GaussianBlur(test_norm, kernel, 0)

    diff = cv2.absdiff(test_blur, ref_blur)
    diff[~valid] = 0

    ssim_map = compute_ssim_map(test_blur, ref_blur)
    ssim_values = np.clip(ssim_map[valid], -1.0, 1.0)
    result.ssim = float(ssim_values.mean())

    # Threshold + morphology.  Open removes one-pixel compression noise,
    # close reconnects a defect that a thin trace splits in two.
    mask = ((diff > threshold) & valid).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask[~valid] = 0

    result.diff_map = diff
    result.diff_mask = mask
    result.valid_pixels = int(valid.sum())
    result.changed_pixel_pct = 100.0 * float(mask.sum()) / max(result.valid_pixels, 1)

    if mask.sum() == 0:
        result.difference_score = 0.0
        result.reason = "No structural difference above tolerance."
        result.heatmap, result.overlay = _render(reference_bgr, aligned_testing_bgr, diff, mask, [], threshold)
        return result

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    ref_area = reference_bgr.shape[0] * reference_bgr.shape[1]
    min_area = max(30, int(min_area_ratio * ref_area))
    max_area = int(MAX_REGION_AREA_RATIO * ref_area)

    regions: list[DiffRegion] = []
    for idx in range(1, count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        x2, y2 = min(reference_bgr.shape[1], x + w), min(reference_bgr.shape[0], y + h)

        component = labels == idx
        component_diff = diff[component]
        mean_diff = float(component_diff.mean())
        max_diff = int(component_diff.max())
        region_ssim = float(np.clip(ssim_map[component], -1.0, 1.0).mean())

        # Severity: how far the average difference is from the tolerance.
        # 255 -> 100; a typical PCB defect region sits around 20-50.
        severity = min(100.0, (mean_diff / 255.0) * 100.0 * 1.6)

        touches_border = x <= 2 or y <= 2 or x2 >= reference_bgr.shape[1] - 2 or y2 >= reference_bgr.shape[0] - 2

        regions.append(
            DiffRegion(
                bbox=[x, y, x2, y2],
                area=area,
                mean_diff=mean_diff,
                max_diff=max_diff,
                severity=severity,
                ssim=region_ssim,
                centroid=[int(centroids[idx][0]), int(centroids[idx][1])],
                touches_border=touches_border,
            )
        )

    regions.sort(key=lambda r: (r.severity, r.area), reverse=True)
    result.regions = regions

    if regions:
        max_severity = max(r.severity for r in regions)
        result.difference_score = min(
            100.0,
            max_severity + min(20.0, result.changed_pixel_pct * 8.0),
        )
        result.reason = (
            f"{len(regions)} structural difference(s) above tolerance "
            f"(SSIM {result.ssim:.3f})."
        )
    else:
        result.difference_score = 0.0
        result.reason = "Differences exist but are below the minimum defect area."

    result.heatmap, result.overlay = _render(
        reference_bgr, aligned_testing_bgr, diff, mask, regions, threshold
    )
    return result


def _render(reference_bgr, aligned_testing_bgr, diff, mask, regions, threshold):
    """Build the heatmap and the annotated difference overlay."""
    h, w = aligned_testing_bgr.shape[:2]

    # Heatmap: amplify the diff around the tolerance so faint defects are
    # visible, then blend it over the aligned testing photo.
    scale = 255.0 / max(float(threshold) * 4.0, 60.0)
    heat_src = np.clip(diff.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    heat_src = cv2.GaussianBlur(heat_src, (5, 5), 0)
    heat_color = cv2.applyColorMap(heat_src, cv2.COLORMAP_JET)

    heatmap = cv2.addWeighted(aligned_testing_bgr, 0.62, heat_color, 0.38, 0)
    heatmap[mask == 0] = cv2.addWeighted(
        aligned_testing_bgr, 0.85, heat_color, 0.15, 0
    )[mask == 0]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(heatmap, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)

    overlay = aligned_testing_bgr.copy()
    cv2.drawContours(overlay, contours, -1, _REGION, 2, cv2.LINE_AA)

    font_scale = max(0.45, min(0.9, w / 1800.0))
    thickness = max(1, int(round(w / 900.0)))
    for i, region in enumerate(regions[:25], start=1):
        x1, y1, x2, y2 = region.bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), _REGION, thickness, cv2.LINE_AA)
        label = f"d{i} {region.severity:.0f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ly1 = max(0, y1 - th - 6)
        cv2.rectangle(overlay, (x1, ly1), (min(w, x1 + tw + 6), y1), _REGION, -1)
        cv2.putText(
            overlay, label, (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA,
        )

    return heatmap, overlay
