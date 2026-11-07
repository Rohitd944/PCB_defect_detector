"""Reference <-> testing PCB alignment.

ORB feature matching + RANSAC homography, with a quality gate so the hybrid
verification engine knows when a reference comparison can be trusted.

The reference project (Junkpup/pcb-qa) uses the same classical-CV approach
inside a PySide6 app.  This module keeps the ideas (ORB + Lowe ratio test +
RANSAC + sanity checks) and returns plain Python data so it can be used from
FastAPI without any Qt dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import math
import numpy as np


# Processing cap.  PCB defects in the sample set are only ~40-70 px wide on
# 2900 px boards, so we must not downscale too aggressively.  2000 px keeps
# them visible while staying fast and memory-light on an M1.
PROC_MAX_DIM = 2000

ORB_FEATURES = 5000
LOWE_RATIO = 0.75
RANSAC_REPROJ_PX = 4.0

MIN_GOOD_MATCHES = 30
MIN_INLIERS = 25
MIN_INLIER_RATIO = 0.30
MAX_REPROJ_ERROR = 4.0
MIN_SCALE, MAX_SCALE = 0.30, 3.50
MIN_COVERAGE = 0.55
MIN_QUAD_RATIO, MAX_QUAD_RATIO = 0.35, 3.0


@dataclass
class AlignmentResult:
    quality: str  # "good" | "weak" | "failed"
    reason: str = ""
    aligned: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None
    homography: Optional[np.ndarray] = None
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_error: float = 0.0
    scale: float = 1.0
    rotation_deg: float = 0.0
    coverage: float = 0.0
    quad_area_ratio: float = 0.0
    ref_scale: float = 1.0
    test_scale: float = 1.0
    ref_size: tuple = (0, 0)
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.quality == "good"

    def to_dict(self) -> dict:
        return {
            "quality": self.quality,
            "trusted": self.ok,
            "reason": self.reason,
            "good_matches": self.good_matches,
            "inliers": self.inliers,
            "inlier_ratio": round(float(self.inlier_ratio), 3),
            "reprojection_error": round(float(self.reprojection_error), 2),
            "scale": round(float(self.scale), 3),
            "rotation_deg": round(float(self.rotation_deg), 2),
            "coverage": round(float(self.coverage), 3),
            "reference_size": {"width": int(self.ref_size[0]), "height": int(self.ref_size[1])},
            "warnings": list(self.warnings),
        }


def resize_for_processing(image: np.ndarray, max_dim: int = PROC_MAX_DIM):
    """Downscale only (never upscale) so small defects are not lost."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image.copy(), 1.0
    scale = float(max_dim) / float(max(h, w))
    resized = cv2.resize(
        image,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _quad_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def align_images(
    reference_bgr: np.ndarray,
    testing_bgr: np.ndarray,
    max_dim: int = PROC_MAX_DIM,
) -> AlignmentResult:
    """Align the testing image onto the reference image's coordinate frame."""
    ref, ref_scale = resize_for_processing(reference_bgr, max_dim)
    test, test_scale = resize_for_processing(testing_bgr, max_dim)

    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    ref_h, ref_w = ref_gray.shape[:2]

    result = AlignmentResult(
        quality="failed",
        ref_scale=ref_scale,
        test_scale=test_scale,
        ref_size=(ref_w, ref_h),
    )

    if ref_gray.size == 0 or test_gray.size == 0:
        result.reason = "Empty reference or testing image."
        return result

    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    kp_ref, desc_ref = orb.detectAndCompute(ref_gray, None)
    kp_test, desc_test = orb.detectAndCompute(test_gray, None)

    if desc_ref is None or desc_test is None:
        result.reason = "Not enough ORB texture in one of the images."
        return result

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(desc_test, desc_ref, k=2)
    good = [m for m in knn if len(m) == 2 and m[0].distance < LOWE_RATIO * m[1].distance]
    result.good_matches = len(good)

    if len(good) < MIN_GOOD_MATCHES:
        result.reason = (
            f"Only {len(good)} reliable ORB matches (need {MIN_GOOD_MATCHES}). "
            "The images may show different boards, a different angle, or too little overlap."
        )
        return result

    src = np.float32([kp_test[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_ref[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ_PX)
    if homography is None or mask is None:
        result.reason = "Homography estimation failed (RANSAC found no consistent transform)."
        return result

    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    result.homography = homography
    result.inliers = inliers
    result.inlier_ratio = inliers / max(len(good), 1)

    if inliers:
        projected = cv2.perspectiveTransform(src, homography).reshape(-1, 2)
        errors = np.linalg.norm(projected - dst.reshape(-1, 2), axis=1)
        result.reprojection_error = float(errors[inlier_mask].mean())

    # Normalised homography decomposition for scale / rotation (good enough for
    # the human-readable metadata; the full H is still used for warping).
    hn = homography / homography[2, 2]
    scale_x = math.hypot(float(hn[0, 0]), float(hn[1, 0]))
    scale_y = math.hypot(float(hn[0, 1]), float(hn[1, 1]))
    result.scale = math.sqrt(max(scale_x * scale_y, 1e-9))
    result.rotation_deg = math.degrees(math.atan2(float(hn[1, 0]), float(hn[0, 0])))

    # How much of the reference frame the warped testing image actually covers.
    valid = cv2.warpPerspective(
        np.ones(test_gray.shape, dtype=np.uint8),
        homography,
        (ref_w, ref_h),
        flags=cv2.INTER_NEAREST,
    )
    result.valid_mask = valid.astype(bool)
    result.coverage = float(result.valid_mask.mean())

    # Warped testing footprint relative to the reference frame.  A crop or
    # extreme zoom shows up here even when the homography itself is valid.
    ref_corners = np.float32([[0, 0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]]).reshape(-1, 1, 2)
    try:
        inv = np.linalg.inv(homography)
        test_quad = cv2.perspectiveTransform(ref_corners, inv).reshape(-1, 2)
        ref_area = float(ref_w * ref_h)
        result.quad_area_ratio = _quad_area(test_quad) / max(ref_area, 1.0)
    except np.linalg.LinAlgError:
        result.quad_area_ratio = 0.0

    result.aligned = cv2.warpPerspective(
        test,
        homography,
        (ref_w, ref_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # ── Quality gate ────────────────────────────────────────────────────────
    warnings = []

    if inliers < MIN_INLIERS or result.inlier_ratio < MIN_INLIER_RATIO:
        result.quality = "failed"
        result.reason = (
            f"Reference alignment is not reliable ({inliers} inliers, "
            f"ratio {result.inlier_ratio:.2f}). Use a photo of the same board."
        )
        return result

    if result.reprojection_error > MAX_REPROJ_ERROR:
        warnings.append(
            f"high reprojection error ({result.reprojection_error:.1f}px)"
        )
    if not (MIN_SCALE <= result.scale <= MAX_SCALE):
        warnings.append(
            f"strong scale mismatch ({result.scale:.2f}x). Test image may be a different crop."
        )
    if result.coverage < MIN_COVERAGE:
        warnings.append(f"low overlap ({result.coverage * 100:.0f}% of reference frame)")
    if not (MIN_QUAD_RATIO <= result.quad_area_ratio <= MAX_QUAD_RATIO):
        warnings.append("testing image covers an unusual portion of the reference frame")

    if warnings:
        result.quality = "weak"
        result.reason = "Reference comparison is limited: " + "; ".join(warnings) + "."
        result.warnings = warnings
    else:
        result.quality = "good"
        result.reason = (
            f"Aligned with {inliers} inliers "
            f"(rotation {result.rotation_deg:+.1f}, scale {result.scale:.2f}x)."
        )

    return result
