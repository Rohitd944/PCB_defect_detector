"""Persistent store for known-good reference PCB images.

References are written to ``backend/references/`` and described by a small
JSON file next to the image.  ORB descriptors are cached in memory so a
repeated inspection does not re-extract features.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

import cv2
import json
import numpy as np
import uuid


class ReferenceStore:
    def __init__(self, directory: Path, max_cached: int = 8):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._max_cached = max_cached
        self._cache: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = Lock()
        self._orb = cv2.ORB_create(nfeatures=5000)

    # ── public API ─────────────────────────────────────────────────────────
    def save(self, image_bgr: np.ndarray, filename: str, name: Optional[str] = None) -> dict:
        ref_id = uuid.uuid4().hex[:10]
        image_path = self.directory / f"{ref_id}.jpg"
        meta_path = self.directory / f"{ref_id}.json"

        ok = cv2.imwrite(str(image_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            raise RuntimeError("Could not write the reference image to disk.")

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)

        meta = {
            "reference_id": ref_id,
            "filename": filename,
            "name": name or filename,
            "width": int(image_bgr.shape[1]),
            "height": int(image_bgr.shape[0]),
            "keypoints": int(len(keypoints) if keypoints else 0),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        entry = {
            "meta": meta,
            "image": image_bgr.copy(),
            "gray": gray,
            "keypoints": keypoints,
            "descriptors": descriptors,
            "path": image_path,
        }
        with self._lock:
            self._cache[ref_id] = entry
            self._cache.move_to_end(ref_id)
            while len(self._cache) > self._max_cached:
                self._cache.popitem(last=False)
        return dict(meta)

    def get(self, reference_id: str) -> Optional[dict]:
        if not reference_id:
            return None
        reference_id = reference_id.strip()
        with self._lock:
            if reference_id in self._cache:
                self._cache.move_to_end(reference_id)
                return self._cache[reference_id]

        image_path = self.directory / f"{reference_id}.jpg"
        meta_path = self.directory / f"{reference_id}.json"
        if not image_path.exists():
            return None

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return None

        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                meta = {}
        else:
            meta = {}
        meta.setdefault("reference_id", reference_id)
        meta.setdefault("filename", image_path.name)
        meta.setdefault("name", meta["filename"])
        meta.setdefault("width", int(image.shape[1]))
        meta.setdefault("height", int(image.shape[0]))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        meta["keypoints"] = int(len(keypoints) if keypoints else 0)

        entry = {
            "meta": meta,
            "image": image,
            "gray": gray,
            "keypoints": keypoints,
            "descriptors": descriptors,
            "path": image_path,
        }
        with self._lock:
            self._cache[reference_id] = entry
            self._cache.move_to_end(reference_id)
            while len(self._cache) > self._max_cached:
                self._cache.popitem(last=False)
        return entry

    def latest(self) -> Optional[dict]:
        candidates = sorted(
            self.directory.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for meta_path in candidates:
            entry = self.get(meta_path.stem)
            if entry is not None:
                return entry
        return None

    def list(self, limit: int = 20) -> list:
        metas = []
        for meta_path in self.directory.glob("*.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            meta["reference_id"] = meta_path.stem
            meta["image_filename"] = f"{meta_path.stem}.jpg"
            metas.append(meta)
        metas.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        return metas[:limit]

    def image_path(self, reference_id: str) -> Optional[Path]:
        path = self.directory / f"{reference_id}.jpg"
        return path if path.exists() else None
