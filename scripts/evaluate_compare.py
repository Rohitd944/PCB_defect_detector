"""
Compare old vs improved model on the merged validation set.

Metrics: precision, recall, F1 (at best-F1 conf), mAP50, mAP50-95,
         false-positive rate (FP/image) on defect-free images.

Usage:
  PYTHONPATH=backend/venv/lib/python3.11/site-packages \
    python3 scripts/evaluate_compare.py \
      --old backend/models/best.pt \
      --new runs/finetune_m1/weights/best.pt \
      --data dataset_merged/data.yaml

Also runs a defect-free FPR spot-check on test_images normals.
"""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def val_metrics(model_path, data_yaml, imgsz=640, device="cpu"):
    from ultralytics import YOLO
    m = YOLO(str(model_path))
    results = m.val(data=str(data_yaml), imgsz=imgsz, device=device, plots=False, verbose=False)
    p = float(results.box.mean(axis=0)[0]) if hasattr(results.box, "mean") else None
    # Ultralytics returns dict-like; extract reliably
    try:
        return {
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
            "mAP50": float(results.box.map50),
            "mAP50_95": float(results.box.map),
            "f1": float(results.box.f1.mean()) if hasattr(results.box.f1, "mean") else None,
        }
    except Exception:
        return {
            "precision": None, "recall": None, "mAP50": None, "mAP50_95": None, "f1": None,
            "raw": str(results)[:500],
        }


def fpr_on_negatives(model_path, neg_dir, conf=0.35, imgsz=640):
    from ultralytics import YOLO
    import glob as _g
    m = YOLO(str(model_path))
    total_fp = 0
    n = 0
    for f in sorted(_g.glob(f"{neg_dir}/*.jpg")):
        r = m.predict(source=f, conf=conf, iou=0.45, imgsz=imgsz, verbose=False)[0]
        total_fp += len(r.boxes)
        n += 1
    per_image = (total_fp / n) if n else 0.0
    return {"fp_per_image": round(per_image, 3), "total_fp": total_fp, "num_images": n}


def main():
    parser = argparse.ArgumentParser(description="Compare old vs new PCB model")
    parser.add_argument("--old", default=str(PROJECT_ROOT / "backend" / "models" / "best.pt"))
    parser.add_argument("--new", default=str(PROJECT_ROOT / "runs" / "finetune_m1" / "weights" / "best.pt"))
    parser.add_argument("--data", default=str(PROJECT_ROOT / "dataset_merged" / "data.yaml"))
    parser.add_argument("--negatives", default=str(PROJECT_ROOT / "test_images"))
    args = parser.parse_args()

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Validating on {args.data} (device={device})...\n")
    old = val_metrics(args.old, args.data, device=device)
    print(f"OLD {Path(args.old).name}: {json.dumps(old, indent=2)}")
    new_path = Path(args.new)
    if new_path.exists():
        new = val_metrics(str(new_path), args.data, device=device)
        print(f"\nNEW {new_path.name}: {json.dumps(new, indent=2)}")
        print(f"\nDelta mAP50: {new['mAP50'] - old['mAP50']:+.4f}, "
              f"mAP50-95: {new['mAP50_95'] - old['mAP50_95']:+.4f}, "
              f"precision: {new['precision'] - old['precision']:+.4f}")
    else:
        print(f"\nNEW model not found at {new_path} — run scripts/finetune_m1.py first.")

    print(f"\nFalse-positive spot-check on defect-free boards in {args.negatives} (conf=0.35):")
    print(f"  OLD: {fpr_on_negatives(args.old, args.negatives, conf=0.35)} FP/image")
    if new_path.exists():
        print(f"  NEW: {fpr_on_negatives(str(new_path), args.negatives, conf=0.35)} FP/image")


if __name__ == "__main__":
    main()
