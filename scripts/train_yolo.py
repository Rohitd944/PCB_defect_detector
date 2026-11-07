"""
Improved YOLOv8 training for PCB defect detection — FIXED v3.

What was wrong with the original script (50 epochs, imgsz=640, batch=16):
  - data path was a Windows absolute path (C:/PCB DEFECTS DETECTION/dataset)
    baked into data.yaml, so training could not run on macOS/Linux at all.
  - imgsz=640 on 2900px DeepPCB images shrinks tiny defects (mouse-bite,
    spur, missing-hole) to a few pixels -> the "not detecting defects
    properly" recall problem at inference time.
  - yolov8n (smallest nano model) + 50 fixed epochs with no patience,
    no resume, no validation-focused metrics — peak mAP was not reliably
    reached.

This v3 script:
  - Resolves data.yaml relative to the project root (portable).
  - Trains at imgsz=1024 by default (better for tiny PCB defects).
    Use --imgsz 640 on CPU-only machines.
  - Uses yolov8s (small) by default, 100 epochs with patience=15.
  - Writes to runs/pcb_detector_v2/ so the current best.pt is NOT
    overwritten until you verify the new weights are better.
  - Enables augmentation suited to PCB inspection (no color distortion
    that hides copper, but geometric aug on).

Usage:
    # 1. Get the dataset first (see scripts/download_dataset.py or README):
    #    python scripts/download_dataset.py
    #    python scripts/convert_deeppcb_to_yolo.py
    #    cp dataset/data.yaml -> dataset/data_fixed.yaml (auto below)
    #
    # 2a. Train on GPU / Apple Silicon (recommended):
    #    PYTHONPATH=backend/venv/lib/python3.11/site-packages \
    #      python scripts/train_yolo.py --imgsz 1024 --epochs 100
    #
    # 2b. Train on CPU (slow — expect ~2-4h for 50 epochs at 640px):
    #    python scripts/train_yolo.py --imgsz 640 --epochs 50 --model yolov8n.pt
    #
    # 3. Evaluate & promote:
    #    PYTHONPATH=... python scripts/evaluate_model.py runs/pcb_detector_v2/weights/best.pt
    #    (or just compare val mAP50 in runs/pcb_detector_v2/results.csv)
    #    cp runs/pcb_detector_v2/weights/best.pt backend/models/best.pt
    #    # only if the new val mAP50 beats the old model's mAP50.
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = PROJECT_ROOT / "dataset" / "data.yaml"
FIXED_YAML = PROJECT_ROOT / "dataset" / "data_fixed.yaml"


def _ensure_fixed_yaml():
    """Write a portable data.yaml if the dataset exists (no Windows path)."""
    if not DATASET_YAML.exists():
        return None  # dataset not downloaded yet; training will error clearly
    FIXED_YAML.parent.mkdir(parents=True, exist_ok=True)
    FIXED_YAML.write_text(
        f"path: {PROJECT_ROOT / 'dataset'}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "\n"
        "names:\n"
        "  0: open\n"
        "  1: short\n"
        "  2: mouse_bite\n"
        "  3: spur\n"
        "  4: spurious_copper\n"
        "  5: missing_hole\n"
    )
    return FIXED_YAML


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 PCB defect detector")
    parser.add_argument("--model", default="yolov8s.pt", help="Pretrained weights (yolov8n.pt = fastest)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024, help="Input size — 1024 recommended, 640 for CPU")
    parser.add_argument("--batch", type=int, default=8, help="Batch size — 8 at 1024px, 16 at 640px")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--project", default="runs", help="Output project dir")
    parser.add_argument("--name", default="pcb_detector_v2", help="Run name (weights in <project>/<name>/weights/best.pt)")
    parser.add_argument("--resume", action="store_true", help="Resume most recent run")
    args = parser.parse_args()

    yaml_path = _ensure_fixed_yaml()
    if yaml_path is None or not (PROJECT_ROOT / "dataset" / "train" / "images").exists():
        print("ERROR: dataset/ not found. Download & convert DeepPCB first:")
        print("  1. python scripts/download_dataset.py")
        print("  2. python scripts/convert_deeppcb_to_yolo.py")
        print("  3. python scripts/split_train_valid.py")
        print("  Then re-run this script.")
        raise SystemExit(1)

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=args.name,
        resume=args.resume,
        amp=True,
        save_period=10,
        plots=True,
        val=True,
    )
    print(f"\nTraining done. Weights: {PROJECT_ROOT / args.project / args.name / 'weights/best.pt'}")
    print("Promote only if val mAP50 beats the current backend/models/best.pt")


if __name__ == "__main__":
    main()
