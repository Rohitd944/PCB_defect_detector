"""
Fresh YOLOv8n training on the merged PCB-defects dataset — Apple Silicon (MPS).

This reproduces the deployed checkpoint (`backend/models/best.pt`): it starts
from COCO-pretrained `yolov8n.pt` (NOT from the old DeepPCB weights) and runs
on `dataset_merged/data.yaml`.

Why a separate script from `finetune_m1.py`?
  - `finetune_m1.py` continues from `backend/models/best.pt` (small LR, frozen
    backbone). It is for nudging an existing model.
  - This script trains a model from the generic COCO backbone, which is what
    produced the 0.889 mAP50 checkpoint. Use it when the domain changes a lot
    (DeepPCB B&W traces -> Huang colour microscope crops).

M1 settings that avoid RAM blow-up / overheating:
  imgsz=640, batch=8, workers=2, cache='ram', amp=False, save_period=10.

Usage:
    python3 scripts/train_fresh_m1.py
    python3 scripts/train_fresh_m1.py --epochs 60 --batch 8
    python3 scripts/train_fresh_m1.py --resume   # continue runs/<name>/weights/last.pt
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "dataset_merged" / "data.yaml"
BASE_WEIGHTS = PROJECT_ROOT / "yolov8n.pt"


def main():
    parser = argparse.ArgumentParser(description="Fresh YOLOv8n training on M1 (MPS)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--model", default=str(BASE_WEIGHTS),
                        help="Starting weights (default: local yolov8n.pt, else downloads)")
    parser.add_argument("--project", default=str(PROJECT_ROOT / "runs"))
    parser.add_argument("--name", default="akhatova_v2")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit(
            f"Merged dataset not found at {DATA_YAML}.\n"
            "Build it first, e.g.:\n"
            "  python3 scripts/prepare_finetune_dataset.py "
            "--src /path/to/PCB_defect --out dataset_merged"
        )

    import torch
    from ultralytics import YOLO

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device} (mps_available={torch.backends.mps.is_available()})", flush=True)
    if device == "cpu":
        print("WARNING: MPS not available — training will be very slow on CPU.")

    model = YOLO(args.model if Path(args.model).exists() else "yolov8n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        lr0=args.lr0,
        patience=args.patience,
        freeze=0,
        project=args.project,
        name=args.name,
        resume=args.resume,
        cache="ram",
        amp=False,          # MPS AMP is unstable on torch 2.12
        save_period=10,
        save=True,
        plots=True,
        val=True,
        mosaic=0.5,         # mosaic=1.0 blurs tiny defects; 0.5 is a good compromise
        hsv_h=0.015,
        hsv_s=0.4,          # keep copper colour a usable cue
        hsv_v=0.3,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        flipud=0.0,
        close_mosaic=10,
        cos_lr=False,
    )
    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
