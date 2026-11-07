"""
Fine-tune YOLOv8n on merged PCB dataset — Apple M1 (MPS) optimized.

Preserves the original DeepPCB model; fine-tunes (not trains from scratch)
with additional compatible data + defect-free negatives to fix:
  - background false positives (4341 background errors in confusion matrix)
  - wrong locations (640px downscale on 2000px+ boards)
  - wrong classes (Open<->Mouse Bite, Short<->Spur)

M1 8GB constraints:
  - device='mps' (torch 2.12.1, verified available)
  - imgsz=640 (keeps model's native scale; 1024 OOMs at batch 8 on 8GB)
  - batch=8, workers=2, cache='ram', amp=False (MPS AMP unstable)
  - lr0=0.002 (fine-tune, below original 0.01), freeze backbone 10 epochs
  - save_period=5 checkpoints + resume support
  - val every epoch, plots saved

Outputs (original preserved separately):
  backend/models/best_original.pt  (backup of original DeepPCB best.pt)
  runs/finetune_m1/weights/best_improved.pt, last.pt
  runs/finetune_m1/results.csv     (per-epoch precision/recall/mAP)

Usage:
  PYTHONPATH=backend/venv/lib/python3.11/site-packages \
    python3 scripts/finetune_m1.py --epochs 30 --imgsz 640 --batch 8
  # Resume after interruption:
  python3 scripts/finetune_m1.py --resume

Note: dataset_merged/ must exist first:
  python3 scripts/prepare_finetune_dataset.py --src /tmp/PCB_defect_hf2
"""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = PROJECT_ROOT / "backend" / "models" / "best.pt"
BACKUP = PROJECT_ROOT / "backend" / "models" / "best_original.pt"
DATA_YAML = PROJECT_ROOT / "dataset_merged" / "data.yaml"


def ensure_backup():
    if not ORIGINAL.exists():
        raise SystemExit(f"Original model not found: {ORIGINAL}")
    if not BACKUP.exists():
        shutil.copy(ORIGINAL, BACKUP)
        print(f"Backed up original -> {BACKUP}")
    else:
        print(f"Backup already exists -> {BACKUP}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n on M1 (MPS)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr0", type=float, default=0.002)
    parser.add_argument("--freeze", type=int, default=10, help="Freeze backbone for N epochs")
    parser.add_argument("--project", default=str(PROJECT_ROOT / "runs"))
    parser.add_argument("--name", default="finetune_m1")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit(
            f"Merged dataset not found at {DATA_YAML}. Run first:\n"
            "  PYTHONPATH=backend/venv/lib/python3.11/site-packages "
            "python3 scripts/prepare_finetune_dataset.py --src /tmp/PCB_defect_hf2"
        )
    ensure_backup()

    import torch
    from ultralytics import YOLO

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device} (mps_available={torch.backends.mps.is_available()})")
    if device == "cpu":
        print("WARNING: MPS not available — training will be very slow on CPU.")

    # Fine-tune FROM the existing DeepPCB weights (not from scratch)
    model = YOLO(str(ORIGINAL))
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        lr0=args.lr0,
        patience=args.patience,
        freeze=args.freeze,
        project=args.project,
        name=args.name,
        resume=args.resume,
        cache="ram",
        amp=False,  # MPS AMP is unstable on torch 2.12
        save_period=5,
        save=True,
        plots=True,
        val=True,
        mosaic=0.5,  # lower than default 1.0 — mosaic hurts tiny-defect localization
        hsv_h=0.015,
        hsv_s=0.4,  # reduce color distortion — copper color is a cue
        hsv_v=0.3,
        degrees=2.0,  # small geometric jitter only (boards are axis-aligned)
        translate=0.05,
        scale=0.2,
        fliplr=0.5,
        flipud=0.0,
    )
    print(f"\nFine-tune done. Compare:")
    print(f"  old: {ORIGINAL}")
    print(f"  new: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    print("Run scripts/evaluate_compare.py to compare precision/recall/F1/mAP + FPR.")


if __name__ == "__main__":
    main()
