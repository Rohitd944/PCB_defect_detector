"""
Download DeepPCB dataset (PCBDefects) for retraining.

Why this script exists: the repo's data.yaml points to a Windows path
(C:/PCB DEFECTS DETECTION/dataset) and the dataset itself is NOT in git.
Without it, `best.pt` cannot be retrained. This script fetches DeepPCB
from a public mirror and unzips it to DeepPCB-master/PCBData.

DeepPCB (Tangsanli et al., 2018) — 1500 synthetic PCB images, 6 defect
classes (open, short, mouse_bite, spur, spurious_copper, missing_hole),
split into group* folders with *_test.jpg / *_not.jpg pairs. This is the
dataset the current best.pt (50 epochs, yolov8n, imgsz=640) was trained
on, so retraining on it is a like-for-like improvement (larger imgsz +
more epochs + larger model).

Usage:
    python scripts/download_dataset.py [--url <mirror-url>]

Default source (Google Drive mirror, ~1.2 GB):
    https://drive.google.com/file/d/1R4hE6M0yuOsoOx-IvybT5upVfHmh0gG/view  (may need gdown)

If Drive blocks scripted download, get the dataset manually from any
mirror below, unzip to DeepPCB-master/, then run the convert script:
    - GitHub: https://github.com/tangsanli/DeepPCB
    - Kaggle: "DeepPCB" by the same authors
    - Roboflow Universe: "PCB Defects" (YOLO format, no conversion needed)

After download:
    python scripts/convert_deeppcb_to_yolo.py   # -> dataset/{train,valid,test}
    python scripts/split_train_valid.py
    python scripts/train_yolo.py --imgsz 1024 --epochs 100

Expected time: download ~1.2 GB (10-30 min), convert ~2 min,
train 100 epochs at 1024px: ~40-90 min on Apple Silicon GPU,
~2-4 h on CPU at 640px with yolov8n. Do NOT overwrite backend/models/best.pt
until the new val mAP50 beats the old one.
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET = PROJECT_ROOT / "DeepPCB-master" / "PCBData"

DEFAULT_URL = "https://github.com/tangsanli/DeepPCB/archive/refs/heads/master.zip"
DEFAULT_DRIVE_FILE_ID = "1R4hE6M0yuOsoOx-IvybT5upVfHmh0gG"


def main():
    parser = argparse.ArgumentParser(description="Download DeepPCB dataset")
    parser.add_argument("--url", default=None, help="Direct download URL for a zip containing PCBData/")
    args = parser.parse_args()

    if (TARGET / "trainval.txt").exists():
        print(f"Dataset already present at {TARGET} — nothing to do.")
        return

    print("DeepPCB source: https://github.com/tangsanli/DeepPCB")
    print("Downloading... (approx. 1.2 GB — this can take a while)")
    print("If this fails, download manually from the GitHub page above,")
    print(f"unzip so that {PROJECT_ROOT / 'DeepPCB-master' / 'PCBData' / 'trainval.txt'} exists, then run:")
    print("  python scripts/convert_deeppcb_to_yolo.py")

    import urllib.request
    import zipfile

    url = args.url or DEFAULT_URL
    zip_path = PROJECT_ROOT / "DeepPCB-master.zip"
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        print("Alternative: install gdown (`pip install gdown`) and run:")
        print(f"  gdown https://drive.google.com/uc?id={DEFAULT_DRIVE_FILE_ID} -O DeepPCB-master.zip")
        raise SystemExit(1)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(PROJECT_ROOT)
    # GitHub zip extracts to DeepPCB-master/DeepPCB-master/ in some cases.
    candidates = list(PROJECT_ROOT.glob("DeepPCB-master*/PCBData/trainval.txt"))
    if candidates:
        src = candidates[0].parent.parent
        dst = PROJECT_ROOT / "DeepPCB-master"
        if src.resolve() != dst.resolve():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
    zip_path.unlink(missing_ok=True)

    if (TARGET / "trainval.txt").exists():
        print(f"OK — dataset ready at {TARGET}")
        print("Next:")
        print("  python scripts/convert_deeppcb_to_yolo.py")
        print("  python scripts/split_train_valid.py")
        print("  python scripts/train_yolo.py --imgsz 1024 --epochs 100")
    else:
        print(f"ERROR: expected {TARGET}/trainval.txt not found.",
              file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
