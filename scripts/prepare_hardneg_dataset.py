"""
Build hard-negative dataset for accuracy fine-tune (M1).

Base: dataset_merged (Huang 2019, 727 images, 34 negatives, 2953 boxes).
Adds hard negatives that trigger the worst false positives:
  - test_images/pcb13.png  (B&W board, every via -> Missing Hole 0.94)
  - test_images/pcb18.png  (B&W coupon, every pad -> Spurious 0.94)
  - test_images/pcb10.jpg  (green normal, tiles FP)
Full images + random defect-free crops (640px) with EMPTY labels.

Output: dataset_hardneg/{train,valid,test} YOLO, 6 classes index-stable
with best_original.pt (0 Open Circuit ... 5 Missing Hole).

Usage:
  PYTHONPATH=backend/venv/lib/python3.11/site-packages \
    python3 scripts/prepare_hardneg_dataset.py
"""

import random
import shutil
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "dataset_merged"
OUT = PROJECT_ROOT / "dataset_hardneg"

HARD_BOARDS = [
    PROJECT_ROOT / "test_images" / "pcb13.png",
    PROJECT_ROOT / "test_images" / "pcb18.png",
    PROJECT_ROOT / "test_images" / "pcb10.jpg",
]


def save_resized(src, dst, max_dim=640):
    img = cv2.imread(str(src))
    if img is None:
        raise FileNotFoundError(src)
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    random.seed(42)
    print(f"Source: {SRC}")
    if OUT.exists():
        shutil.rmtree(OUT)
    print(f"Copying positives to {OUT} ...")
    shutil.copytree(SRC, OUT)
    # Remove cache + temp leftovers from copy
    for p in OUT.rglob("*.cache"):
        p.unlink()

    # Count before
    for split in ["train", "valid", "test"]:
        n = len(list((OUT / split / "images").glob("*")))
        print(f"  {split} before: {n}")

    # Add hard negatives: full boards + crops
    rng = random.Random(42)
    added = 0
    for board in HARD_BOARDS:
        if not board.exists():
            print(f"Missing {board}, skip")
            continue
        # Full board -> train (model must see whole-board as background)
        dst = OUT / "train" / "images" / f"hardneg_{board.name}.jpg"
        save_resized(board, dst)
        open(OUT / "train" / "labels" / f"hardneg_{board.stem}.txt", "w").close()
        added += 1
        # 8 random 640 crops (defect-free regions) -> train
        img = cv2.imread(str(board))
        H, W = img.shape[:2]
        crop = 640
        pieces = 0
        tries = 0
        while pieces < 8 and tries < 30:
            tries += 1
            if H < crop or W < crop:
                patch = img
            else:
                x0 = rng.randint(0, W - crop)
                y0 = rng.randint(0, H - crop)
                patch = img[y0:y0 + crop, x0:x0 + crop]
            name = f"hardneg_{board.stem}_crop{pieces}.jpg"
            cv2.imwrite(str(OUT / "train" / "images" / name), patch, [cv2.IMWRITE_JPEG_QUALITY, 90])
            open(OUT / "train" / "labels" / f"{Path(name).stem}.txt", "w").close()
            pieces += 1
            added += 1

    print(f"Added {added} hard-negative images to train.")

    # Also add 1 full board to valid (for FPR monitoring, not training eval)
    for board in HARD_BOARDS[:1]:
        dst = OUT / "valid" / "images" / f"hardneg_{board.name}.jpg"
        save_resized(board, dst)
        open(OUT / "valid" / "labels" / f"hardneg_{board.stem}.txt", "w").close()
    print("Added 1 hard-negative to valid for FPR monitoring.")

    # Rewrite data.yaml with model-identical names (spaces) to keep
    # predictor colors / display correct after promotion.
    yaml = (
        f"path: {OUT}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "\n"
        "names:\n"
        "  0: Open Circuit\n"
        "  1: Short Circuit\n"
        "  2: Mouse Bite\n"
        "  3: Spur\n"
        "  4: Spurious Copper\n"
        "  5: Missing Hole\n"
    )
    (OUT / "data.yaml").write_text(yaml)

    for split in ["train", "valid", "test"]:
        imgs = list((OUT / split / "images").glob("*"))
        negs = sum(1 for f in (OUT / split / "labels").glob("*.txt") if f.stat().st_size == 0)
        print(f"  {split}: {len(imgs)} images ({negs} defect-free)")
    print(f"data.yaml -> {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
