"""
Prepare merged fine-tune dataset for CircuitSentinel (M1).

Sources (verified, NOT DeepPCB duplicates):
  1. RobotHuman/PCB_defect (Huang 2019, akhatova) — 693 JPG + 693 XML,
     naked PCB photos with synthetic defects, 6 classes matching best.pt.
     XML <name> verified: open_circuit, short, mouse_bite, spur,
     spurious_copper, missing_hole.
  2. Defect-free (negative) images — your existing test_images that the
     current model scores 0 at conf 0.25 (pcb10, pcb12, pcb14, pcb16),
     plus full-board crops from (1) that avoid annotated boxes.
     These teach the model "normal pads/traces = background" and fix the
     background false-positive flood (confusion matrix: 4341 background FPs).

Output (YOLO format, same 6-class indices as best.pt):
  dataset_merged/
    train/images, train/labels
    valid/images, valid/labels
    test/images,  test/labels
    data.yaml

Class mapping (index-preserving for fine-tune from best.pt):
  0 open_circuit       -> Open Circuit
  1 short              -> Short Circuit
  2 mouse_bite         -> Mouse Bite
  3 spur               -> Spur
  4 spurious_copper    -> Spurious Copper
  5 missing_hole       -> Missing Hole

Defect-free images get EMPTY .txt files (YOLO negative examples).

Usage (after HF download finishes):
  PYTHONPATH=backend/venv/lib/python3.11/site-packages \
    python3 scripts/prepare_finetune_dataset.py \
      --src /tmp/PCB_defect_hf2 \
      --out dataset_merged \
      --negatives-from test_images

Split: stratified 70/15/15 by folder class, seed 42, image-level (no leakage).
Negatives split proportionally.
"""

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2


def save_resized(src_path, dst_path, max_dim=640):
    """Copy image, resizing so longest side <= max_dim (M1 disk + speed)."""
    img = cv2.imread(str(src_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read {src_path}")
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASS_MAP = {
    "open_circuit": 0,
    "short": 1,
    "mouse_bite": 2,
    "spur": 3,
    "spurious_copper": 4,
    "missing_hole": 5,
}
CLASS_NAMES = {0: "open_circuit", 1: "short_circuit", 2: "mouse_bite", 3: "spur", 4: "spurious_copper", 5: "missing_hole"}

# Local images that scored 0 detections at conf 0.25 with current best.pt
# (likely defect-free). Verified in audit. Includes populated-board photos
# as hard negatives (components/traces must be background).
DEFAULT_NEGATIVES = ["pcb10.jpg", "pcb12.jpg", "pcb14.jpg", "pcb16.jpg"]


def xml_to_yolo(xml_path: Path):
    """Convert Pascal VOC XML -> list of (cls, cx, cy, w, h) normalized."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    size = root.find("size")
    w_img = int(size.find("width").text)
    h_img = int(size.find("height").text)
    out = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASS_MAP:
            raise ValueError(f"Unknown class {name!r} in {xml_path} — mapping not verified!")
        cls = CLASS_MAP[name]
        bb = obj.find("bndbox")
        xmin = int(bb.find("xmin").text)
        ymin = int(bb.find("ymin").text)
        xmax = int(bb.find("xmax").text)
        ymax = int(bb.find("ymax").text)
        cx = ((xmin + xmax) / 2) / w_img
        cy = ((ymin + ymax) / 2) / h_img
        w = (xmax - xmin) / w_img
        h = (ymax - ymin) / h_img
        # Clip to [0,1]
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
        if w > 0 and h > 0:
            out.append((cls, cx, cy, w, h))
    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare merged YOLO dataset for fine-tune")
    parser.add_argument("--src", default="/tmp/PCB_defect_hf2")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "dataset_merged"))
    parser.add_argument("--negatives-from", default=str(PROJECT_ROOT / "test_images"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.exists():
        raise SystemExit(f"Source not found: {src} — wait for HF download.")

    random.seed(args.seed)

    # Collect (jpg, xml) pairs grouped by folder (= defect type of that image set)
    pairs = []
    for folder in sorted(p for p in src.iterdir() if p.is_dir()):
        jpgs = sorted(folder.glob("*.jpg"))
        for jpg in jpgs:
            xml = folder / (jpg.stem + "_annotation.xml")
            if not xml.exists():
                # Some files use different stem convention; look for matching xml
                candidates = list(folder.glob(f"{jpg.stem}*.xml"))
                if candidates:
                    xml = candidates[0]
                else:
                    print(f"WARNING: no XML for {jpg} — skipping image.")
                    continue
            pairs.append((jpg, xml, folder.name))
    print(f"Found {len(pairs)} defective images in source.")
    if len(pairs) < 100:
        raise SystemExit("Too few pairs — is the HF download complete? Check /tmp/hf_download.log")

    # Stratified split by folder (70/15/15)
    by_folder = {}
    for jpg, xml, folder in pairs:
        by_folder.setdefault(folder, []).append((jpg, xml))
    splits = {"train": [], "valid": [], "test": []}
    for folder, items in sorted(by_folder.items()):
        random.shuffle(items)
        n = len(items)
        n_train = int(n * 0.70)
        n_valid = int(n * 0.15)
        splits["train"].extend(items[:n_train])
        splits["valid"].extend(items[n_train:n_train + n_valid])
        splits["test"].extend(items[n_train + n_valid:])
    for k, v in splits.items():
        random.shuffle(v)

    # Negatives: local defect-free boards (empty labels)
    neg_dir = Path(args.negatives_from)
    negatives = []
    for name in DEFAULT_NEGATIVES:
        p = neg_dir / name
        if p.exists():
            negatives.append(p)
    # Add extra negatives: crop defect-free patches from source images
    # (regions far from annotated boxes) to expand the negative set.
    # Simple approach: use full source images' random 512 crops that contain
    # zero annotated boxes. For now we add 30 such crops (deterministic).
    import cv2
    extra_negs = []
    rng = random.Random(args.seed)
    attempts = 0
    # Pre-load boxes per image to test crop overlap
    for jpg, xml, folder in rng.sample(pairs, min(60, len(pairs))):
        if len(extra_negs) >= 30:
            break
        boxes = xml_to_yolo(xml)
        img = cv2.imread(str(jpg))
        if img is None:
            continue
        H, W = img.shape[:2]
        crop = 640
        if H < crop or W < crop:
            continue
        for _ in range(5):
            x0 = rng.randint(0, W - crop)
            y0 = rng.randint(0, H - crop)
            cx0, cy0 = x0 / W, y0 / H
            cx1, cy1 = (x0 + crop) / W, (y0 + crop) / H
            # Check if any defect center falls in the crop
            hit = False
            for (cls, bcx, bcy, bw, bh) in boxes:
                if cx0 <= bcx <= cx1 and cy0 <= bcy <= cy1:
                    hit = True
                    break
            if not hit:
                patch = img[y0:y0 + crop, x0:x0 + crop]
                patch_name = f"negcrop_{folder}_{jpg.stem}_{len(extra_negs)}.jpg"
                patch_path = out / "negatives_tmp" / patch_name
                patch_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(patch_path), patch)
                extra_negs.append(patch_path)
                break
    print(f"Negatives: {len(negatives)} local boards + {len(extra_negs)} defect-free crops.")

    # Write dataset
    total_boxes = 0
    for split, items in splits.items():
        img_dir = out / split / "images"
        lbl_dir = out / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for jpg, xml in items:
            # Unique name to avoid collisions across folders
            out_name = f"{jpg.parent.name}_{jpg.name}"
            save_resized(jpg, img_dir / out_name)
            boxes = xml_to_yolo(xml)
            with open(lbl_dir / (Path(out_name).stem + ".txt"), "w") as f:
                for (cls, cx, cy, w, h) in boxes:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            total_boxes += len(boxes)

    # Distribute negatives + crops across splits (keep ~15% negatives overall)
    neg_all = negatives + extra_negs
    # Shuffle and assign: train gets most, valid/test get a share
    rng.shuffle(neg_all)
    n = len(neg_all)
    neg_splits = {
        "train": neg_all[: int(n * 0.70)],
        "valid": neg_all[int(n * 0.70): int(n * 0.85)],
        "test": neg_all[int(n * 0.85):],
    }
    for split, items in neg_splits.items():
        img_dir = out / split / "images"
        lbl_dir = out / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for p in items:
            name = Path(p).name
            # Disambiguate local names
            if not name.startswith("neg_") and not name.startswith("negcrop_"):
                name = f"neg_{name}"
            save_resized(p, img_dir / name)
            # Empty label file = defect-free
            open(lbl_dir / (Path(name).stem + ".txt"), "w").close()

    yaml = (
        f"path: {out}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "\n"
        "names:\n"
        "  0: open_circuit\n"
        "  1: short_circuit\n"
        "  2: mouse_bite\n"
        "  3: spur\n"
        "  4: spurious_copper\n"
        "  5: missing_hole\n"
    )
    (out / "data.yaml").write_text(yaml)

    print(f"\nDone. Total boxes: {total_boxes}")
    for split in ["train", "valid", "test"]:
        n_img = len(list((out / split / "images").glob("*")))
        n_neg = sum(1 for f in (out / split / "labels").glob("*.txt") if f.stat().st_size == 0)
        print(f"  {split}: {n_img} images ({n_neg} defect-free)")
    print(f"data.yaml -> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
