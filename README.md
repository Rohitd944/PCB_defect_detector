# 🔍 CircuitSentinel — AI-Powered PCB Defect Detection

> End-to-end PCB quality inspection: upload a board photo, get defect boxes, a class for each defect, confidence scores, an annotated image and a downloadable PDF inspection report.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![YOLOv8](https://img.shields.io/badge/YOLOv8-nano-FF6F00)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

**CircuitSentinel** is a full-stack PCB inspection system built around a fine-tuned **YOLOv8n** detector. It detects six common PCB manufacturing defects directly from an image — no special lighting rig or camera required.

The project includes:

- a **React + Vite** single-page inspection workspace (upload or webcam capture),
- a **FastAPI** inference API with tuned post-processing,
- a **hybrid reference-comparison mode** that verifies YOLO detections against a known-good board (ORB/RANSAC alignment + SSIM/structural diff),
- a reproducible **training pipeline** (dataset preparation, Apple-Silicon MPS fine-tuning, evaluation),
- **PDF report generation** and annotated-image export.

The detector was retrained from scratch on the **PCB Defects dataset** (Huang, 2019 / `akhatova/pcb-defects` on Kaggle) and reaches **mAP50 ≈ 0.89** on its validation split — versus **0.00 mAP50** for the original DeepPCB-only checkpoint on that data.

---

## ✨ Features

- 📤 **Upload or camera capture** — inspect an existing photo or a live webcam frame.
- 🤖 **Six-class defect detection** — Open Circuit, Short Circuit, Mouse Bite, Spur, Spurious Copper, Missing Hole.
- 🎯 **Tuned post-processing** — per-class confidence thresholds, area/aspect filters, class-aware NMS and density caps to suppress false positives.
- 📊 **Inspection summary** — defect count, peak/average confidence, class distribution, PASS / MANUAL INSPECTION / FAIL verdict.
- 🖼️ **Annotated result image** — every kept detection drawn on the board.
- 📄 **PDF report** — inspection ID, timestamp, statistics, AI recommendation and detection table.
- 🎚️ **Confidence control** — Sensitive / Balanced / Strict presets in the UI.
- 🍎 **Apple Silicon ready** — training and inference run on MPS (no NVIDIA GPU required).
- 🧬 **Hybrid reference compare** — upload a known-good board and a testing board; ORB/RANSAC alignment and SSIM/structural diff verify each YOLO detection.
- ✅ **PASS / FAIL / REVIEW verdict** — confirmed defects fail the board, unmatched structural differences go to manual review, clean boards pass even when YOLO raises a false positive.
- 🗺️ **Difference heatmap + side-by-side comparison** — annotated testing image, aligned comparison and JET heatmap are generated for every inspection.
- 🎚️ **Difference sensitivity control** — ignore lighting/rotation/scale/pixel noise while still catching 40–100 px defects.

---

## 📸 Screenshots

| Home | Detection result |
|---|---|
| ![Home](screenshots/home1.png) | ![Detection](screenshots/detection.png) |

| Inspection report | PDF export |
|---|---|
| ![Report](screenshots/report1.png) | ![PDF](screenshots/pdf.png) |

---

## 🧠 Model

| Property | Value |
|---|---|
| Architecture | YOLOv8n (Ultralytics) |
| Parameters | ~3.0 M |
| Input size | 640 × 640 |
| Classes | 6 |
| Training data | PCB Defects dataset (Huang, 2019) — 693 annotated images |
| Training hardware | Apple M1 (MPS) |
| Weights | `backend/models/best.pt` (6 MB) |

### Validation metrics (best checkpoint, epoch 45)

| Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **0.904** | **0.865** | **0.882** | **0.889** | **0.457** |

For comparison, the previous DeepPCB-only checkpoint scored **mAP50 0.000** on this validation split (it never saw these defect scales/textures).

### Defect classes

| ID | Class | Description |
|---|---|---|
| 0 | Open Circuit | Broken / interrupted copper trace |
| 1 | Short Circuit | Unwanted connection between traces |
| 2 | Mouse Bite | Bite-shaped notch on a trace edge |
| 3 | Spur | Thin unwanted copper protrusion |
| 4 | Spurious Copper | Isolated copper remnant |
| 5 | Missing Hole | Missing through-hole / via |

> **Class-name normalisation:** training configs sometimes use snake_case (`open_circuit`), while older weights use display names (`Open Circuit`). `backend/predictor.py` normalises both so thresholds, colours and API responses stay consistent.

---

## 🏗️ Architecture

```text
                 PCB image (upload / webcam)
                            │
                            ▼
                 React + Vite frontend
                            │  POST /predict (multipart)
                            ▼
                    FastAPI backend
                            │
                            ▼
              YOLOv8n inference (imgsz 640)
                            │
                            ▼
        Post-processing: per-class thresholds,
        area/aspect filters, class-aware NMS, caps
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
     Annotated image               Inspection analysis
     (/results/...)                (counts, verdict, PDF)
```

### Hybrid reference comparison

```text
     Reference PCB (known good)          Testing PCB
              │                              │
              └──────────────┬───────────────┘
                             ▼
                  ORB features + Lowe ratio test
                             │
                             ▼
              RANSAC homography (test → reference)
                             │
                   alignment quality gate
              (matches / inliers / reproj / scale / overlap)
                             │
                             ▼
          Exposure normalisation + Gaussian blur
                             │
              ┌──────────────┴───────────────┐
              ▼                              ▼
     SSIM + absolute difference        Structural diff regions
              │                              │
              └──────────────┬───────────────┘
                             ▼
                 YOLO ↔ diff verification engine
          (confirmed / rejected / unverified / anomaly)
                             │
                             ▼
              PASS / FAIL / REVIEW + annotated images
```

The YOLO model still decides *what* a defect is; the reference comparison decides whether the board actually changed. A YOLO box that overlaps a real structural difference is **confirmed**, one with no structural support is **rejected** (model false positive), and an unmatched difference is an **anomaly** for manual review.

---

## 📂 Project structure

```text
CircuitSentinel/
├── backend/
│   ├── app.py                  # FastAPI app + /predict, /reference, /inspect/hybrid
│   ├── predictor.py            # YOLO loading, inference, post-processing
│   ├── services/
│   │   ├── analysis.py    # Verdict, stats, recommendations
│   │   ├── alignment.py   # ORB/RANSAC alignment + quality gates
│   │   ├── diff_detection.py   # SSIM, structural diff, heatmap
│   │   ├── hybrid_inspection.py # YOLO + reference fusion, PASS/FAIL/REVIEW
│   │   └── reference_store.py  # Known-good reference persistence
│   ├── models/
│   │   ├── best.pt             # deployed model (tracked)
│   │   └── best_original.pt    # previous model backup (git-ignored)
│   ├── results/                # generated annotated images
│   ├── uploads/                # uploaded inputs
│   ├── references/             # saved known-good reference boards (git-ignored)
│   ├── requirements.txt
│   └── test_predictor.py
├── frontend/
│   ├── src/
│   │   ├── components/         # workspace, report panel, etc.
│   │   ├── pages/              # Home
│   │   └── services/api.js
│   ├── package.json
│   └── vite.config.js
├── scripts/
│   ├── download_dataset.py         # fetch / verify the Kaggle dataset
│   ├── prepare_finetune_dataset.py # XML → YOLO, resize, negatives
│   ├── train_fresh_m1.py           # fresh training (deployed checkpoint)
│   ├── finetune_m1.py              # Apple-Silicon fine-tuning
│   ├── evaluate_compare.py         # metrics + old-vs-new comparison
│   ├── prepare_hardneg_dataset.py  # hard-negative dataset builder
│   ├── train_yolo.py               # generic training entry point
│   └── convert_deeppcb_to_yolo.py
├── dataset_merged/             # generated training data (git-ignored)
├── test_images/                # sample boards
├── evaluation/                 # PR/F1/confusion plots
├── screenshots/
├── data.yaml
├── dockerfile.yaml
├── README.md
└── LICENSE
```

---

## 🚀 Quick start

### Prerequisites

- Python **3.11+**
- Node.js **18+** and npm
- (Recommended) Apple Silicon Mac, or any CPU — CUDA is optional

### 1. Clone

```bash
git clone https://github.com/Rohitd944/PCB_defect_detector.git
cd PCB_defect_detector
```

### 2. Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

Backend: <http://127.0.0.1:8000> · Swagger UI: <http://127.0.0.1:8000/docs>

### 3. Frontend

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Frontend: <http://127.0.0.1:5173>

> The frontend calls `http://localhost:8000` by default. If your backend runs elsewhere, update the base URL in `frontend/src/services/api.js`.

---

## 🖥️ Usage

### Single-image inspection (YOLO)

1. Open the frontend and choose **Upload** (or **Camera**).
2. Select a PCB image (`.jpg`, `.jpeg`, `.png`).
3. Pick a confidence preset — *Balanced (25%)* is the default recommendation.
4. Click **Analyze PCB**.
5. Review the annotated image, detection list and inspection summary.
6. Download the **PDF report** or the **annotated image**.

### Reference comparison (hybrid)

1. Switch the mode toggle to **Reference Compare**.
2. In card **A · Reference**, upload or capture the known-good board. It is stored on the server and its ORB features are extracted immediately.
3. In card **B · Testing**, upload or capture the board under inspection.
4. Optionally adjust **YOLO Confidence** and **Difference Sensitivity**.
5. Click **Run Hybrid Inspection**.
6. Review the **PASS / FAIL / REVIEW** verdict, SSIM/difference metrics, annotated testing image, reference-vs-testing comparison and difference heatmap. The detection table shows which YOLO boxes were confirmed or rejected.

---

## 🔌 API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Welcome / service info |
| `GET` | `/health` | Health check (`model_loaded`, class count) |
| `GET` | `/model-info` | Model path and class map |
| `POST` | `/predict` | Run detection on an uploaded image |
| `POST` | `/reference` | Save a known-good reference PCB image |
| `GET` | `/reference` | List saved reference boards |
| `GET` | `/reference/latest` | Most recently saved reference board |
| `GET` | `/reference/{id}` | Metadata for one reference board |
| `POST` | `/inspect/hybrid` | YOLO + reference-comparison inspection |
| `GET` | `/results/{file}` | Static annotated images |
| `GET` | `/references/{file}` | Static reference board images |

### `POST /predict`

Multipart form fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `file` | file | — | JPG / JPEG / PNG |
| `confidence` | float | `0.25` | Clamped to 0.05–0.95 |
| `iou` | float | `0.40` | Clamped to 0.10–0.90 |

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@test_images/pcb17.jpg" \
  -F "confidence=0.25"
```

Response (trimmed):

```json
{
  "status": "success",
  "filename": "pcb17.jpg",
  "total_detections": 1,
  "detections": [
    { "type": "Missing Hole", "confidence": 0.836, "bbox": [x1, y1, x2, y2] }
  ],
  "result_image_url": "http://127.0.0.1:8000/results/...",
  "analysis": {
    "inspection_status": "MANUAL INSPECTION",
    "verdict": "DEFECTIVE",
    "distribution": { "Missing Hole": 1 }
  },
  "confidence_threshold": 0.25
}
```

### `POST /inspect/hybrid`

Inspect a testing board against a known-good reference.

Multipart form fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `file` | file | — | Testing PCB (JPG / JPEG / PNG) |
| `reference` | file | — | Optional reference upload (saved automatically) |
| `reference_id` | string | latest | ID returned by `POST /reference` |
| `confidence` | float | `0.25` | YOLO threshold, clamped 0.05–0.95 |
| `iou` | float | `0.40` | NMS IoU, clamped 0.10–0.90 |
| `diff_threshold` | float | `18.0` | Lower = more sensitive, clamped 6–60 |

Example:

```bash
# 1) save the known-good board
curl -X POST http://127.0.0.1:8000/reference \
  -F "file=@test_images/pcb2.jpg" -F "name=pcb2 golden"

# 2) inspect a testing board against it
curl -X POST http://127.0.0.1:8000/inspect/hybrid \
  -F "file=@test_images/pcb4.jpg" \
  -F "reference_id=<id from step 1>" \
  -F "confidence=0.25" \
  -F "diff_threshold=18"
```

Response (trimmed):

```json
{
  "status": "success",
  "inspection_id": "PCB-HY-20261107-180456-2492",
  "verdict": "FAIL",
  "verdict_reason": "5 defect(s) confirmed by reference comparison.",
  "confidence_level": "high",
  "alignment": {
    "quality": "good",
    "inliers": 3137,
    "inlier_ratio": 0.999,
    "rotation_deg": 0.0,
    "scale": 1.0,
    "coverage": 1.0
  },
  "diff": {
    "ssim": 0.9987,
    "difference_score": 44.8,
    "changed_pixel_pct": 0.099,
    "region_count": 10
  },
  "summary": {
    "total_detections": 5,
    "confirmed": 5,
    "rejected": 0,
    "unverified": 0,
    "anomalies": 5
  },
  "detections": [
    {
      "type": "Missing Hole",
      "confidence": 0.81,
      "verification": "confirmed",
      "reference_difference": 32.5,
      "location": { "x": 85.7, "y": 20.3 }
    }
  ],
  "images": {
    "annotated_url": "http://127.0.0.1:8000/results/..._annotated.jpg",
    "comparison_url": "http://127.0.0.1:8000/results/..._comparison.jpg",
    "heatmap_url": "http://127.0.0.1:8000/results/..._heatmap.jpg"
  }
}
```

**Verdict policy**

| Verdict | Meaning |
|---|---|
| `FAIL` | At least one YOLO defect overlaps a real structural difference |
| `REVIEW` | Structural difference(s) without a YOLO match, a large global change, or unreliable alignment |
| `PASS` | Testing board matches the reference within tolerance (unsupported YOLO boxes are rejected) |

---

## 🧪 Training pipeline

The training data is **not committed** (keeps the repo lean). Rebuild it with the scripts below.

### 1. Get the dataset

Download **PCB Defects** (`akhatova/pcb-defects`) from Kaggle and place the extracted folder somewhere local, e.g. `/tmp/PCB_defect`. The folder should contain one sub-folder per class (`Missing_hole`, `Mouse_bite`, `Open_circuit`, `Short`, `Spur`, `Spurious_copper`) with paired `.jpg` + `.xml` files.

> `scripts/download_dataset.py` is the legacy helper for the older **DeepPCB** baseline, not this dataset.

### 2. Build a YOLO dataset

Converts Pascal-VOC XML → YOLO labels, resizes images to ≤ 640 px, and optionally adds defect-free negatives:

```bash
python3 scripts/prepare_finetune_dataset.py \
  --src /path/to/PCB_defect \
  --out dataset_merged \
  --negatives-from test_images
```

Output: `dataset_merged/{train,valid,test}/{images,labels}` + `data.yaml`.

### 3. Train on Apple Silicon (MPS)

**Fresh training** (this is how the deployed checkpoint was produced — starts from COCO `yolov8n.pt`, not the old DeepPCB weights):

```bash
python3 scripts/train_fresh_m1.py --epochs 60 --batch 8
```

**Fine-tuning** an existing checkpoint (lower LR, optional frozen backbone):

```bash
python3 scripts/finetune_m1.py --epochs 30 --batch 8 --lr0 0.002 --freeze 10
```

Both use `imgsz=640`, `device=mps` (falls back to CPU), `amp=False`, `workers=2`, `cache=ram` and checkpoint periodically so an 8 GB M1 stays stable.

### 4. Evaluate

```bash
python3 scripts/evaluate_compare.py \
  --old backend/models/best_original.pt \
  --new runs/akhatova_v2/weights/best.pt \
  --data dataset_merged/data.yaml
```

Reports Precision, Recall, F1, mAP50, mAP50-95 and false positives per image on a defect-free set.

---

## ⚙️ Inference post-processing

Raw YOLO output is filtered in `backend/predictor.py` to reduce false alarms:

1. **Per-class confidence floors** — Open/Missing `0.30`, Short/Spurious `0.35`, Mouse Bite/Spur `0.40`.
2. **Area filter** — reject boxes below `0.005%` or above `1.5%` of image area (dataset GT median ≈ `0.04–0.10%`).
3. **Aspect filter** — reject extreme slivers (`> 6:1`), which are usually trace edges.
4. **Class-aware NMS** — suppress same-class duplicates at IoU > 0.5 and cross-class overlaps at IoU > 0.7.
5. **Density caps** — at most 5 detections per class and 12 per board (dataset GT mean ≈ 4.3 boxes/image).
6. **Whole-image inference** — tiling is available but **off by default**; it amplified false positives in testing.

---

## 📊 Evaluation notes

| Evaluation set | Old (DeepPCB) | New (fine-tuned) |
|---|---|---|
| PCB Defects validation (107 images) | mAP50 0.000 | **mAP50 0.889** |
| Real green boards — defect-free (4 imgs) | **0 FP** | 1.5 FP/img |
| Real green boards — defective (5 imgs) | 4/5 detected | 4/5 detected |

The fine-tuned model is clearly stronger on microscope-style / Huang-domain boards. On full-board green PCB photos it is **more sensitive but also more prone to false positives** than the old checkpoint. A mixed fine-tune (green-board negatives + Huang positives) is the recommended next step before using it in production on this camera domain.

Confusion/PR/F1 curves are in [`evaluation/`](evaluation).

### Hybrid reference comparison — measured runs

All numbers below are from actual local runs on the images in `test_images/`, not estimates.

| Reference | Testing | Verdict | YOLO verification | SSIM | Diff regions |
|---|---|---|---|---|---|
| `pcb2.jpg` | `pcb4.jpg` | **FAIL** | 5/5 Missing Hole confirmed | 0.9987 | 10 |
| `pcb2.jpg` | `pcb5.jpg` | **FAIL** | 5/5 Missing Hole confirmed | 0.9986 | 10 |
| `pcb6.jpg` | `pcb7.jpg` | **FAIL** | 4/4 Open Circuit confirmed | 0.9991 | 10 |
| `pcb8.jpg` | `pcb9.jpg` | **FAIL** | 5/5 Missing Hole confirmed | 0.9986 | 8 |
| `pcb2.jpg` | `pcb2` rotated 3° + 1.05× + brightness +18 | **PASS** | 4 YOLO candidates rejected | 0.9893 | 0 |
| `pcb2.jpg` | `pcb10.jpg` (different board) | **REVIEW** | alignment failed (9 inliers, 0.14 ratio) | — | — |

Tolerance checks that produce **0 diff regions** on an otherwise identical board: JPEG quality 40, Gaussian noise σ≤6, 1–2 px translation, mild blur, ±3° rotation, 1.05× scale and ±18 brightness.

Typical runtime on an M1 (processing capped at 2000 px): **≈7–9 s** per hybrid inspection including YOLO inference, alignment, diff and image rendering.

---

## ⚠️ Limitations

- Trained on close-up PCB imagery; full-board photos with strong perspective/glare are out-of-distribution.
- Very small defects (< 10 px) remain hard at `imgsz=640`.
- Hybrid mode expects the reference and testing images to show the **same board**. Different boards, crops or camera angles are reported as `REVIEW` rather than guessed.
- Reference comparison is 2-D; it cannot verify defects hidden underneath components or on the far side of the board.
- No polygon/segmentation output — boxes only.
- Inspection verdicts are advisory; a human should confirm rejects.
- The dataset is third-party; check its license before commercial use.

---

## 🗺️ Roadmap

- [ ] Mixed-domain fine-tune (green boards as hard negatives)
- [ ] Defect segmentation masks
- [ ] Inspection history (SQLite/PostgreSQL) + analytics dashboard
- [ ] Batch upload and line-scan support
- [ ] ESP32-CAM / conveyor integration
- [ ] ONNX / CoreML export for edge deployment

---

## 🧰 Tech stack

**Frontend:** React 19, Vite, Axios, jsPDF, html2canvas, react-webcam, react-toastify
**Backend:** FastAPI, Uvicorn, OpenCV, Pillow, python-multipart
**ML:** Ultralytics YOLOv8, PyTorch 2.12 (MPS/CUDA/CPU), NumPy
**Tooling:** Git, Docker, VS Code

---

## 👨‍💻 Author

**Rohit Deshmukh**

- GitHub: [@Rohitd944](https://github.com/Rohitd944)
- Repository: [PCB_defect_detector](https://github.com/Rohitd944/PCB_defect_detector)

---

## 📜 License

Released under the **MIT License** — see [LICENSE](LICENSE).

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- PCB Defects dataset — Huang, 2019 (`akhatova/pcb-defects`, Kaggle)
- [FastAPI](https://fastapi.tiangolo.com/) · [React](https://react.dev/) · [OpenCV](https://opencv.org/)
- Original UI foundation by Anchal Sharma and Hardik Gupta

---

## ⭐ Support

If this project helped you, please give it a **star** on GitHub — it helps others find it.
