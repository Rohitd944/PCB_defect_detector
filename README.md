# PCB Defect Detection

AI-powered PCB quality inspection. Upload a board photo (or capture one with the camera) and a fine-tuned YOLOv8n model finds manufacturing defects, draws bounding boxes, and exports an annotated image plus a PDF report. A hybrid mode compares a testing board with a known-good reference to confirm each defect.

## Screenshots

| Home | Detection results |
|---|---|
| ![Home](screenshots/dashboard-home.png) | ![Detection results](screenshots/detection-results.png) |

## What it does

- **Single Image (YOLO)** — upload JPG/JPEG/PNG or use the webcam; adjust confidence with Sensitive / Balanced / Strict presets, then click **Analyze PCB**.
- **Reference Compare (HYBRID)** — upload a known-good board and a testing board; ORB/RANSAC alignment + SSIM difference verify each YOLO detection and return a **PASS / FAIL / REVIEW** verdict.
- Detects six classes: Open Circuit, Short Circuit, Mouse Bite, Spur, Spurious Copper, Missing Hole.
- Shows an annotated image, detection list, confidence scores and summary, with PDF report and image download.

## Tech stack

- **Frontend:** React + Vite, Axios, react-webcam, jsPDF
- **Backend:** FastAPI, Uvicorn, OpenCV, Pillow
- **Model:** Ultralytics YOLOv8n fine-tuned on the PCB Defects dataset (~0.89 mAP50), runs on Apple MPS / CPU

## Run locally

Backend:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

Frontend (in a second terminal):

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Open <http://127.0.0.1:5173>. The frontend talks to `http://localhost:8000`.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health check |
| `POST /predict` | Detect defects in an uploaded image |
| `POST /reference` | Save a known-good reference board |
| `POST /inspect/hybrid` | YOLO + reference comparison inspection |
| `GET /results/{file}` | Fetch generated annotated images |

## Layout

- `backend/` — FastAPI app, YOLO predictor, alignment / diff / verdict services
- `frontend/` — React UI (upload, confidence filter, viewer, detections, report)
- `scripts/` — dataset preparation, M1 training and evaluation
- `test_images/` — sample PCB boards
- `screenshots/` — UI screenshots used above

## Author

Rohit Deshmukh · [@Rohitd944](https://github.com/Rohitd944)
