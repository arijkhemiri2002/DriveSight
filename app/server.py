"""DriveSight demo app - local web server.

Run from the repository root:
    python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
then open http://127.0.0.1:8000

Images are processed in memory and never written to disk. The server only listens on 127.0.0.1.
Models are the three final V2 checkpoints in weights/final/ (loaded lazily, one inference at a time).
"""
import io
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "False")   # never let Ultralytics pip-install anything from this server

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
WEIGHTS = ROOT / "weights" / "final"

MODELS = {  # key -> (label, file, note)
    "yolov8s": ("YOLOv8s", "yolov8s_v2_best.pt", "selected model, best on vulnerable road users"),
    "yolo11s": ("YOLO11s", "yolo11s_v2_best.pt", "highest recall"),
    "yolo11n": ("YOLO11n", "yolo11n_v2_cls_pw025_best.pt", "fastest, smallest"),
}
DEFAULT_MODEL = "yolov8s"
IMGSZ, NMS_IOU, MIN_CONF = 1152, 0.70, 0.05      # the V2 evaluation settings; the UI filters by confidence
MAX_BYTES, MAX_SIDE = 15 * 1024 * 1024, 4096
CLASS_NAMES = ["pedestrian", "rider", "car", "truck", "bus", "motorcycle", "bicycle", "traffic light", "traffic sign"]

app = FastAPI(title="DriveSight", docs_url=None, redoc_url=None)
_cache, _lock = {}, threading.Lock()


_SIGNATURES = (bytes([0xFF, 0xD8, 0xFF]), bytes([0x89]) + b"PNG", b"GIF87a", b"GIF89a", b"BM")   # JPEG, PNG, GIF, BMP (WebP checked below)


def _is_supported_image(data: bytes) -> bool:
    """Check the file signature ourselves: Ultralytics patches PIL.Image.open so that an unreadable file would make it
    try to pip-install an HEIF plugin. Unsupported or non-image files are rejected before PIL sees them."""
    return data.startswith(_SIGNATURES) or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")


def _device():
    import torch
    return "0" if torch.cuda.is_available() else "cpu"


def _model(key):
    if key not in MODELS:
        raise HTTPException(400, "unknown model")
    if key not in _cache:
        from ultralytics import YOLO
        path = WEIGHTS / MODELS[key][1]
        if not path.exists():
            raise HTTPException(500, f"checkpoint not found: {path.name}")
        _cache[key] = YOLO(str(path))
    return _cache[key]


@app.get("/api/models")
def models():
    return {"default": DEFAULT_MODEL, "device": "GPU" if _device() == "0" else "CPU", "classes": CLASS_NAMES,
            "models": [{"key": k, "label": v[0], "note": v[2], "available": (WEIGHTS / v[1]).exists()}
                       for k, v in MODELS.items()]}


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), model: str = Form(DEFAULT_MODEL)):
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image too large (15 MB max)")
    if not _is_supported_image(data):
        raise HTTPException(400, "unsupported file: use a JPG, PNG, WebP, GIF or BMP image")
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        raise HTTPException(400, "this file is not a readable image")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    w, h = img.size
    with _lock:                                   # one inference at a time on the 4 GB GPU
        net = _model(model)
        t0 = time.perf_counter()
        res = net.predict(img, imgsz=IMGSZ, conf=MIN_CONF, iou=NMS_IOU, device=_device(), verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
    dets = []
    for b in res.boxes:
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        c = int(b.cls[0])
        dets.append({"cls": c, "name": CLASS_NAMES[c], "conf": round(float(b.conf[0]), 4),
                     "box": [round(x1 / w, 5), round(y1 / h, 5), round(x2 / w, 5), round(y2 / h, 5)]})
    dets.sort(key=lambda d: -d["conf"])
    return {"model": MODELS[model][0], "width": w, "height": h, "ms": round(ms, 1), "detections": dets}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
