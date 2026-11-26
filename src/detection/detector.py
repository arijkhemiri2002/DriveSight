"""
Thin wrapper around an Ultralytics YOLO model.

`predict(frames)` takes a list of BGR uint8 frames (as decoded by OpenCV) and returns one
float32 array per frame with columns [x1, y1, x2, y2, conf, cls] in pixels of the input frame.
Ultralytics treats numpy inputs as BGR, the same convention the models saw at validation
time (images read from disk with cv2).

The class order of the loaded checkpoint is checked against CLASS_NAMES: a model trained
with another class order would silently produce wrong labels otherwise.
"""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np

from src.detection.detect_config import CLASS_NAMES, DetectConfig


class Detector(Protocol):
    names: tuple[str, ...]
    def predict(self, frames: list[np.ndarray]) -> list[np.ndarray]: ...


class YoloDetector:
    def __init__(self, cfg: DetectConfig):
        import torch
        from ultralytics import YOLO
        self.cfg = cfg
        if cfg.device == "auto":
            self.device = "0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.device
        self.device_type = "cpu" if self.device == "cpu" else "cuda"
        self.half = (self.device_type == "cuda") if cfg.half == "auto" else (cfg.half.lower() == "true")
        self.model = YOLO(str(cfg.weights_path))
        names = self.model.names
        names = tuple(names[i] for i in sorted(names)) if isinstance(names, dict) else tuple(names)
        if names != CLASS_NAMES:
            raise ValueError(f"Checkpoint class order {names} != expected {CLASS_NAMES}")
        self.names = names
        self.gpu_name = torch.cuda.get_device_name(0) if self.device_type == "cuda" else "cpu"
        self.infer_seconds = 0.0
        self.frames_seen = 0

    def predict(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        if not frames:
            return []
        t0 = time.perf_counter()
        kw = {"half": True} if self.half else {}       # only pass `half` when used (newer releases rename it)
        results = self.model.predict(frames, imgsz=self.cfg.imgsz, conf=self.cfg.conf_floor,
                                     iou=self.cfg.nms_iou, max_det=self.cfg.max_det,
                                     device=self.device, verbose=False, **kw)
        out = []
        for r in results:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append(np.zeros((0, 6), np.float32))
                continue
            arr = np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                                  b.cls.cpu().numpy()[:, None]], axis=1).astype(np.float32)
            out.append(arr)
        self.infer_seconds += time.perf_counter() - t0
        self.frames_seen += len(frames)
        return out

    def warmup(self, frame: np.ndarray) -> None:
        self.predict([frame])
        self.infer_seconds, self.frames_seen = 0.0, 0
