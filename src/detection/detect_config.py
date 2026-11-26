"""
Configuration for DriveSight Phase 3: YOLO inference on the Phase 2 sampling grid.

Inference settings are the ones used to validate the V2 models (configs/v2_train_recipe.yaml,
app/server.py): imgsz 1152, NMS IoU 0.70. Detections are kept down to conf 0.05 so that the
operating threshold can be chosen later on the VALIDATION split (never on test), and so that a
ByteTrack-style tracker (Phase 4) can use low-score boxes in its second association step.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.features.preprocess_config import PROJECT_ROOT, PreprocessConfig

DETECTION_SCHEMA_VERSION = "3.0"

# Order defines the class ids of the V2 models (configs/v2_data.yaml).
CLASS_NAMES = ("pedestrian", "rider", "car", "truck", "bus", "motorcycle", "bicycle",
               "traffic light", "traffic sign")
VRU_CLASSES = ("pedestrian", "rider", "motorcycle", "bicycle")     # "VRU-4" in the V2 evaluation

MODEL_FILES = {
    "yolov8s": "yolov8s_v2_best.pt",            # selected model (best VRU-4 AP50-95)
    "yolo11s": "yolo11s_v2_best.pt",
    "yolo11n": "yolo11n_v2_cls_pw025_best.pt",
}


@dataclass(frozen=True)
class DetectConfig:
    model_key: str = "yolov8s"
    weights_dir: Path = PROJECT_ROOT / "weights" / "final"
    imgsz: int = 1152
    conf_floor: float = 0.05        # storage floor, NOT the operating threshold
    nms_iou: float = 0.70
    max_det: int = 300
    batch: int = 8                  # frames per forward pass (speed only, results unchanged)
    device: str = "auto"            # "auto" -> GPU 0 if available, else cpu
    half: str = "auto"              # "auto" -> FP16 on GPU, FP32 on CPU
    border_px: int = 2              # a box within this distance of the image edge is "truncated"
    qc_fraction: float = 0.05       # same hash rule as Phase 2 -> same QC sequences
    pre: PreprocessConfig = field(default_factory=PreprocessConfig)

    @property
    def weights_path(self) -> Path:
        return self.weights_dir / MODEL_FILES[self.model_key]

    # ---- output layout (next to the Phase 2 outputs of the same split) -------------
    def det_dir(self, split: str) -> Path:
        return self.pre.split_dir(split) / "detections"

    def seq_det_dir(self, split: str) -> Path:
        return self.det_dir(split) / "sequence_detections"

    def checkpoints_dir(self, split: str) -> Path:
        return self.det_dir(split) / "sequence_checkpoints"

    def detections_path(self, split: str) -> Path:
        return self.det_dir(split) / "detections.parquet"

    def frame_summary_path(self, split: str) -> Path:
        return self.det_dir(split) / "frame_summary.csv"

    def qc_dir(self, split: str) -> Path:
        return self.pre.split_dir(split) / "qc" / "detections"

    def output_hash(self, *, weights_sha256: str, device_type: str, half: bool,
                    ultralytics_version: str) -> str:
        """Everything that can change a detection. Batch size is excluded (results identical)."""
        d = {"schema": DETECTION_SCHEMA_VERSION, "model": self.model_key, "weights": weights_sha256,
             "imgsz": self.imgsz, "conf_floor": self.conf_floor, "nms_iou": self.nms_iou,
             "max_det": self.max_det, "border_px": self.border_px, "device_type": device_type,
             "half": half, "ultralytics": ultralytics_version,
             "phase2": self.pre.output_hash()}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def describe(self) -> dict:
        d = asdict(self)
        d.pop("pre")
        return {k: str(v) for k, v in d.items()}
