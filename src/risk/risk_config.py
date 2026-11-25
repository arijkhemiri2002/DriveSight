"""
Configuration for the DriveSight risk-estimation stage (src/risk).

Task: predict, for each sampled video frame, whether the driver will brake hard
within the next HORIZON_SEC seconds ("imminent braking"), from:
  - objects detected by the selected V2 detector (YOLOv8s),
  - the BDDA attention (gaze) map,
  - causal GPS kinematics (speed, past acceleration).

Label source: GPS deceleration. BDDA clips were cut around a braking event
(about 6.5 s before / 3.5 s after), so every usable clip should contain one.
The BDDA gaze map is the average of in-lab observers, i.e. where an attentive
driver would look - not the recording driver's own gaze.

All paths are relative to PROJECT_ROOT (the repository root).
Nothing under data/BDDA or data/processed is ever modified.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---- Inputs (read-only) ---------------------------------------------------------
BDDA_ROOT = PROJECT_ROOT / "data" / "BDDA"
MANIFEST_CSV = PROJECT_ROOT / "reports" / "video_dataset" / "dataset_manifest.csv"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"      # output of src/features/video_preprocessing.py
DETECTOR_WEIGHTS = PROJECT_ROOT / "weights" / "final" / "yolov8s_v2_best.pt"
SPLITS = ("training", "validation", "test")

# ---- Outputs (git-ignored: /data/ and /reports/) ------------------------------------
RISK_DATA_DIR = PROJECT_ROOT / "data" / "risk"            # large intermediate tables
RISK_REPORT_DIR = PROJECT_ROOT / "reports" / "risk"       # metrics, figures, markdown reports
LABELS_SEQ_CSV = RISK_DATA_DIR / "braking_events.csv"     # one row per sequence
DETECTIONS_CSV = RISK_DATA_DIR / "detections.csv"         # one row per detected box
DETECTIONS_DONE = RISK_DATA_DIR / "detections_done.txt"   # finished sequences (resume support)
FEATURES_CSV = RISK_DATA_DIR / "frame_features.csv"       # one row per frame (features + label)

# ---- Braking label --------------------------------------------------------------
HARD_BRAKE_DECEL = 2.0        # m/s^2. GPS is ~1 Hz, which smooths peaks; 2.0 m/s^2 ~ 0.2 g
MIN_SPEED_BEFORE_BRAKE = 1.5  # m/s. Ignore "braking" that starts from (almost) standstill
MAX_GPS_INTERVAL = 2.5        # s. Do not compute acceleration across larger GPS gaps
HORIZON_SEC = 2.0             # positive label: braking onset within the next HORIZON_SEC seconds
NO_EVENT_POLICY = "exclude"   # "exclude" or "negative": what to do with clips where no hard braking is found
                              # (BDDA clips were selected for braking, so a missing event usually means GPS noise)

# ---- Detection ------------------------------------------------------------------
DET_IMGSZ = 1152              # V2 evaluation settings
DET_NMS_IOU = 0.70
DET_MIN_CONF = 0.10           # keep everything >= 0.10 in detections.csv; feature thresholds below
DET_BATCH = 8

VRU_CLASSES = ("pedestrian", "rider", "motorcycle", "bicycle")
VEHICLE_CLASSES = ("car", "truck", "bus")
ALL_CLASSES = ("pedestrian", "rider", "car", "truck", "bus", "motorcycle",
               "bicycle", "traffic light", "traffic sign")
CONF_VRU = 0.15               # same defaults as the demo app
CONF_OTHER = 0.25

# ---- Frame geometry --------------------------------------------------------------
IMG_W, IMG_H = 1280, 720      # camera frames and resized gaze maps share this size
GAZE_READ_SCALE = 4           # gaze maps are read at 1/4 resolution (320x180) for speed

# Ego-path corridor (approximation of the lane ahead, normalised coordinates):
# a box is "in path" if its bottom edge is below PATH_Y_MIN and its bottom-centre x satisfies
# |x - 0.5| <= PATH_HALF_W_TOP + PATH_HALF_W_GROWTH * (y_bottom - PATH_Y_MIN) / (1 - PATH_Y_MIN)
PATH_Y_MIN = 0.45
PATH_HALF_W_TOP = 0.06      # about one lane ahead: x in [0.22, 0.78] at the bottom edge, narrower further up
PATH_HALF_W_GROWTH = 0.22

# ---- Temporal features ------------------------------------------------------------
SAMPLING_FPS = 5.0            # must match src/features/preprocess_config.py
DELTA_SHORT_SEC = 0.2         # one sampled frame back
DELTA_LONG_SEC = 1.0

# ---- Training / evaluation ------------------------------------------------------------
SEED = 0
N_BOOTSTRAP = 1000            # sequence-level (cluster) bootstrap resamples
TIME_BIN_SEC = 1.0            # bin width for the "time-matched" AUC (see train_risk_model.py)
