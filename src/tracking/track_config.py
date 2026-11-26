"""
Configuration for DriveSight Phase 4: ByteTrack on the stored Phase 3 detections.

No video is decoded and YOLO is not re-run: the input is
data/processed/{split}/detections/sequence_detections/{video_id}.parquet (one row per box, conf >= 0.05)
plus {video_id}_frames.csv (one row per sampled frame, including frames without any box).

Time base. Phase 2 samples every 200 ms (5 Hz). In Ultralytics 8.4.189, BYTETracker takes no frame_rate:
`track_buffer` is used directly as a number of frames (max_frames_lost = args.track_buffer). The default 30
assumes 30 fps (1 s); at 5 Hz it would keep a lost track for 6 s. The buffer is therefore set here in
SECONDS and converted with the Phase 2 sampling rate: track_buffer = round(track_buffer_sec * 5).

Every setting that can change a track is part of `output_hash`, together with the Phase 3 config hash,
so a change upstream or here invalidates old tracks automatically.
Thresholds are tuned on the VALIDATION split only (src/tracking/tune_tracking.py), never on test.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.features.preprocess_config import PreprocessConfig
from src.detection.detect_config import DetectConfig

TRACK_SCHEMA_VERSION = "4.0"


# Ultralytics bytetrack.yaml defaults, with the 30-frame buffer expressed as 1 s. Reference point of the tuning.
ULTRALYTICS_DEFAULTS = {"track_high_thresh": 0.25, "track_low_thresh": 0.10, "new_track_thresh": 0.25,
                        "match_thresh": 0.80, "fuse_score": True, "track_buffer_sec": 1.0}


@dataclass(frozen=True)
class TrackConfig:
    # ---- ByteTrack (Ultralytics names). FROZEN setting chosen on the validation split by tune_tracking.py ------
    # (reports/video_dataset/phase4_tuning_validation.md, rule fixed in advance; justification in
    # docs/phase4_tracking.md section 8). Change only with a new tuning run on validation, never on test.
    track_high_thresh: float = 0.25    # first association uses boxes with conf >= this            (default 0.25)
    track_low_thresh: float = 0.10     # second association (low-score boxes): low < conf < high   (default 0.10, not tuned)
    new_track_thresh: float = 0.50     # an unmatched box starts a track only if conf >= this      (default 0.25)
    match_thresh: float = 0.90         # first-association cost threshold (1 - IoU, fused with the score) (default 0.80)
    fuse_score: bool = True            # Ultralytics default, not tuned
    track_buffer_sec: float = 0.4      # how long a lost track can be recovered: 0.4 s = 2 frames  (default 1 s)
    # ---- Output -------------------------------------------------------------------------------------------
    # Ultralytics confirms a new track only when it is matched again in the next frame. The first box of a
    # confirmed track is kept (it is part of the object's trajectory); tracks that are never confirmed
    # (single, unmatched boxes) are dropped and counted in the checkpoint.
    keep_birth_box: bool = True
    det: DetectConfig = field(default_factory=DetectConfig)

    @property
    def pre(self) -> PreprocessConfig:
        return self.det.pre

    @property
    def sampling_fps(self) -> float:
        return self.pre.sampling_fps

    @property
    def track_buffer(self) -> int:
        """Lost-track lifetime in frames at the Phase 2 rate (BYTETracker counts frames, not seconds)."""
        return max(1, int(round(self.track_buffer_sec * self.sampling_fps)))

    def tracker_args(self):
        """The argument object BYTETracker.__init__ reads (ultralytics/trackers/byte_tracker.py)."""
        from types import SimpleNamespace
        return SimpleNamespace(track_high_thresh=self.track_high_thresh, track_low_thresh=self.track_low_thresh,
                               new_track_thresh=self.new_track_thresh, track_buffer=self.track_buffer,
                               match_thresh=self.match_thresh, fuse_score=self.fuse_score)

    # ---- output layout (next to the Phase 2/3 outputs of the same split; they are never modified) ---------
    def tracks_dir(self, split: str) -> Path:
        return self.pre.split_dir(split) / "tracks"

    def seq_tracks_dir(self, split: str) -> Path:
        return self.tracks_dir(split) / "sequence_tracks"

    def checkpoints_dir(self, split: str) -> Path:
        return self.tracks_dir(split) / "sequence_checkpoints"

    def tracks_path(self, split: str) -> Path:
        return self.tracks_dir(split) / "tracks.parquet"

    def summary_path(self, split: str) -> Path:
        return self.tracks_dir(split) / "track_summary.csv"

    def params(self) -> dict:
        return {"track_high_thresh": self.track_high_thresh, "track_low_thresh": self.track_low_thresh,
                "new_track_thresh": self.new_track_thresh, "match_thresh": self.match_thresh,
                "fuse_score": self.fuse_score, "track_buffer_sec": self.track_buffer_sec,
                "track_buffer_frames": self.track_buffer, "keep_birth_box": self.keep_birth_box}

    def output_hash(self, *, phase3_hash: str, ultralytics_version: str) -> str:
        d = {"schema": TRACK_SCHEMA_VERSION, "tracker": "bytetrack", **self.params(),
             "sampling_fps": self.sampling_fps, "ultralytics": ultralytics_version, "phase3": phase3_hash}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def describe(self) -> dict:
        return {k: str(v) for k, v in asdict(self).items() if k != "det"}
