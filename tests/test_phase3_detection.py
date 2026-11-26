"""Phase 3 tests. A fake detector encodes the index of the frame it received in its boxes, which
proves that every detection is joined to the right Phase 2 row. A real-YOLO test runs only when
ultralytics and the final weights are available."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.detection.box_features import box_table
from src.detection.detect_config import CLASS_NAMES, DetectConfig, PROJECT_ROOT
from src.detection import run_detection as rd
from src.detection import verify_detection as vd
from src.features import run_preprocessing as rp
from tests.synthetic import decode_cam_index


class FakeDetector:
    names = CLASS_NAMES
    device_type, half, version, weights_sha = "cpu", False, "fake", "fake"

    def __init__(self):
        self.infer_seconds, self.frames_seen, self.calls = 0.0, 0, []

    def predict(self, frames):
        self.calls.append(len(frames))
        out = []
        for f in frames:
            i = decode_cam_index(f)
            out.append(np.array([
                [100 + i, 300, 160 + i, 450, 0.91, 0],          # pedestrian: x1 encodes the frame index
                [600, 350, 900, 600, 0.40, 2],                  # car
                [1200, 500, 1300, 730, 0.07, 3],                # truck cut by the image border
            ], np.float32))
        self.frames_seen += len(frames)
        return out


def args(dataset, out: Path, *extra):
    return ["--bdda-root", str(dataset["bdda"]), "--manifest", str(dataset["manifest"]),
            "--processed-root", str(out), *extra]


@pytest.fixture()
def processed(dataset, tmp_path):
    out = tmp_path / "processed"
    assert rp.main(args(dataset, out, "--qc-fraction", "0")) == 0      # Phase 2 first
    return out


def test_box_table_geometry():
    d = np.array([[1200, 500, 1300, 730, 0.07, 3], [100, 300, 160, 450, 0.91, 0]], np.float32)
    t = box_table(d, 1280, 720)
    assert list(t["cls_name"]) == ["pedestrian", "truck"]                 # sorted by confidence
    assert list(t["det_idx"]) == [0, 1]
    tr = t.iloc[1]
    assert tr["x2"] == 1280 and tr["y2"] == 720 and tr["touches_border"]   # clipped to the image
    p = t.iloc[0]
    assert p["is_vru"] and not p["touches_border"]
    assert p["cx_n"] == pytest.approx(130 / 1280, abs=1e-5)
    assert p["bottom_y_n"] == pytest.approx(450 / 720, abs=1e-5)
    assert p["x_offset_n"] == pytest.approx(130 / 1280 - 0.5, abs=1e-5)
    assert len(box_table(np.zeros((0, 6)), 1280, 720)) == 0


def test_detection_end_to_end_join_resume_and_verify(dataset, processed, tmp_path):
    det = FakeDetector()
    assert rd.main(args(dataset, processed, "--qc-fraction", "1.0", "--batch", "8"), detector=det) == 0
    assert max(det.calls) == 8
    cfg = DetectConfig(pre=rp.PreprocessConfig(processed_root=processed))
    for split in ("training", "validation", "test"):
        b = pd.read_parquet(cfg.detections_path(split))
        f = pd.read_csv(cfg.frame_summary_path(split), dtype={"video_id": str})
        p2 = pd.read_csv(cfg.pre.aligned_records_path(split), dtype={"video_id": str})
        assert len(f) == len(p2) and (f["n_det"] == 3).all()
        # the pedestrian box x1 = 100 + index of the frame the detector actually received
        ped = b[b["cls_name"] == "pedestrian"].merge(f, on=["video_id", "t_ms"])
        assert ((ped["x1"] - 100).round().astype(int) == ped["cam_frame_idx"]).all()
        assert len(list(cfg.qc_dir(split).glob("*.jpg"))) == f["video_id"].nunique()
    # resume: nothing recomputed
    det2 = FakeDetector()
    assert rd.main(args(dataset, processed), detector=det2) == 0
    assert det2.frames_seen == 0
    # independent verification passes, then fails after tampering
    vargs = args(dataset, processed, "--out", str(tmp_path / "v.md"))
    assert vd.main(vargs) == 0
    p = cfg.detections_path("test")
    b = pd.read_parquet(p)
    b.loc[b.index[0], "cls_name"] = "car" if b.loc[b.index[0], "cls_name"] != "car" else "bus"
    b.to_parquet(p, index=False)
    assert vd.main(vargs) == 1


def test_missing_phase2_is_a_problem(dataset, tmp_path):
    out = tmp_path / "processed"
    assert rd.main(args(dataset, out, "--split", "test"), detector=FakeDetector()) == 0
    ck = json.loads((out / "test" / "detections" / "sequence_checkpoints" / "6.json").read_text())
    assert ck["status"] == "problem" and "Phase 2" in ck["problems"][0]


def test_phase2_change_invalidates_detections(dataset, processed):
    det = FakeDetector()
    assert rd.main(args(dataset, processed, "--split", "test"), detector=det) == 0
    cfg = DetectConfig(pre=rp.PreprocessConfig(processed_root=processed))
    h = cfg.output_hash(weights_sha256="fake", device_type="cpu", half=False, ultralytics_version="fake")
    assert rd.det_checkpoint_valid(cfg, "test", "6", h)
    other = DetectConfig(pre=rp.PreprocessConfig(processed_root=processed, sampling_fps=10.0))
    h2 = other.output_hash(weights_sha256="fake", device_type="cpu", half=False, ultralytics_version="fake")
    assert h2 != h and not rd.det_checkpoint_valid(other, "test", "6", h2)


# ------------------------------------------------------------------ real model (optional)
WEIGHTS = PROJECT_ROOT / "weights" / "final" / "yolov8s_v2_best.pt"


@pytest.mark.skipif(not WEIGHTS.exists(), reason="final weights not present")
def test_real_yolo_on_synthetic_frames(dataset, processed):
    pytest.importorskip("ultralytics")
    from src.detection.detector import YoloDetector
    cfg = DetectConfig(pre=rp.PreprocessConfig(processed_root=processed), device="cpu", imgsz=640)
    det = YoloDetector(cfg)
    assert det.names == CLASS_NAMES
    frames = [np.full((720, 1280, 3), 90, np.uint8)] * 2
    out = det.predict(frames)
    assert len(out) == 2 and all(o.shape[1] == 6 for o in out)


def test_degenerate_box_dropped_and_repair_restores_fixed_output(dataset, processed, tmp_path):
    from src.detection import repair_degenerate_boxes as rep
    from src.features.run_preprocessing import sha256_file
    # fixed code: a box beyond the right edge is clipped to zero width and dropped
    t = box_table(np.array([[1300, 10, 1350, 50, 0.6, 2], [10, 10, 50, 50, 0.5, 2]], np.float32), 1280, 720)
    assert len(t) == 1 and t["det_idx"].tolist() == [0]

    assert rd.main(args(dataset, processed, "--split", "test"), detector=FakeDetector()) == 0
    cfg = DetectConfig(pre=rp.PreprocessConfig(processed_root=processed))
    bpath, fpath = cfg.seq_det_dir("test") / "6.parquet", cfg.seq_det_dir("test") / "6_frames.csv"
    good_b, good_f = pd.read_parquet(bpath), pd.read_csv(fpath, dtype={"video_id": str})
    # simulate the first run's output: one degenerate box inserted with det_idx 1 in frame 0
    b = good_b.copy()
    row = b.iloc[[0]].copy()
    row[["x1", "x2", "conf", "cls_name", "cls_id", "is_vru"]] = [1280.0, 1280.0, 0.5, "car", 2, False]
    b.loc[(b["sample_idx"] == 0) & (b["det_idx"] >= 1), "det_idx"] += 1
    row["det_idx"] = 1
    b = rd.typed(pd.concat([b, row]).sort_values(["sample_idx", "det_idx"]).reset_index(drop=True))
    f = good_f.copy()
    f.loc[f["sample_idx"] == 0, "n_det"] += 1
    rd.atomic_write_parquet(b, bpath, cfg)
    f.to_csv(fpath, index=False)
    ckp = cfg.checkpoints_dir("test") / "6.json"
    ck = json.loads(ckp.read_text())
    ck.update(boxes_sha256=sha256_file(bpath), frames_sha256=sha256_file(fpath))
    ckp.write_text(json.dumps(ck))
    rd.merge_split(cfg, pd.read_csv(dataset["manifest"], dtype={"video_id": str}), "test", ck["config_hash"], {})
    vargs = args(dataset, processed, "--split", "test", "--out", str(tmp_path / "v.md"))
    assert vd.main(vargs) == 1                                   # verifier catches it
    assert rep.main(["--processed-root", str(processed), "--out", str(tmp_path / "r.md")]) == 0
    assert vd.main(vargs) == 0                                   # repaired
    pd.testing.assert_frame_equal(pd.read_parquet(bpath), good_b)   # identical to fixed-code output
    after_f = pd.read_csv(fpath, dtype={"video_id": str})
    pd.testing.assert_frame_equal(after_f[good_f.columns], good_f, check_dtype=False)
    assert json.loads(ckp.read_text())["repairs"][0]["removed"][0]["x1"] == 1280.0
