"""Phase 4 tests on synthetic data.

Unit tests drive the tracking core with hand-made boxes whose true identity is known (an object moving,
a one-frame gap, two objects crossing, a low-score frame, a short buffer). The end-to-end test runs
Phase 2, Phase 3 (with the fake detector of the Phase 3 tests) and Phase 4, then the independent verifier,
including resume, tampering and a change of the Phase 3 input.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ultralytics")

from src.tracking.track_config import TrackConfig, ULTRALYTICS_DEFAULTS
from src.tracking.tracker_core import track_sequence, summarize_tracks, Detections
from src.tracking import run_tracking as rt
from src.tracking import verify_tracking as vt
from src.tracking import evaluate_tracking as et
from src.tracking import tune_tracking as tt
from src.tracking import draw_tracks as dt
from src.detection import run_detection as rd
from src.detection.detect_config import DetectConfig
from src.features import run_preprocessing as rp
from tests.test_phase3_detection import FakeDetector

CLS = ["pedestrian", "rider", "car", "truck", "bus", "motorcycle", "bicycle", "traffic light", "traffic sign"]


def box(k, i, x, y, w=80, h=160, conf=0.9, cls=0, vid="v"):
    x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    return dict(split="validation", video_id=vid, sample_idx=k, t_ms=200 * k, det_idx=i, cls_id=cls, cls_name=CLS[cls],
                is_vru=CLS[cls] in ("pedestrian", "rider", "motorcycle", "bicycle"), conf=conf, x1=x1, y1=y1, x2=x2,
                y2=y2, cx_n=x / 1280, cy_n=y / 720, w_n=w / 1280, h_n=h / 720, area_frac=w * h / (1280 * 720),
                bottom_y_n=y2 / 720, x_offset_n=x / 1280 - 0.5, touches_border=False)


UL = dict(ULTRALYTICS_DEFAULTS)          # behaviour tests use the untuned Ultralytics values (1 s buffer)


def cfg_ul(**kw):
    return TrackConfig(**{**UL, **kw})


def run(rows, n, cfg=None):
    b = pd.DataFrame(rows)
    f = pd.DataFrame({"sample_idx": range(n), "t_ms": [200 * k for k in range(n)]})
    return track_sequence(b, f, cfg or cfg_ul())


# ------------------------------------------------------------------ configuration
def test_buffer_is_converted_from_seconds_to_5hz_frames():
    assert cfg_ul().track_buffer == 5                               # 1.0 s at 5 Hz (not 30 frames = 6 s)
    assert TrackConfig(track_buffer_sec=0.4).track_buffer == 2
    a = cfg_ul().tracker_args()
    assert a.track_buffer == 5 and a.track_high_thresh == 0.25 and a.fuse_score is True


def test_frozen_setting_is_the_validation_choice():
    d = TrackConfig()
    assert (d.track_high_thresh, d.new_track_thresh, d.match_thresh, d.track_buffer_sec) == (0.25, 0.50, 0.90, 0.4)
    assert d.track_buffer == 2 and d.track_low_thresh == UL["track_low_thresh"] and d.fuse_score


def test_detections_object_supports_what_bytetracker_reads():
    d = Detections.from_boxes(pd.DataFrame([box(0, 0, 100, 200), box(0, 1, 500, 300, conf=0.2)]))
    assert len(d) == 2 and d.xywh.shape == (2, 4)
    assert np.allclose(d.xywh[0], [100, 200, 80, 160])               # centre x, y, width, height
    sub = d[d.conf > 0.5]
    assert len(sub) == 1 and sub.conf[0] == pytest.approx(0.9)


# ------------------------------------------------------------------ tracking behaviour
def test_one_moving_object_keeps_one_id_and_birth_box_is_kept():
    t, s = run([box(k, 0, 200 + 30 * k, 400) for k in range(10)], 10)
    assert t["track_id"].nunique() == 1 and len(t) == 10               # every frame, including the first
    assert t["is_birth"].sum() == 1 and t.loc[t["is_birth"], "t_ms"].iloc[0] == 0
    assert list(t.sort_values("t_ms")["track_frame"]) == list(range(10))
    assert s["n_unconfirmed_dropped"] == 0 and s["n_boxes_conf50_tracked"] == 10


def test_one_frame_gap_is_bridged():
    t, _ = run([box(k, 0, 200 + 30 * k, 400) for k in range(10) if k != 5], 10)
    assert t["track_id"].nunique() == 1 and len(t) == 9
    sm = summarize_tracks(t)
    assert sm.loc[0, "n_gaps"] == 1 and sm.loc[0, "max_gap_frames"] == 1 and sm.loc[0, "span_frames"] == 10


def test_gap_longer_than_buffer_starts_a_new_track():
    rows = [box(k, 0, 200 + 10 * k, 400) for k in range(12) if not 4 <= k <= 7]   # 4 missing frames
    t, _ = run(rows, 12, cfg_ul(track_buffer_sec=0.4))                              # buffer 2 frames
    assert t["track_id"].nunique() == 2
    t2, _ = run(rows, 12, cfg_ul(track_buffer_sec=2.0))                             # buffer 10 frames
    assert t2["track_id"].nunique() == 1


def test_low_score_frame_is_recovered_by_second_association():
    t, _ = run([box(k, 0, 200 + 30 * k, 400, conf=0.15 if k == 5 else 0.9) for k in range(10)], 10)
    assert t["track_id"].nunique() == 1 and len(t) == 10               # the 0.15 box is in the track


def test_two_crossing_objects_keep_their_identity():
    rows = []
    for k in range(24):                                                # they overlap around frames 10-13
        rows.append(box(k, 0, 300 + 25 * k, 400))                      # moving right
        rows.append(box(k, 1, 875 - 25 * k, 415))                      # moving left
    t, s = run(rows, 24)
    assert t["track_id"].nunique() == 2 and len(t) == 48
    for _, g in t.groupby("track_id"):
        x = g.sort_values("t_ms")["cx_n"].to_numpy()
        assert (np.diff(x) > 0).all() or (np.diff(x) < 0).all()        # each track follows ONE object


def test_isolated_box_never_confirmed_is_dropped_and_counted():
    rows = [box(k, 0, 200 + 20 * k, 400) for k in range(8)] + [box(3, 1, 1000, 200, conf=0.9, cls=2)]
    t, s = run(rows, 8)
    assert t["track_id"].nunique() == 1 and s["n_unconfirmed_dropped"] == 1
    assert not ((t["t_ms"] == 600) & (t["det_idx"] == 1)).any()


def test_frames_without_boxes_age_tracks():
    """Empty frames must still be passed to the tracker, otherwise lost tracks would never expire."""
    rows = [box(k, 0, 300, 400) for k in range(3)] + [box(k, 0, 300, 400) for k in range(9, 12)]
    t, _ = run(rows, 12, cfg_ul(track_buffer_sec=0.4))                # 6 empty frames > buffer 2
    assert t["track_id"].nunique() == 2


def test_det_idx_links_back_to_the_right_box():
    rows = []
    for k in range(6):                                                 # det_idx order differs from x order
        rows.append(box(k, 1, 200 + 10 * k, 400, cls=0))
        rows.append(box(k, 0, 900, 300, cls=2, conf=0.95))
    t, _ = run(rows, 6)
    ped = t[t["cls_name"] == "pedestrian"]
    assert (ped["det_idx"] == 1).all() and ped["track_id"].nunique() == 1


# ------------------------------------------------------------------ end to end
def args(dataset, out: Path, *extra):
    return ["--bdda-root", str(dataset["bdda"]), "--manifest", str(dataset["manifest"]),
            "--processed-root", str(out), *extra]


@pytest.fixture()
def detected(dataset, tmp_path):
    out = tmp_path / "processed"
    assert rp.main(args(dataset, out, "--qc-fraction", "0")) == 0
    assert rd.main(args(dataset, out, "--qc-fraction", "0"), detector=FakeDetector()) == 0
    return out


def test_tracking_end_to_end_resume_verify_and_tamper(dataset, detected, tmp_path):
    assert rt.main(args(dataset, detected)) == 0
    pre = rp.PreprocessConfig(processed_root=detected)
    cfg = TrackConfig(det=DetectConfig(pre=pre))
    for split in ("training", "validation", "test"):
        tr = pd.read_parquet(cfg.tracks_path(split))
        sm = pd.read_csv(cfg.summary_path(split), dtype={"video_id": str})
        f3 = pd.read_csv(cfg.det.frame_summary_path(split), dtype={"video_id": str})
        # fake detector: pedestrian 0.91 moving right, static car 0.40, truck 0.07. With the frozen setting
        # (new_track_thresh 0.50) only the pedestrian starts a track; the truck is below track_low_thresh.
        assert set(tr["cls_name"]) == {"pedestrian"}
        per_seq = sm.groupby("video_id")["track_id"].nunique()
        assert (per_seq == 1).all()
        n_frames = f3.groupby("video_id").size()
        assert (sm.groupby("video_id")["n_boxes"].max() == n_frames.reindex(per_seq.index)).all()
        ped = sm[sm["cls_name"] == "pedestrian"]
        assert (ped["d_cx_n"] > 0).all() and (ped["n_classes"] == 1).all()
    # resume: everything skipped, outputs unchanged
    before = {p: p.stat().st_mtime_ns for p in detected.rglob("tracks/sequence_tracks/*")}
    assert rt.main(args(dataset, detected)) == 0
    assert before == {p: p.stat().st_mtime_ns for p in detected.rglob("tracks/sequence_tracks/*")}
    # verification passes
    vargs = args(dataset, detected, "--out", str(tmp_path / "v.md"))
    assert vt.main(vargs) == 0
    assert "ALL CHECKS PASSED" in (tmp_path / "v.md").read_text()
    # evaluation and tuning run (tuning on validation only)
    assert et.main(["--processed-root", str(detected), "--split", "validation", "--out", str(tmp_path / "e.md")]) == 0
    assert tt.main(args(dataset, detected, "--quick", "--out", str(tmp_path / "t.md"))) == 0
    tune = pd.read_csv(tmp_path / "t.csv")
    assert len(tune) == len(tt.grid(tt.QUICK)) and tune["score"].is_monotonic_decreasing
    # visual check: drawn on the decoded camera frames
    assert dt.main(args(dataset, detected, "--split", "validation", "--videos", "1744")) == 0
    qc = detected / "validation" / "qc" / "tracks"
    assert (qc / "1744_tracks.mp4").stat().st_size > 0 and (qc / "1744_sheet.jpg").exists()
    # tampering: a tracked box moved -> verification fails
    p = cfg.tracks_path("test")
    tr = pd.read_parquet(p)
    tr.loc[tr.index[0], "x1"] += 5
    tr.to_parquet(p, index=False)
    assert vt.main(vargs) == 1


def test_phase3_change_invalidates_tracks(dataset, detected):
    assert rt.main(args(dataset, detected, "--split", "test")) == 0
    pre = rp.PreprocessConfig(processed_root=detected)
    cfg = TrackConfig(det=DetectConfig(pre=pre))
    h = json.loads((cfg.checkpoints_dir("test") / "6.json").read_text())["config_hash"]
    assert rt.track_checkpoint_valid(cfg, "test", "6", h)
    # with the Ultralytics defaults the 0.40 car is tracked too
    assert rt.main(args(dataset, detected, "--split", "test", "--new-track-thresh", "0.25", "--no-merge")) == 0
    assert json.loads((cfg.checkpoints_dir("test") / "6.json").read_text())["n_tracks"] == 2
    assert rt.main(args(dataset, detected, "--split", "test")) == 0
    # another tracker setting -> another hash
    other = replace(cfg, match_thresh=0.7)
    assert other.output_hash(phase3_hash="x", ultralytics_version="v") != cfg.output_hash(phase3_hash="x", ultralytics_version="v")
    # Phase 3 file rewritten (even with valid Phase 3 hashes) -> the old tracks are no longer valid
    bpath = cfg.det.seq_det_dir("test") / "6.parquet"
    b = pd.read_parquet(bpath)
    b.loc[b.index[0], "conf"] = 0.5
    rd.atomic_write_parquet(rd.typed(b), bpath, cfg.det)
    ckp = cfg.det.checkpoints_dir("test") / "6.json"
    ck = json.loads(ckp.read_text())
    ck["boxes_sha256"] = rp.sha256_file(bpath)
    ckp.write_text(json.dumps(ck))
    assert not rt.track_checkpoint_valid(cfg, "test", "6", h)


def test_missing_phase3_is_a_problem(dataset, tmp_path):
    out = tmp_path / "processed"
    assert rp.main(args(dataset, out, "--qc-fraction", "0", "--split", "test")) == 0
    assert rd.main(args(dataset, out, "--qc-fraction", "0", "--split", "test"), detector=FakeDetector()) == 0
    (out / "test" / "detections" / "sequence_detections" / "6_frames.csv").unlink()
    assert rt.main(args(dataset, out, "--split", "test")) == 0
    ck = json.loads((out / "test" / "tracks" / "sequence_checkpoints" / "6.json").read_text())
    assert ck["status"] == "problem" and "Phase 3" in ck["problems"][0]
