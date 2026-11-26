"""Phase 2 tests on synthetic BDDA-like data. Run from the repo root:  python -m pytest -q tests"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.features.preprocess_config import PreprocessConfig
from src.features.frame_extractor import SequentialReader, sample_plan, seek_read
from src.features.gps_aligner import load_gps_track, align_gps
from src.features.gaze_crop import gaze_descriptors
from src.features import run_preprocessing as rp
from tests.synthetic import make_sequence, decode_cam_index, gaze_blob_x, write_gps


def cli(dataset, out: Path, *extra) -> int:
    return rp.main(["--bdda-root", str(dataset["bdda"]), "--manifest", str(dataset["manifest"]),
                    "--processed-root", str(out), "--qc-fraction", "0.0", *extra])


def cfg_for(dataset, out: Path) -> PreprocessConfig:
    return PreprocessConfig(bdda_root=dataset["bdda"], manifest_csv=dataset["manifest"],
                            processed_root=out, qc_fraction=0.0)


# --------------------------------------------------------------------------- sampling plan
def test_plan_stays_inside_both_videos():
    plan = sample_plan(300, 30.0, 292, 29.0, 5.0)
    assert plan[-1].t_sec == pytest.approx(9.8)          # 10.0 would be past frame 299 (9.967 s)
    assert all(p.cam_idx <= 299 and p.gaze_idx <= 291 for p in plan)
    assert len(plan) == 50
    assert [p.t_ms for p in plan[:3]] == [0, 200, 400]


def test_plan_bdda_typical_29_97():
    plan = sample_plan(300, 29.97, 292, 29.0, 5.0)
    # last camera frame at 299/29.97 = 9.977 s -> last target 9.8 s
    assert plan[-1].t_sec == pytest.approx(9.8)
    assert max(abs(p.cam_idx / 29.97 - p.t_sec) for p in plan) <= 0.5 / 29.97 + 1e-9


def test_plan_residual_bounded_by_half_frame():
    for fps in (29.92, 29.97, 30.0, 59.94, 60.0):
        plan = sample_plan(int(fps * 10), fps, 292, 29.0, 5.0)
        assert max(abs(p.cam_idx / fps - p.t_sec) for p in plan) <= 0.5 / fps + 1e-9
        assert max(abs(p.gaze_idx / 29.0 - p.t_sec) for p in plan) <= 0.5 / 29.0 + 1e-9


# --------------------------------------------------------------------------- frame accuracy
def test_sequential_reader_returns_exact_frames(dataset):
    path = dataset["bdda"] / "training" / "camera_videos" / "2.mp4"
    plan = sample_plan(300, 30.0, 292, 29.0, 5.0)
    with SequentialReader(path) as r:
        for p in plan:
            f = r.read_at(p.cam_idx)
            assert decode_cam_index(f) == p.cam_idx
            assert r.last_container_t == pytest.approx(p.cam_idx / 30.0, abs=1e-3)
        assert r.count_total_frames() == 300


def test_sequential_reader_is_forward_only(dataset):
    path = dataset["bdda"] / "training" / "camera_videos" / "2.mp4"
    with SequentialReader(path) as r:
        r.read_at(10)
        assert r.read_at(10) is not None     # same index twice is fine (cache)
        with pytest.raises(ValueError):
            r.read_at(5)


def test_sequential_matches_seek_on_synthetic(dataset):
    path = dataset["bdda"] / "training" / "camera_videos" / "4.mp4"     # 59.94 fps
    cap = cv2.VideoCapture(str(path))
    with SequentialReader(path) as r:
        for p in sample_plan(600, 59.94, 292, 29.0, 5.0):
            a, b = r.read_at(p.cam_idx), seek_read(cap, p.cam_idx)
            assert decode_cam_index(a) == decode_cam_index(b) == p.cam_idx
    cap.release()


# --------------------------------------------------------------------------- GPS
def test_invalid_speed_never_leaks(dataset):
    cfg = cfg_for(dataset, dataset["root"] / "unused")
    tr = load_gps_track(dataset["bdda"] / "validation" / "gps_jsons" / "1744.json", cfg)
    assert tr.stats["gps_speed_invalid"] == 4
    assert np.all(np.isnan(tr.speed[~tr.speed_valid]))
    assert np.nanmin(tr.speed) >= 0
    # acceleration only between two consecutive valid records (records 3->4, 4->... etc.)
    # record 1 (-1) and record 3 (follows -1) must have NaN acceleration
    assert np.isnan(tr.accel[1]) and np.isnan(tr.accel[2]) and np.isnan(tr.accel[3])
    assert tr.accel[4] == pytest.approx(1.0)              # 4.0 -> 5.0 over 1 s
    t = np.arange(0, 9.81, 0.2)
    g = align_gps(tr, t, cfg)
    assert (g["gps_speed_m_s"].dropna() >= 0).all()
    assert (g["gps_speed_kmh"].dropna() >= 0).all()
    assert not g.loc[g["gps_speed_m_s"].isna() & g["gps_matched"], "gps_speed_valid"].any()
    assert (g["gps_speed_interp_m_s"].dropna() >= 0).all()
    # interpolation spans the invalid pair (0.5 s, 3.5 s) only if <= 2 s apart: it is 3 s -> NaN
    assert np.isnan(g.loc[np.isclose(t, 2.0), "gps_speed_interp_m_s"]).all()
    # between valid 3.5 s (4.0) and 4.5 s (5.0): t = 4.0 -> 4.5
    assert g.loc[np.isclose(t, 4.0), "gps_speed_interp_m_s"].iloc[0] == pytest.approx(4.5)


def test_gps_nearest_and_gap(dataset):
    cfg = cfg_for(dataset, dataset["root"] / "unused")
    tr = load_gps_track(dataset["bdda"] / "training" / "gps_jsons" / "2.json", cfg)
    g = align_gps(tr, np.array([0.0, 0.4, 0.6, 9.8]), cfg)
    assert list(g["gps_rec_t_sec"]) == [0.5, 0.5, 0.5, 9.5]
    assert g["gps_gap_sec"].max() <= 0.5 + 1e-9
    assert g["gps_matched"].all()


def test_gps_course_and_duplicates(tmp_path):
    cfg = PreprocessConfig(processed_root=tmp_path)
    p = tmp_path / "g.json"
    write_gps(p, 5, [1, 2, 3, 4, 5], courses=[10, -1, 370, 90, 180])
    d = json.loads(p.read_text())
    d["locations"].append(dict(d["locations"][0]))         # exact duplicate timestamp
    p.write_text(json.dumps(d))
    tr = load_gps_track(p, cfg)
    assert tr.stats["gps_dup_timestamps"] == 1
    assert tr.stats["gps_course_invalid"] == 2
    assert np.isnan(tr.course[1]) and np.isnan(tr.course[2])


# --------------------------------------------------------------------------- gaze
def test_gaze_descriptors_track_the_blob():
    from tests.synthetic import make_gaze_frame
    cfg = PreprocessConfig()
    for idx in (0, 100, 291):
        d = gaze_descriptors(make_gaze_frame(idx, 292), cfg)
        assert d["gaze_peak_x"] == pytest.approx(gaze_blob_x(idx, 292), abs=0.01)
        assert d["gaze_centroid_y"] == pytest.approx(0.4, abs=0.02)
        assert d["gaze_n_regions"] == 1 and not d["gaze_empty"] and d["gaze_bar_max"] == 0
    d = gaze_descriptors(np.zeros((768, 1024, 3), np.uint8), cfg)
    assert d["gaze_empty"] and np.isnan(d["gaze_peak_x"])


# --------------------------------------------------------------------------- end to end
def test_end_to_end_resume_and_merge(dataset, tmp_path):
    out = tmp_path / "processed"
    cfg = cfg_for(dataset, out)
    # 1) interrupted run: only one sequence
    assert cli(dataset, out, "--limit", "1") == 0
    assert len(list(out.glob("*/logs/processing_log.csv"))) == 1
    # 2) full run resumes: the first sequence is skipped, the rest processed
    assert cli(dataset, out) == 0
    logs = pd.concat(pd.read_csv(p) for p in out.glob("*/logs/processing_log.csv"))
    assert len(logs) == 4 and logs["video_id"].astype(str).is_unique
    # 3) rerun: nothing reprocessed
    assert cli(dataset, out) == 0
    logs2 = pd.concat(pd.read_csv(p) for p in out.glob("*/logs/processing_log.csv"))
    assert len(logs2) == 4

    man = pd.read_csv(dataset["manifest"], dtype={"video_id": str})
    total = 0
    for split in ("training", "validation", "test"):
        a = pd.read_csv(cfg.aligned_records_path(split), dtype={"video_id": str})
        assert (a["split"] == split).all()
        assert set(a["video_id"]) == set(man.loc[man["split"] == split, "video_id"])
        assert not a.duplicated(["video_id", "t_ms"]).any()
        assert a["cam_ok"].all() and a["gaze_ok"].all()
        assert (a["cam_residual_sec"].abs() <= 0.5 / 29.9 + 1e-6).all()
        assert (a["gaze_residual_sec"].abs() <= 0.5 / 29 + 1e-6).all()
        assert (a["gps_speed_m_s"].dropna() >= 0).all()
        total += len(a)
    # every synthetic clip ends at 9.97-9.99 s -> last target 9.8 s -> 50 rows each
    assert total == 4 * 50
    v = pd.read_csv(cfg.aligned_records_path("validation"), dtype={"video_id": str})
    assert (~v["gps_speed_valid"]).sum() > 0                # invalid records are flagged...
    assert v.loc[~v["gps_speed_valid"], "gps_speed_m_s"].isna().all()   # ...and not passed on
    # gaze descriptor follows the blob of the selected gaze frame
    exp = v["gaze_frame_idx"].map(lambda i: gaze_blob_x(i, 292))
    assert (v["gaze_peak_x"] - exp).abs().max() < 0.01
    # no JPEG frames anywhere, only metadata
    assert not list(out.rglob("frames"))


def test_corrupted_csv_is_recomputed(dataset, tmp_path):
    out = tmp_path / "processed"
    assert cli(dataset, out, "--split", "test") == 0
    cfg = cfg_for(dataset, out)
    csv = cfg.sequence_records_dir("test") / "6.csv"
    csv.write_text(csv.read_text()[:500])                  # truncate = simulated corruption
    assert cli(dataset, out, "--split", "test") == 0
    log = pd.read_csv(out / "test" / "logs" / "processing_log.csv")
    assert len(log) == 2                                   # processed again
    assert len(pd.read_csv(csv)) == 50


def test_config_change_invalidates_checkpoints(dataset, tmp_path):
    out = tmp_path / "processed"
    cfg = cfg_for(dataset, out)
    assert cli(dataset, out, "--split", "test") == 0
    other = PreprocessConfig(processed_root=out, sampling_fps=10.0)
    assert other.output_hash() != cfg.output_hash()
    assert not rp.checkpoint_is_valid(other, "test", "6", other.output_hash())


def test_refuses_to_write_into_raw_data(dataset):
    cfg = cfg_for(dataset, dataset["bdda"] / "processed")
    with pytest.raises(PermissionError):
        cfg.assert_safe_output(dataset["bdda"] / "training" / "x.csv")
    with pytest.raises(PermissionError):
        cli(dataset, dataset["bdda"] / "processed")


def test_refuses_old_pilot_layout(dataset, tmp_path):
    out = tmp_path / "processed"
    (out / "training" / "frames" / "camera").mkdir(parents=True)
    with pytest.raises(SystemExit):
        cli(dataset, out, "--split", "training")


def test_qc_overlay_written_only_for_selected(dataset, tmp_path):
    out = tmp_path / "processed"
    rp.main(["--bdda-root", str(dataset["bdda"]), "--manifest", str(dataset["manifest"]),
             "--processed-root", str(out), "--qc-fraction", "1.0", "--split", "test"])
    files = list((out / "test" / "qc" / "overlays").glob("*.jpg"))
    assert len(files) == 1 and files[0].name == "6_t05000.jpg"
    img = cv2.imread(str(files[0]))
    assert img.shape[1] == 3 * 640


# --------------------------------------------------------------------------- check scripts
def test_gps_scan_finds_negative_speeds(dataset, tmp_path):
    from src.features import gps_validity_scan as scan
    assert scan.main(["--bdda-root", str(dataset["bdda"]), "--manifest", str(dataset["manifest"]),
                      "--out-dir", str(tmp_path)]) == 0
    df = pd.read_csv(tmp_path / "gps_validity_scan.csv", dtype={"video_id": str, "negative_speed_values": str})
    r = df.set_index("video_id")
    assert r.at["1744", "n_speed_negative"] == 4 and r.at["1744", "negative_speed_values"] == "-1"
    assert df["in_clean_manifest"].all() and df["any_issue"].sum() == 1


def test_frame_accuracy_script(dataset, tmp_path):
    from src.features import check_frame_accuracy as fa
    out = tmp_path / "fa.md"
    assert fa.main(["--bdda-root", str(dataset["bdda"]), "--manifest", str(dataset["manifest"]),
                    "--videos", "2", "1744", "--out", str(out)]) == 0
    assert "agree on every sampled frame" in out.read_text()


def test_verifier_passes_then_detects_tampering(dataset, tmp_path):
    from src.features import verify_preprocessing as vp
    out = tmp_path / "processed"
    assert cli(dataset, out) == 0
    args = ["--bdda-root", str(dataset["bdda"]), "--manifest", str(dataset["manifest"]),
            "--processed-root", str(out), "--out", str(tmp_path / "v.md")]
    assert vp.main(args) == 0
    # tamper: inject a negative speed into the merged file -> verifier must fail
    cfg = cfg_for(dataset, out)
    p = cfg.aligned_records_path("validation")
    a = pd.read_csv(p, dtype={"video_id": str})
    a.loc[a.index[0], "gps_speed_m_s"] = -1.0
    a.to_csv(p, index=False)
    assert vp.main(args) == 1
    assert "FAIL" in (tmp_path / "v.md").read_text()


def test_broken_video_is_isolated(tmp_path):
    bdda = tmp_path / "BDDA"
    rows = [make_sequence(bdda, "test", "6"), make_sequence(bdda, "test", "7")]
    (bdda / "test" / "camera_videos" / "7.mp4").write_bytes(b"not a video")
    man = tmp_path / "m.csv"
    pd.DataFrame(rows).to_csv(man, index=False)
    out = tmp_path / "processed"
    args = ["--bdda-root", str(bdda), "--manifest", str(man), "--processed-root", str(out), "--qc-fraction", "0"]
    assert rp.main(args) == 0
    ck = json.loads((out / "test" / "metadata" / "sequence_checkpoints" / "7.json").read_text())
    assert ck["status"] == "problem" and "could not open" in ck["problems"][0]
    a = pd.read_csv(out / "test" / "metadata" / "aligned_records.csv", dtype={"video_id": str})
    assert set(a["video_id"]) == {"6"}                    # the broken one is excluded from the merge
    from src.features import verify_preprocessing as vp
    assert vp.main(args[:6] + ["--out", str(tmp_path / "v.md")]) == 1   # and the verifier reports it


def test_bar_mass_fraction():
    cfg = PreprocessConfig()
    img = np.zeros((768, 1024), np.uint8)
    img[300:340, 500:540] = 200          # 1600 px inside the crop
    img[80:96, 500:540] = 200            # 640 px in the top bar, touching the boundary
    d = gaze_descriptors(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cfg)
    assert d["gaze_bar_mass_frac"] == pytest.approx(640 / 2240, abs=1e-4)
    assert d["gaze_bar_max"] == 200


def test_gaze_bar_diagnostic_runs(dataset, tmp_path):
    from src.features import gaze_bar_diagnostic as gb
    assert gb.main(["--bdda-root", str(dataset["bdda"]), "--manifest", str(dataset["manifest"]),
                    "--out-dir", str(tmp_path), "--n-random", "1"]) == 0
    assert "Verdict: A" in (tmp_path / "gaze_bar_diagnostic.md").read_text()
    pytest.importorskip("matplotlib")
    assert (tmp_path / "gaze_bar_diagnostic.png").exists()
