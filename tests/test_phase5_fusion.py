"""Phase 5 tests on synthetic data (no video, no BDDA).

A small processed root is written by hand: Phase 2 rows (gaze descriptors, nearest 1 Hz GPS record) and
Phase 4 tracked boxes whose behaviour is known (a car ahead that grows before the braking, a pedestrian
looked at or not). Unit tests check the labels, the causal GPS, object motion and attention; the
end-to-end tests run the fusion, the independent verifier (including tampering and a deliberate
look-ahead leak) and the model training with its one-shot test protection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fusion import run_fusion as rf
from src.fusion import train_risk as trk
from src.fusion import verify_fusion as vf
from src.fusion import fusion_core as fc
from src.fusion.fusion_config import FusionConfig
from src.fusion.fusion_core import ALL_FEATURES, fuse_sequence, object_features
from src.fusion.gps_labels import causal_kinematics, find_braking_event, gps_records

CLS = ["pedestrian", "rider", "car", "truck", "bus", "motorcycle", "bicycle", "traffic light", "traffic sign"]


# ------------------------------------------------------------------------------------ synthetic data
def make_rows(split, vid, speeds, duration=9.8, gaze=(0.5, 0.55), spread=0.08):
    """Phase 2 rows at 5 Hz; `speeds[k]` is the GPS speed (m/s) of the record at t = k + 0.5 s."""
    rec_t = np.arange(len(speeds)) + 0.5
    out = []
    for k, t in enumerate(np.round(np.arange(0, duration + 1e-9, 0.2), 1)):
        j = int(np.argmin(np.abs(rec_t - t)))
        out.append(dict(split=split, video_id=str(vid), sample_idx=k, t_sec=t, t_ms=int(round(t * 1000)),
                        gaze_ok=True, gaze_empty=False, gaze_mean=3.0, gaze_active_frac=0.04,
                        gaze_peak_x=gaze[0], gaze_peak_y=gaze[1], gaze_centroid_x=gaze[0], gaze_centroid_y=gaze[1],
                        gaze_spread=spread, gaze_n_regions=1, gps_matched=True, gps_rec_t_sec=rec_t[j],
                        gps_speed_m_s=speeds[j], gps_speed_valid=True))
    return pd.DataFrame(out)


def tbox(split, vid, k, det, tid, x, y, w, h, cls=2, conf=0.9):
    x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    return dict(split=split, video_id=str(vid), sample_idx=k, t_ms=200 * k, det_idx=det, track_id=tid, track_frame=0,
                is_birth=False, cls_id=cls, cls_name=CLS[cls], is_vru=CLS[cls] in fc.VRU_CLASSES, conf=conf,
                x1=x1, y1=y1, x2=x2, y2=y2, cx_n=x / 1280, cy_n=y / 720, w_n=w / 1280, h_n=h / 720,
                area_frac=w * h / (1280 * 720), bottom_y_n=y2 / 720, x_offset_n=x / 1280 - 0.5, touches_border=False)


def make_tracks(split, vid, n=50, brake=True):
    """A car ahead in the ego path; it grows quickly from t = 3 s if the clip has a braking event."""
    rows = []
    for k in range(n):
        t = k * 0.2
        h = 100 * (np.exp(0.6 * (t - 3.0)) if (brake and t > 3.0) else 1.0)
        rows.append(tbox(split, vid, k, 0, 1, 640, 450, 1.4 * h, min(h, 300)))
        rows.append(tbox(split, vid, k, 1, 2, 200, 420, 40, 100, cls=0, conf=0.8))     # pedestrian at the side
    return pd.DataFrame(rows)


BRAKE = [8, 8, 8, 8, 8, 8, 4, 2, 2, 2]          # hard decel between the records at 5.5 s and 6.5 s
MILD = [8, 8, 8, 8, 8, 8, 7, 6, 6, 6]           # 1 m/s^2 only


def write_root(root: Path, n_per_split=None, seed=0):
    rng = np.random.default_rng(seed)
    n_per_split = n_per_split or {"training": 24, "validation": 12, "test": 10}
    vid = 1
    for split, n in n_per_split.items():
        recs, trks = [], []
        for i in range(n):
            brake = i % 3 != 0
            speeds = [s + rng.normal(0, 0.05) for s in (BRAKE if brake else MILD)]
            recs.append(make_rows(split, vid, speeds))
            trks.append(make_tracks(split, vid, brake=brake))
            vid += 1
        md, td = root / split / "metadata", root / split / "tracks"
        md.mkdir(parents=True)
        td.mkdir(parents=True)
        pd.concat(recs).to_csv(md / "aligned_records.csv", index=False)
        (md / "_schema.json").write_text(json.dumps({"schema_version": "2.1", "config_hash": "p2", "missing": []}))
        pd.concat(trks).to_parquet(td / "tracks.parquet", index=False)
        (td / "_schema.json").write_text(json.dumps({"schema_version": "4.0", "config_hash": "p4", "missing": []}))
    return root


@pytest.fixture()
def root(tmp_path):
    return write_root(tmp_path / "processed")


def cfg_for(tmp_path, root, **kw):
    return FusionConfig(processed_root=str(root), reports_dir=str(tmp_path / "reports"),
                        models_dir=str(tmp_path / "models"), **kw)


def common_args(tmp_path, root):
    return ["--processed-root", str(root), "--reports-dir", str(tmp_path / "reports")]


# ------------------------------------------------------------------------------------------- labels
def test_braking_onset_is_the_start_of_the_hard_interval():
    rows = make_rows("training", 1, BRAKE)
    ev = find_braking_event(gps_records(rows), FusionConfig())
    assert ev["has_event"] and ev["onset_sec"] == pytest.approx(5.5) and ev["peak_decel"] == pytest.approx(4.0)


def test_mild_braking_and_standstill_are_not_hard_braking():
    cfg = FusionConfig()
    assert not find_braking_event(gps_records(make_rows("training", 1, MILD)), cfg)["has_event"]
    slow = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]        # -1 m/s^2 ... and below min speed anyway
    assert not find_braking_event(gps_records(make_rows("training", 1, slow)), cfg)["has_event"]


def test_gps_kinematics_never_read_a_future_record():
    rows = make_rows("training", 1, BRAKE)
    g = gps_records(rows)
    k = causal_kinematics(g, np.array([0.2, 5.4, 6.0, 6.6]), FusionConfig())
    assert np.isnan(k["speed_kmh"][0])                               # first record is at 0.5 s
    assert k["speed_kmh"][1] == pytest.approx(8 * 3.6)               # 5.4 s: last record 4.5 s
    assert k["accel_past"][2] == pytest.approx(0.0)                  # 6.0 s: braking record (6.5 s) not seen yet
    assert k["accel_past"][3] == pytest.approx(-4.0)                 # 6.6 s: now it is


def test_labels_scope_and_warmup():
    cfg = FusionConfig()
    fr, _, ev = fuse_sequence("training", "1", make_rows("training", 1, BRAKE), make_tracks("training", 1), cfg)
    pos = fr[fr["y"] == 1]["t_sec"]
    assert pos.min() == pytest.approx(3.6) and pos.max() == pytest.approx(5.4)   # 5 Hz samples in [3.5, 5.5]
    assert not fr[fr["t_sec"] < 1.4]["in_scope"].any()                            # warm-up
    assert not fr[fr["t_sec"] > 5.5]["in_scope"].any()                            # after the onset
    fr2, _, _ = fuse_sequence("training", "2", make_rows("training", 2, MILD), make_tracks("training", 2, brake=False), cfg)
    assert (fr2["y"] == 0).all() and fr2[fr2["t_sec"] >= 1.4]["in_scope"].all()   # mild clip: negatives


# ----------------------------------------------------------------------------------- object features
def test_growing_object_gives_positive_scale_rate_and_contact_time():
    cfg = FusionConfig()
    rows = make_rows("training", 1, BRAKE)
    o = object_features(make_tracks("training", 1), rows, cfg)
    car = o[(o["track_id"] == 1)].set_index("t_ms")
    assert np.isnan(car.loc[400, "scale_rate"])                      # no box 0.6-1.4 s earlier
    assert car.loc[2000, "scale_rate"] == pytest.approx(0.0, abs=1e-9)
    assert car.loc[4400, "scale_rate"] == pytest.approx(0.6, rel=0.02)   # exp(0.6 t): log-growth 0.6 / s
    assert car.loc[4400, "ttc_sec"] == pytest.approx(1 / 0.6, rel=0.02)
    assert car.loc[4400, "in_path"] and not o[o["track_id"] == 2]["in_path"].any()


def test_attention_is_high_on_the_looked_at_object_only():
    cfg = FusionConfig()
    rows = make_rows("training", 1, BRAKE, gaze=(200 / 1280, 420 / 720), spread=0.05)   # gaze on the pedestrian
    o = object_features(make_tracks("training", 1), rows, cfg)
    ped, car = o[o["track_id"] == 2], o[o["track_id"] == 1]
    assert (ped["attention"] == 1.0).all() and ped["gaze_peak_in_box"].all()
    early = car["t_ms"] < 4000                                       # later the synthetic car covers the gaze
    assert (car.loc[early, "attention"] < 0.01).all()
    fr, _, _ = fuse_sequence("training", "1", rows, make_tracks("training", 1), cfg)
    assert fr["vru_attention"].dropna().eq(1.0).all()
    assert (fr.loc[fr["t_ms"] < 4000, "lead_attention"] < 0.01).all()


def test_low_confidence_boxes_are_not_counted():
    cfg = FusionConfig()
    t = make_tracks("training", 1)
    t.loc[t["track_id"] == 2, "conf"] = 0.10                         # pedestrian below conf_vru = 0.15
    fr, _, _ = fuse_sequence("training", "1", make_rows("training", 1, BRAKE), t, cfg)
    assert (fr["n_vru"] == 0).all() and (fr["vru_present"] == 0).all() and (fr["n_vehicle"] == 1).all()


def test_no_clip_time_among_features():
    assert not (vf.TIME_COLUMNS & set(ALL_FEATURES))
    assert "gps_age_sec" not in ALL_FEATURES


def test_same_time_auc_ignores_clip_time():
    t = np.repeat(np.arange(10) * 200, 20)
    y = (t >= 1000).astype(int)
    rng = np.random.default_rng(0)
    y2 = rng.integers(0, 2, len(t))
    assert np.isnan(trk.same_time_auc(y, t.astype(float), t))       # each time has one class only
    assert trk.same_time_auc(y2, t.astype(float), t) == pytest.approx(0.5)   # a pure time score is 0.5
    assert trk.same_time_auc(y2, y2.astype(float), t) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------- end to end
def test_fusion_end_to_end_verify_and_resume(tmp_path, root, capsys):
    rf.main(common_args(tmp_path, root))
    cfg = cfg_for(tmp_path, root)
    fr = pd.read_parquet(cfg.frame_features_path("training"))
    assert len(fr) == len(pd.read_csv(cfg.records_path("training")))
    vf.main(common_args(tmp_path, root) + ["--n-causal", "4"])
    assert "ALL CHECKS PASSED" in capsys.readouterr().out
    rf.main(common_args(tmp_path, root))
    assert capsys.readouterr().out.count("up to date") == 3


def test_verifier_detects_tampering_and_changed_inputs(tmp_path, root, capsys):
    rf.main(common_args(tmp_path, root))
    cfg = cfg_for(tmp_path, root)
    p = cfg.frame_features_path("validation")
    df = pd.read_parquet(p)
    df.loc[df.index[10], "y"] = 1 - df.loc[df.index[10], "y"]
    df.to_parquet(p, index=False)
    with pytest.raises(SystemExit):
        vf.main(common_args(tmp_path, root) + ["--n-causal", "2", "--split", "validation"])
    out = capsys.readouterr().out
    assert "frame_features.parquet intact" in out and "y consistent" in out
    rf.main(common_args(tmp_path, root) + ["--split", "validation"])          # hash mismatch -> recomputed
    tr = pd.read_parquet(cfg.tracks_path("test"))
    tr.loc[tr.index[0], "conf"] = 0.5
    tr.to_parquet(cfg.tracks_path("test"), index=False)
    with pytest.raises(SystemExit):
        vf.main(common_args(tmp_path, root) + ["--n-causal", "2", "--split", "test"])
    assert "inputs unchanged" in capsys.readouterr().out


def test_verifier_catches_a_look_ahead_leak(tmp_path, root, capsys, monkeypatch):
    real = fc.frame_features

    def leaky(rows, obj, cfg):                    # puts the speed of the LAST row into every row
        f = real(rows, obj, cfg)
        f["speed_trend"] = float(rows["gps_speed_m_s"].iloc[-1])
        return f

    monkeypatch.setattr(fc, "frame_features", leaky)
    rf.main(common_args(tmp_path, root) + ["--split", "training"])
    with pytest.raises(SystemExit):
        vf.main(common_args(tmp_path, root) + ["--n-causal", "6", "--split", "training"])
    out = capsys.readouterr().out
    assert "features unchanged when the future is removed" in out and "speed_trend" in out


def test_refuses_to_write_into_upstream_folders(tmp_path):
    from src.fusion.fusion_config import assert_safe_output
    with pytest.raises(RuntimeError):
        assert_safe_output(tmp_path / "data" / "processed" / "training" / "tracks" / "x.parquet")
    with pytest.raises(RuntimeError):
        assert_safe_output(tmp_path / "data" / "BDDA" / "x.csv")
    assert_safe_output(tmp_path / "data" / "processed" / "training" / "fusion" / "x.parquet")


def test_training_report_models_and_one_shot_test(tmp_path, root, capsys):
    rf.main(common_args(tmp_path, root))
    args = common_args(tmp_path, root) + ["--models-dir", str(tmp_path / "models"), "--n-bootstrap", "10"]
    trk.main(args)
    rep = tmp_path / "reports"
    assert (rep / "phase5_model_validation.md").exists() and (tmp_path / "models" / "risk_model_fusion.joblib").exists()
    v = json.loads((rep / "phase5_model_validation.json").read_text())
    assert v["metrics"]["time-only"]["same_time_AUC"] == pytest.approx(0.5)
    assert v["metrics"][v["selected"]]["same_time_AUC"] > 0.8         # the growing car is learnable
    import joblib
    b = joblib.load(tmp_path / "models" / "risk_model_vision_v5.joblib")
    assert b["features"] == fc.VISION_FEATURES
    trk.main(args + ["--final-test"])
    assert (rep / "phase5_model_test.md").exists()
    with pytest.raises(SystemExit):
        trk.main(args + ["--final-test"])                            # second look at test is refused
    trk.main(args + ["--final-test", "--allow-retest"])
    assert json.loads((rep / "phase5_model_test.json").read_text())["test_runs"] == 2


# ------------------------------------------------------------------------------------- demo app
def test_demo_app_uses_the_phase5_vision_model(tmp_path, root, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("ultralytics")
    rf.main(common_args(tmp_path, root))
    trk.main(common_args(tmp_path, root) + ["--models-dir", str(tmp_path / "models"), "--n-bootstrap", "5"])
    import app.risk_engine as re_

    if not hasattr(re_, "V5_MODEL"):
        pytest.skip("demo app without the Phase 5 integration (front not in this branch)")
    monkeypatch.setattr(re_, "V5_MODEL", tmp_path / "models" / "risk_model_vision_v5.joblib")
    video = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
    for _ in range(30 * 8):
        w.write(np.zeros((360, 640, 3), np.uint8))
    w.release()

    class Box:                                   # what ultralytics results expose
        def __init__(self, xyxy, conf, cls):
            import torch
            self.xyxy, self.conf, self.cls = torch.tensor(xyxy), torch.tensor(conf), torch.tensor(cls)

        def __len__(self):
            return len(self.conf)

    class Net:                                   # fake detector: a car ahead that grows after 3 s
        calls = 0

        def predict(self, imgs, **kw):
            out = []
            for _ in imgs:
                t = Net.calls * 0.2
                Net.calls += 1
                h = 50 * (np.exp(0.6 * (t - 3)) if t > 3 else 1.0)
                h = min(h, 150)
                out.append(type("R", (), {"boxes": Box([[320 - 0.7 * h, 225 - h / 2, 320 + 0.7 * h, 225 + h / 2]],
                                                       [0.9], [2.0])})())
            return out

    res = re_.analyse(Net(), str(video), device="cpu", imgsz=640, iou=0.7, min_conf=0.05,
                      class_names=CLS, speed_kmh=40.0, max_sec=8.0)
    assert res["method"] == "learned-v5"
    risks = [f["risk"] for f in res["frames"]]
    assert len(risks) == len(res["frames"]) > 30 and all(0 <= r <= 1 for r in risks)
    assert np.mean(risks[20:28]) > np.mean(risks[7:14])          # growing car -> higher risk
    assert any("closing in fast" in f["reasons"] for f in res["frames"])
    ids = {d["id"] for f in res["frames"] for d in f["dets"] if "id" in d}
    assert len(ids) == 1                                          # one car, one track id from start to end
    assert res["tracking"]["tracks"] == 1 and all("sees" in f for f in res["frames"])
    late = res["frames"][22]                                     # t = 4.4 s, the box still grows
    assert late["dets"][0]["inpath"] and late["dets"][0]["ttc"] < 2.5 and late["sees"]["min_ttc"] < 2.5
