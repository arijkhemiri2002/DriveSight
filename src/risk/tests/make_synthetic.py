"""
Self-check for src/risk without the real BDDA data.

Builds a synthetic sandbox NEXT TO the repository (../DriveSight_sandbox by default): fake GPS
files with a braking event, fake gaze maps, fake detections, and a copy of src/. Nothing in the
repository is touched. In the synthetic clips, an in-path car grows and attracts the gaze during
the 2.5 s before braking, so a working pipeline must score clearly above the time-only reference.

Run from the repository root:
    python src/risk/tests/make_synthetic.py
then follow the commands it prints.
"""
import json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd, cv2

repo = Path(__file__).resolve().parents[3]
root = Path(sys.argv[1]) if len(sys.argv) > 1 else repo.parent / "DriveSight_sandbox"
if root.exists(): shutil.rmtree(root)
shutil.copytree(repo / "src", root / "src")
rng = np.random.default_rng(1)
W, H = 1280, 720
man, det_rows, done = [], [], []
splits = {"training": 60, "validation": 25, "test": 25}
vid = 0
for split, n in splits.items():
    recs = []
    for _ in range(n):
        vid += 1; v = str(vid)
        k0 = int(rng.integers(4, 8))            # onset at 0.5 + k0 s
        onset = 0.5 + k0
        no_event = rng.random() < 0.08
        times = 0.5 + np.arange(11)
        speed = np.full(11, 12.0) + rng.normal(0, 0.2, 11)
        if not no_event:
            speed[k0 + 1:] = speed[k0] - 3.2 * np.arange(1, 11 - k0)
            speed = np.clip(speed, 0, None)
        if rng.random() < 0.05: speed[2] = -1   # invalid reading
        start = 1_500_000_000_000
        gps = {"startTime": start, "endTime": start + 10000,
               "locations": [{"timestamp": int(start + t * 1000), "speed": float(s), "course": 90.0,
                              "latitude": 37.0, "longitude": -122.0, "accuracy": 5} for t, s in zip(times, speed)]}
        gp = root / "data/BDDA" / split / "gps_jsons" / f"{v}.json"; gp.parent.mkdir(parents=True, exist_ok=True)
        gp.write_text(json.dumps(gps))
        man.append({"split": split, "video_id": vid, "cam_duration": 10.0, "gaze_duration": 10.05})
        hazard = (not no_event) and rng.random() < 0.75
        for i in range(51):
            t = round(i * 0.2, 3)
            gz = np.zeros((H, W), np.float32)
            ys, xs = np.mgrid[0:H:4, 0:W:4]
            cx, cy = 640, 330
            if hazard and onset - 2.5 <= t <= onset + 1:
                a = 0.02 + 0.12 * np.clip((t - (onset - 2.5)) / 2.5, 0, 1)
                bw, bh = np.sqrt(a) * W * 1.2, np.sqrt(a) * H * 0.8
                x1, y2 = 640 - bw / 2, 400 + 250 * a / 0.14
                det_rows.append([split, v, t, "car", 0.8, x1, y2 - bh, x1 + bw, y2]); cx, cy = 640, y2 - bh / 2
                if rng.random() < 0.3:
                    det_rows.append([split, v, t, "pedestrian", 0.5, 820, 380, 860, 480])
            for _ in range(rng.integers(0, 4)):
                x = rng.uniform(0, 1100)
                det_rows.append([split, v, t, str(rng.choice(["car", "traffic sign", "truck"])), float(rng.uniform(0.2, 0.9)),
                                 x, 300, x + 80, 360])
            cx += rng.normal(0, 30); cy += rng.normal(0, 20)
            g = np.exp(-(((np.arange(W)[None] - cx) / 90) ** 2 + ((np.arange(H)[:, None] - cy) / 70) ** 2))
            gpth = root / "data/processed" / split / "frames/gaze" / v / f"{int(round(t*1000)):06d}.jpg"
            gpth.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(gpth), cv2.cvtColor((g * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, 60])
            recs.append({"split": split, "video_id": vid, "time_seconds": t,
                         "camera_frame_path": f"C:\\Users\\arijk\\fake\\{v}\\{i}.jpg",   # foreign absolute path
                         "gaze_frame_path": "C:\\nope.jpg", "gps_speed_kmh": 0})
        done.append(f"{split}/{v}")
    md = root / "data/processed" / split / "metadata"; md.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_csv(md / "aligned_records.csv", index=False)
(root / "reports/video_dataset").mkdir(parents=True, exist_ok=True)
pd.DataFrame(man).to_csv(root / "reports/video_dataset/dataset_manifest.csv", index=False)
(root / "data/risk").mkdir(parents=True, exist_ok=True)
pd.DataFrame(det_rows, columns=["split", "video_id", "time_seconds", "cls", "conf", "x1", "y1", "x2", "y2"]).to_csv(
    root / "data/risk/detections.csv", index=False)
(root / "data/risk/detections_done.txt").write_text("\n".join(done) + "\n")
print(f"Sandbox ready: {root}")
print("Now run:")
print(f"  cd \"{root}\"")
for c in ("braking_labels.py", "build_features.py", "train_risk_model.py --final-test"):
    print(f"  python src/risk/{c}")
print("Then open reports/risk/risk_model_report.md and reports/risk/figures/ in the sandbox.")
