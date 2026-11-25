"""
Step 2 - Run the selected V2 detector (YOLOv8s) on every extracted camera frame.

Uses the V2 evaluation settings (imgsz 1152, NMS IoU 0.70) and keeps every box with
conf >= DET_MIN_CONF; per-class confidence thresholds are applied later, in build_features.py.

Output: data/risk/detections.csv, one row per box:
    split, video_id, time_seconds, cls, conf, x1, y1, x2, y2   (pixels, 1280x720 frame)
Resumable: finished sequences are listed in data/risk/detections_done.txt and skipped on rerun.
~59k frames at ~31 ms/frame on an RTX 2050 -> roughly 30-40 min for the full set.

Run from the repository root:
    python src/risk/detect_objects.py [--weights weights/final/yolov8s_v2_best.pt] [--device 0]
"""

from __future__ import annotations

import argparse
import csv
import os
import time

os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import risk_config as C  # noqa: E402
from records import load_records  # noqa: E402

FIELDS = ["split", "video_id", "time_seconds", "cls", "conf", "x1", "y1", "x2", "y2"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(C.DETECTOR_WEIGHTS))
    ap.add_argument("--device", default=None, help="'0' for the first GPU, 'cpu'; default: auto")
    ap.add_argument("--batch", type=int, default=C.DET_BATCH)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    model = YOLO(args.weights)
    names = model.names  # {idx: name}
    unknown = set(names.values()) - set(C.ALL_CLASSES)
    if unknown:
        raise SystemExit(f"Unexpected classes in {args.weights}: {unknown}")

    recs = load_records()
    recs = recs[recs["camera_path"] != ""]
    C.RISK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    done = set(C.DETECTIONS_DONE.read_text().split()) if C.DETECTIONS_DONE.exists() else set()
    new_file = not C.DETECTIONS_CSV.exists()
    groups = [(k, g) for k, g in recs.groupby(["split", "video_id"], sort=True) if f"{k[0]}/{k[1]}" not in done]
    print(f"Device {device} | {len(recs)} frames | {len(groups)} sequences to process ({len(done)} already done)")

    t0 = time.time()
    with open(C.DETECTIONS_CSV, "a", newline="", encoding="utf-8") as f, open(C.DETECTIONS_DONE, "a") as fd:
        w = csv.writer(f)
        if new_file:
            w.writerow(FIELDS)
        for n, ((split, vid), g) in enumerate(groups, 1):
            paths = g["camera_path"].tolist()
            times = dict(zip(paths, g["time_seconds"]))
            rows = []
            for i in range(0, len(paths), args.batch):
                chunk = paths[i:i + args.batch]
                results = model.predict(chunk, imgsz=C.DET_IMGSZ, iou=C.DET_NMS_IOU, conf=C.DET_MIN_CONF,
                                        device=device, verbose=False)
                for p, r in zip(chunk, results):
                    b = r.boxes
                    if b is None or len(b) == 0:
                        continue
                    xyxy, cls, conf = b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()
                    for (x1, y1, x2, y2), c, s in zip(xyxy, cls, conf):
                        rows.append([split, vid, times[p], names[int(c)], round(float(s), 4),
                                     round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)])
            w.writerows(rows)
            f.flush()
            fd.write(f"{split}/{vid}\n")
            fd.flush()
            if n % 25 == 0 or n == len(groups):
                el = time.time() - t0
                print(f"  {n}/{len(groups)} sequences, {el / 60:.1f} min elapsed, "
                      f"~{el / n * (len(groups) - n) / 60:.1f} min left")
    print(f"Detections -> {C.DETECTIONS_CSV}")


if __name__ == "__main__":
    main()
