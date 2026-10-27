#!/usr/bin/env python3
"""
Convert the local BDD100K-style JSON annotations into a YOLO detection dataset.

Source data is read-only:
  data/dataset/images/train
  data/dataset/labels_json/train

Generated output:
  data/yolo/
    data.yaml
    images/{train,val,test}/...
    labels/{train,val,test}/...

The script refuses to overwrite an existing output directory. It maps the historical
nine classes as follows:
  pedestrian <- person
  rider <- rider
  car <- car
  truck <- truck
  bus <- bus
  motorcycle <- motor
  bicycle <- bike
  traffic light <- traffic light
  traffic sign <- traffic sign

All other BDD100K categories (lanes/areas/train etc.) are intentionally ignored.
"""
from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
SOURCE_IMAGES = REPO / "data" / "dataset" / "images" / "train"
SOURCE_LABELS = REPO / "data" / "dataset" / "labels_json" / "train"
OUTPUT = REPO / "data" / "yolo"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Order is important: it defines YOLO class IDs 0..8.
YOLO_CLASSES = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]
CATEGORY_MAP = {
    "person": "pedestrian",
    "rider": "rider",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motor": "motorcycle",
    "bike": "bicycle",
    "traffic light": "traffic light",
    "traffic sign": "traffic sign",
}
CLASS_TO_ID = {name: idx for idx, name in enumerate(YOLO_CLASSES)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def safe_stem(path: Path) -> str:
    return path.stem


def box_to_yolo(box: dict, width: int, height: int) -> tuple[float, float, float, float] | None:
    try:
        x1, y1 = float(box["x1"]), float(box["y1"])
        x2, y2 = float(box["x2"]), float(box["y2"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(map(lambda v: v == v and abs(v) != float("inf"), (x1, y1, x2, y2))):
        return None

    # Clip boxes to image bounds and reject degenerate boxes.
    x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
    y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None

    xc = ((x1 + x2) / 2.0) / width
    yc = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return xc, yc, bw, bh


def main() -> None:
    if not SOURCE_IMAGES.is_dir() or not SOURCE_LABELS.is_dir():
        raise FileNotFoundError(
            "Source directories not found. Expected:\n"
            f"  {SOURCE_IMAGES}\n  {SOURCE_LABELS}"
        )
    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {OUTPUT}\n"
            "Move/rename it manually after reviewing it, or choose a new OUTPUT path."
        )
    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
        raise ValueError("Train/val/test ratios must add up to 1.")

    images = sorted(p for p in SOURCE_IMAGES.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    labels_by_stem = {p.stem: p for p in SOURCE_LABELS.glob("*.json")}
    pairs = [(img, labels_by_stem[img.stem]) for img in images if img.stem in labels_by_stem]
    missing_json = [img.name for img in images if img.stem not in labels_by_stem]
    if not pairs:
        raise RuntimeError("No matching image/JSON pairs were found.")
    if missing_json:
        raise RuntimeError(
            f"{len(missing_json)} images have no matching JSON. Example: {missing_json[:5]}"
        )

    # Deterministic shuffle; same source pair always goes to the same split for seed 42.
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    split_pairs = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    for split in ("train", "val", "test"):
        (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=False)
        (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=False)

    class_counts = Counter()
    object_count = 0
    ignored_count = 0
    invalid_box_count = 0
    empty_label_count = 0

    for split, split_items in split_pairs.items():
        for img_path, json_path in split_items:
            # Hardlink avoids duplicating image bytes when source and output share a volume.
            # Fall back to copy2 if hardlinks are unavailable.
            dest_img = OUTPUT / "images" / split / img_path.name
            try:
                dest_img.hardlink_to(img_path)
            except OSError:
                shutil.copy2(img_path, dest_img)

            with Image.open(img_path) as im:
                width, height = im.size

            data = json.loads(json_path.read_text(encoding="utf-8"))
            frames = data.get("frames") or []
            objects = frames[0].get("objects", []) if frames else []
            lines = []

            for obj in objects:
                category = obj.get("category")
                class_name = CATEGORY_MAP.get(category)
                if class_name is None:
                    ignored_count += 1
                    continue
                yolo_box = box_to_yolo(obj.get("box2d", {}), width, height)
                if yolo_box is None:
                    invalid_box_count += 1
                    continue
                class_id = CLASS_TO_ID[class_name]
                lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in yolo_box))
                class_counts[class_name] += 1
                object_count += 1

            (OUTPUT / "labels" / split / f"{img_path.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            if not lines:
                empty_label_count += 1

    yaml_lines = [
        f"path: {OUTPUT.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(YOLO_CLASSES)}",
        "names:",
    ]
    yaml_lines.extend(f"  {idx}: {name!r}" for idx, name in enumerate(YOLO_CLASSES))
    (OUTPUT / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print("Conversion completed.")
    print(f"Source image/JSON pairs: {n}")
    print("Split sizes:", {key: len(value) for key, value in split_pairs.items()})
    print(f"YOLO objects retained: {object_count}")
    print(f"Objects in other categories ignored: {ignored_count}")
    print(f"Invalid/degenerate boxes skipped: {invalid_box_count}")
    print(f"Images with no retained boxes: {empty_label_count}")
    print("Retained object counts by class:")
    for class_name in YOLO_CLASSES:
        print(f"  {class_name}: {class_counts[class_name]}")
    print(f"YAML: {OUTPUT / 'data.yaml'}")
    print("Original images and JSON annotations were not modified.")


if __name__ == "__main__":
    main()
