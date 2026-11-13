from pathlib import Path
from collections import Counter
from PIL import Image
import json
import math
import random
import shutil


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROOT = PROJECT_ROOT / "data" / "bdd100k"

IMAGES_ROOT = ROOT / "images" / "10k"
LABELS_ROOT = ROOT / "labels" / "100k"

OUTPUT = ROOT / "yolo_dataset"
OUTPUT_YAML = OUTPUT / "data.yaml"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RANDOM_SEED = 42

# Keep the same nine classes as the previous DriveSight model.
CLASS_NAMES = [
    "pedestrian",     # 0
    "rider",          # 1
    "car",            # 2
    "truck",          # 3
    "bus",            # 4
    "motorcycle",     # 5
    "bicycle",        # 6
    "traffic light",  # 7
    "traffic sign",   # 8
]

# Map both modern and older BDD100K category names.
CLASS_TO_ID = {
    "pedestrian": 0,
    "person": 0,       # Older BDD100K name

    "rider": 1,

    "car": 2,
    "truck": 3,
    "bus": 4,

    "motorcycle": 5,
    "motor": 5,        # Older BDD100K name

    "bicycle": 6,
    "bike": 6,         # Older BDD100K name

    "traffic light": 7,
    "traffic sign": 8,
}


# ============================================================
# 2. CHECK INPUT PATHS AND EXISTING OUTPUT
# ============================================================

if not IMAGES_ROOT.exists():
    raise FileNotFoundError(
        f"Images directory not found:\n{IMAGES_ROOT}"
    )

if not LABELS_ROOT.exists():
    raise FileNotFoundError(
        f"Labels directory not found:\n{LABELS_ROOT}"
    )

# Protect existing generated data from accidental overwriting.
if OUTPUT.exists() and any(OUTPUT.iterdir()):
    raise RuntimeError(
        f"\nOutput folder already contains files:\n{OUTPUT}\n\n"
        "Rename the existing yolo_dataset folder to "
        "yolo_dataset_old, then run this script again."
    )


# ============================================================
# 3. INDEX JSON ANNOTATIONS
# ============================================================

print("Indexing JSON annotations...")

labels_by_stem = {}

for json_path in LABELS_ROOT.rglob("*.json"):
    stem = json_path.stem.lower()

    if stem in labels_by_stem:
        raise RuntimeError(
            f"Duplicate JSON annotation filename: {stem}"
        )

    labels_by_stem[stem] = json_path

print("JSON files indexed:", len(labels_by_stem))


# ============================================================
# 4. MATCH IMAGES TO ANNOTATIONS
# ============================================================

records = []
seen_image_stems = set()

for source_split in ["train", "val", "test"]:

    image_folder = IMAGES_ROOT / source_split

    if not image_folder.exists():
        print(f"Warning: missing image folder: {image_folder}")
        continue

    for image_path in image_folder.rglob("*"):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        stem = image_path.stem.lower()

        annotation_path = labels_by_stem.get(stem)

        # Keep only image files with a matching JSON.
        if annotation_path is None:
            continue

        if stem in seen_image_stems:
            raise RuntimeError(
                f"Duplicate image filename: {stem}"
            )

        seen_image_stems.add(stem)

        records.append({
            "stem": stem,
            "image": image_path,
            "annotation": annotation_path,
            "source_split": source_split,

            # Practical grouping heuristic. This is not confirmed
            # to represent the original video ID.
            "group": stem.split("-")[0],
        })

print("\nMatching image-label pairs:", len(records))

source_counts = Counter(
    record["source_split"] for record in records
)

print("Matched source images:", dict(source_counts))

if not records:
    raise RuntimeError("No matching image-annotation pairs found.")


# ============================================================
# 5. CREATE TRAIN / VAL / TEST SPLITS
# ============================================================

# Keep matched images from the original test folder for testing.
train_source = [
    r for r in records
    if r["source_split"] == "train"
]

test_source = [
    r for r in records
    if r["source_split"] == "test"
]

if not train_source or not test_source:
    raise RuntimeError(
        "Matched source training or test images are missing."
    )

train_groups = {r["group"] for r in train_source}
test_groups = {r["group"] for r in test_source}

# Avoid putting an identical grouping prefix in train and test.
group_overlap = train_groups & test_groups

if group_overlap:
    raise RuntimeError(
        f"{len(group_overlap)} filename prefixes appear in both "
        "train and test. Inspect these groups before proceeding."
    )

# Split the matched original training images approximately 80/20,
# keeping each filename-prefix group entirely in one split.
group_sizes = Counter(r["group"] for r in train_source)

groups = list(train_groups)
random.Random(RANDOM_SEED).shuffle(groups)

target_val_count = round(len(train_source) * 0.20)

val_groups = set()
val_count = 0

for group in groups:
    if val_count >= target_val_count:
        break

    val_groups.add(group)
    val_count += group_sizes[group]

splits = {
    "train": [
        r for r in train_source
        if r["group"] not in val_groups
    ],

    "val": [
        r for r in train_source
        if r["group"] in val_groups
    ],

    "test": test_source,
}

print("\nPlanned dataset splits:")

for split_name, split_records in splits.items():
    print(f"{split_name.upper()}: {len(split_records)} images")


# ============================================================
# 6. CONVERT BOXES TO YOLO FORMAT
# ============================================================

class_counts = Counter()
skipped_categories = Counter()

invalid_boxes = 0
empty_label_files = 0

successful_images = Counter()
failed_annotations = []

for split_name, split_records in splits.items():

    image_output_dir = OUTPUT / "images" / split_name
    label_output_dir = OUTPUT / "labels" / split_name

    image_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    for record in split_records:

        image_path = record["image"]
        annotation_path = record["annotation"]
        stem = record["stem"]

        try:
            # -----------------------------------------------
            # Read JSON annotation
            # -----------------------------------------------

            with annotation_path.open("r", encoding="utf-8") as f:
                annotation = json.load(f)

            if not isinstance(annotation, dict):
                raise ValueError("JSON root is not an object")

            annotated_name = Path(
                str(annotation.get("name", ""))
            ).stem.lower()

            if annotated_name != stem:
                raise ValueError(
                    f"Image name mismatch: {annotated_name} != {stem}"
                )

            # -----------------------------------------------
            # Read original image dimensions
            # -----------------------------------------------

            with Image.open(image_path) as image:
                width, height = image.size

                if width <= 0 or height <= 0:
                    raise ValueError("Invalid image dimensions")

            # Use local counters so failed files don't affect
            # the final class-count report.
            local_class_counts = Counter()
            local_skipped_categories = Counter()
            local_invalid_boxes = 0

            yolo_lines = []

            # -----------------------------------------------
            # Read objects and convert bounding boxes
            # -----------------------------------------------

            for frame in annotation.get("frames", []):

                for obj in frame.get("objects", []):

                    category = str(
                        obj.get("category", "")
                    ).strip().lower()

                    # Skip segmentation categories and classes
                    # not supported by our nine-class model.
                    if category not in CLASS_TO_ID:
                        local_skipped_categories[
                            category or "<missing>"
                        ] += 1
                        continue

                    box = obj.get("box2d")

                    if not isinstance(box, dict):
                        local_invalid_boxes += 1
                        continue

                    try:
                        x1 = float(box["x1"])
                        y1 = float(box["y1"])
                        x2 = float(box["x2"])
                        y2 = float(box["y2"])

                        coordinates = [x1, y1, x2, y2]

                        if not all(
                            math.isfinite(value)
                            for value in coordinates
                        ):
                            raise ValueError(
                                "Non-finite bounding-box coordinates"
                            )

                        # Clip coordinates to image boundaries.
                        x1 = max(0.0, min(x1, width))
                        x2 = max(0.0, min(x2, width))
                        y1 = max(0.0, min(y1, height))
                        y2 = max(0.0, min(y2, height))

                        box_width = x2 - x1
                        box_height = y2 - y1

                        if box_width <= 0 or box_height <= 0:
                            raise ValueError("Empty bounding box")

                        # YOLO format:
                        # class_id x_center y_center width height
                        # All four coordinates are normalized to [0, 1].
                        x_center = (x1 + x2) / (2 * width)
                        y_center = (y1 + y2) / (2 * height)

                        normalized_width = box_width / width
                        normalized_height = box_height / height

                        normalized_box = [
                            x_center,
                            y_center,
                            normalized_width,
                            normalized_height,
                        ]

                        if not all(
                            0.0 <= value <= 1.0
                            for value in normalized_box
                        ):
                            raise ValueError(
                                "Normalized box outside [0, 1]"
                            )

                        class_id = CLASS_TO_ID[category]

                        yolo_lines.append(
                            f"{class_id} "
                            + " ".join(
                                f"{value:.6f}"
                                for value in normalized_box
                            )
                        )

                        # Count under the canonical class name.
                        local_class_counts[
                            CLASS_NAMES[class_id]
                        ] += 1

                    except (KeyError, TypeError, ValueError):
                        local_invalid_boxes += 1

            # -----------------------------------------------
            # Write YOLO label and copy original image
            # -----------------------------------------------

            label_output_path = (
                label_output_dir / f"{stem}.txt"
            )

            image_output_path = (
                image_output_dir / image_path.name
            )

            label_output_path.write_text(
                "\n".join(yolo_lines),
                encoding="utf-8",
            )

            shutil.copy2(image_path, image_output_path)

            # Add counts only after the file was processed.
            class_counts.update(local_class_counts)
            skipped_categories.update(local_skipped_categories)

            invalid_boxes += local_invalid_boxes
            successful_images[split_name] += 1

            if not yolo_lines:
                empty_label_files += 1

        except Exception as exc:
            failed_annotations.append(
                (stem, str(exc))
            )


# ============================================================
# 7. CREATE YOLO DATA.YAML
# ============================================================

yaml_content = [
    f"path: '{OUTPUT.resolve().as_posix()}'",
    "train: images/train",
    "val: images/val",
    "test: images/test",
    "",
    "names:",
]

for class_id, class_name in enumerate(CLASS_NAMES):
    yaml_content.append(
        f"  {class_id}: '{class_name}'"
    )

OUTPUT_YAML.write_text(
    "\n".join(yaml_content) + "\n",
    encoding="utf-8",
)


# ============================================================
# 8. FINAL REPORT
# ============================================================

print("\n========== CONVERSION REPORT ==========")

print("Output directory:", OUTPUT)
print("YOLO configuration:", OUTPUT_YAML)

print("\nSuccessfully copied images and converted labels:")

for split_name in ["train", "val", "test"]:
    print(
        f"  {split_name.upper()}: "
        f"{successful_images[split_name]} images"
    )

print("\nBounding boxes by YOLO class:")

for class_name in CLASS_NAMES:
    print(
        f"  {class_name}: {class_counts[class_name]} boxes"
    )

print("\nInvalid boxes skipped:", invalid_boxes)
print("Images with empty label files:", empty_label_files)
print("Failed annotations:", len(failed_annotations))

if skipped_categories:
    print("\nUnsupported categories:")

    for category, count in skipped_categories.most_common():
        print(f"  {category}: {count}")

if failed_annotations:
    print("\nFirst annotation failures:")

    for stem, error in failed_annotations[:10]:
        print(f"  {stem}: {error}")

print("\nFinished. Verify the report before training.")