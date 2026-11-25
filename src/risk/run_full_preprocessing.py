"""
Step 0 - Run the existing video preprocessing (src/features/video_preprocessing.py) on ALL
1,161 clean sequences instead of the 9-sequence pilot, without editing preprocess_config.py.

  - frames go to data/processed/<split>/frames/{camera,gaze}/<video_id>/<ms>.jpg (as in the pilot)
  - aligned records to data/processed/<split>/metadata/aligned_records.csv
  - the pilot summary (reports/video_dataset/pilot_preprocessing_summary.md) is kept; the full-run
    summary is written to reports/video_dataset/full_preprocessing_summary.md

Expect ~59k frames per modality (~13 GB) and a few hours. Run from the repository root:
    python src/risk/run_full_preprocessing.py
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

FEATURES_DIR = Path(__file__).resolve().parents[1] / "features"
sys.path.insert(0, str(FEATURES_DIR))

from preprocess_config import PreprocessConfig  # noqa: E402
from video_preprocessing import PROJECT_ROOT, VideoPreprocessor  # noqa: E402


class FullRunPreprocessor(VideoPreprocessor):
    def _write_pilot_summary(self) -> None:
        summary_dir = PROJECT_ROOT / "reports" / "video_dataset"
        pilot = summary_dir / "pilot_preprocessing_summary.md"
        backup = pilot.read_bytes() if pilot.exists() else None
        super()._write_pilot_summary()
        full = summary_dir / "full_preprocessing_summary.md"
        text = pilot.read_text(encoding="utf-8").replace("# Preprocessing Pilot Summary",
                                                         "# Preprocessing Summary (full run)", 1)
        full.write_text(text, encoding="utf-8")
        if backup is not None:
            pilot.write_bytes(backup)
        print(f"Full-run summary -> {full}")


def main() -> None:
    cfg = dataclasses.replace(PreprocessConfig(), pilot_mode=False)
    FullRunPreprocessor(cfg).run()


if __name__ == "__main__":
    main()
