"""
Build a clean dataset manifest from the temporal synchronization audit.

Reads reports/video_dataset/audit_all_sequences.csv and keeps only the
sequences where sync_ok = True. Preserves the training / validation / test
splits. Does not touch any original data files.

Outputs (in reports/video_dataset/):
* dataset_manifest.csv   — one row per retained sequence
* dataset_manifest_report.md — human-readable summary

Run:  python src/features/build_clean_manifest.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports" / "video_dataset"
AUDIT_CSV = REPORT_DIR / "audit_all_sequences.csv"
MANIFEST_CSV = REPORT_DIR / "dataset_manifest.csv"
REPORT_MD = REPORT_DIR / "dataset_manifest_report.md"

SPLITS = ["training", "validation", "test"]


def build_manifest() -> pd.DataFrame:
    audit = pd.read_csv(AUDIT_CSV)
    kept = audit[audit["sync_ok"] == True].copy()  # noqa: E712
    kept = kept.sort_values(["split", "video_id"], key=lambda s: s.astype(str)).reset_index(drop=True)
    return audit, kept


def validate_manifest(audit: pd.DataFrame, kept: pd.DataFrame) -> dict:
    stats = {
        "total_audited": len(audit),
        "retained": len(kept),
        "excluded": len(audit) - len(kept),
    }
    per_split = []
    for split in SPLITS:
        sub_a = audit[audit["split"] == split]
        sub_k = kept[kept["split"] == split]
        per_split.append({
            "split": split,
            "audited": len(sub_a),
            "retained": len(sub_k),
            "excluded": len(sub_a) - len(sub_k),
        })
    stats["per_split"] = per_split

    # Every retained sequence has readable camera & gaze + reliable GPS
    bad_cam = kept[(kept["cam_readable"] != True) | (kept["resolution_camera_ok"] != True)]  # noqa: E712
    bad_gaze = kept[(kept["gaze_readable"] != True) | (kept["resolution_gaze_ok"] != True)]  # noqa: E712
    bad_gps = kept[
        (kept["gps_readable"] != True)
        | (kept["gps_sorted"] != True)
        | (kept["gps_dups"].fillna(0) > 0)
        | (kept["gps_covers_video"] != True)
        | (kept["gps_interval_ok"] != True)
    ]  # noqa: E712
    stats["bad_camera"] = len(bad_cam)
    stats["bad_gaze"] = len(bad_gaze)
    stats["bad_gps"] = len(bad_gps)

    # No video ID in multiple splits
    id_counts = kept["video_id"].astype(str).value_counts()
    dup_ids = id_counts[id_counts > 1]
    stats["duplicate_ids_across_splits"] = len(dup_ids)
    stats["dup_ids"] = dup_ids.index.tolist()

    # All retained are sync_ok and complete
    stats["all_retained_sync_ok"] = bool((kept["sync_ok"] == True).all())  # noqa: E712
    stats["all_retained_complete"] = bool((kept["complete"] == True).all())

    return stats


def write_manifest(kept: pd.DataFrame) -> None:
    keep_cols = [
        "split", "video_id",
        "cam_w", "cam_h", "cam_fps", "cam_frames", "cam_duration",
        "gaze_w", "gaze_h", "gaze_fps", "gaze_frames", "gaze_duration",
        "gps_records", "gps_duration", "gps_median_interval",
        "gps_max_interval", "gps_first_offset", "gps_last_offset",
        "cam_gaze_duration_diff",
    ]
    out = kept[keep_cols].copy()
    out["camera_path"] = out.apply(
        lambda r: f"data/BDDA/{r['split']}/camera_videos/{r['video_id']}.mp4", axis=1
    )
    out["gaze_path"] = out.apply(
        lambda r: f"data/BDDA/{r['split']}/gazemap_videos/{r['video_id']}_pure_hm.mp4", axis=1
    )
    out["gps_path"] = out.apply(
        lambda r: f"data/BDDA/{r['split']}/gps_jsons/{r['video_id']}.json", axis=1
    )
    out.to_csv(MANIFEST_CSV, index=False)


def write_report(audit: pd.DataFrame, kept: pd.DataFrame, stats: dict) -> None:
    lines: list[str] = []
    lines.append("# Clean Dataset Manifest — Sync-OK Sequences Only")
    lines.append("")
    lines.append("Source: `reports/video_dataset/audit_all_sequences.csv`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total sequences audited: {stats['total_audited']}")
    lines.append(f"- Retained (sync_ok = True): {stats['retained']}")
    lines.append(f"- Excluded (sync_ok = False or incomplete): {stats['excluded']}")
    lines.append("")

    lines.append("## Per-split breakdown")
    lines.append("")
    lines.append("| Split | Audited | Retained | Excluded | Retention rate |")
    lines.append("|-------|---------|----------|----------|----------------|")
    for row in stats["per_split"]:
        rate = 100.0 * row["retained"] / row["audited"] if row["audited"] else 0.0
        lines.append(
            f"| {row['split']} | {row['audited']} | {row['retained']} | "
            f"{row['excluded']} | {rate:.1f}% |"
        )
    lines.append("")

    lines.append("## Retained-sequence guarantees")
    lines.append("")
    lines.append(f"- All retained sequences have sync_ok = True: {stats['all_retained_sync_ok']}")
    lines.append(f"- All retained sequences are complete (camera + gaze + GPS): {stats['all_retained_complete']}")
    lines.append(f"- Retained sequences with readable camera video & valid resolution: "
                 f"{stats['retained'] - stats['bad_camera']} / {stats['retained']}")
    lines.append(f"- Retained sequences with readable gaze video & valid resolution: "
                 f"{stats['retained'] - stats['bad_gaze']} / {stats['retained']}")
    lines.append(f"- Retained sequences with reliable GPS synchronization: "
                 f"{stats['retained'] - stats['bad_gps']} / {stats['retained']}")
    lines.append(f"- Video IDs appearing in multiple splits: {stats['duplicate_ids_across_splits']}")
    if stats["dup_ids"]:
        lines.append(f"  Duplicate IDs: {stats['dup_ids']}")
    lines.append("")
    lines.append("## Original data")
    lines.append("")
    lines.append("- No original files were deleted or modified.")
    lines.append("- The 274 excluded sequences remain in `data/BDDA/` but are not part of this manifest.")
    lines.append("- The manifest in `dataset_manifest.csv` is the authoritative list for preprocessing.")
    lines.append("")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {REPORT_MD}")


def main() -> None:
    audit, kept = build_manifest()
    stats = validate_manifest(audit, kept)
    write_manifest(kept)
    write_report(audit, kept, stats)

    print(f"\nManifest: {MANIFEST_CSV} ({len(kept)} sequences)")
    print(f"Retained per split: "
          f"{ {r['split']: r['retained'] for r in stats['per_split']} }")
    print(f"Excluded per split: "
          f"{ {r['split']: r['excluded'] for r in stats['per_split']} }")


if __name__ == "__main__":
    main()
