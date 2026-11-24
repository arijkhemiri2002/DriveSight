# Clean Dataset Manifest — Sync-OK Sequences Only

Source: `reports/video_dataset/audit_all_sequences.csv`

## Summary

- Total sequences audited: 1435
- Retained (sync_ok = True): 1161
- Excluded (sync_ok = False or incomplete): 274

## Per-split breakdown

| Split | Audited | Retained | Excluded | Retention rate |
|-------|---------|----------|----------|----------------|
| training | 926 | 800 | 126 | 86.4% |
| validation | 203 | 95 | 108 | 46.8% |
| test | 306 | 266 | 40 | 86.9% |

## Retained-sequence guarantees

- All retained sequences have sync_ok = True: True
- All retained sequences are complete (camera + gaze + GPS): True
- Retained sequences with readable camera video & valid resolution: 1161 / 1161
- Retained sequences with readable gaze video & valid resolution: 1161 / 1161
- Retained sequences with reliable GPS synchronization: 1161 / 1161
- Video IDs appearing in multiple splits: 0

## Original data

- No original files were deleted or modified.
- The 274 excluded sequences remain in `data/BDDA/` but are not part of this manifest.
- The manifest in `dataset_manifest.csv` is the authoritative list for preprocessing.
