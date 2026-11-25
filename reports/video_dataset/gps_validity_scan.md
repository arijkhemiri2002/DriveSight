# GPS validity scan

Files scanned: 1435 (clean manifest: 1161)

Rules: speed valid if 0 <= speed <= 60.0 m/s; course valid if 0 <= course < 360; position valid if lat/lon in range, not (0,0), accuracy >= 0.

## Records with each issue

| Issue | All files: records | All files: files | Clean manifest: records | Clean manifest: files |
|---|---|---|---|---|
| n_dup_timestamps | 0 | 0 | 0 | 0 |
| n_unsorted | 0 | 0 | 0 | 0 |
| n_speed_negative | 29 | 18 | 27 | 16 |
| n_speed_missing | 0 | 0 | 0 | 0 |
| n_speed_too_high | 0 | 0 | 0 | 0 |
| n_course_invalid | 3 | 3 | 1 | 1 |
| n_accuracy_negative | 0 | 0 | 0 | 0 |
| n_position_zero | 0 | 0 | 0 | 0 |
| n_position_out_of_range | 0 | 0 | 0 | 0 |

Unreadable files: 0

## Clean-manifest sequences with any issue

| Split | Video | n_dup_timestamps | n_unsorted | n_speed_negative | n_speed_missing | n_speed_too_high | n_course_invalid | n_accuracy_negative | n_position_zero | n_position_out_of_range | negative speed values |
|---|---|---|---|---|---|---|---|---|---|---|---|
| training | 1027 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | -1 |
| training | 1219 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | -1 |
| training | 1253 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | -1 |
| training | 159 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | -1 |
| training | 1598 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | -1 |
| training | 1734 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | -1 |
| training | 303 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | -1 |
| training | 315 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | -1 |
| training | 634 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | -1 |
| training | 845 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | -1 |
| training | 865 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | -1 |
| validation | 1744 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | -1 |
| test | 1261 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | -1 |
| test | 1458 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | -1 |
| test | 220 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |  |
| test | 486 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | -1 |
| test | 503 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | -1 |

Handling in the pipeline: invalid values are set to NaN with a *_valid flag; acceleration is only computed between two valid consecutive records.
