# Frame accuracy check: seek (pilot) vs sequential decoding (Phase 2)

Sequences: 11 | comparisons: 1360 | identical: 1360

**Verdict: seek and sequential decoding agree on every sampled frame**

| split      |   video_id | modality   |   fps |   frames_compared |   identical |   seek_failed |   max_abs_diff | offsets_idx:off   |
|:-----------|-----------:|:-----------|------:|------------------:|------------:|--------------:|---------------:|:------------------|
| test       |          6 | camera     | 30    |                50 |          50 |             0 |              0 |                   |
| test       |          6 | gaze       | 29    |                50 |          50 |             0 |              0 |                   |
| test       |         25 | camera     | 29.97 |                50 |          50 |             0 |              0 |                   |
| test       |         25 | gaze       | 29    |                50 |          50 |             0 |              0 |                   |
| test       |         33 | camera     | 30    |                50 |          50 |             0 |              0 |                   |
| test       |         33 | gaze       | 29    |                50 |          50 |             0 |              0 |                   |
| test       |        100 | camera     | 59.94 |                50 |          50 |             0 |              0 |                   |
| test       |        100 | gaze       | 29    |                50 |          50 |             0 |              0 |                   |
| training   |          2 | camera     | 29.97 |                50 |          50 |             0 |              0 |                   |
| training   |          2 | gaze       | 29    |                50 |          50 |             0 |              0 |                   |
| training   |          4 | camera     | 29.97 |                50 |          50 |             0 |              0 |                   |
| training   |          4 | gaze       | 29    |                50 |          50 |             0 |              0 |                   |
| training   |          5 | camera     | 30    |                50 |          50 |             0 |              0 |                   |
| training   |          5 | gaze       | 29    |                50 |          50 |             0 |              0 |                   |
| training   |       1698 | camera     | 30    |               160 |         160 |             0 |              0 |                   |
| training   |       1698 | gaze       | 29    |               160 |         160 |             0 |              0 |                   |
| validation |       1737 | camera     | 29.97 |                50 |          50 |             0 |              0 |                   |
| validation |       1737 | gaze       | 29    |                50 |          50 |             0 |              0 |                   |
| validation |       1744 | camera     | 29.97 |                70 |          70 |             0 |              0 |                   |
| validation |       1744 | gaze       | 29    |                70 |          70 |             0 |              0 |                   |
| validation |       1746 | camera     | 29.97 |                50 |          50 |             0 |              0 |                   |
| validation |       1746 | gaze       | 29    |                50 |          50 |             0 |              0 |                   |
