# Temporal Synchronization Audit — DriveSight BDDA

Dataset root: `C:\Users\arijk\Desktop\DriveSight\DriveSight\data\BDDA`
Splits: training, validation, test

## 1. Sequence coverage

| Split | Sequences | Complete | Missing camera | Missing gaze | Missing GPS | Sync OK | Sync problems |
|-------|-----------|----------|----------------|--------------|-------------|---------|---------------|
| training | 926 | 926 | 0 | 0 | 0 | 800 | 126 |
| validation | 203 | 200 | 0 | 3 | 0 | 95 | 108 |
| test | 306 | 303 | 0 | 3 | 0 | 266 | 40 |

- Total sequences: 1435
- Complete sequences (3 modalities): 1429
- Sequences with synchronization problems: 274

## 2. Camera / gaze FPS distributions

| Modality | FPS | Count |
|----------|-----|-------|
| camera | 29.83 | 1 |
| camera | 29.92 | 2 |
| camera | 29.97 | 1002 |
| camera | 30.0 | 263 |
| camera | 59.83 | 1 |
| camera | 59.94 | 135 |
| camera | 60.0 | 23 |
| camera | 120.0 | 1 |
| gaze | 29.0 | 1429 |

## 3. Frame count and duration distributions

- Camera duration median: 10.01s
- Gaze duration median: 10.07s
- Camera frames median: 300.0
- Gaze frames median: 292.0
- Camera FPS: mean 33.38, median 29.97, range [29.83, 120.0]
- Gaze FPS: mean 29.00, median 29.00, range [29.0, 29.0]

## 4. GPS statistics

- GPS records median per sequence: 10.0
- GPS median interval median: 1.000s
- GPS max interval max: 6.000s
- Sequences with GPS covering video timeline: 1165
- Sequences with consistent GPS sampling: 1169

## 5. Resolution and FPS plausibility

- Camera resolution 1280x720 OK: 1428
- Gaze resolution 1024x768 OK: 1429
- Camera FPS plausible: 1427
- Gaze FPS plausible: 1429

## 6. Synchronization tolerances used

- Duration camera<->gaze tolerance: 1.0s (a few frames of container slack)
- Start offset tolerance: 0.5s (both videos start at t=0)
- Camera nominal FPS: {29.97, 30, 59.94}, tolerance 2.0fps
- Gaze nominal FPS: ~29-30, tolerance 2.0fps
- GPS nominal sampling: ~1 Hz; median interval tolerance 0.6s, max gap 2.5s
- GPS coverage: first record within 2.0s of video start, last record within 2.0s of video end
- Frame<->GPS association: nearest neighbor accepted only within 1.0s

## 7. Sequences with synchronization problems

| Split | Video ID | Problems |
|-------|----------|----------|
| training | 3 | GPS median interval 2.00s != ~1s; GPS timeline does not cover video duration (first offset 6.5, last gap 1.5) |
| training | 23 | GPS median interval 3.13s != ~1s; GPS max gap 4.03s > 2.5s |
| training | 52 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 64 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 72 | GPS median interval 2.00s != ~1s; GPS max gap 3.00s > 2.5s |
| training | 82 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 118 | GPS median interval 2.00s != ~1s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 127 | GPS median interval 2.00s != ~1s; GPS max gap 3.00s > 2.5s |
| training | 146 | camera fps 120.0 not plausible |
| training | 176 | GPS median interval 2.00s != ~1s; GPS max gap 3.00s > 2.5s |
| training | 177 | GPS median interval 2.00s != ~1s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 183 | GPS median interval 2.00s != ~1s; GPS max gap 4.00s > 2.5s |
| training | 194 | GPS timeline does not cover video duration (first offset 6.5, last gap 3.5) |
| training | 226 | GPS median interval 2.00s != ~1s; GPS max gap 3.00s > 2.5s |
| training | 239 | GPS median interval 2.00s != ~1s; GPS max gap 4.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 243 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 245 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 246 | GPS median interval 2.00s != ~1s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 250 | GPS median interval 2.00s != ~1s; GPS max gap 3.00s > 2.5s |
| training | 251 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s |
| training | 253 | GPS median interval 2.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 266 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s |
| training | 269 | GPS median interval 4.00s != ~1s; GPS max gap 6.00s > 2.5s |
| training | 277 | GPS median interval 4.49s != ~1s; GPS max gap 5.98s > 2.5s |
| training | 291 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 316 | GPS median interval 4.00s != ~1s; GPS max gap 6.00s > 2.5s |
| training | 356 | GPS median interval 2.00s != ~1s; GPS max gap 3.00s > 2.5s |
| training | 358 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 359 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s |
| training | 365 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.499, last gap 3.5) |
| training | 375 | GPS median interval 2.00s != ~1s; GPS max gap 3.00s > 2.5s |
| training | 378 | GPS median interval 2.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.501, last gap 3.5) |
| training | 403 | GPS median interval 5.00s != ~1s; GPS max gap 5.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 406 | GPS median interval 3.50s != ~1s; GPS max gap 5.00s > 2.5s |
| training | 430 | GPS median interval 2.00s != ~1s |
| training | 458 | GPS timeline does not cover video duration (first offset 6.5, last gap 3.5) |
| training | 465 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 6.5, last gap 0.5) |
| training | 469 | GPS timeline does not cover video duration (first offset 6.5, last gap 3.5) |
| training | 470 | GPS median interval 3.00s != ~1s; GPS max gap 4.00s > 2.5s; GPS timeline does not cover video duration (first offset 2.5, last gap 1.5) |
| training | 475 | GPS timeline does not cover video duration (first offset 6.5, last gap 3.5) |
| training | 489 | GPS median interval 2.00s != ~1s; GPS timeline does not cover video duration (first offset 6.5, last gap 1.5) |
| training | 491 | camera-gaze duration diff 6.74s > 1.0s |
| training | 502 | GPS median interval 2.04s != ~1s; GPS timeline does not cover video duration (first offset 0.449, last gap 3.5) |
| training | 508 | GPS median interval 2.99s != ~1s; GPS max gap 3.02s > 2.5s |
| training | 523 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 539 | GPS median interval 2.00s != ~1s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 558 | GPS median interval 5.00s != ~1s; GPS max gap 5.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 568 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.499, last gap 3.5) |
| training | 612 | GPS median interval 5.00s != ~1s; GPS max gap 5.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 613 | GPS median interval 3.00s != ~1s; GPS max gap 4.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 622 | GPS median interval 2.00s != ~1s; GPS max gap 4.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 632 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 640 | GPS median interval 2.00s != ~1s; GPS max gap 3.00s > 2.5s |
| training | 656 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 659 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 666 | GPS median interval 3.00s != ~1s; GPS max gap 4.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 672 | GPS median interval 2.00s != ~1s; GPS max gap 3.00s > 2.5s |
| training | 673 | GPS median interval 3.00s != ~1s; GPS max gap 4.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 698 | GPS median interval 4.97s != ~1s; GPS max gap 4.97s > 2.5s; GPS timeline does not cover video duration (first offset 1.525, last gap 3.5) |
| training | 702 | GPS median interval 4.00s != ~1s; GPS max gap 5.00s > 2.5s |
| training | 713 | GPS median interval 4.50s != ~1s; GPS max gap 6.00s > 2.5s |
| training | 759 | GPS median interval 2.00s != ~1s; GPS max gap 3.00s > 2.5s |
| training | 777 | GPS median interval 2.99s != ~1s; GPS max gap 3.00s > 2.5s |
| training | 788 | GPS median interval 2.50s != ~1s; GPS max gap 4.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 794 | GPS median interval 3.00s != ~1s; GPS max gap 4.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 821 | GPS median interval 6.00s != ~1s; GPS max gap 6.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 828 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 832 | GPS median interval 5.00s != ~1s; GPS max gap 5.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 836 | GPS median interval 4.96s != ~1s; GPS max gap 4.96s > 2.5s; GPS timeline does not cover video duration (first offset 1.537, last gap 3.5) |
| training | 871 | GPS median interval 6.00s != ~1s; GPS max gap 6.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 889 | GPS median interval 2.00s != ~1s; GPS timeline does not cover video duration (first offset 2.5, last gap 3.5) |
| training | 900 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 904 | GPS median interval 2.00s != ~1s; GPS max gap 3.00s > 2.5s |
| training | 922 | camera resolution NonexNone != 1280x720; camera fps None not plausible |
| training | 927 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 936 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 949 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s |
| training | 950 | GPS median interval 2.00s != ~1s; GPS max gap 3.00s > 2.5s |
| training | 979 | GPS median interval 4.00s != ~1s; GPS max gap 5.00s > 2.5s |
| training | 993 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 994 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 999 | GPS median interval 2.84s != ~1s; GPS max gap 3.45s > 2.5s; GPS timeline does not cover video duration (first offset 0.813, last gap 3.5) |
| training | 1000 | GPS median interval 6.00s != ~1s; GPS max gap 6.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 1040 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s |
| training | 1056 | GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 1080 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 1082 | GPS timeline does not cover video duration (first offset 6.5, last gap 3.5) |
| training | 1084 | GPS timeline does not cover video duration (first offset 6.5, last gap 3.5) |
| training | 1090 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 1102 | GPS median interval 2.00s != ~1s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 1104 | GPS timeline does not cover video duration (first offset 6.5, last gap 3.5) |
| training | 1116 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 1128 | GPS median interval 6.00s != ~1s; GPS max gap 6.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 1154 | GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 1186 | GPS median interval 3.02s != ~1s; GPS max gap 3.03s > 2.5s; GPS timeline does not cover video duration (first offset 0.452, last gap 3.5) |
| training | 1230 | GPS median interval 2.00s != ~1s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 1235 | GPS median interval 2.00s != ~1s; GPS timeline does not cover video duration (first offset 6.5, last gap 1.5) |
| training | 1243 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 1304 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 6.5, last gap 0.5) |
| training | 1310 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s |
| training | 1326 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 1333 | GPS median interval 2.00s != ~1s; GPS max gap 3.00s > 2.5s |
| training | 1337 | GPS median interval 2.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 1344 | GPS median interval 4.00s != ~1s; GPS max gap 5.00s > 2.5s |
| training | 1348 | GPS median interval 2.00s != ~1s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 1359 | GPS median interval 1.97s != ~1s; GPS max gap 3.06s > 2.5s |
| training | 1371 | GPS median interval 2.00s != ~1s; GPS timeline does not cover video duration (first offset 6.5, last gap 1.5) |
| training | 1375 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 6.5, last gap 0.501) |
| training | 1387 | GPS median interval 3.00s != ~1s; GPS max gap 4.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 1392 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 1401 | GPS median interval 3.50s != ~1s; GPS max gap 5.00s > 2.5s |
| training | 1440 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s |
| training | 1451 | GPS median interval 4.00s != ~1s; GPS max gap 5.00s > 2.5s |
| training | 1472 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s |
| training | 1487 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 3.5, last gap 3.5) |
| training | 1494 | GPS median interval 2.00s != ~1s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 1499 | GPS median interval 4.00s != ~1s; GPS max gap 6.00s > 2.5s |
| training | 1512 | GPS median interval 2.00s != ~1s; GPS max gap 4.00s > 2.5s |
| training | 1523 | GPS median interval 5.00s != ~1s; GPS max gap 5.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 1585 | GPS median interval 2.49s != ~1s; GPS max gap 2.94s > 2.5s; GPS timeline does not cover video duration (first offset 1.528, last gap 3.5) |
| training | 1605 | GPS median interval 4.00s != ~1s; GPS max gap 5.00s > 2.5s |
| training | 1677 | GPS median interval 2.00s != ~1s; GPS max gap 3.00s > 2.5s |
| training | 1680 | GPS timeline does not cover video duration (first offset 6.5, last gap 3.5) |
| training | 1691 | GPS median interval 5.00s != ~1s; GPS max gap 5.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| training | 1709 | GPS median interval 2.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| training | 1741 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 6.5, last gap 0.5) |
| validation | 1738 | missing modality: gaze |
| validation | 1739 | missing modality: gaze |
| validation | 1741 | missing modality: gaze |
| validation | 1755 | GPS median interval 3.00s != ~1s; GPS max gap 4.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| validation | 1760 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| validation | 1765 | GPS median interval 2.46s != ~1s; GPS max gap 2.99s > 2.5s |
| validation | 1776 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s |
| validation | 1787 | GPS timeline does not cover video duration (first offset 6.5, last gap 3.5) |
| validation | 1797 | GPS median interval 6.00s != ~1s; GPS max gap 6.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| validation | 1802 | GPS timeline does not cover video duration (first offset 0.493, last gap 3.5) |
| validation | 1806 | GPS median interval 2.00s != ~1s; GPS timeline does not cover video duration (first offset 4.5, last gap 3.5) |
| validation | 1824 | GPS timeline does not cover video duration (first offset 6.5, last gap 3.5) |
| validation | 1833 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 3.5, last gap 0.5) |
| validation | 1840 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| validation | 1841 | GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| validation | 1847 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s |
| validation | 1853 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| validation | 1891 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1892 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1893 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1894 |  |
| validation | 1896 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1899 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1900 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1903 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1905 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1906 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1907 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1910 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1911 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1912 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1913 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1915 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1916 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1917 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1918 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1919 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1920 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1921 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1922 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1924 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1925 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1927 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1928 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1930 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1932 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1933 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1935 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1936 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1938 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1939 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1941 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1942 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1943 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1944 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1945 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1946 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1948 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1950 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1951 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1952 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1953 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1954 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1955 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1956 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1957 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1958 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1959 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1960 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1961 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1962 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1963 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1964 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1965 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1966 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1967 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1968 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1969 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1970 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1972 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1975 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1978 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1979 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1980 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1981 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1984 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1985 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1986 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1987 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1988 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1989 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1990 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1992 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1993 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1995 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1996 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 1998 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 2000 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 2001 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 2002 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 2006 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 2008 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 2010 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 2011 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 2012 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 2013 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 2015 | GPS timeline does not cover video duration (first offset None, last gap None) |
| validation | 2017 | GPS timeline does not cover video duration (first offset None, last gap None) |
| test | 1 | missing modality: gaze |
| test | 9 | GPS timeline does not cover video duration (first offset 0.5, last gap 2.5) |
| test | 19 | GPS median interval 6.00s != ~1s; GPS max gap 6.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| test | 24 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| test | 89 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| test | 93 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 6.5, last gap 0.5) |
| test | 115 | GPS median interval 6.00s != ~1s; GPS max gap 6.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| test | 192 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| test | 284 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| test | 327 | GPS max gap 3.01s > 2.5s |
| test | 442 | GPS max gap 3.00s > 2.5s |
| test | 456 | GPS timeline does not cover video duration (first offset 6.5, last gap 3.5) |
| test | 480 | GPS median interval 5.00s != ~1s; GPS max gap 5.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| test | 724 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s |
| test | 735 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 6.5, last gap 0.5) |
| test | 742 | GPS median interval 3.00s != ~1s; GPS max gap 4.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| test | 810 | GPS timeline does not cover video duration (first offset 6.5, last gap 3.5) |
| test | 885 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| test | 887 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 6.5, last gap 0.5) |
| test | 899 | GPS timeline does not cover video duration (first offset 6.5, last gap 3.5) |
| test | 1037 | missing modality: gaze |
| test | 1045 | missing modality: gaze |
| test | 1112 | GPS median interval 2.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| test | 1133 | GPS median interval 2.00s != ~1s; GPS timeline does not cover video duration (first offset 4.5, last gap 3.5) |
| test | 1138 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| test | 1150 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s |
| test | 1167 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| test | 1215 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| test | 1250 | GPS median interval 2.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| test | 1303 | GPS median interval 4.00s != ~1s; GPS max gap 4.00s > 2.5s; GPS timeline does not cover video duration (first offset 2.5, last gap 3.5) |
| test | 1362 | GPS median interval 2.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| test | 1368 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 6.5, last gap 0.5) |
| test | 1420 | GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| test | 1424 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| test | 1436 | GPS median interval 6.00s != ~1s; GPS max gap 6.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| test | 1466 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |
| test | 1579 | GPS median interval 3.00s != ~1s; GPS max gap 4.00s > 2.5s; GPS timeline does not cover video duration (first offset 0.5, last gap 3.5) |
| test | 1611 | GPS median interval 3.00s != ~1s; GPS max gap 3.00s > 2.5s |
| test | 1627 | GPS median interval 3.99s != ~1s; GPS max gap 5.05s > 2.5s |
| test | 1650 | GPS median interval 2.50s != ~1s; GPS max gap 3.00s > 2.5s; GPS timeline does not cover video duration (first offset 1.5, last gap 3.5) |

## 8. Decision gate

- Synchronized sequences: 1161 / 1435 (80.9%)
- **CASE B — Synchronization is NOT reliable.** Stopping before preprocessing. See section 7 for affected sequences.
