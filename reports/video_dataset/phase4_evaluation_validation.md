# Phase 4 evaluation without ground truth - validation

BDDA has no annotated trajectories. These indicators describe the tracks; they do not measure identity accuracy (that needs ground truth, e.g. the BDD100K MOT set).

Tracker settings: `{'track_high_thresh': 0.25, 'track_low_thresh': 0.1, 'new_track_thresh': 0.5, 'match_thresh': 0.9, 'fuse_score': True, 'track_buffer_sec': 0.4, 'track_buffer_frames': 2, 'keep_birth_box': True}`

| Indicator | All | VRU | Vehicles |
|---|---|---|---|
| Tracks | 4153 | 583 | 2797 |
| Median length (frames) | 10 | 6 | 12 |
| Mean length (frames) | 14.7 | 9.1 | 16.6 |
| Tracks of 1-2 frames | 10.8% | 19.2% | 7.2% |
| Tracks of >= 10 frames (2 s) | 53.3% | 33.4% | 59.2% |
| Tracks with a class change | 12.3% | 16.1% | 13.9% |
| Tracks with a bridged gap | 22.6% | 29.2% | 24.2% |
| Consecutive pairs with centre jump > 0.1 | 0.3% | 0.3% | 0.3% |
| Consecutive pairs with no overlap (IoU = 0) | 0.1% | 0.6% | 0.1% |
| Boxes with conf >= 0.5 in a track | 83.0% | 65.6% | 92.3% |

- Jump threshold 0.1 (normalised image units) and confidence 0.5 are reporting choices, not tuned values.
- At 5 Hz, a near object can move most of its own width between two frames; IoU-based association then fails (measured on synthetic boxes: tracking holds up to ~60 % of the box width per frame). Fast, close objects are the expected source of fragmentation.
