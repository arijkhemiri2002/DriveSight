# Phase 4 evaluation without ground truth - training

BDDA has no annotated trajectories. These indicators describe the tracks; they do not measure identity accuracy (that needs ground truth, e.g. the BDD100K MOT set).

Tracker settings: `{'track_high_thresh': 0.25, 'track_low_thresh': 0.1, 'new_track_thresh': 0.5, 'match_thresh': 0.9, 'fuse_score': True, 'track_buffer_sec': 0.4, 'track_buffer_frames': 2, 'keep_birth_box': True}`

| Indicator | All | VRU | Vehicles |
|---|---|---|---|
| Tracks | 32435 | 3557 | 22367 |
| Median length (frames) | 11 | 6 | 12 |
| Mean length (frames) | 15.1 | 9.1 | 16.8 |
| Tracks of 1-2 frames | 10.3% | 18.6% | 6.6% |
| Tracks of >= 10 frames (2 s) | 55.4% | 34.9% | 60.3% |
| Tracks with a class change | 11.3% | 16.4% | 12.5% |
| Tracks with a bridged gap | 20.8% | 23.1% | 23.2% |
| Consecutive pairs with centre jump > 0.1 | 0.2% | 0.1% | 0.3% |
| Consecutive pairs with no overlap (IoU = 0) | 0.1% | 0.4% | 0.1% |
| Boxes with conf >= 0.5 in a track | 84.3% | 66.9% | 92.8% |

- Jump threshold 0.1 (normalised image units) and confidence 0.5 are reporting choices, not tuned values.
- At 5 Hz, a near object can move most of its own width between two frames; IoU-based association then fails (measured on synthetic boxes: tracking holds up to ~60 % of the box width per frame). Fast, close objects are the expected source of fragmentation.
