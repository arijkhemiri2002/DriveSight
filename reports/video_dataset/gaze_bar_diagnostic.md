# Gaze crop diagnostic (letterbox bars)

Sequences: 71 (extended pilot + 60 random, seed 0) | gaze frames: 3925

- Mean share of gaze mass in the bars: 0.6874%
- Sequences with any frame > 5 % of mass in bars: 21
- Share of bar mass within 24 px of the crop boundary: 53.4%
- Max value deeper than 24 px inside the bars: 167
- Mean intensity at the boundary row / mean inside the crop: 0.10

**Verdict: A (spill-over at the image edge): the crop is correct; the lost share is small and is recorded per row in gaze_bar_mass_frac.**

Figure: gaze_bar_diagnostic.png. Per-sequence table: gaze_bar_diagnostic.csv.
