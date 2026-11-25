"""
Diagnostic: is the gaze crop (rows 96:672 of the 1024x768 map) losing attention signal?

The pilot found non-zero gaze values in the black letterbox bars in 5 of 11 sequences
(max 11-79 out of 255). Two explanations predict different patterns:

  A. spill-over: the heat-map blur of fixations near the top/bottom image edge extends
     past the edge. The bar signal sits right against the crop boundary, decays away
     from it, and is a small share of the total mass. The crop is right; a little
     edge attention is lost.
  B. misalignment: the video does not occupy rows 96:672. Signal appears deep inside
     the bars, or the intensity profile is shifted. The crop must be corrected.

For a sample of sequences this script reads every 5 Hz gaze frame at full resolution
and measures:
  * the mean vertical intensity profile (rows 0..767)
  * the share of the gaze mass in the bars, per frame
  * the share of the bar mass within 24 px of the crop boundary (near the image edge)
It writes a figure and a short report (read-only on the raw data).

Run:  python -m src.features.gaze_bar_diagnostic                 (pilot + 60 random sequences)
      python -m src.features.gaze_bar_diagnostic --n-random 150
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.features.preprocess_config import PreprocessConfig
from src.features.manifest_loader import load_manifest, select_pilot, sequence_paths, sort_sequences
from src.features.frame_extractor import SequentialReader, sample_plan

NEAR_PX = 24      # "near the boundary" = within 24 px (1/4 of the 96 px bar)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-random", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bdda-root", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--out-dir", type=Path)
    a = ap.parse_args(argv)
    kw = {k: v for k, v in (("bdda_root", a.bdda_root), ("manifest_csv", a.manifest)) if v}
    cfg = PreprocessConfig(**kw)
    out_dir = a.out_dir or cfg.reports_dir
    cfg.assert_safe_output(out_dir)
    man = load_manifest(cfg)
    pilot = select_pilot(man)
    rest = man[~man["video_id"].isin(pilot["video_id"])]
    rnd = rest.sample(n=min(a.n_random, len(rest)), random_state=a.seed)
    sel = sort_sequences(pd.concat([pilot, rnd]))
    top, bot = cfg.gaze_crop_top, cfg.gaze_crop_bottom
    H = cfg.expected_gaze_h

    profile = np.zeros(H, np.float64)
    n_frames = 0
    rows = []
    worst = (-1.0, None, None, None)          # (bar frac, gaze frame, camera frame, label)
    for r in sel.itertuples():
        paths = sequence_paths(r.split, r.video_id, cfg)
        plan = sample_plan(int(r.cam_frames), float(r.cam_fps), int(r.gaze_frames), float(r.gaze_fps),
                           cfg.sampling_fps)
        fr_fracs, near_fracs, deep_max = [], [], 0
        with SequentialReader(paths["gaze"]) as g:
            for p in plan:
                f = g.read_at(p.gaze_idx)
                if f is None:
                    continue
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64)
                profile += gray.mean(axis=1)
                n_frames += 1
                crop_m = gray[top:bot].sum()
                top_bar, bot_bar = gray[:top], gray[bot:]
                bar_m = top_bar.sum() + bot_bar.sum()
                frac = bar_m / (bar_m + crop_m) if (bar_m + crop_m) > 0 else 0.0
                fr_fracs.append(frac)
                if bar_m > 0:
                    near = top_bar[top - NEAR_PX:].sum() + bot_bar[:NEAR_PX].sum()
                    near_fracs.append(near / bar_m)
                deep_max = max(deep_max, int(top_bar[:top - NEAR_PX].max()), int(bot_bar[NEAR_PX:].max()))
                if frac > worst[0]:
                    worst = (frac, f.copy(), p.cam_idx, f"{r.split}/{r.video_id} t={p.t_sec:.1f}s")
        fr = np.array(fr_fracs) if fr_fracs else np.array([0.0])
        rows.append({"split": r.split, "video_id": r.video_id, "frames": len(fr_fracs),
                     "bar_mass_frac_mean": fr.mean(), "bar_mass_frac_max": fr.max(),
                     "frames_over_5pct": int((fr > 0.05).sum()),
                     "bar_mass_near_boundary": float(np.mean(near_fracs)) if near_fracs else np.nan,
                     "deep_bar_max": deep_max})
        print(f"{r.split}/{r.video_id}: mean {fr.mean():.4f} max {fr.max():.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "gaze_bar_diagnostic.csv", index=False)
    profile /= max(n_frames, 1)

    # verdict
    total_mean = float((df["bar_mass_frac_mean"] * df["frames"]).sum() / df["frames"].sum())
    near = float(df["bar_mass_near_boundary"].dropna().mean()) if df["bar_mass_near_boundary"].notna().any() else float("nan")
    inside_mean = profile[top:bot].mean()
    edge_ratio = max(profile[top - 1], profile[bot]) / inside_mean if inside_mean > 0 else float("nan")
    spill = (np.isnan(near) or near >= 0.5) and total_mean < 0.01
    verdict = ("A (spill-over at the image edge): the crop is correct; the lost share is small and is "
               "recorded per row in gaze_bar_mass_frac." if spill else
               "UNCLEAR / possible misalignment: inspect the figure before using gaze features.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
        ax[0].semilogy(np.arange(H), np.maximum(profile, 1e-4))
        for y in (top, bot):
            ax[0].axvline(y, color="r", ls="--")
        ax[0].set_title("Mean gaze intensity per row (log)")
        ax[0].set_xlabel("row (0 = top of the 1024x768 map)")
        ax[1].hist(np.clip(df["bar_mass_frac_max"], 0, None), bins=30)
        ax[1].set_title("Per-sequence max share of gaze mass in bars")
        if worst[1] is not None:
            img = cv2.cvtColor(worst[1], cv2.COLOR_BGR2RGB)
            ax[2].imshow(img)
            ax[2].axhline(top, color="r", ls="--")
            ax[2].axhline(bot, color="r", ls="--")
            ax[2].set_title(f"Worst frame {worst[3]} ({worst[0]:.1%} in bars)")
        for x in ax[2:]:
            x.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "gaze_bar_diagnostic.png", dpi=90)
        plt.close(fig)
    except ImportError:
        pass

    lines = ["# Gaze crop diagnostic (letterbox bars)", "",
             f"Sequences: {len(df)} (extended pilot + {len(rnd)} random, seed {a.seed}) | gaze frames: {n_frames}", "",
             f"- Mean share of gaze mass in the bars: {total_mean:.4%}",
             f"- Sequences with any frame > 5 % of mass in bars: {int((df['frames_over_5pct'] > 0).sum())}",
             f"- Share of bar mass within {NEAR_PX} px of the crop boundary: {near:.1%}",
             f"- Max value deeper than {NEAR_PX} px inside the bars: {int(df['deep_bar_max'].max())}",
             f"- Mean intensity at the boundary row / mean inside the crop: {edge_ratio:.2f}", "",
             f"**Verdict: {verdict}**", "",
             "Figure: gaze_bar_diagnostic.png. Per-sequence table: gaze_bar_diagnostic.csv."]
    (out_dir / "gaze_bar_diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
