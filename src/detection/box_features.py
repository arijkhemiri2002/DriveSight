"""
Geometric descriptors of detected boxes (derived features, not ground truth).

All coordinates are normalised by the image size (1280x720) so they are resolution-free.

Distance PROXIES (no calibration, no metres):
  * bottom_y_n: y of the box bottom edge. With a forward camera over a roughly flat road, the
    ground-contact point of an object moves down the image as the object gets closer. Only an
    ORDINAL proxy: it ignores slopes, pitch, mounting height (BDD cameras differ per vehicle)
    and occlusion of the contact point.
  * h_n: box height. Inversely related to distance for objects of a given real height, which
    varies per class and per instance (a child versus an adult, a car versus a truck).
These are useful to rank objects within one clip and class, and they are documented as such.
They must not be presented as distances.

  x_offset_n : horizontal offset of the box centre from the image centre (-0.5 .. 0.5).
               Not the ego lane: lanes curve and the camera can be off-centre.
  touches_border : the box is cut by the image edge, so its size and bottom are unreliable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.detection.detect_config import CLASS_NAMES, VRU_CLASSES

BOX_COLUMNS = ["det_idx", "cls_id", "cls_name", "is_vru", "conf", "x1", "y1", "x2", "y2",
               "cx_n", "cy_n", "w_n", "h_n", "area_frac", "aspect_wh", "bottom_y_n",
               "x_offset_n", "touches_border"]

_VRU_IDS = {CLASS_NAMES.index(c) for c in VRU_CLASSES}


def box_table(dets: np.ndarray, img_w: int, img_h: int, border_px: int = 2) -> pd.DataFrame:
    """dets: (n, 6) [x1, y1, x2, y2, conf, cls] in pixels -> one row per box."""
    if dets is None or len(dets) == 0:
        return pd.DataFrame(columns=BOX_COLUMNS)
    d = np.asarray(dets, dtype=np.float64)
    x1 = np.clip(d[:, 0], 0, img_w)
    y1 = np.clip(d[:, 1], 0, img_h)
    x2 = np.clip(d[:, 2], 0, img_w)
    y2 = np.clip(d[:, 3], 0, img_h)
    # Drop boxes that are degenerate once clipped to the image (zero width/height, e.g. a box
    # lying entirely beyond the image edge). Rounding to 0.1 px happens below, so the test is
    # done on rounded coordinates.
    keep = (np.round(x2, 1) > np.round(x1, 1)) & (np.round(y2, 1) > np.round(y1, 1))
    if not keep.all():
        d, x1, y1, x2, y2 = d[keep], x1[keep], y1[keep], x2[keep], y2[keep]
        if len(d) == 0:
            return pd.DataFrame(columns=BOX_COLUMNS)
    w, h = x2 - x1, y2 - y1
    cls = d[:, 5].astype(int)
    order = np.argsort(-d[:, 4], kind="stable")          # det_idx 0 = most confident
    df = pd.DataFrame({
        "cls_id": cls, "cls_name": [CLASS_NAMES[c] for c in cls],
        "is_vru": [c in _VRU_IDS for c in cls], "conf": np.round(d[:, 4], 4),
        "x1": np.round(x1, 1), "y1": np.round(y1, 1), "x2": np.round(x2, 1), "y2": np.round(y2, 1),
        "cx_n": np.round((x1 + x2) / 2 / img_w, 5), "cy_n": np.round((y1 + y2) / 2 / img_h, 5),
        "w_n": np.round(w / img_w, 5), "h_n": np.round(h / img_h, 5),
        "area_frac": np.round(w * h / (img_w * img_h), 6),
        "aspect_wh": np.round(np.where(h > 0, w / np.where(h > 0, h, 1), np.nan), 4),
        "bottom_y_n": np.round(y2 / img_h, 5),
        "x_offset_n": np.round((x1 + x2) / 2 / img_w - 0.5, 5),
        "touches_border": (x1 <= border_px) | (y1 <= border_px) | (x2 >= img_w - border_px) | (y2 >= img_h - border_px),
    }).iloc[order].reset_index(drop=True)
    df.insert(0, "det_idx", np.arange(len(df)))
    return df[BOX_COLUMNS]
