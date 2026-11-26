"""Shared fixtures: a small synthetic BDDA-like dataset (built once per test session)."""
from __future__ import annotations

import pandas as pd
import pytest

from tests.synthetic import make_sequence


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("bdda")
    bdda = root / "BDDA"
    rows = [
        make_sequence(bdda, "training", "2"),
        make_sequence(bdda, "training", "4", cam_fps=59.94, cam_n=600),
        # validation/1744 reproduces the real issue: invalid speed = -1.0 records
        make_sequence(bdda, "validation", "1744", speeds=[3.0, -1.0, -1.0, 4.0, 5.0, -1.0, 6.0, 6.5, 7.0, -1.0]),
        make_sequence(bdda, "test", "6"),
    ]
    manifest = root / "dataset_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return {"root": root, "bdda": bdda, "manifest": manifest}


