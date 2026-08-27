"""Loading and cross-cable matching of arrival-time pick annotations.

`call_id` in the raw annotation CSVs is only unique within one source file
(one cable, one ~50s time window) -- it resets per file. To work with the
full dataset we need a call identity that's unique across the whole
collection, and a way to find the calls that were picked up on BOTH cables
(the same window's North and South files sharing a `call_id`).
"""
import re
from pathlib import Path

import pandas as pd

_FNAME_RE = re.compile(r"annotated_calls_(north|south)_(?P<window>.+)\.csv$")
_BATCH_DIR_RE = re.compile(r"batch(\d+)$")


def load_annotations(data_root):
    """Load every annotation CSV under data_root/annotations/batch*/.

    Adds `batch` (int), `window` (str, the shared North/South timestamp key
    parsed from the filename) and `global_call_id` (str, unique across the
    whole dataset) columns.
    """
    data_root = Path(data_root)
    frames = []
    for batch_dir in sorted((data_root / "annotations").glob("batch*")):
        batch_match = _BATCH_DIR_RE.match(batch_dir.name)
        if not batch_match:
            raise ValueError(f"unexpected annotations directory name: {batch_dir}")
        batch = int(batch_match.group(1))

        for csv_path in sorted(batch_dir.glob("*.csv")):
            fname_match = _FNAME_RE.match(csv_path.name)
            if not fname_match:
                raise ValueError(f"unexpected annotation filename: {csv_path}")
            window = fname_match.group("window")

            df = pd.read_csv(csv_path)
            df["batch"] = batch
            df["window"] = window
            frames.append(df)

    picks = pd.concat(frames, ignore_index=True)
    picks["cable"] = picks["cable"].str.lower()
    picks["global_call_id"] = (
        "b" + picks["batch"].astype(str) + ":" + picks["window"] + ":" + picks["call_id"].astype(str)
    )
    return picks


def matched_cross_cable_calls(picks):
    """Return sorted (batch, window, call_id) keys picked up on both cables."""
    key_cols = ["batch", "window", "call_id"]
    north_keys = set(map(tuple, picks.loc[picks["cable"] == "north", key_cols].itertuples(index=False)))
    south_keys = set(map(tuple, picks.loc[picks["cable"] == "south", key_cols].itertuples(index=False)))
    return sorted(north_keys & south_keys)


def call_picks(picks, batch, window, call_id, cable=None):
    """Rows for one call, optionally restricted to a single cable."""
    mask = (picks["batch"] == batch) & (picks["window"] == window) & (picks["call_id"] == call_id)
    if cable is not None:
        mask &= picks["cable"] == cable
    return picks.loc[mask]
