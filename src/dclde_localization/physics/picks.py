"""Load the annotated arrival picks and give every call a stable key.

Source is the committed snapshot fixtures/das_calls.csv.gz — all
11,117 picks, no network access needed.

The dataset writes the same window timestamp three ways (colons in the .nc
filename, dashes in the bucket object name, underscores in the annotation CSV
name), so keys are always underscore-normalised.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PICKS_SNAPSHOT = "fixtures/das_calls.csv.gz"

_TS = re.compile(r"(\d{4}-\d{2}-\d{2})[_ ](\d{2})[:_-](\d{2})[:_-](\d{2})")


def call_key(filename: str, call_id: int) -> str:
    found = _TS.findall(str(filename))
    if not found:
        raise ValueError(f"no timestamp in filename: {filename!r}")
    # Silently taking the first of several timestamps would assign a wrong key,
    # which is worse than failing: the key is the join axis for everything.
    if len({f for f in found}) > 1:
        raise ValueError(f"ambiguous timestamps in filename: {filename!r} -> {found}")
    date, hh, mm, ss = found[0]
    return f"{date}_{hh}_{mm}_{ss}/{int(call_id)}"


def load_picks(path: str | Path = PICKS_SNAPSHOT) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["cable"] = df["cable"].str.strip().str.capitalize()
    df["call"] = [call_key(f, c) for f, c in zip(df["filename"], df["call_id"])]
    df["window"] = df["call"].str.split("/").str[0]
    keep = ["call", "window", "call_id", "cable", "dist", "time", "snr", "call_type", "batch"]
    return df[keep].reset_index(drop=True)


def attach_xyz(picks: pd.DataFrame, cables: dict) -> pd.DataFrame:
    # Fail fast on a label we have no geometry for. Leaving those rows as NaN
    # would silently drop picks from every downstream join instead.
    unknown = set(picks["cable"].unique()) - set(cables)
    if unknown:
        raise ValueError(f"unknown cable label(s) in picks: {sorted(unknown)}")

    out = picks.copy()
    out[["px", "py", "pz"]] = np.nan
    for name, cable in cables.items():
        m = out["cable"] == name
        if not m.any():
            continue
        x, y, z = cable.chan_to_xyz(out.loc[m, "dist"].to_numpy())
        out.loc[m, "px"] = x
        out.loc[m, "py"] = y
        out.loc[m, "pz"] = z
    return out
