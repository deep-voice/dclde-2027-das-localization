"""The pluggable results contract, plus the frame guard.

`frame` is mandatory and never inferred: the reference implementation shipped
lat/lon roughly 270 km west of the cable (its local coordinates were fine, its
local->UTM->geographic conversion was not). The guard turns that class of bug
into an automatic check on every contributed file.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from dclde_localization.physics.geometry import VALID_FRAMES

VALID_KINDS = ("estimate", "ground_truth", "synthetic_truth")
FRAME_GUARD_THRESHOLD_M = 25_000.0


@dataclass(frozen=True)
class Solution:
    call: str
    x: float
    y: float
    z: float | None = None
    z_fixed: bool | None = None
    t0: float | None = None
    wrms: float | None = None
    n_picks: int | None = None
    cables_used: tuple[str, ...] = ()
    unc: dict | None = None


@dataclass(frozen=True)
class ResultSet:
    name: str
    kind: str
    frame: str
    produced_by: str
    c_assumed: float | None
    notes: str
    solutions: tuple[Solution, ...]


def _finite(value, label):
    # bool is a subclass of int in Python, so isinstance(True, int) is True
    # and math.isfinite(True) is True too -- without this explicit check a
    # contributed {"x": true} loads silently as x = 1.0. This module's entire
    # job is refusing malformed third-party input by name, so a boolean
    # sneaking through as a number is exactly the class of bug it exists to
    # catch.
    if isinstance(value, bool) or value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"non-finite {label}: {value!r}")
    return float(value)


def load_result_set(path: str | Path) -> ResultSet:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    for field in ("name", "kind", "frame", "solutions"):
        if field not in raw:
            raise ValueError(f"result set is missing required field: {field}")
    if raw["frame"] not in VALID_FRAMES:
        raise ValueError(f"unknown frame: {raw['frame']!r}; expected one of {VALID_FRAMES}")
    if raw["kind"] not in VALID_KINDS:
        raise ValueError(f"unknown kind: {raw['kind']!r}; expected one of {VALID_KINDS}")

    sols = []
    for s in raw["solutions"]:
        if "call" not in s:
            raise ValueError("solution is missing its call key")
        sols.append(
            Solution(
                call=str(s["call"]),
                x=_finite(s.get("x"), "x"),
                y=_finite(s.get("y"), "y"),
                z=None if s.get("z") is None else float(s["z"]),
                z_fixed=s.get("z_fixed"),
                t0=s.get("t0"),
                wrms=s.get("wrms"),
                n_picks=s.get("n_picks"),
                cables_used=tuple(s.get("cables_used", ())),
                unc=s.get("unc"),
            )
        )

    return ResultSet(
        name=str(raw["name"]),
        kind=raw["kind"],
        frame=raw["frame"],
        produced_by=str(raw.get("produced_by", "")),
        c_assumed=raw.get("c_assumed"),
        notes=str(raw.get("notes", "")),
        solutions=tuple(sols),
    )


def to_local(rs: ResultSet, transform) -> ResultSet:
    if rs.frame == "local_xy":
        return rs
    xs = np.array([s.x for s in rs.solutions], dtype=float)
    ys = np.array([s.y for s in rs.solutions], dtype=float)
    lx, ly = transform.to_local(rs.frame, xs, ys)
    sols = tuple(
        replace(s, x=float(a), y=float(b)) for s, a, b in zip(rs.solutions, lx, ly)
    )
    return replace(rs, frame="local_xy", solutions=sols)


@dataclass(frozen=True)
class GuardReport:
    median_offset_m: float
    suspect: bool


def frame_guard(rs_local: ResultSet, cables: dict) -> GuardReport:
    """Median distance from each solution to the nearest cable channel."""
    if rs_local.frame != "local_xy":
        raise ValueError("frame_guard expects a set already converted to local_xy")

    cx = np.concatenate([c.x for c in cables.values()])
    cy = np.concatenate([c.y for c in cables.values()])
    # Decimate the cable for the distance scan; 40 m resolution is far below
    # the 25 km threshold this guard tests.
    cx, cy = cx[::20], cy[::20]

    d = []
    for s in rs_local.solutions:
        d.append(np.sqrt(((cx - s.x) ** 2 + (cy - s.y) ** 2).min()))
    med = float(np.median(d)) if d else float("inf")
    return GuardReport(med, med > FRAME_GUARD_THRESHOLD_M)


def baseline_csv_to_result_set(csv_path: str | Path, name: str) -> ResultSet:
    """Import the reference implementation's released localization CSV.

    Its x_local/y_local are trustworthy (median 1.4 km from the nearest cable
    channel); its lat/lon and x_utm/y_utm are not, so they are ignored.
    Every row has z = -30.0, hence z_fixed=True.
    """
    df = pd.read_csv(csv_path)
    sols = tuple(
        Solution(
            call=str(r.get("call", f"row{i}")),
            x=float(r["x_local"]),
            y=float(r["y_local"]),
            z=float(r["z_local"]),
            z_fixed=True,
            wrms=float(r["wrms"]),
            unc={"type": "circle", "radius": float(r["deltax"])},
        )
        for i, r in df.iterrows()
    )
    return ResultSet(
        name=name,
        kind="estimate",
        frame="local_xy",
        produced_by="Goestchel, Wilcock & Abadi 2026, released code",
        c_assumed=1480.0,
        notes=(
            "Imported from batch1_localizations_with_coords.csv. Published lat/lon and "
            "x_utm/y_utm are discarded: they place calls ~270 km west of the cable. "
            "z is fixed at -30 m for all rows."
        ),
        solutions=sols,
    )
