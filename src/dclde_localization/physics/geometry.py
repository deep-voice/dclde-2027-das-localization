"""DAS cable geometry and coordinate frames.

The dataset's "local" XY frame is UTM zone 10N with the x axis flipped:
    utm_x = X_OFFSET - x        utm_y = Y_OFFSET + y
Both offsets have zero variance across all 77,958 channels of both cables, so
this is an identity, not a fit. The offsets are still derived from the geometry
table and asserted constant, so a revised dataset cannot slip past unnoticed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

CABLES = ("North", "South")
CHAN_IDX0 = 942
CHAN_SPACING_M = 2.041904688
VALID_FRAMES = ("local_xy", "utm10n", "wgs84")

UTM10N_EPSG = 32610
WGS84_EPSG = 4326

_TO_WGS84 = Transformer.from_crs(UTM10N_EPSG, WGS84_EPSG, always_xy=True)
_FROM_WGS84 = Transformer.from_crs(WGS84_EPSG, UTM10N_EPSG, always_xy=True)


@dataclass(frozen=True)
class CableGeometry:
    name: str
    chan_m0: float
    spacing_m: float
    x: np.ndarray
    y: np.ndarray
    depth: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    utm_x: np.ndarray
    utm_y: np.ndarray

    @property
    def chan_m(self) -> np.ndarray:
        return self.chan_m0 + np.arange(len(self.x)) * self.spacing_m

    def chan_to_xyz(self, dist_m) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map along-cable distance to local (x, y, depth), clamped at the ends."""
        d = np.asarray(dist_m, dtype=float)
        m = self.chan_m
        return (
            np.interp(d, m, self.x),
            np.interp(d, m, self.y),
            np.interp(d, m, self.depth),
        )


def load_cable(name: str, csv_path: str | Path) -> CableGeometry:
    df = pd.read_csv(csv_path).sort_values("chan_idx").reset_index(drop=True)

    idx = df["chan_idx"].to_numpy()
    if idx[0] != CHAN_IDX0 or not np.array_equal(np.diff(idx), np.ones(len(idx) - 1)):
        raise ValueError(f"{name}: chan_idx must start at {CHAN_IDX0} and be contiguous")

    chan_m = df["chan_m"].to_numpy()
    steps = np.diff(chan_m)
    spacing = float(steps.mean())
    if steps.std() > 1e-6:
        raise ValueError(f"{name}: channel spacing is not uniform (std={steps.std():.3e})")

    return CableGeometry(
        name=name,
        chan_m0=float(chan_m[0]),
        spacing_m=spacing,
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        depth=df["depth"].to_numpy(),
        lat=df["lat"].to_numpy(),
        lon=df["lon"].to_numpy(),
        utm_x=df["utm_x"].to_numpy(),
        utm_y=df["utm_y"].to_numpy(),
    )


def decimate_indices(n: int, stride: int) -> np.ndarray:
    """Indices for a decimated cable, always including the final channel.

    A plain ``[::stride]`` leaves the last ``(n-1) % stride`` channels beyond the
    final kept sample, where interpolation clamps instead of interpolating — 1.8 m
    of position error on North and 6.3 m on South at stride 4. Both the bundle and
    its tests must decimate through this function so they cannot diverge.
    """
    return np.unique(np.r_[np.arange(n)[::stride], n - 1])


def load_cables(geometry_dir: str | Path) -> dict[str, CableGeometry]:
    d = Path(geometry_dir)
    return {n: load_cable(n, d / f"{n.lower()}_DAS_multicoord.csv") for n in CABLES}


@dataclass(frozen=True)
class FrameTransform:
    x_offset: float
    y_offset: float

    @classmethod
    def from_cable(cls, cable: CableGeometry, tol_m: float = 1e-3) -> FrameTransform:
        xs = cable.utm_x + cable.x
        ys = cable.utm_y - cable.y
        if xs.std() > tol_m or ys.std() > tol_m:
            raise ValueError(
                "local<->UTM relation is not a pure flip+translate "
                f"(x std={xs.std():.3e}, y std={ys.std():.3e})"
            )
        return cls(x_offset=float(xs.mean()), y_offset=float(ys.mean()))

    def local_to_utm(self, x, y):
        return self.x_offset - np.asarray(x, float), self.y_offset + np.asarray(y, float)

    def utm_to_local(self, ux, uy):
        return self.x_offset - np.asarray(ux, float), np.asarray(uy, float) - self.y_offset

    def local_to_wgs84(self, x, y):
        ux, uy = self.local_to_utm(x, y)
        lon, lat = _TO_WGS84.transform(ux, uy)
        return np.asarray(lat), np.asarray(lon)

    def wgs84_to_local(self, lat, lon):
        ux, uy = _FROM_WGS84.transform(np.asarray(lon, float), np.asarray(lat, float))
        return self.utm_to_local(ux, uy)

    def to_local(self, frame: str, a, b):
        """Convert a coordinate pair in `frame` to local XY.

        For 'wgs84' the pair is (lat, lon); otherwise it is (x, y) / (easting, northing).
        """
        if frame == "local_xy":
            return np.asarray(a, float), np.asarray(b, float)
        if frame == "utm10n":
            return self.utm_to_local(a, b)
        if frame == "wgs84":
            return self.wgs84_to_local(a, b)
        raise ValueError(f"unknown frame: {frame!r}; expected one of {VALID_FRAMES}")
