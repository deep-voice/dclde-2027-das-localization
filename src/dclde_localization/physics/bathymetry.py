"""GMRT bathymetry: fetch, decode the legacy GMT grid, reproject to local XY.

GMRT's GridServer returns a COARDS/legacy GMT netCDF where z is a flat array
stored top-down with x_range/y_range/dimension headers rather than coordinate
variables, so it needs decoding by hand.

Validated against the cables' own reported depth: median offset -0.6 m,
r = 0.99995 over 46,440 south-cable channels. That is an independent check on
both the bathymetry and the cable geolocation, so it is kept as a test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import xarray as xr

GMRT_BOUNDS = (-125.35, -123.90, 44.95, 45.40)
GMRT_URL = "https://www.gmrt.org/services/GridServer"
NO_DATA = -32768


def fetch_gmrt(bounds=GMRT_BOUNDS, dest: str | Path = "fixtures/gmrt_bathy.nc") -> Path:
    min_lon, max_lon, min_lat, max_lat = bounds
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(
        GMRT_URL,
        params={
            "minlongitude": min_lon, "maxlongitude": max_lon,
            "minlatitude": min_lat, "maxlatitude": max_lat,
            "format": "netcdf", "resolution": "default", "layer": "topo",
        },
        timeout=180,
    )
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


@dataclass(frozen=True)
class GeoGrid:
    lon: np.ndarray
    lat: np.ndarray
    z: np.ndarray  # (nlat, nlon), ascending latitude, NaN for no-data


def decode_gmt_grid(path: str | Path) -> GeoGrid:
    ds = xr.open_dataset(path)
    nx, ny = (int(v) for v in ds["dimension"].values)
    x_range = ds["x_range"].values
    y_range = ds["y_range"].values
    z = ds["z"].values.reshape(ny, nx)[::-1]  # stored top-down
    return GeoGrid(
        lon=np.linspace(float(x_range[0]), float(x_range[1]), nx),
        lat=np.linspace(float(y_range[0]), float(y_range[1]), ny),
        z=z.astype(float),
    )


@dataclass(frozen=True)
class LocalGrid:
    x0: float
    y0: float
    dx: float
    dy: float
    nx: int
    ny: int
    z: np.ndarray  # int16 metres, NO_DATA sentinel


def _sample(grid: GeoGrid, lat, lon) -> np.ndarray:
    """Nearest-neighbour sample; out-of-bounds and no-data return NaN.

    Index arithmetic on the regular grid, not searchsorted. `searchsorted(a, v) - 1`
    returns the LOWER neighbour for every interior query, never the nearer one,
    which biases every sample toward the grid's south-west by up to a full native
    cell (173 m here, ~87 m on average) and wrongly excludes a query landing
    exactly on the first row or column. Measured against the south cable, the two
    differ on 74.9% of samples. `grid.lon`/`grid.lat` are linspaces, so the
    rounding below is exact.
    """
    dlon = grid.lon[1] - grid.lon[0]
    dlat = grid.lat[1] - grid.lat[0]
    i = np.round((np.asarray(lon, float) - grid.lon[0]) / dlon).astype(int)
    j = np.round((np.asarray(lat, float) - grid.lat[0]) / dlat).astype(int)
    ok = (i >= 0) & (i < len(grid.lon)) & (j >= 0) & (j < len(grid.lat))
    out = np.full(np.shape(lat), np.nan, dtype=float)
    out[ok] = grid.z[j[ok], i[ok]]
    return out


def reproject_to_local(grid: GeoGrid, transform, nx: int = 663, ny: int = 294) -> LocalGrid:
    """Resample the geographic grid onto a regular local-XY grid.

    The output box is taken from the four geographic corners. That is only a
    rectangle in the local frame up to projection distortion; over this domain
    (~115 x 50 km, within 2.5 deg of UTM 10N's central meridian) the bowing is a
    few metres against ~173 m cells, so nothing clips. It would not hold for a
    much larger or more off-meridian domain.
    """
    corner_lat = grid.lat[[0, 0, -1, -1]]
    corner_lon = grid.lon[[0, -1, 0, -1]]
    cx, cy = transform.wgs84_to_local(corner_lat, corner_lon)
    x0, x1 = float(cx.min()), float(cx.max())
    y0, y1 = float(cy.min()), float(cy.max())

    dx = (x1 - x0) / (nx - 1)
    dy = (y1 - y0) / (ny - 1)
    gx, gy = np.meshgrid(x0 + dx * np.arange(nx), y0 + dy * np.arange(ny))
    lat, lon = transform.local_to_wgs84(gx.ravel(), gy.ravel())
    z = _sample(grid, lat, lon).reshape(ny, nx)

    zi = np.where(np.isfinite(z), np.clip(np.round(z), -32767, 32767), NO_DATA)
    return LocalGrid(x0, y0, dx, dy, nx, ny, zi.astype(np.int16))


@dataclass(frozen=True)
class ValidationReport:
    n: int
    median_offset_m: float
    correlation: float


def validate_against_cable(grid: GeoGrid, cable) -> ValidationReport:
    z = _sample(grid, cable.lat, cable.lon)
    ok = np.isfinite(z)
    return ValidationReport(
        n=int(ok.sum()),
        median_offset_m=float(np.median(z[ok] - cable.depth[ok])),
        correlation=float(np.corrcoef(z[ok], cable.depth[ok])[0, 1]),
    )


@dataclass(frozen=True)
class PathProfile:
    """Water depth sampled along a straight horizontal path.

    range_m: horizontal distance from the start, 0 .. total, `n_steps` points.
    depth_m: water depth, POSITIVE DOWN, at each of those ranges.
    ok:      whether the profile is usable. When False, `range_m` and `depth_m`
             are EMPTY -- a failed path must not hand back something that looks
             like an answer.
    reason:  why it failed; empty when `ok`.

    A path that leaves the grid, crosses a no-data hole, or crosses land fails
    rather than interpolating over the gap. Interpolating across a no-data hole
    in bathymetry invents a waveguide, and the whole point of this module is
    that the waveguide's shape along the path is the physics.
    """

    range_m: np.ndarray
    depth_m: np.ndarray
    ok: bool
    reason: str = ""


def _failed(reason: str) -> PathProfile:
    empty = np.empty(0, dtype=float)
    return PathProfile(range_m=empty, depth_m=empty, ok=False, reason=reason)


def depth_along_path(grid: LocalGrid, x0: float, y0: float, x1: float, y1: float,
                     *, n_steps: int = 64) -> PathProfile:
    """Sample water depth along the straight path from (x0, y0) to (x1, y1).

    Nearest-neighbour on the regular local grid, matching `_sample`'s convention
    (see its docstring for why nearest-neighbour rather than a floor lookup).
    """
    if n_steps < 2:
        raise ValueError(f"n_steps must be at least 2, got {n_steps}")

    total = float(np.hypot(x1 - x0, y1 - y0))
    if total <= 0.0:
        return _failed("zero-length path: source and receiver are at the same "
                       "horizontal position")

    s = np.linspace(0.0, 1.0, int(n_steps))
    xs = x0 + s * (x1 - x0)
    ys = y0 + s * (y1 - y0)

    i = np.round((xs - grid.x0) / grid.dx).astype(int)
    j = np.round((ys - grid.y0) / grid.dy).astype(int)
    inside = (i >= 0) & (i < grid.nx) & (j >= 0) & (j < grid.ny)
    if not inside.all():
        return _failed(f"path leaves the bathymetry grid: {int((~inside).sum())} "
                       f"of {int(n_steps)} samples outside")

    elev = grid.z[j, i].astype(float)
    if np.any(elev == NO_DATA):
        return _failed(f"path crosses a no-data hole in the bathymetry: "
                       f"{int((elev == NO_DATA).sum())} of {int(n_steps)} samples")
    if np.any(elev >= 0.0):
        return _failed(f"path crosses land: {int((elev >= 0.0).sum())} of "
                       f"{int(n_steps)} samples are at or above sea level")

    return PathProfile(range_m=s * total, depth_m=-elev, ok=True)
