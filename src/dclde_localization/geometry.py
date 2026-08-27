"""Cable geometry loading and distance-along-cable -> physical position mapping.

Each DAS cable's geometry CSV maps `chan_m` (distance along the fiber, in
meters, from the shore end) to lat/lon/UTM/local-planar (x, y) coordinates
and seafloor `depth` at that point. Annotation picks record a `dist` in the
same units/reference as `chan_m`, so a pick's physical position is obtained
by interpolating the geometry table at that distance.
"""
from pathlib import Path

import pandas as pd
from scipy.interpolate import interp1d

CABLES = ("north", "south")


def load_geometry(data_root, cable):
    """Load the raw geometry table for one cable ("north" or "south")."""
    path = Path(data_root) / "geometry" / f"{cable}_DAS_multicoord.csv"
    return pd.read_csv(path)


class CableInterpolator:
    """Maps distance-along-cable (meters) to physical (x, y, depth)."""

    def __init__(self, geo):
        self._interp_x = interp1d(geo["chan_m"], geo["x"], kind="linear", fill_value="extrapolate")
        self._interp_y = interp1d(geo["chan_m"], geo["y"], kind="linear", fill_value="extrapolate")
        self._interp_depth = interp1d(
            geo["chan_m"], geo["depth"], kind="linear", fill_value="extrapolate"
        )

    def xy(self, dist):
        return self._interp_x(dist), self._interp_y(dist)

    def xyz(self, dist):
        return self._interp_x(dist), self._interp_y(dist), self._interp_depth(dist)


def load_interpolators(data_root):
    """Return {"north": CableInterpolator, "south": CableInterpolator}."""
    return {cable: CableInterpolator(load_geometry(data_root, cable)) for cable in CABLES}
