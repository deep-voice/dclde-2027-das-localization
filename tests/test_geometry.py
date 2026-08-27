import numpy as np
import pandas as pd

from dclde_localization.geometry import CableInterpolator


def straight_cable_geometry(length_m=50000, n_points=500, y=0.0, depth=-500.0):
    chan_m = np.linspace(0, length_m, n_points)
    return pd.DataFrame(
        {
            "chan_m": chan_m,
            "x": chan_m,
            "y": np.full(n_points, y),
            "depth": np.full(n_points, depth),
        }
    )


def test_interpolates_known_points():
    geo = straight_cable_geometry(y=1000.0, depth=-750.0)
    interp = CableInterpolator(geo)

    x, y = interp.xy(np.array([0.0, 25000.0, 50000.0]))
    np.testing.assert_allclose(x, [0.0, 25000.0, 50000.0], atol=1.0)
    np.testing.assert_allclose(y, [1000.0, 1000.0, 1000.0], atol=1.0)

    x, y, z = interp.xyz(np.array([10000.0]))
    np.testing.assert_allclose(z, [-750.0], atol=1.0)


def test_extrapolates_beyond_cable_ends():
    geo = straight_cable_geometry()
    interp = CableInterpolator(geo)

    x, _ = interp.xy(np.array([-1000.0, 60000.0]))
    assert x[0] < 0
    assert x[1] > 50000
