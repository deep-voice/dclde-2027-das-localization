import numpy as np
import pandas as pd
import pytest

from dclde_localization.geometry import CableInterpolator
from dclde_localization.localize import DEFAULT_SOUND_SPEED, jackknife_uncertainty, localize_2d, localize_3d


def straight_cable(length_m=60000, n_points=500, y=0.0, depth=-500.0):
    chan_m = np.linspace(0, length_m, n_points)
    geo = pd.DataFrame(
        {"chan_m": chan_m, "x": chan_m, "y": np.full(n_points, y), "depth": np.full(n_points, depth)}
    )
    return CableInterpolator(geo)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _simulate(interp, dist_samples, whale_x, whale_y, t0=0.0, c=DEFAULT_SOUND_SPEED, noise_std=0.0, rng=None):
    rng = rng or np.random.default_rng()
    cx, cy = interp.xy(dist_samples)
    dists = np.sqrt((whale_x - cx) ** 2 + (whale_y - cy) ** 2)
    times = t0 + dists / c + rng.normal(0.0, noise_std, size=len(dist_samples))
    return cx, cy, times


def test_localize_2d_recovers_known_position_noiseless(rng):
    cable = straight_cable()
    whale_x, whale_y = 30000.0, 8000.0
    dist_samples = rng.uniform(10000, 50000, size=25)
    cx, cy, t = _simulate(cable, dist_samples, whale_x, whale_y, t0=3.0, rng=rng)

    result = localize_2d(cx, cy, t, x_range=(0, 60000), y_range=(0, 20000), grid_size=60)

    error = np.hypot(result["x"] - whale_x, result["y"] - whale_y)
    assert error < 50.0
    assert result["residual"] < 1e-2


def test_localize_2d_bounded_error_with_noise(rng):
    cable = straight_cable()
    whale_x, whale_y = 30000.0, 8000.0
    dist_samples = rng.uniform(10000, 50000, size=25)
    cx, cy, t = _simulate(cable, dist_samples, whale_x, whale_y, t0=3.0, noise_std=0.01, rng=rng)

    result = localize_2d(cx, cy, t, x_range=(0, 60000), y_range=(0, 20000), grid_size=60)

    error = np.hypot(result["x"] - whale_x, result["y"] - whale_y)
    assert error < 500.0


def test_localize_3d_recovers_known_position(rng):
    north = straight_cable(y=5000.0, depth=-500.0)
    south = straight_cable(y=0.0, depth=-1500.0)
    whale_x, whale_y, whale_z = 30000.0, 2000.0, -50.0

    n_dist = rng.uniform(10000, 50000, size=15)
    s_dist = rng.uniform(10000, 50000, size=15)
    ncx, ncy = north.xy(n_dist)
    scx, scy = south.xy(s_dist)
    _, _, ncz = north.xyz(n_dist)
    _, _, scz = south.xyz(s_dist)

    def times(cx, cy, cz):
        dists = np.sqrt((whale_x - cx) ** 2 + (whale_y - cy) ** 2 + (whale_z - cz) ** 2)
        return 2.0 + dists / DEFAULT_SOUND_SPEED

    picks_x = np.concatenate([ncx, scx])
    picks_y = np.concatenate([ncy, scy])
    picks_z = np.concatenate([ncz, scz])
    picks_t = np.concatenate([times(ncx, ncy, ncz), times(scx, scy, scz)])

    result = localize_3d(picks_x, picks_y, picks_z, picks_t, x_range=(0, 60000), y_range=(0, 10000), grid_size=40)

    error_xy = np.hypot(result["x"] - whale_x, result["y"] - whale_y)
    assert error_xy < 200.0
    assert abs(result["z"] - whale_z) < 200.0


def test_jackknife_uncertainty_grows_with_noise(rng):
    cable = straight_cable()
    whale_x, whale_y = 30000.0, 8000.0
    dist_samples = rng.uniform(10000, 50000, size=20)

    cx, cy, t_low = _simulate(cable, dist_samples, whale_x, whale_y, noise_std=0.001, rng=rng)
    _, _, t_high = _simulate(cable, dist_samples, whale_x, whale_y, noise_std=0.05, rng=rng)

    fit_low = localize_2d(cx, cy, t_low, x_range=(0, 60000), y_range=(0, 20000), grid_size=60)
    fit_high = localize_2d(cx, cy, t_high, x_range=(0, 60000), y_range=(0, 20000), grid_size=60)

    unc_low = jackknife_uncertainty(cx, cy, t_low, fit_low["x"], fit_low["y"], fit_low["t0"])
    unc_high = jackknife_uncertainty(cx, cy, t_high, fit_high["x"], fit_high["y"], fit_high["t0"])

    assert unc_high["position_std"] > unc_low["position_std"]
