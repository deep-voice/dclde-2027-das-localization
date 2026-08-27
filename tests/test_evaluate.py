import numpy as np
import pandas as pd

from dclde_localization.evaluate import synthetic_trial
from dclde_localization.geometry import CableInterpolator


def straight_cable(length_m=60000, n_points=500, y=0.0, depth=-500.0):
    chan_m = np.linspace(0, length_m, n_points)
    geo = pd.DataFrame(
        {"chan_m": chan_m, "x": chan_m, "y": np.full(n_points, y), "depth": np.full(n_points, depth)}
    )
    return CableInterpolator(geo)


def test_synthetic_trial_low_noise_is_accurate():
    interpolators = {"north": straight_cable(y=5000.0), "south": straight_cable(y=0.0)}
    dist_ranges = {"north": (10000, 50000), "south": (10000, 50000)}
    rng = np.random.default_rng(1)

    error = synthetic_trial(
        interpolators,
        cables=("north", "south"),
        whale_x=30000.0,
        whale_y=2000.0,
        dist_ranges=dist_ranges,
        n_picks_per_cable=15,
        time_noise_std=0.001,
        localize_kwargs={"x_range": (0, 60000), "y_range": (0, 10000), "grid_size": 60},
        rng=rng,
    )

    assert error < 100.0


def test_synthetic_trial_sound_speed_error_degrades_accuracy():
    interpolators = {"north": straight_cable(y=5000.0)}
    dist_ranges = {"north": (10000, 50000)}

    error_matched = synthetic_trial(
        interpolators,
        cables=("north",),
        whale_x=30000.0,
        whale_y=2000.0,
        dist_ranges=dist_ranges,
        true_c=1480.0,
        assumed_c=1480.0,
        localize_kwargs={"x_range": (0, 60000), "y_range": (0, 10000), "grid_size": 60},
        rng=np.random.default_rng(2),
    )
    error_mismatched = synthetic_trial(
        interpolators,
        cables=("north",),
        whale_x=30000.0,
        whale_y=2000.0,
        dist_ranges=dist_ranges,
        true_c=1450.0,
        assumed_c=1480.0,
        localize_kwargs={"x_range": (0, 60000), "y_range": (0, 10000), "grid_size": 60},
        rng=np.random.default_rng(2),
    )

    assert error_mismatched > error_matched
