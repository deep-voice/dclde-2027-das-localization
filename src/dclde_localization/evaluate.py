"""Self-evaluation harness.

No official evaluation metric, baseline, or ground truth exists yet for this
challenge (confirmed against the DCLDE 2027 site and dataset -- even the
training annotation CSVs contain only arrival *picks*, never a true whale
position). Until the organizers release the held-out test set, two proxies
stand in:

1. Synthetic ground truth: place a whale at a KNOWN position on the real
   cable geometry, simulate picks with configurable noise, run the
   localizer, and measure the true error in meters. This is the only source
   of a real accuracy number right now.
2. Cross-cable consistency: for calls picked up on both cables, compare the
   single-cable (L-1) estimates against the combined (L-2) estimate as an
   empirical, real-data sanity check.
"""
import numpy as np
import pandas as pd

from .localize import localize_2d
from .picks import call_picks, matched_cross_cable_calls


def simulate_picks(interpolator, dist_samples, whale_x, whale_y, t0=0.0, c=1480.0, time_noise_std=0.0, rng=None):
    """Simulate arrival-time picks for a whale at a known (whale_x, whale_y).

    `dist_samples` are the along-cable distances (m) at which a pick is
    "made" -- pass a random subset of real chan_m values to mimic realistic
    pick sparsity and geometry.
    """
    rng = rng or np.random.default_rng()
    cx, cy = interpolator.xy(dist_samples)
    dists = np.sqrt((whale_x - cx) ** 2 + (whale_y - cy) ** 2)
    times = t0 + dists / c + rng.normal(0.0, time_noise_std, size=len(dist_samples))
    return np.asarray(cx), np.asarray(cy), times


def synthetic_trial(
    interpolators,
    cables,
    whale_x,
    whale_y,
    dist_ranges,
    n_picks_per_cable=15,
    t0=0.0,
    true_c=1480.0,
    assumed_c=1480.0,
    time_noise_std=0.0,
    localize_kwargs=None,
    rng=None,
):
    """Run one synthetic localization trial and return the true position error (m).

    `cables` is e.g. ("north",) for an L-1 trial or ("north", "south") for
    L-2. `true_c` is the sound speed used to generate the data and
    `assumed_c` is what the localizer is told to use -- set them apart to
    test sensitivity to sound-speed-model error.
    """
    rng = rng or np.random.default_rng()
    localize_kwargs = localize_kwargs or {}

    all_cx, all_cy, all_t = [], [], []
    for cable in cables:
        lo, hi = dist_ranges[cable]
        dist_samples = rng.uniform(lo, hi, size=n_picks_per_cable)
        cx, cy, t = simulate_picks(
            interpolators[cable], dist_samples, whale_x, whale_y, t0=t0, c=true_c, time_noise_std=time_noise_std, rng=rng
        )
        all_cx.append(cx)
        all_cy.append(cy)
        all_t.append(t)

    picks_x = np.concatenate(all_cx)
    picks_y = np.concatenate(all_cy)
    picks_t = np.concatenate(all_t)

    result = localize_2d(picks_x, picks_y, picks_t, c=assumed_c, **localize_kwargs)
    return float(np.hypot(result["x"] - whale_x, result["y"] - whale_y))


def cross_cable_consistency(picks, interpolators, localize_kwargs=None):
    """For every call seen on both cables, compare L-1(each cable) to L-2(combined).

    Returns a DataFrame with one row per matched call: the three fitted
    positions, and the distance from each single-cable estimate to the
    combined estimate (an empirical, real-data proxy for accuracy since no
    true position is available).
    """
    localize_kwargs = localize_kwargs or {}
    rows = []

    for batch, window, call_id in matched_cross_cable_calls(picks):
        north = call_picks(picks, batch, window, call_id, cable="north")
        south = call_picks(picks, batch, window, call_id, cable="south")

        north_cx, north_cy = interpolators["north"].xy(north["dist"].values)
        south_cx, south_cy = interpolators["south"].xy(south["dist"].values)

        result_n = localize_2d(north_cx, north_cy, north["time"].values, **localize_kwargs)
        result_s = localize_2d(south_cx, south_cy, south["time"].values, **localize_kwargs)
        result_both = localize_2d(
            np.concatenate([north_cx, south_cx]),
            np.concatenate([north_cy, south_cy]),
            np.concatenate([north["time"].values, south["time"].values]),
            **localize_kwargs,
        )

        rows.append(
            {
                "batch": batch,
                "window": window,
                "call_id": call_id,
                "n_picks_north": len(north),
                "n_picks_south": len(south),
                "x_north": result_n["x"],
                "y_north": result_n["y"],
                "x_south": result_s["x"],
                "y_south": result_s["y"],
                "x_combined": result_both["x"],
                "y_combined": result_both["y"],
                "north_vs_combined_m": float(np.hypot(result_n["x"] - result_both["x"], result_n["y"] - result_both["y"])),
                "south_vs_combined_m": float(np.hypot(result_s["x"] - result_both["x"], result_s["y"] - result_both["y"])),
            }
        )

    return pd.DataFrame(rows)
