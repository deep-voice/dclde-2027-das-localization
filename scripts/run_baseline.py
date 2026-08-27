#!/usr/bin/env python3
"""Run the physics-based TDOA baseline over the full annotation dataset.

For every whale call in the dataset (grouped by batch + time window +
call_id), localizes it with L-1 (each cable independently, when picked up
on that cable) and L-2 (both cables combined, when picked up on both), then
reports summary error/residual statistics as the baseline result to beat.

There's no official ground truth to score against yet (see
dclde_localization.evaluate), so "error" here is the L1 residual converted
to an approximate distance via `mean(|residual|) * c` -- the same proxy the
explainer notebook uses -- not a true position error.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dclde_localization.geometry import load_interpolators  # noqa: E402
from dclde_localization.localize import DEFAULT_SOUND_SPEED, localize_2d  # noqa: E402
from dclde_localization.picks import load_annotations  # noqa: E402

MIN_PICKS = 5
X_RANGE = (0, 110000)
Y_RANGE = (5000, 35000)
GRID_SIZE = 60


def residual_error_m(result, c):
    return float(np.mean(np.abs(result["per_pick_residuals"])) * c)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=str(Path(__file__).parent.parent / "dclde_2027_das-finwhale-localization"),
        help="Directory containing annotations/ and geometry/ (default: the downloaded full dataset)",
    )
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    args = parser.parse_args()

    print(f"Loading annotations and geometry from {args.data_root} ...")
    picks = load_annotations(args.data_root)
    interpolators = load_interpolators(args.data_root)
    print(f"Loaded {len(picks)} picks across {picks['batch'].nunique()} batches, "
          f"{picks.groupby(['batch', 'window', 'call_id', 'cable']).ngroups} (call, cable) pick-sets.")

    rows = []
    groups = picks.groupby(["batch", "window", "call_id"])
    for (batch, window, call_id), group in groups:
        by_cable = {cable: sub for cable, sub in group.groupby("cable")}
        if len(group) < MIN_PICKS:
            continue

        cx_all, cy_all, t_all = [], [], []
        per_cable_result = {}
        for cable, sub in by_cable.items():
            cx, cy = interpolators[cable].xy(sub["dist"].values)
            cx_all.append(cx)
            cy_all.append(cy)
            t_all.append(sub["time"].values)
            if len(sub) >= MIN_PICKS:
                r = localize_2d(cx, cy, sub["time"].values, x_range=X_RANGE, y_range=Y_RANGE, grid_size=args.grid_size)
                per_cable_result[cable] = residual_error_m(r, DEFAULT_SOUND_SPEED)

        row = {
            "batch": batch,
            "window": window,
            "call_id": call_id,
            "n_picks": len(group),
            "cables": ",".join(sorted(by_cable)),
            "error_north_m": per_cable_result.get("north"),
            "error_south_m": per_cable_result.get("south"),
        }

        if len(by_cable) > 1:
            r_combined = localize_2d(
                np.concatenate(cx_all), np.concatenate(cy_all), np.concatenate(t_all),
                x_range=X_RANGE, y_range=Y_RANGE, grid_size=args.grid_size,
            )
            row["error_combined_m"] = residual_error_m(r_combined, DEFAULT_SOUND_SPEED)

        rows.append(row)

    results = pd.DataFrame(rows)
    print(f"\nLocalized {len(results)} calls (>= {MIN_PICKS} picks) out of {groups.ngroups} total.\n")

    for col in ["error_north_m", "error_south_m", "error_combined_m"]:
        vals = results[col].dropna()
        if len(vals) == 0:
            continue
        print(f"{col:20s} n={len(vals):4d}  mean={vals.mean():7.1f} m  median={vals.median():7.1f} m  "
              f"p90={vals.quantile(0.9):7.1f} m")


if __name__ == "__main__":
    main()
