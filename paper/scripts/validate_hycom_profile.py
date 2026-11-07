"""Regenerate the two committed HYCOM profiles and the HYCOM-vs-in-situ
validation table, and check the profiles against the real pick-depth range.

This script *is* the fetch for both roles the sound-speed research needs, and
they are deliberately kept separate (review round 2 finding: conflating them
silently reintroduced Task 1's clamped-extrapolation bug through the data):

  1. VALIDATION profile, at the OOI mooring (44.37, -124.95) -- the only site
     this task can check HYCOM against a real in-situ measurement. Only
     ~600 m deep on HYCOM's grid.
  2. EXPERIMENT profile, at the South cable's own deepest channel (see
     `soundspeed.reanalysis.SOUTH_CABLE_DEEP_END`) -- reaches real HYCOM data
     to 1500 m, close enough to the cables' -1575 m to leave only a small
     sliver of clamped extrapolation. This one was NEVER checked against
     in-situ data; see the epistemics note in `soundspeed.reanalysis`'s
     module docstring before treating its agreement with in-situ as anything
     more than inferred.

`PROFILES` in `experiments.runner` loads both committed CSVs this script
writes, with no network at experiment time -- per the sound-speed plan's "do
not fetch at experiment time" rule, which exists so an experiment's input
cannot silently vary between runs or depend on live servers being up.

Requires live network access to two public, no-account endpoints:
  - OOI ERDDAP (in-situ CTD, QARTOD-filtered)      -- soundspeed.erddap
  - HYCOM GOFS 3.1 NCSS (reanalysis T/S, unpacked) -- soundspeed.reanalysis

Writes:
  - paper/results/profile_validation.csv
    (per-depth HYCOM-vs-in-situ comparison table at the OOI site, primary window)
  - paper/results/hycom_profile_oregon_nov2021.csv
    (validation-site HYCOM profile -- PROFILES["hycom_oregon_nov"])
  - paper/results/hycom_profile_south_cable_deep.csv
    (experiment HYCOM profile -- PROFILES["hycom_south_deep"])
  - paper/results/hycom_spatial_consistency_check.csv
    (South-cable-deep vs. South-cable-midpoint HYCOM fetches, same date,
    compared over their common depth band -- an internal cross-check of how
    much the reanalysis profile varies along the cable, NOT a validation:
    both sides are HYCOM, neither is a measurement)

Run (as a module, like the other scripts in this package, so `viz` and
`scripts` resolve from the repo root):
    uv run python -m paper.scripts.validate_hycom_profile

The validated band is derived from whatever the in-situ fetch actually
returns, never hard-coded: OOI's QARTOD-passing shallow edge is
window-dependent (see `soundspeed.erddap.NOV_2021_STATUS`), so a literal
depth here would silently drift out of sync with reality. Pick-depth coverage
is likewise computed from the committed pick snapshot (`dclde_localization.physics.picks`) and the
committed cable geometry (`geometry.load_cables`), not asserted as a number.

Only one profile product (HYCOM) is used throughout. The project spec calls
for two independent profile products so a conclusion can be checked against
both; that requirement is UNMET as of this script (see Task 9 report,
Finding 3) -- WOA23 or GLORYS12V1 would be the second, and neither is fetched
here.
"""

from __future__ import annotations

import numpy as np

from dclde_localization.physics.geometry import load_cables
from dclde_soundspeed.soundspeed.erddap import fetch_profile
from dclde_soundspeed.soundspeed.reanalysis import (
    HYCOM_EXPERIMENT_PROFILE_PATH, HYCOM_PROFILE_PATH, SOUTH_CABLE_DEEP_END,
    check_sound_speed_sanity, compare_profiles, coverage_stats,
    fetch_hycom_profile, save_profile, validation_summary)
from dclde_localization.physics.picks import attach_xyz, load_picks

# `experiments.runner.write_results` is imported lazily inside `main()`,
# after both committed profile CSVs have been written by `save_profile`.
# `runner.py` eagerly loads both profiles by path at import time (so
# experiments never fetch over the network) -- importing it here, at module
# scope, before those files exist on a fresh checkout would be exactly the
# chicken-and-egg failure this script exists to avoid: it IS what creates the
# files runner.py expects to already be there.

VALIDATION_SITE = (44.37, -124.95)
HYCOM_DATE = "2021-11-04"
# Primary: 1-6 Nov, temporally matched to the single HYCOM daily snapshot above.
PRIMARY_WINDOW = ("2021-11-01", "2021-11-06")
# Sensitivity check: wider window, more QC-passing rows, shallower edge --
# does the disagreement number move much if the window changes?
WIDE_WINDOW = ("2021-10-25", "2021-11-15")
GEOMETRY_DIR = "fixtures"

# A median disagreement above this would exceed the effect this whole
# research programme is chasing -- stop rather than commit a base profile
# that isn't usable.
STOP_THRESHOLD_MS = 5.0


def _report_validation(label: str, insitu, hycom) -> dict:
    df = compare_profiles(hycom, insitu, insitu.z_down.min(), insitu.z_down.max())
    summary = validation_summary(df)
    print(f"{label}: in-situ {insitu.source}")
    print(f"  validated band {df['z_down'].min():.0f}..{df['z_down'].max():.0f} m, "
          f"n={summary['n']} compare points")
    print(f"  median_abs_diff_ms={summary['median_abs_diff_ms']:.3f}  "
          f"p90_abs_diff_ms={summary['p90_abs_diff_ms']:.3f}  "
          f"max_abs_diff_ms={summary['max_abs_diff_ms']:.3f}")
    return {"df": df, "summary": summary}


def _pick_depths() -> np.ndarray:
    """Real DAS pick depths (positive-down), from the committed snapshot and
    committed cable geometry -- no network, no assumed numbers."""
    cables = load_cables(GEOMETRY_DIR)
    picks = attach_xyz(load_picks(), cables)
    return -picks["pz"].to_numpy(dtype=float)  # repo frame is negative-down


def main() -> None:
    # --- 1. Validation profile, at the OOI site (compute only; write after
    #        the stop-threshold check and after the experiment profile is
    #        fetched, so both profile files exist before runner.py is
    #        imported for write_results -- see the module-level comment). ---
    hycom_validation = fetch_hycom_profile(*VALIDATION_SITE, HYCOM_DATE)
    check_sound_speed_sanity(hycom_validation)
    print(f"hycom validation-site ({HYCOM_DATE}, {VALIDATION_SITE}): "
          f"{hycom_validation.source}")

    insitu_primary = fetch_profile("ce04ospd_deep", *PRIMARY_WINDOW, *VALIDATION_SITE)
    primary = _report_validation(f"PRIMARY {PRIMARY_WINDOW[0]}..{PRIMARY_WINDOW[1]}",
                                  insitu_primary, hycom_validation)

    insitu_wide = fetch_profile("ce04ospd_deep", *WIDE_WINDOW, *VALIDATION_SITE)
    _report_validation(f"SENSITIVITY {WIDE_WINDOW[0]}..{WIDE_WINDOW[1]}",
                        insitu_wide, hycom_validation)

    if primary["summary"]["median_abs_diff_ms"] > STOP_THRESHOLD_MS:
        raise SystemExit(
            f"STOP: median disagreement "
            f"{primary['summary']['median_abs_diff_ms']:.2f} m/s exceeds "
            f"{STOP_THRESHOLD_MS} m/s -- reanalysis is not usable as a base "
            f"profile. Not writing results or committing any profile.")

    validation_path = save_profile(hycom_validation, HYCOM_PROFILE_PATH)
    print(f"wrote {validation_path} (source: {hycom_validation.source})")

    # --- 2. Experiment profile, at the South cable's deepest channel --------
    print(f"\nfetching experiment profile at South cable deep end "
          f"{SOUTH_CABLE_DEEP_END} ...")
    hycom_experiment = fetch_hycom_profile(*SOUTH_CABLE_DEEP_END, HYCOM_DATE)
    check_sound_speed_sanity(hycom_experiment)  # loud failure, not a silent fallback
    print(f"hycom experiment-site ({HYCOM_DATE}, {SOUTH_CABLE_DEEP_END}): "
          f"{hycom_experiment.source}")

    experiment_path = save_profile(hycom_experiment, HYCOM_EXPERIMENT_PROFILE_PATH)
    print(f"wrote {experiment_path} (source: {hycom_experiment.source})")

    # Both committed profile files exist now, so runner.py's eager
    # load_profile(...) calls at import time will succeed.
    from dclde_soundspeed.experiments.runner import write_results

    out = write_results(primary["df"], "profile_validation")
    print(f"\nwrote {out}")

    depths = _pick_depths()
    cov = coverage_stats(hycom_experiment, depths)
    print(f"pick-depth coverage: {cov['n_covered']}/{cov['n']} picks "
          f"({cov['pct_covered']:.1f}%) within the profile's sampled range "
          f"(0..{cov['profile_z_max_m']:.0f} m); "
          f"deepest uncovered pick needs {cov['max_extrapolation_m']:.0f} m "
          f"of clamped extrapolation beyond that")

    old_cov = coverage_stats(hycom_validation, depths)
    print(f"(for comparison, the validation-site profile covers only "
          f"{old_cov['n_covered']}/{old_cov['n']} picks, "
          f"{old_cov['pct_covered']:.1f}%)")

    # --- 3. Spatial-consistency spot check (HYCOM vs. HYCOM, not a validation) --
    # Does the reanalysis profile vary meaningfully along the cable? Fetch a
    # second point at the cable's own midpoint (derived from geometry, not a
    # literal lat/lon) and compare the two HYCOM fetches over the depth band
    # they share.
    south = load_cables(GEOMETRY_DIR)["South"]
    mid_i = len(south.lat) // 2
    mid_lat, mid_lon = float(south.lat[mid_i]), float(south.lon[mid_i])
    print(f"\nfetching spatial spot-check at South cable midpoint "
          f"({mid_lat:.4f}, {mid_lon:.4f}) ...")
    hycom_mid = fetch_hycom_profile(mid_lat, mid_lon, HYCOM_DATE)
    check_sound_speed_sanity(hycom_mid)

    spot_df = compare_profiles(hycom_experiment, hycom_mid,
                                hycom_mid.z_down.min(), hycom_mid.z_down.max())
    spot_summary = validation_summary(spot_df)
    print(f"deep-end vs. midpoint HYCOM (both {HYCOM_DATE}), band "
          f"{spot_df['z_down'].min():.0f}..{spot_df['z_down'].max():.0f} m, "
          f"n={spot_summary['n']}: median={spot_summary['median_abs_diff_ms']:.3f} m/s "
          f"p90={spot_summary['p90_abs_diff_ms']:.3f} m/s "
          f"max={spot_summary['max_abs_diff_ms']:.3f} m/s")
    print("(this is HYCOM-vs-HYCOM at two locations, not a validation against "
          "measurement -- it only bounds how much the choice of fetch point "
          "along the cable matters)")
    spot_path = write_results(spot_df, "hycom_spatial_consistency_check")
    print(f"wrote {spot_path}")

    print("\nNOTE: only HYCOM was checked here. The spec wants two independent "
          "profile products before trusting a conclusion; that is not yet met.")
    print("NOTE: hycom_south_deep (the experiment profile) was never checked "
          "against in-situ data. Its credibility is inferred from the "
          "validation-site agreement above, not measured at its own location.")


if __name__ == "__main__":
    main()
