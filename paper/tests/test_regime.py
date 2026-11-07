import numpy as np
import pandas as pd
import pytest

from dclde_soundspeed.experiments.regime import (
    _profile_validation_note, _solve_and_residualize, depth_sweep, residual_table,
    shadow_zone_table, stratify)
from dclde_soundspeed.experiments.runner import PROFILES
from dclde_localization.physics.sound_speed_models import ConstantC, PathAverage
from dclde_soundspeed.soundspeed.modes import Seabed
from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile


def _picks_for_a_known_source(n=30, src=(35000.0, 26000.0, -30.0), t0=5.0):
    px = np.linspace(12000.0, 60000.0, n)
    py = np.full(n, 20000.0)
    pz = np.linspace(-80.0, -1500.0, n)
    t = ConstantC().travel_time(*src, px, py, pz) + t0
    return pd.DataFrame({"call": "c1", "px": px, "py": py, "pz": pz,
                         "time": t, "snr": np.linspace(8.0, 25.0, n)})


def test_residuals_are_numerically_zero_for_a_perfectly_matched_model():
    # If the model that generated the arrivals leaves residuals, the estimator
    # or the residual definition is broken, and every stratification built on
    # it is measuring that bug.
    df = residual_table(_picks_for_a_known_source(), ConstantC(), source_z=-30.0, min_picks=8)
    assert np.abs(df["residual_s"]).max() < 1e-3


def test_residual_table_carries_the_axes_the_stratification_needs():
    df = residual_table(_picks_for_a_known_source(), ConstantC(), source_z=-30.0, min_picks=8)
    assert {"call", "residual_s", "range_m", "channel_z", "snr"} <= set(df.columns)
    assert (df["range_m"] > 0).all()


def test_stratify_reports_its_denominator_per_cell():
    df = residual_table(_picks_for_a_known_source(), ConstantC(), source_z=-30.0, min_picks=8)
    out = stratify(df, range_bins=[0, 10_000, 25_000, 100_000])
    assert "n" in out.columns
    assert out["n"].sum() == len(df)


def test_calls_below_the_pick_threshold_are_excluded_not_silently_solved():
    small = _picks_for_a_known_source(n=4)
    df = residual_table(small, ConstantC(), source_z=-30.0, min_picks=8)
    assert df.empty


def test_stratify_drops_gross_outliers_from_the_mean_but_still_counts_them():
    # A single mis-associated pick with a multi-second residual must not be
    # allowed to carry a cell's mean (review round 1, Finding 1) -- but it
    # must still be visible in the table, not silently vanish.
    df = residual_table(_picks_for_a_known_source(), ConstantC(), source_z=-30.0, min_picks=8)
    contaminated = df.copy()
    contaminated.loc[0, "residual_s"] = 43.5  # gross mis-association, as seen on real data
    cut = stratify(contaminated, range_bins=[0, 10_000, 25_000, 100_000], clip_s=1.0)
    raw = stratify(contaminated, range_bins=[0, 10_000, 25_000, 100_000], clip_s=None)
    assert cut["n_dropped_outlier"].sum() == 1
    assert cut["n"].sum() == len(contaminated) - 1
    assert raw["n_dropped_outlier"].sum() == 0
    assert (cut["mean_ms"].dropna() < 1000).all()
    assert (raw["mean_ms"].dropna() > 1000).any()
    assert "clip_s" in cut.columns and "median_ms" in cut.columns


def test_stratify_crosses_range_bins_with_snr_quartiles():
    # Coverage gap (Task 14 review): stratify's snr_quartiles=True branch
    # (adding an "snr_q" grouping key via pd.qcut) was never exercised by any
    # test -- every existing stratify call used the default False.
    df = residual_table(_picks_for_a_known_source(n=40), ConstantC(), source_z=-30.0, min_picks=8)
    out = stratify(df, range_bins=[0, 25_000, 100_000], snr_quartiles=True)
    assert "snr_q" in out.columns
    assert set(out["snr_q"].astype(str)) <= {"Q1", "Q2", "Q3", "Q4"}
    # Every row's own "n" (post-cut count) still sums to the full input,
    # crossed or not -- the denominator contract must survive the extra key.
    assert out["n"].sum() + out["n_dropped_outlier"].sum() == len(df)


def test_solver_error_is_counted_not_left_to_crash_the_sweep(monkeypatch):
    # Finding 3: the old bare `except Exception` had no test exercising it.
    # Narrowing to ValueError and recording the count in `_solve_and_residualize`
    # replaces "trust the comment" with something a test can actually check.
    def boom(*args, **kwargs):
        raise ValueError("degenerate pick geometry")

    monkeypatch.setattr("dclde_soundspeed.experiments.regime.solve", boom)
    picks = _picks_for_a_known_source()
    rows, counts = _solve_and_residualize(picks, ConstantC(), source_z=-30.0, min_picks=8)
    assert rows == []
    assert counts == {"too_few_picks": 0, "no_basin": 0, "solver_error": 1, "solved": 0}

    df = residual_table(picks, ConstantC(), source_z=-30.0, min_picks=8)
    assert df.empty


def test_depth_sweep_accounts_for_every_call_at_every_depth():
    picks = _picks_for_a_known_source()
    df = depth_sweep(picks, ConstantC(), [-30.0, -60.0])
    for _, row in df.iterrows():
        assert (row["n_too_few_picks"] + row["n_no_basin"] + row["n_solver_error"]
                + row["n_calls_solved"]) == row["n_calls_total"]
        assert row["n_calls_total"] == picks["call"].nunique()
        assert row["clip_s"] == 1.0


def test_depth_sweep_profile_validated_uses_the_shared_classifier():
    # Task 12 review, Finding 2: depth_sweep used to have its own
    # profile_validated check that matched a substring occurring in none of
    # the real PROFILES source strings, so it silently fell through to
    # "unknown" for every profile-carrying model -- including
    # hycom_oregon_nov, the one profile genuinely validated against in-situ
    # CTD. All the depth_sweep tests above use ConstantC, which never
    # exercises this branch at all. This does, so the classifier can't
    # silently regress to "unknown" again without failing a test.
    picks = _picks_for_a_known_source()
    model = PathAverage(PROFILES["hycom_oregon_nov"])
    df = depth_sweep(picks, model, [-30.0])
    assert df["profile_validated"].iloc[0].startswith("yes")


# --- Task 14 review, Important 8: every branch of the classifier, not just
# the one "yes" branch that happened to be exercised above. In particular,
# hycom_south_deep's honest "not validated in-situ" disclosure is the one two
# of the plan's three headline findings (S3.3, S3.5) actually depend on, and
# it had no test at all before this. ---------------------------------------

def test_profile_validation_note_for_every_named_profile():
    assert _profile_validation_note(PROFILES["oregon_nov"]).startswith(
        "no -- hand-typed indicative")
    assert _profile_validation_note(PROFILES["hycom_oregon_nov"]).startswith("yes")
    # The branch the review found untested: hycom_south_deep is the profile
    # section 3.3 and 3.5's headline numbers actually run on, and it must
    # read as NOT validated in-situ, not silently fall through to "unknown"
    # (the exact failure mode of the classifier this replaced).
    note = _profile_validation_note(PROFILES["hycom_south_deep"])
    assert note.startswith("no --")
    assert "not validated in-situ" in note


def test_profile_validation_note_has_no_profile_and_unclassified_branches():
    # No profile at all (e.g. a ConstantC model) is "n/a", never "unknown" --
    # the two mean different things ("there is nothing to validate" vs. "there
    # is a profile and its status was never recorded").
    assert _profile_validation_note(None) == "n/a (no profile; constant speed)"

    unclassified = SoundSpeedProfile(np.array([0.0, 100.0]), np.array([1480.0, 1480.0]),
                                     "some-ad-hoc-profile")
    assert _profile_validation_note(unclassified) == "unknown"


def test_shadow_zone_table_reports_the_untested_not_validated_branch():
    # Same classifier, exercised through shadow_zone_table's own call site
    # (a second, independent call site from depth_sweep's) with the actual
    # hycom_south_deep profile the committed regime_shadow_zone.csv uses.
    got = shadow_zone_table(PROFILES["hycom_south_deep"], SWEEP, 20.0,
                            [(30.0, 150.0, 5000.0)])
    assert got["profile_validated"].iloc[0].startswith("no --")


OREGON = SoundSpeedProfile(
    np.array([0.0, 100.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0]),
    np.array([1491.0, 1480.0, 1478.0, 1477.0, 1479.0, 1484.0, 1490.0]),
    "oregon-nov")
SWEEP = [Seabed(c=1600.0, rho=1.6, alpha=0.1), Seabed(c=1800.0, rho=2.0, alpha=0.3)]


def test_shadow_zone_table_reports_both_the_ray_limit_and_the_mode_count():
    got = shadow_zone_table(OREGON, SWEEP, 20.0,
                            [(30.0, 150.0, 5000.0), (30.0, 500.0, 20000.0)])
    assert {"z_src", "z_rcv", "range_m", "ray_max_direct_m", "n_modes_min",
            "n_modes_max", "ray_says_shadowed"} <= set(got.columns)
    assert got["ray_says_shadowed"].all()


def test_seabed_sweep_is_reported_as_a_spread_not_a_single_value():
    # Seabed geoacoustics are unknown here. Fixing them at invented values and
    # reporting one number would present a guess as a measurement.
    #
    # n_modes_max >= n_modes_min holds by construction of how the table is
    # built (max()/min() over the same `counts` list) regardless of whether
    # the seabed sweep actually changes anything, so that alone cannot fail
    # and does not test the claim in this test's name (Task 14 review). The
    # real claim -- that sweeping the seabed produces a genuine SPREAD, not
    # the same count reported twice under two column names -- needs the
    # sweep to actually contain seabeds whose sound speed differs enough to
    # change the mode count, and needs that difference in the output.
    # The 150 m/5 km geometry alone (checked directly) happens to give the
    # same count (2) for both seabeds -- too shallow/close a case for the
    # 1600 vs 1800 m/s difference to change how many modes are trapped -- so
    # a second, deeper/farther geometry is included specifically because it
    # is known to show a real spread (5-8 modes), not chosen to make the
    # assertion pass by accident.
    got = shadow_zone_table(OREGON, SWEEP, 20.0,
                            [(30.0, 150.0, 5000.0), (30.0, 500.0, 20000.0)])
    assert (got["n_modes_max"] >= got["n_modes_min"]).all()
    assert (got["n_modes_max"] > got["n_modes_min"]).any(), \
        "the two seabeds in SWEEP differ enough (1600 vs 1800 m/s) that at " \
        "least one geometry should show a real spread, not an identical count"


def test_no_geometries_gives_an_empty_table_rather_than_raising():
    assert shadow_zone_table(OREGON, SWEEP, 20.0, []).empty
