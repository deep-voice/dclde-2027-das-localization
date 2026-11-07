from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from dclde_soundspeed.experiments import ablation as ablation_mod
from dclde_soundspeed.experiments.ablation import (
    ablation_report, classify_c_plausibility, effective_c_report, fit_free_c,
    free_c_flip_diagnostics, raytrace_fallback_fraction_at_best_fit,
    raytrace_fallback_over_search_grid, solve_all)
from dclde_localization.physics.solve import Solution, solve
from dclde_localization.physics.sound_speed_models import ConstantC
from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile
from dclde_soundspeed.soundspeed.raytrace import RayTrace


def _picks(n=30, src=(35000.0, 26000.0, -30.0), c=1480.0, t0=5.0, call="c1"):
    px = np.linspace(12000.0, 60000.0, n)
    py = np.full(n, 20000.0)
    pz = np.linspace(-80.0, -1500.0, n)
    t = ConstantC(c).travel_time(*src, px, py, pz) + t0
    return pd.DataFrame({"call": call, "px": px, "py": py, "pz": pz,
                         "time": t, "snr": 15.0})


def test_free_c_recovers_the_speed_that_generated_the_arrivals():
    # The estimator must be able to find c before any ablation using it means
    # anything.
    df = fit_free_c(_picks(c=1455.0), source_z=-30.0,
                    c_grid=np.arange(1430.0, 1510.0, 2.5))
    assert abs(float(df["best_c"].iloc[0]) - 1455.0) <= 5.0


def test_free_c_reduces_or_matches_the_fixed_c_misfit():
    # A free parameter can never fit worse than a fixed one at its own optimum.
    # If it does, the search grid or the objective is wrong.
    df = fit_free_c(_picks(c=1455.0), source_z=-30.0,
                    c_grid=np.arange(1430.0, 1510.0, 2.5))
    assert float(df["best_misfit"].iloc[0]) <= float(df["fixed_misfit"].iloc[0]) + 1e-9


def test_ablation_report_never_merges_flips_into_the_delta_statistic():
    # The reporting contract from spec section 5. A merged median would hide the
    # only interesting failure mode.
    #
    # `n == n_stable + n_flipped` holds by construction of how Comparison is
    # built (ablation.py's own `n_stable=n - flips`), so it cannot fail
    # regardless of whether the reported numbers are actually right (Task 14
    # review) -- it was not testing the "never merges" claim in this test's
    # name at all. This fixture is a single straight cable, whose two mirror
    # lobes fit almost equally well (see test_the_two_lobes_fit_almost_
    # equally_well in test_solve.py); checked directly, a 1 m/s change in c
    # is enough to flip which lobe ranks first here -- a concrete, and
    # rather pointed, illustration of exactly the failure mode this
    # reporting contract exists for: with n_stable=0, `stable_delta_median_m`
    # must come back NaN, not silently average in a kilometre-scale flip as
    # if it were a small, stable shift.
    base = solve_all(_picks(call="a"), ConstantC(1480.0), source_z=-30.0)
    variant = solve_all(_picks(call="a"), ConstantC(1479.0), source_z=-30.0)
    rep = ablation_report(base, variant, "tiny-c-change")
    assert {"label", "n", "n_stable", "n_flipped", "flip_fraction",
            "stable_delta_median_m", "stable_delta_p90_m"} <= set(rep)
    assert rep["n"] == 1
    assert rep["n_flipped"] == 1
    assert rep["n_stable"] == 0
    assert rep["flip_fraction"] == 1.0
    assert np.isnan(rep["stable_delta_median_m"]), \
        "no stable calls survived -- the delta statistic must be NaN, not a merged number"


def test_solve_all_skips_calls_below_the_pick_threshold():
    small = _picks(n=4, call="tiny")
    assert solve_all(small, ConstantC(), source_z=-30.0, min_picks=8) == {}


# --- Review round 1 additions (Finding 4): R8 narrowing, `flipped`, and the
# `_solve_or_none` drop-on-failure path had no coverage. ------------------

def test_r8_coarse_to_fine_search_agrees_with_a_dense_brute_force_grid():
    # Ruling R8's original justification ("misfit is smooth and
    # near-unimodal in c") does not hold in general -- the real 524-call run
    # has a 37.5% basin-flip rate, meaning the top-ranked-basin identity (and
    # so the misfit-vs-c curve) can have a kink. This is the "add a check"
    # half of that finding: a spot check against a materially denser (81 vs.
    # the 3x9=27 R8 uses), single-pass brute-force grid over the same outer
    # bracket, for one representative call. It is evidence the search behaves
    # sanely here, not a proof it always finds the global optimum.
    picks = _picks(c=1467.0)
    lo, hi = 1430.0, 1530.0
    df = fit_free_c(picks, source_z=-30.0, c_grid=np.array([lo, hi]))

    # n_coarse=41 (not solve()'s default 161) for the 81-solve brute-force
    # comparison loop below: this spot check only needs to find which of 81
    # candidate c values fits best to within a few m/s, not localization-grade
    # position precision, and 81 solves at the default 161x161 grid is most
    # of this test file's runtime for no accuracy this check actually needs
    # (Task 14 review; measured ~2.5 s at n_coarse=41 vs. much longer at the
    # default). fit_free_c's own 27 solves above are left at the default --
    # that IS the production code path under test.
    brute_best_misfit, brute_best_c = np.inf, np.nan
    for c in np.linspace(lo, hi, 81):
        sol = solve(picks["px"].to_numpy(), picks["py"].to_numpy(), picks["pz"].to_numpy(),
                    picks["time"].to_numpy(), ConstantC(float(c)), sz=-30.0, n_coarse=41)
        if sol.best.misfit < brute_best_misfit:
            brute_best_misfit, brute_best_c = sol.best.misfit, float(c)

    assert abs(float(df["best_c"].iloc[0]) - brute_best_c) <= 5.0
    # R8's finer resolution should never leave it with a worse misfit than
    # the coarser brute-force grid found.
    assert float(df["best_misfit"].iloc[0]) <= brute_best_misfit + 1e-6


def test_fit_free_c_flip_threshold_controls_the_flipped_column():
    # Real mirror-lobe flips are geometry-dependent and awkward to construct
    # on demand; the threshold logic itself is not. A threshold of 0 must
    # flag any nonzero shift as a flip, and a huge threshold must flag none --
    # this pins the `flipped = shift_m > flip_threshold_m` wiring independent
    # of whether this particular fixture happens to produce a real flip.
    picks = _picks(c=1455.0)
    df_sensitive = fit_free_c(picks, source_z=-30.0,
                              c_grid=np.arange(1430.0, 1510.0, 2.5), flip_threshold_m=0.0)
    df_lenient = fit_free_c(picks, source_z=-30.0,
                            c_grid=np.arange(1430.0, 1510.0, 2.5), flip_threshold_m=1e9)
    assert bool(df_sensitive["flipped"].iloc[0]) == (float(df_sensitive["shift_m"].iloc[0]) > 0.0)
    assert not bool(df_lenient["flipped"].iloc[0])


def test_solve_or_none_returns_none_when_solve_finds_no_usable_basin():
    # solve.py's documented edge case: a pick geometry whose only candidate
    # refinement lands on its own box boundary is discarded, leaving
    # Solution.basins empty. `_solve_or_none` must turn that into None, not
    # let `.best` raise IndexError.
    with patch.object(ablation_mod, "solve",
                      return_value=Solution(basins=(), n_picks=10, model_name="x")):
        got = ablation_mod._solve_or_none(np.zeros(10), np.zeros(10), np.zeros(10),
                                          np.zeros(10), ConstantC(), -30.0)
    assert got is None


def test_solve_or_none_returns_none_on_expected_solver_error():
    with patch.object(ablation_mod, "solve", side_effect=ValueError("degenerate grid")):
        got = ablation_mod._solve_or_none(np.zeros(10), np.zeros(10), np.zeros(10),
                                          np.zeros(10), ConstantC(), -30.0)
    assert got is None


def test_solve_all_drops_a_call_solve_cannot_produce_a_basin_for():
    # Same edge case as above, exercised through solve_all's own loop rather
    # than _solve_or_none directly: a call above min_picks whose solve()
    # yields no basin must be dropped, not crash the whole ablation.
    with patch.object(ablation_mod, "solve",
                      return_value=Solution(basins=(), n_picks=10, model_name="x")):
        out = solve_all(_picks(n=10, call="no-basin"), ConstantC(), source_z=-30.0, min_picks=8)
    assert out == {}


# --- Review round 1 additions (Finding 1): the plausibility classifier. ---

# --- Task 14 review: promote the one-off report scripts that built
# ablation_free_c_flip_diagnostics.csv and ablation_effective_c*.csv into
# tested functions (Critical 1), and fix RayTrace.last_fallback_fraction's
# bias toward zero after a solve loop (Important 4). ------------------------

def test_free_c_flip_diagnostics_reports_separation_columns_for_flipped_calls():
    # A genuine mirror-lobe flip is awkward to construct on demand (see the
    # flip_threshold test above); flip_threshold_m=0.0 forces this call's
    # ordinary nonzero shift to count as "flipped" so the diagnostic path runs
    # and its output shape/columns can be checked without needing a real
    # kilometre-scale ambiguity.
    picks = _picks(c=1455.0)
    fit = fit_free_c(picks, source_z=-30.0, c_grid=np.arange(1430.0, 1510.0, 2.5),
                     flip_threshold_m=0.0)
    assert bool(fit["flipped"].iloc[0])
    diag = free_c_flip_diagnostics(fit, picks, source_z=-30.0)
    assert len(diag) == 1
    assert {"call", "best_c", "shift_m", "fixed_separation_resolved", "fixed_separation_m",
            "best_c_separation_resolved", "best_c_separation_m", "fixed_n_basins",
            "best_c_n_basins", "fixed_best_provenance", "best_c_best_provenance"} <= set(diag.columns)
    assert diag["fixed_best_provenance"].iloc[0] in ("local_min", "fallback", "unresolved")


def test_free_c_flip_diagnostics_is_empty_when_nothing_flipped():
    picks = _picks(c=1455.0)
    fit = fit_free_c(picks, source_z=-30.0, c_grid=np.arange(1430.0, 1510.0, 2.5),
                     flip_threshold_m=1e9)
    diag = free_c_flip_diagnostics(fit, picks, source_z=-30.0)
    assert diag.empty


_SMALL_PROFILE = SoundSpeedProfile(
    np.array([0.0, 100.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0]),
    np.array([1491.0, 1480.0, 1478.0, 1477.0, 1479.0, 1484.0, 1490.0]),
    "small-oregon")


def test_raytrace_fallback_at_best_fit_reports_a_fraction_and_its_denominator():
    picks = _picks(n=20, call="c1")
    model = RayTrace(_SMALL_PROFILE, n_z=5, n_r=5)
    solutions = solve_all(picks, model, source_z=-30.0)
    frac, n = raytrace_fallback_fraction_at_best_fit(model, solutions, picks)
    assert 0.0 <= frac <= 1.0
    assert n == len(picks)  # one call, all its picks queried once


def test_raytrace_fallback_over_search_grid_pools_every_grid_query():
    # The whole point of this function: the grid-pooled fraction has a much
    # larger denominator than the best-fit-only one (161*161 candidate source
    # positions per pick, not one), and resetting the record first means an
    # earlier at-best-fit call on the same instance cannot leak into it.
    picks = _picks(n=20, call="c1")
    model = RayTrace(_SMALL_PROFILE, n_z=5, n_r=5)
    solutions = solve_all(picks, model, source_z=-30.0)
    raytrace_fallback_fraction_at_best_fit(model, solutions, picks)  # leaves a record behind

    n_coarse = 9  # small grid so the test stays fast
    frac, n = raytrace_fallback_over_search_grid(model, solutions, picks, n_coarse=n_coarse)
    assert 0.0 <= frac <= 1.0
    assert n == n_coarse * n_coarse * len(picks)


def test_effective_c_report_matches_ablation_report_and_states_dependency():
    # End-to-end driver for section 7.3 (Critical 1): must reproduce the same
    # numbers ablation_report/compare would give directly, plus the profile
    # provenance and the raytrace-vs-path_average dependency note.
    picks = _picks(n=20, call="c1")
    summary, detail = effective_c_report(picks, _SMALL_PROFILE, source_z=-30.0)
    assert list(summary["label"]) == ["path_average vs constant", "raytrace vs constant"]
    assert (summary["profile_source"] == _SMALL_PROFILE.source).all()
    assert summary.loc[summary["label"] == "path_average vs constant", "dependency_note"].iloc[0] \
        == "independent of the raytrace row"
    rt_note = summary.loc[summary["label"] == "raytrace vs constant", "dependency_note"].iloc[0]
    assert "NOT independent" in rt_note
    assert {"call", "delta_m_path_average", "flipped_path_average",
            "delta_m_raytrace", "flipped_raytrace"} <= set(detail.columns)
    assert len(detail) == 1


def test_classify_c_plausibility_labels_edge_pinned_and_physical_rows():
    profile = SoundSpeedProfile(np.array([0.0, 1500.0]), np.array([1477.5, 1497.5]), "test")
    df = pd.DataFrame({
        "best_c": [1430.0, 1487.5, 1510.0, 1530.0],
        "outer_bracket_lo": 1430.0, "outer_bracket_hi": 1530.0,
    })
    out = classify_c_plausibility(df, profile)
    assert out["edge_pinned"].tolist() == [True, False, False, True]
    assert out["physically_plausible"].tolist() == [False, True, False, False]
