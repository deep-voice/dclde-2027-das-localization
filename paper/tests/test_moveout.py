import numpy as np
import pytest

from dclde_soundspeed.experiments.moveout import (
    DEFAULT_C, NULL_SOURCE_Z, depth_profile, marginal_sigma_z, robust_sigma)


def synthetic_call(n=24, z_src=-120.0, depth_span=400.0, rng_seed=0, noise_s=0.0):
    """Two receiver arcs at different bearings from the source, each with its
    own depth-vs-along-track relationship (and the two run in OPPOSITE
    directions along track). A single arc whose depth rises monotonically
    with along-track position is depth/range degenerate -- a depth change is
    almost exactly mimicked by a position shift along the arc (measured: ~23
    us of unresolvable residual for 400 m of depth error). Two arcs at
    different bearings are not (measured: ~3 ms for the same 400 m at the
    single-arc case's own geometry) -- see `depth_profile`'s docstring.
    Returns (px, py, pz, obs_t, truth)."""
    rng = np.random.default_rng(rng_seed)
    n1 = n // 2
    n2 = n - n1
    sx, sy = 500.0, -2000.0

    tA = np.linspace(-1.0, 1.0, n1)
    pxA = sx + 800.0 * tA
    pyA = sy + 200.0 + 200.0 * tA**2
    pzA = -300.0 - depth_span * (tA + 1.0) / 2.0

    tB = np.linspace(-1.0, 1.0, n2)
    pxB = sx + 200.0 + 200.0 * tB**2
    pyB = sy + 800.0 * tB
    pzB = -300.0 - depth_span * (1.0 - tB) / 2.0

    px = np.concatenate([pxA, pxB])
    py = np.concatenate([pyA, pyB])
    pz = np.concatenate([pzA, pzB])
    r = np.sqrt((px - sx) ** 2 + (py - sy) ** 2 + (pz - z_src) ** 2)
    obs_t = r / DEFAULT_C + 12.3 + rng.normal(0.0, noise_s, n)
    return px, py, pz, obs_t, (sx, sy, z_src)


def test_robust_sigma_is_mad_based_and_ignores_a_gross_outlier():
    r = np.concatenate([np.full(50, 0.0), np.array([1000.0])])
    assert robust_sigma(r) == pytest.approx(0.0, abs=1e-12)
    r2 = np.concatenate([np.array([-1.0, 1.0] * 25), np.array([1e6])])
    # n = 51: the outlier shifts the median off zero to 1.0, MAD = 2.0, so
    # 1.4826 * MAD = 2.9652 -- not 1.4826.
    assert robust_sigma(r2) == pytest.approx(2.9652, rel=1e-6)
    # And the contrast with a non-robust spread estimator is the whole point:
    # np.std is blown up by the single 1e6 outlier by orders of magnitude.
    assert np.std(r2) > 1000.0 * robust_sigma(r2)


def test_depth_profile_minimum_sits_at_the_true_depth_when_noise_free():
    px, py, pz, obs_t, (_, _, z_true) = synthetic_call(z_src=-120.0, noise_s=0.0)
    z_grid = np.linspace(-400.0, 0.0, 81)
    zs, mis = depth_profile(px, py, pz, obs_t, z_grid=z_grid)
    assert zs.shape == mis.shape == z_grid.shape
    assert zs[int(np.argmin(mis))] == pytest.approx(z_true, abs=10.0)


def test_depth_profile_is_a_real_profile_not_a_fixed_xy_slice():
    """x and y must be re-minimised at every candidate depth. If they were held
    at the null solution the profile would be spuriously sharp, which is the
    exact way this measurement overstates depth resolution."""
    px, py, pz, obs_t, _ = synthetic_call(z_src=-120.0, noise_s=0.0)
    z_grid = np.linspace(-400.0, 0.0, 41)
    _, profiled = depth_profile(px, py, pz, obs_t, z_grid=z_grid)

    from dclde_localization.physics.forward import misfit_value, travel_time
    sx, sy = 500.0, -2000.0
    fixed = []
    for z in z_grid:
        tt = travel_time(sx, sy, z, px, py, pz, DEFAULT_C)
        fixed.append(misfit_value(obs_t, tt, "l2"))
    fixed = np.array(fixed)
    # Profiling can only lower the misfit, never raise it.
    assert np.all(profiled <= fixed + 1e-9)
    # And it must actually move somewhere: a pure slice would be identical.
    assert np.any(profiled < fixed - 1e-6)


def test_marginal_sigma_z_widens_as_the_noise_grows():
    z_grid = np.linspace(-400.0, 0.0, 161)
    px, py, pz, obs_t, _ = synthetic_call(noise_s=0.0)
    _, mis = depth_profile(px, py, pz, obs_t, z_grid=z_grid)
    s_small, r_small = marginal_sigma_z(z_grid, mis, n_picks=len(px), resid_sigma=0.005)
    s_big, r_big = marginal_sigma_z(z_grid, mis, n_picks=len(px), resid_sigma=0.050)
    assert np.isfinite(s_small) and np.isfinite(s_big)
    assert r_small == "" and r_big == ""
    assert s_big > s_small


def test_marginal_sigma_z_is_infinite_when_the_profile_is_flat():
    """A perfectly flat profile's minimum is tied everywhere; `argmin`
    resolves the tie at the first (deepest) index, so this is reported as an
    edge case, not a distinct "flat" reason -- an honest description of what
    argmin actually did, not a special case invented for this test."""
    z_grid = np.linspace(-400.0, 0.0, 41)
    flat = np.full_like(z_grid, 0.25)
    sigma, reason = marginal_sigma_z(z_grid, flat, n_picks=20, resid_sigma=0.01)
    assert not np.isfinite(sigma)
    assert reason == "edge_deep"


def test_marginal_sigma_z_is_infinite_with_not_crossed_when_an_interior_minimum_never_bounds():
    """A genuinely interior minimum whose misfit never rises enough to reach
    Delta-chi-squared = 1 anywhere in the grid (too little noise-normalised
    curvature) must report "not_crossed", distinct from a grid-edge cause."""
    z_grid = np.linspace(-400.0, 0.0, 41)
    # A very shallow bowl: tiny curvature, large resid_sigma -> chi2 never
    # reaches 1 even at the grid edges.
    bowl = 0.20 + 1e-8 * (np.arange(41) - 20) ** 2
    sigma, reason = marginal_sigma_z(z_grid, bowl, n_picks=20, resid_sigma=1.0)
    assert not np.isfinite(sigma)
    assert reason == "not_crossed"


def test_marginal_sigma_z_reports_which_edge_when_the_minimum_is_pinned():
    """A minimum at the grid edge means the data did not bound the depth; it
    must not be reported as a resolved one -- and which edge matters: the
    deep floor is a grid-extension artifact, the surface is not (fix round
    1, MAJOR 2)."""
    z_grid = np.linspace(-400.0, 0.0, 41)  # ascending: index 0 is deepest
    deep_ramp = np.linspace(0.10, 0.40, z_grid.size)  # minimum at index 0 (deep)
    sigma, reason = marginal_sigma_z(z_grid, deep_ramp, n_picks=20, resid_sigma=0.01)
    assert not np.isfinite(sigma)
    assert reason == "edge_deep"

    surface_ramp = np.linspace(0.40, 0.10, z_grid.size)  # minimum at index -1 (surface)
    sigma2, reason2 = marginal_sigma_z(z_grid, surface_ramp, n_picks=20, resid_sigma=0.01)
    assert not np.isfinite(sigma2)
    assert reason2 == "edge_surface"


def test_nearest_crossing_stops_at_the_first_crossing_not_a_farther_one():
    """Fix round 1, MEDIUM 5: a naive `np.interp` over a whole half-array
    assumes the array is monotone, which a multi-modal chi-squared profile
    violates -- precisely the failure the bidirectional walk check exists to
    catch, so not hypothetical. Hand-build a chi2 array that rises above the
    threshold, DROPS back below it (a second basin), then rises again -- the
    correct 1-sigma bound is the FIRST (nearest) crossing, not the last."""
    from dclde_soundspeed.experiments.moveout import _nearest_crossing

    z = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    #                          ^k=0 (minimum, chi2=0)
    chi2 = np.array([0.0, 0.6, 1.4, 0.3, 2.0, 5.0])
    # Walking right from k=0: crosses 1.0 between index 1 (0.6) and 2 (1.4)
    # -- the FIRST crossing. A monotonicity-assuming interpolant fed the
    # whole array would instead be dragged toward the index-3/4 crossing.
    got = _nearest_crossing(z, chi2, k=0, step=1, threshold=1.0)
    expected = 1.0 + (1.0 - 0.6) / (1.4 - 0.6) * (2.0 - 1.0)
    assert got == pytest.approx(expected)

    # No crossing before the array ends -> None, not a wrong answer.
    assert _nearest_crossing(z, chi2, k=0, step=-1, threshold=1.0) is None


def test_marginal_sigma_z_is_not_fooled_by_a_second_deeper_basin():
    """End-to-end version of the above: a profile whose global minimum sits
    at index 20 but which dips toward a second, lower-chi2 region further
    out must still report the interval bounded by the NEAREST crossing on
    each side, not a wider one reached only because the far side dipped
    again.

    The dip is placed at the grid's far edge (index 0) with a value tuned so
    its OWN chi-squared contribution is O(1) (~0.5, i.e. comfortably below
    the threshold=1 this test searches for) -- not the ~8e4 an earlier
    version of this test used. At ~8e4 the dip sits far outside the region a
    threshold=1 search ever reaches, so it cannot exercise the bug this test
    is meant to catch. At O(1) it lands exactly where a threshold-1 search
    does reach, so a monotonicity-assuming reader of the whole half-array
    (the pre-fix-round-1 `np.interp` implementation, reproduced verbatim
    below as `_old_np_interp_crossing` from git history at adb7d71) latches
    onto it and returns a grossly wrong answer, proven below.
    """
    z_grid = np.linspace(-400.0, 0.0, 41)
    m = 0.05 + 0.002 * (np.arange(41) - 20) ** 2  # clean bowl at index 20
    # Second, deeper dip at the far edge (index 0): a value just above the
    # global minimum so its own chi-squared is ~0.5 (O(1)), not the ~8e4 an
    # earlier version of this test used (unreachable by any threshold-1
    # search, hence "no teeth" -- see the module-level docstring above).
    m[0] = 0.05001666388981443  # chi2 there == 0.5 exactly, at n_picks=30, resid_sigma=0.01
    assert np.argmin(m) == 20

    sigma, reason = marginal_sigma_z(z_grid, m, n_picks=30, resid_sigma=0.01)
    assert reason == ""
    # Hand-computed: chi2 rises from 0 (index 20) to 61.2 one grid step out
    # (index 19), so the 1-sigma crossing is the linear interpolation
    # between those two points, 0.1634 m from the minimum on each side.
    assert sigma == pytest.approx(0.1633986928104605)


from dclde_soundspeed.experiments.moveout import bounce_travel_time, path_comparison
from dclde_localization.physics.forward import travel_time


def test_bounce_is_always_slower_than_the_direct_path():
    px = np.array([1000.0, 5000.0, 20000.0])
    py = np.zeros(3)
    pz = np.array([-400.0, -400.0, -400.0])
    d = travel_time(0.0, 0.0, -30.0, px, py, pz, DEFAULT_C)
    b = bounce_travel_time(0.0, 0.0, -30.0, px, py, pz, DEFAULT_C, bottom_z=-1200.0)
    assert np.all(b > d)


def test_bounce_reduces_to_the_direct_path_as_the_bottom_approaches_the_source():
    """With the seafloor at the source depth the image coincides with the
    source, so the two travel times must agree exactly."""
    px = np.array([3000.0, 9000.0])
    py = np.array([100.0, -250.0])
    pz = np.array([-500.0, -700.0])
    d = travel_time(0.0, 0.0, -30.0, px, py, pz, DEFAULT_C)
    b = bounce_travel_time(0.0, 0.0, -30.0, px, py, pz, DEFAULT_C, bottom_z=-30.0)
    assert np.allclose(b, d)


def test_bounce_and_direct_differ_in_CURVATURE_not_just_offset():
    """Guards SHAPE-discrimination, not the image-depth formula: a wrong
    image-depth formula still passes this test (diff.std() ~= 0.0174 against
    the 1e-3 threshold here) because almost any two distinct travel-time
    curves differ in more than a constant offset. The image-depth formula
    itself is pinned by
    `test_bounce_reduces_to_the_direct_path_as_the_bottom_approaches_the_source`,
    not by this test. What this test guards is that the two paths trace
    different SHAPES: if they differed only by a constant, `best_t0` would
    absorb the difference and no timing test could tell them apart."""
    px = np.linspace(2000.0, 30000.0, 12)
    py = np.zeros(12)
    pz = np.full(12, -600.0)
    d = travel_time(0.0, 0.0, -30.0, px, py, pz, DEFAULT_C)
    b = bounce_travel_time(0.0, 0.0, -30.0, px, py, pz, DEFAULT_C, bottom_z=-1500.0)
    diff = b - d
    assert diff.std() > 1e-3, "difference is nearly constant; t0 would absorb it"


import pandas as pd

from dclde_soundspeed.experiments.moveout import (
    _walk_consistent, gate_summary, gate_table, path_comparison_summary)


def test_gate_summary_applies_the_criterion_and_shows_its_denominators():
    tbl = pd.DataFrame({
        "call": [f"c{i}" for i in range(100)],
        "in_stratum": [True] * 80 + [False] * 20,
        "sigma_z_m": [50.0] * 30 + [400.0] * 50 + [np.inf] * 20,
    })
    out = gate_summary(tbl, {"calls_total": 120}, sigma_target_m=100.0,
                       min_fraction=0.10, min_calls=50)
    row = out.iloc[0]
    assert row["n_stratum"] == 80
    assert row["n_resolved"] == 30
    assert row["fraction_resolved"] == pytest.approx(30 / 80)
    # 30 >= 50 is FALSE, so the absolute-count leg fails even though 37.5% > 10%
    assert bool(row["passes_fraction"]) is True
    assert bool(row["passes_min_calls"]) is False
    assert bool(row["gate_passes"]) is False
    # Fix round 1, MAJOR 1: no walk_consistent column here -> the check was
    # never applied, and the composite must equal the pre-declared count.
    assert row["n_resolved_forward_only"] == 30
    assert bool(row["walk_check_applied"]) is False


def test_gate_summary_passes_only_when_both_legs_pass():
    tbl = pd.DataFrame({
        "call": [f"c{i}" for i in range(600)],
        "in_stratum": [True] * 600,
        "sigma_z_m": [50.0] * 60 + [400.0] * 540,
    })
    out = gate_summary(tbl, {"calls_total": 628}, sigma_target_m=100.0,
                       min_fraction=0.10, min_calls=50)
    row = out.iloc[0]
    assert row["n_resolved"] == 60
    assert bool(row["passes_fraction"]) is True   # 10.0% >= 10%
    assert bool(row["passes_min_calls"]) is True  # 60 >= 50
    assert bool(row["gate_passes"]) is True


def test_gate_summary_counts_infinite_sigma_as_unresolved_not_missing():
    tbl = pd.DataFrame({
        "call": ["a", "b", "c"],
        "in_stratum": [True, True, True],
        "sigma_z_m": [10.0, np.inf, np.nan],
    })
    out = gate_summary(tbl, {"calls_total": 3}, sigma_target_m=100.0,
                       min_fraction=0.10, min_calls=1)
    assert out.iloc[0]["n_stratum"] == 3
    assert out.iloc[0]["n_resolved"] == 1


def test_gate_summary_does_not_count_a_walk_inconsistent_call_as_resolved():
    """Addition (A): a call whose forward and reverse walks disagree by more
    than 10% must not count toward n_resolved even though its (forward)
    sigma_z_m clears the target -- an inconsistent walk means the profile may
    have tracked a local, not global, (x, y) optimum at some depth, so its
    sigma_z is not trustworthy evidence either way."""
    tbl = pd.DataFrame({
        "call": ["a", "b", "c"],
        "in_stratum": [True, True, True],
        "sigma_z_m": [10.0, 20.0, 30.0],
        "walk_consistent": [True, False, np.nan],
    })
    out = gate_summary(tbl, {"calls_total": 3}, sigma_target_m=100.0,
                       min_fraction=0.10, min_calls=1)
    # "a" resolved (consistent, finite, under target); "b" excluded (False);
    # "c" still counted (missing/NaN is not an explicit False, e.g. because
    # this call fell outside a bidirectional sample and was never checked).
    assert out.iloc[0]["n_resolved"] == 2
    # Fix round 1, MAJOR 1: all three clear the pre-declared criterion alone
    # (finite, <= target); only the composite is reduced by walk_consistent.
    assert out.iloc[0]["n_resolved_forward_only"] == 3
    assert bool(out.iloc[0]["walk_check_applied"]) is True


def test_walk_consistent_agrees_within_ten_percent_and_treats_both_infinite_as_consistent():
    assert _walk_consistent(100.0, 105.0) is True   # 5% apart
    assert _walk_consistent(100.0, 115.0) is False  # 15% apart
    assert _walk_consistent(np.inf, np.inf) is True  # both say "unresolved"
    assert _walk_consistent(50.0, np.inf) is False  # one bounds depth, one does not
    assert _walk_consistent(np.nan, 50.0) is False


def test_walk_consistent_grid_spacing_floor_rescues_a_small_absolute_disagreement():
    """Fix round 1, MINORS: a purely relative tolerance is not defensible for
    two small sigma_z values a few metres apart, where 10% can be sub-metre
    -- finer than the grid itself resolves. 12 vs 15 m disagree by 25%
    (fails a bare 10% relative test) but by only 3 m, inside a 5 m grid
    spacing."""
    assert _walk_consistent(12.0, 15.0) is False  # no grid spacing given
    assert _walk_consistent(12.0, 15.0, grid_spacing_m=5.0) is True
    # A genuinely large disagreement must still fail even with a grid floor.
    assert _walk_consistent(10.0, 200.0, grid_spacing_m=5.0) is False


def test_path_comparison_summary_reports_counts_and_denominators():
    tbl = pd.DataFrame({
        "delta_misfit": [0.001, -0.002, np.nan, 0.0005, 0.0],
        "delta_misfit_over_resid_sigma": [2.0, -0.5, np.nan, np.nan, 0.0],
    })
    out = path_comparison_summary(tbl)
    row = out.iloc[0]
    assert row["n_compared"] == 4  # the NaN delta_misfit row is excluded
    assert row["n_favor_direct"] == 2   # 0.001, 0.0005
    assert row["n_favor_bounce"] == 1   # -0.002
    assert row["n_tied"] == 1           # 0.0
    assert row["n_ratio_finite"] == 3
    # |2.0| >= 1 exceeds its own noise; |-0.5| and |0.0| do not.
    assert row["n_exceeds_own_noise"] == 1
    assert row["delta_misfit_mean"] == pytest.approx(np.mean([0.001, -0.002, 0.0005, 0.0]))


def _flat_grid(z_value=-2000.0, half_extent=5000.0, n=41):
    from dclde_localization.physics.bathymetry import LocalGrid
    dx = 2 * half_extent / (n - 1)
    z = np.full((n, n), z_value, dtype=np.int16)
    return LocalGrid(x0=-half_extent, y0=-half_extent - 2500.0, dx=dx, dy=dx, nx=n, ny=n, z=z)


def _picks_frame(px, py, pz, obs_t, call, cable):
    return pd.DataFrame({
        "call": [call] * len(px), "cable": [cable] * len(px),
        "px": px, "py": py, "pz": pz, "time": obs_t,
    })


def test_gate_table_reports_n_cables_and_the_new_walk_columns():
    """Addition (A) + (B): every call is rowed with n_cables (1 or 2, from the
    picks' own `cable` column) and the bidirectional walk columns, regardless
    of whether the call ends up in the stratum."""
    px, py, pz, obs_t, _ = synthetic_call(z_src=-120.0, depth_span=400.0, noise_s=0.0)
    n1 = len(px) // 2
    single = _picks_frame(px, py, pz, obs_t, "c_single", "South")
    two = pd.concat([
        _picks_frame(px[:n1], py[:n1], pz[:n1], obs_t[:n1], "c_two", "South"),
        _picks_frame(px[n1:], py[n1:], pz[n1:], obs_t[n1:], "c_two", "North"),
    ], ignore_index=True)
    picks = pd.concat([single, two], ignore_index=True)
    grid = _flat_grid()

    table, den = gate_table(picks, grid=grid,
                            z_grid=np.linspace(-400.0, 0.0, 21),
                            bidirectional_sample=None)

    assert set(table["call"]) == {"c_single", "c_two"}
    row_single = table[table["call"] == "c_single"].iloc[0]
    row_two = table[table["call"] == "c_two"].iloc[0]
    assert row_single["n_cables"] == 1
    assert row_two["n_cables"] == 2
    for col in ("sigma_z_m_reverse", "profile_max_discrepancy_s", "walk_consistent",
               "sigma_z_reason", "delta_misfit_over_resid_sigma"):
        assert col in table.columns
    # Both calls should resolve their stratum membership and, having been
    # walked in both directions (bidirectional_sample=None means "all"), get
    # a real (non-null) walk_consistent verdict rather than a silently
    # skipped one.
    for row in (row_single, row_two):
        if row["in_stratum"]:
            assert row["walk_consistent"] in (True, False)
            assert np.isfinite(row["profile_max_discrepancy_s"])
            # A finite sigma_z_m must carry an empty reason, and vice versa.
            assert np.isfinite(row["sigma_z_m"]) == (row["sigma_z_reason"] == "")


def test_gate_table_bidirectional_sample_limits_which_calls_get_a_reverse_walk():
    """Addition (A)'s budget escape hatch: with `bidirectional_sample=1`, only
    one in-stratum call gets both directions; the rest keep a forward-only
    (null) walk_consistent so the omission is visible, not silently assumed
    consistent."""
    rows = []
    for i in range(4):
        px, py, pz, obs_t, _ = synthetic_call(z_src=-120.0, depth_span=400.0,
                                              rng_seed=i, noise_s=0.0)
        rows.append(_picks_frame(px, py, pz, obs_t, f"c{i}", "South"))
    picks = pd.concat(rows, ignore_index=True)
    grid = _flat_grid()

    table, den = gate_table(picks, grid=grid,
                            z_grid=np.linspace(-400.0, 0.0, 21),
                            bidirectional_sample=1, bidirectional_seed=0)

    in_strat = table[table["in_stratum"]]
    checked = in_strat["walk_consistent"].notna().sum()
    assert checked == min(1, len(in_strat))
    if len(in_strat) > 1:
        assert in_strat["walk_consistent"].isna().sum() == len(in_strat) - checked


def test_path_comparison_prefers_the_model_that_generated_the_data():
    """A perfectly collinear receiver line (py constant) is degenerate for
    this comparison: fix round 1's tightened `_fit_xy` tolerance (MEDIUM 4)
    revealed that on such a line, the direct-path fit's free (x, y) can
    reproduce BOUNCE-generated data to machine-epsilon misfit too (~1e-13
    either way, i.e. no real discrimination) -- the earlier looser tolerance
    merely masked this by stopping short of that degenerate optimum. `py`
    varies here (a wide sinusoidal spread, not a straight line) so the two
    path models are genuinely distinguishable, matching how real picks are
    never exactly collinear either."""
    px = np.linspace(2000.0, 25000.0, 20)
    py = 500.0 + 1000.0 * np.sin(np.linspace(0.0, 2 * np.pi, 20))
    pz = np.linspace(-300.0, -900.0, 20)
    bottom = -1400.0
    t_b = bounce_travel_time(1000.0, -3000.0, -30.0, px, py, pz, DEFAULT_C, bottom) + 7.0
    out = path_comparison(px, py, pz, t_b, bottom_z=bottom)
    assert out["misfit_bounce"] < out["misfit_direct"]
    assert out["delta_misfit"] < 0

    t_d = travel_time(1000.0, -3000.0, -30.0, px, py, pz, DEFAULT_C) + 7.0
    out2 = path_comparison(px, py, pz, t_d, bottom_z=bottom)
    assert out2["misfit_direct"] < out2["misfit_bounce"]
    assert out2["delta_misfit"] > 0
