import numpy as np
import pandas as pd
import pytest

from dclde_soundspeed.experiments.propagation import (
    geometric_tl, predicted_tl_table)
from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile

SWEEP = [
    {"c": 1550.0, "rho": 1.5, "alpha": 0.05},
    {"c": 1900.0, "rho": 2.1, "alpha": 0.40},
]
GEOMS = [
    (30.0, 500.0, 10_000.0, 1000.0, 1000.0),
    (30.0, 1000.0, 30_000.0, 1400.0, 1200.0),
]


def isovelocity(depth_m=1500.0, c=1480.0, n=64):
    z = np.linspace(0.0, depth_m, n)
    return SoundSpeedProfile(z_down=z, c=np.full(n, c), source="test isovelocity")


def test_geometric_tl_is_cylindrical_spreading_plus_absorption():
    # 10 log10(r) + alpha*r/1000, with alpha in dB/km.
    assert geometric_tl(1000.0, alpha_db_per_km=0.0) == pytest.approx(30.0)
    assert geometric_tl(10_000.0, alpha_db_per_km=0.0) == pytest.approx(40.0)
    assert geometric_tl(10_000.0, alpha_db_per_km=1.0) == pytest.approx(50.0)


def test_geometric_tl_grows_monotonically():
    r = np.array([1e3, 5e3, 20e3, 40e3])
    tl = geometric_tl(r, alpha_db_per_km=0.1)
    assert np.all(np.diff(tl) > 0)


def test_table_has_one_row_per_geometry_and_seabed():
    df = predicted_tl_table(isovelocity(), SWEEP, 20.0, GEOMS, n_steps=6)
    assert len(df) == len(GEOMS) * len(SWEEP)


def test_table_carries_the_columns_the_findings_quote():
    df = predicted_tl_table(isovelocity(), SWEEP, 20.0, GEOMS, n_steps=6)
    for col in ("source_z_m", "receiver_z_m", "range_m", "seabed_c",
                "seabed_alpha", "n_modes_start", "n_modes_end", "n_stripped",
                "n_unresolved_steps", "excitation_rms",
                "tl_modal_coherent_db", "tl_modal_incoherent_db",
                "tl_geometric_db", "path_ok"):
        assert col in df.columns, col


def test_a_faster_seabed_traps_more_modes():
    df = predicted_tl_table(isovelocity(), SWEEP, 20.0, GEOMS[:1], n_steps=6)
    slow = df[df.seabed_c == 1550.0].n_modes_start.iloc[0]
    fast = df[df.seabed_c == 1900.0].n_modes_start.iloc[0]
    assert fast >= slow


def test_stripped_modes_are_reported_not_hidden():
    """The second geometry shoals from 1400 m to 1200 m. Whatever the count
    does, the column must exist and be internally consistent."""
    df = predicted_tl_table(isovelocity(), SWEEP, 20.0, GEOMS, n_steps=8)
    assert (df.n_modes_end == df.n_modes_start - df.n_stripped).all()


def test_excitation_rms_falls_as_the_source_moves_toward_the_surface():
    """phi is pinned to zero at the surface, so a shallower source drives the
    modes less. This is the term that decides whether a whale at -30 m matters."""
    deep = predicted_tl_table(isovelocity(), SWEEP[:1], 20.0,
                              [(300.0, 500.0, 10_000.0, 1000.0, 1000.0)], n_steps=6)
    shallow = predicted_tl_table(isovelocity(), SWEEP[:1], 20.0,
                                 [(3.0, 500.0, 10_000.0, 1000.0, 1000.0)], n_steps=6)
    assert shallow.excitation_rms.iloc[0] < deep.excitation_rms.iloc[0]


from dclde_soundspeed.experiments.propagation import analysis_frame


def test_analysis_frame_reports_every_denominator_it_drops_on():
    """The denominators dict must account for every call: eligible + below-min
    + unsolved + flipped + path-failed + geometry-error == total. A silently
    dropped call is how a rate acquires the wrong denominator."""
    from dclde_localization.physics.geometry import load_cables
    from dclde_localization.physics.bathymetry import decode_gmt_grid, reproject_to_local
    from dclde_localization.physics.geometry import FrameTransform
    from dclde_localization.physics.sound_speed_models import ConstantC
    from dclde_localization.physics.picks import attach_xyz, load_picks

    cables = load_cables("fixtures")
    picks = attach_xyz(load_picks(), cables)
    # A small, deterministic subset -- this test guards the bookkeeping, not
    # the science, and the full solve is minutes long.
    keep = sorted(picks["call"].unique())[:6]
    picks = picks[picks["call"].isin(keep)].copy()

    grid = reproject_to_local(decode_gmt_grid("fixtures/gmrt_bathy.nc"),
                              FrameTransform.from_cable(cables["South"]))
    frame, den = analysis_frame(picks, cables, grid, ConstantC(1480.0), source_z=-30.0)

    assert den["calls_total"] == len(keep)
    accounted = (den["calls_eligible"] + den["calls_below_min_picks"]
                 + den["calls_unsolved"] + den["calls_flipped"]
                 + den["calls_path_failed"] + den["calls_geometry_error"])
    assert accounted == den["calls_total"], (den, accounted)
    assert den["picks_eligible"] == len(frame)


def test_geometry_error_is_counted_not_fatal(monkeypatch):
    """A `DegenerateGeometryError` from das_axial_response's own
    degenerate-geometry refusal (e.g. a solved source that coincides
    horizontally with one of its own channels) must be caught per call,
    counted in calls_geometry_error, and must not abort the whole run --
    it must not lose calls already processed before or after it."""
    import dclde_soundspeed.experiments.propagation as prop_mod
    from dclde_localization.physics.geometry import load_cables
    from dclde_localization.physics.bathymetry import decode_gmt_grid, reproject_to_local
    from dclde_localization.physics.geometry import FrameTransform
    from dclde_localization.physics.sound_speed_models import ConstantC
    from dclde_soundspeed.soundspeed.tl import DegenerateGeometryError
    from dclde_localization.physics.picks import attach_xyz, load_picks

    cables = load_cables("fixtures")
    picks = attach_xyz(load_picks(), cables)
    keep = sorted(picks["call"].unique())[:6]
    picks = picks[picks["call"].isin(keep)].copy()
    grid = reproject_to_local(decode_gmt_grid("fixtures/gmrt_bathy.nc"),
                              FrameTransform.from_cable(cables["South"]))

    calls_seen = []
    real_axial = prop_mod._axial_response_column

    def flaky_axial(sub, cables_arg, source_xy):
        calls_seen.append(sub["call"].iloc[0] if "call" in sub.columns else None)
        if len(calls_seen) == 2:
            # Caught by TYPE, not message text -- see propagation.py's
            # comment at the catch site and the reworded-message test below.
            raise DegenerateGeometryError("zero-length source-to-channel "
                                          "vector: synthetic degeneracy")
        return real_axial(sub, cables_arg, source_xy)

    monkeypatch.setattr(prop_mod, "_axial_response_column", flaky_axial)
    frame, den = analysis_frame(picks, cables, grid, ConstantC(1480.0), source_z=-30.0)

    assert den["calls_geometry_error"] == 1
    accounted = (den["calls_eligible"] + den["calls_below_min_picks"]
                 + den["calls_unsolved"] + den["calls_flipped"]
                 + den["calls_path_failed"] + den["calls_geometry_error"])
    assert accounted == den["calls_total"], (den, accounted)
    # State directly, not just via the identity, that processing continued
    # past the failing call: a call seen strictly after the failing one must
    # still have made it into the frame.
    failing_call = calls_seen[1]
    later_calls_seen = calls_seen[2:]
    assert later_calls_seen, "test setup needs >= 3 calls processed to prove continuation"
    frame_calls = set(frame["call"].unique())
    assert frame_calls, "no call reached the frame at all"
    assert frame_calls - {failing_call}, "only the failing call's slot is non-empty"
    assert any(c in frame_calls for c in later_calls_seen), (
        "no call processed AFTER the failing one made it into the frame -- "
        "continuation was not actually exercised")


def test_geometry_error_is_counted_regardless_of_message_wording(monkeypatch):
    """The regression this whole change prevents: catching must be by TYPE,
    not by matching `das_axial_response`'s current message text. A
    `DegenerateGeometryError` with an entirely different message must still
    be counted -- if this ever goes back to string-matching, this test is
    what would catch it (this repo has already shipped that exact fragility
    once; see commit babbf50)."""
    import dclde_soundspeed.experiments.propagation as prop_mod
    from dclde_localization.physics.geometry import load_cables
    from dclde_localization.physics.bathymetry import decode_gmt_grid, reproject_to_local
    from dclde_localization.physics.geometry import FrameTransform
    from dclde_localization.physics.sound_speed_models import ConstantC
    from dclde_soundspeed.soundspeed.tl import DegenerateGeometryError
    from dclde_localization.physics.picks import attach_xyz, load_picks

    cables = load_cables("fixtures")
    picks = attach_xyz(load_picks(), cables)
    keep = sorted(picks["call"].unique())[:6]
    picks = picks[picks["call"].isin(keep)].copy()
    grid = reproject_to_local(decode_gmt_grid("fixtures/gmrt_bathy.nc"),
                              FrameTransform.from_cable(cables["South"]))

    def broken_axial(sub, cables_arg, source_xy):
        raise DegenerateGeometryError("some entirely different wording, "
                                      "nothing like the original message")

    monkeypatch.setattr(prop_mod, "_axial_response_column", broken_axial)
    frame, den = analysis_frame(picks, cables, grid, ConstantC(1480.0), source_z=-30.0)

    assert den["calls_geometry_error"] > 0
    accounted = (den["calls_eligible"] + den["calls_below_min_picks"]
                 + den["calls_unsolved"] + den["calls_flipped"]
                 + den["calls_path_failed"] + den["calls_geometry_error"])
    assert accounted == den["calls_total"], (den, accounted)


def test_non_geometry_valueerror_propagates(monkeypatch):
    """A plain `ValueError` -- NOT a `DegenerateGeometryError` -- must
    propagate and crash loudly rather than being silently refiled as
    calls_geometry_error. E.g. `_axial_response_column`'s own RangeIndex
    invariant check, or a ValueError from corrupted upstream data."""
    import dclde_soundspeed.experiments.propagation as prop_mod
    from dclde_localization.physics.geometry import load_cables
    from dclde_localization.physics.bathymetry import decode_gmt_grid, reproject_to_local
    from dclde_localization.physics.geometry import FrameTransform
    from dclde_localization.physics.sound_speed_models import ConstantC
    from dclde_localization.physics.picks import attach_xyz, load_picks

    cables = load_cables("fixtures")
    picks = attach_xyz(load_picks(), cables)
    keep = sorted(picks["call"].unique())[:6]
    picks = picks[picks["call"].isin(keep)].copy()
    grid = reproject_to_local(decode_gmt_grid("fixtures/gmrt_bathy.nc"),
                              FrameTransform.from_cable(cables["South"]))

    def broken_axial(sub, cables_arg, source_xy):
        raise ValueError("could not convert string to float: 'NaN-ish garbage'")

    monkeypatch.setattr(prop_mod, "_axial_response_column", broken_axial)
    with pytest.raises(ValueError, match="NaN-ish garbage"):
        analysis_frame(picks, cables, grid, ConstantC(1480.0), source_z=-30.0)


def test_range_is_source_to_channel_not_along_cable_distance():
    """`dist` is distance along the fibre. Using it as range would be wrong by
    tens of kilometres, and it is the single easiest mistake to make here."""
    from dclde_soundspeed.experiments.propagation import _range_to_channels

    sx, sy = 0.0, 0.0
    px = np.array([3000.0, 0.0])
    py = np.array([0.0, 4000.0])
    pz = np.array([-500.0, -500.0])
    got = _range_to_channels(sx, sy, -30.0, px, py, pz)
    assert got[0] == pytest.approx(np.hypot(3000.0, 470.0))
    assert got[1] == pytest.approx(np.hypot(4000.0, 470.0))


def test_axial_response_assigns_each_pick_its_own_cable_tangent():
    """Regression test for the per-pick tangent mapping.

    `test_axial_response_column_is_bounded_and_varies` cannot catch a swap:
    exchanging two picks' tangents within a call still yields valid,
    non-constant cos^2 values, so both its assertions still pass (this is
    exactly the class of bug Task 5's mutants exploited: 31/31 tests passed
    with a silently mis-paired result). This test instead hand-computes the
    expected axial response for two picks on the SAME cable with genuinely
    different tangents, using an independent re-implementation of the
    finite-difference tangent formula (not a call into `_cable_tangents` or
    `_axial_response_column`), and asserts each pick's result matches its
    OWN channel's tangent exactly.
    """
    import numpy as np
    import pandas as pd
    from dclde_soundspeed.experiments.propagation import _axial_response_column

    class CurvedCable:
        """A synthetic cable whose tangent direction genuinely varies along
        its length, so a tangent borrowed from the wrong pick changes the
        answer by a large, detectable amount."""
        spacing_m = 1.0

        def chan_to_xyz(self, d):
            d = np.asarray(d, dtype=float)
            x = d
            y = 0.0005 * d ** 2
            z = np.full_like(d, -1000.0)
            return x, y, z

    cable = CurvedCable()
    cables = {"North": cable}
    dists = np.array([500.0, 1500.0])
    px, py, _ = cable.chan_to_xyz(dists)
    sub = pd.DataFrame({"call": ["c1", "c1"], "cable": ["North", "North"],
                        "dist": dists, "px": px, "py": py})

    def expected_axial(dist, cx, cy, sx=0.0, sy=0.0, step=1.0):
        x0, y0, _ = cable.chan_to_xyz(dist - step)
        x1, y1, _ = cable.chan_to_xyz(dist + step)
        tx, ty = x1 - x0, y1 - y0
        tn = np.hypot(tx, ty)
        tx, ty = tx / tn, ty / tn
        dx, dy = cx - sx, cy - sy
        dn = np.hypot(dx, dy)
        cos_theta = (dx * tx + dy * ty) / dn
        return min(1.0, cos_theta ** 2)

    expected = [expected_axial(dists[0], px[0], py[0]),
               expected_axial(dists[1], px[1], py[1])]
    # The two picks must land on genuinely different values, or a swap
    # between them would be invisible even to this exact check.
    assert abs(expected[0] - expected[1]) > 0.05

    got = _axial_response_column(sub, cables, (0.0, 0.0))
    np.testing.assert_allclose(got, expected, rtol=1e-9)


def test_axial_response_assigns_each_pick_its_own_cable_tangent_across_cables():
    """Sibling to `test_axial_response_assigns_each_pick_its_own_cable_tangent`,
    covering the case that test does NOT: picks on TWO DIFFERENT cables in
    one call (166 of the 628 real calls have exactly this shape, and Task 11
    runs all of them). The original risk here was cross-cable, not
    within-cable: does `groupby("cable")` correctly isolate and reassemble
    North's and South's picks, or can a North pick end up scored against
    South's tangent (or vice versa)? Two different curved cable geometries
    make such a cross-assignment produce a visibly, hand-verifiably wrong
    number rather than a coincidentally similar one.
    """
    import numpy as np
    import pandas as pd
    from dclde_soundspeed.experiments.propagation import _axial_response_column

    class NorthCable:
        spacing_m = 1.0

        def chan_to_xyz(self, d):
            d = np.asarray(d, dtype=float)
            return d, 0.0005 * d ** 2, np.full_like(d, -1000.0)

    class SouthCable:
        spacing_m = 1.0

        def chan_to_xyz(self, d):
            d = np.asarray(d, dtype=float)
            return 2000.0 + 0.0003 * d ** 2, d, np.full_like(d, -1000.0)

    north, south = NorthCable(), SouthCable()
    cables = {"North": north, "South": south}

    n_dist, s_dist = 500.0, 600.0
    nx, ny, _ = north.chan_to_xyz(n_dist)
    sx_, sy_, _ = south.chan_to_xyz(s_dist)
    sub = pd.DataFrame({
        "cable": ["North", "South"],
        "dist": [n_dist, s_dist],
        "px": [float(nx), float(sx_)],
        "py": [float(ny), float(sy_)],
    })

    def expected_axial(cable_obj, dist, cx, cy, sx=0.0, sy=0.0, step=1.0):
        x0, y0, _ = cable_obj.chan_to_xyz(dist - step)
        x1, y1, _ = cable_obj.chan_to_xyz(dist + step)
        tx, ty = x1 - x0, y1 - y0
        tn = np.hypot(tx, ty)
        tx, ty = tx / tn, ty / tn
        dx, dy = cx - sx, cy - sy
        dn = np.hypot(dx, dy)
        cos_theta = (dx * tx + dy * ty) / dn
        return min(1.0, cos_theta ** 2)

    expected = [expected_axial(north, n_dist, float(nx), float(ny)),
               expected_axial(south, s_dist, float(sx_), float(sy_))]
    # The two picks must land on genuinely different values, and each must
    # be far from what its OWN pick would get from the OTHER cable's
    # tangent, or a cross-cable swap would be invisible to this check.
    cross_swapped = [expected_axial(south, s_dist, float(nx), float(ny)),
                     expected_axial(north, n_dist, float(sx_), float(sy_))]
    assert abs(expected[0] - cross_swapped[0]) > 0.1
    assert abs(expected[1] - cross_swapped[1]) > 0.1

    got = _axial_response_column(sub, cables, (0.0, 0.0))
    np.testing.assert_allclose(got, expected, rtol=1e-9)


def test_axial_response_column_is_bounded_and_varies():
    from dclde_localization.physics.geometry import load_cables
    from dclde_localization.physics.bathymetry import decode_gmt_grid, reproject_to_local
    from dclde_localization.physics.geometry import FrameTransform
    from dclde_localization.physics.sound_speed_models import ConstantC
    from dclde_localization.physics.picks import attach_xyz, load_picks

    cables = load_cables("fixtures")
    picks = attach_xyz(load_picks(), cables)
    keep = sorted(picks["call"].unique())[:6]
    picks = picks[picks["call"].isin(keep)].copy()
    grid = reproject_to_local(decode_gmt_grid("fixtures/gmrt_bathy.nc"),
                              FrameTransform.from_cable(cables["South"]))
    frame, _ = analysis_frame(picks, cables, grid, ConstantC(1480.0), source_z=-30.0)
    if frame.empty:
        pytest.skip("no eligible calls in this subset")
    assert frame.axial_response.between(0.0, 1.0).all()
    assert frame.axial_response.std() > 0


from dclde_soundspeed.experiments.propagation import max_range_table


def _frame(rows):
    return pd.DataFrame(rows)


def test_max_range_is_per_call_and_picks_the_furthest_pick():
    frame = _frame([
        {"call": "a", "range_m": 5000.0, "snr": 12.0, "receiver_z_m": 300.0},
        {"call": "a", "range_m": 22_000.0, "snr": 7.0, "receiver_z_m": 800.0},
        {"call": "b", "range_m": 9000.0, "snr": 15.0, "receiver_z_m": 400.0},
    ])
    out = max_range_table(frame, {"calls_total": 3})
    assert len(out) == 2
    a = out[out.call == "a"].iloc[0]
    assert a.max_range_m == pytest.approx(22_000.0)
    assert a.min_range_m == pytest.approx(5000.0)
    assert a.max_range_snr == pytest.approx(7.0)
    assert a.receiver_z_at_max_m == pytest.approx(800.0)
    assert a.n_picks == 2


def test_denominators_travel_with_the_table_so_a_rate_cannot_lose_them():
    frame = _frame([{"call": "a", "range_m": 5000.0, "snr": 12.0,
                     "receiver_z_m": 300.0}])
    den = {"calls_total": 628, "calls_eligible": 1}
    out = max_range_table(frame, den)
    assert out.attrs["denominators"] == den


def test_an_empty_frame_gives_an_empty_table_not_a_crash():
    out = max_range_table(_frame([]), {"calls_total": 0})
    assert out.empty
    assert list(out.columns)[:3] == ["call", "n_picks", "max_range_m"]


from dclde_soundspeed.experiments.propagation import censored_fit, decay_fit_table


def _synthetic(n_calls=40, per_call=30, slope=-8.0, sigma=2.0, seed=0):
    """Known slope, known threshold. log10(range) is the regressor."""
    rng = np.random.default_rng(seed)
    call_index, x, y = [], [], []
    for c in range(n_calls):
        intercept = rng.normal(60.0, 5.0)          # per-call source level
        lr = rng.uniform(np.log10(3e3), np.log10(6e4), per_call)
        obs = intercept + slope * lr + rng.normal(0.0, sigma, per_call)
        call_index += [c] * per_call
        x.append(lr)
        y.append(obs)
    return np.array(call_index), np.concatenate(x), np.concatenate(y)


def test_censored_fit_recovers_a_known_slope_when_nothing_is_censored():
    ci, x, y = _synthetic()
    X = x[:, None]
    out = censored_fit(y, X, ci, threshold=-1e9)   # threshold below all data
    assert out["converged"]
    assert out["n_censored"] == 0
    assert out["beta"][0] == pytest.approx(-8.0, abs=0.4)


def test_the_naive_uncensored_fit_is_biased_and_the_censored_one_is_not():
    """The censoring correction must have teeth -- ON A GENUINELY CENSORED
    SAMPLE, which is what this synthetic setup builds, NOT what the real DAS
    data is.

    WHAT THIS TEST ESTABLISHES AND WHAT IT DOES NOT (see
    `CAVEAT_CENSORED_VS_TRUNCATED` for the full statement). The `corrected`
    fit here is built over the FULL sample -- `x[:, None]` and `ci` for every
    row, `np.where(seen, y, thresh)` substituting the bound for sub-threshold
    values, `censored_mask=~seen` -- so the estimator is TOLD exactly how
    many sub-threshold observations exist and at what x. That is a genuinely
    left-CENSORED sample by construction, and this test correctly shows the
    censored likelihood removes the naive fit's bias on ONE. It does NOT
    establish that the same correction removes the bias on a left-TRUNCATED
    sample (the real DAS picks: sub-threshold rows are simply absent, and the
    estimator is never told they existed at all) -- that is a different
    problem this module's Tobit likelihood is not built to solve, and no
    test in this file claims otherwise.
    """
    ci, x, y = _synthetic()
    # Censoring the lowest 35% (as an earlier draft of this test did) turned
    # out not to bite hard enough with this seed/noise level to be a
    # dependable stress test -- the naive slope came out only mildly shallow.
    # Half the sample censored is still a realistic detection-threshold
    # scenario and gives the naive fit a bias with real teeth.
    thresh = float(np.quantile(y, 0.5))            # censor the lowest 50%
    seen = y >= thresh

    naive = censored_fit(y[seen], x[seen, None], ci[seen], threshold=-1e9)
    corrected = censored_fit(np.where(seen, y, thresh), x[:, None], ci,
                             threshold=thresh, censored_mask=~seen)

    assert corrected["converged"]
    # Truncation flattens the apparent decay: the naive slope is too shallow.
    assert naive["beta"][0] > -8.0 + 1.0, (
        f"naive slope {naive['beta'][0]:.2f} was not detectably biased; "
        "this test has no teeth unless it is")
    assert corrected["beta"][0] == pytest.approx(-8.0, abs=0.7)
    assert abs(corrected["beta"][0] + 8.0) < abs(naive["beta"][0] + 8.0)


def test_per_call_intercepts_absorb_arbitrary_per_call_level_offsets():
    """Absolute level is unusable (denoised product, unknown gain), so the fit
    must be invariant to shifting each call by an arbitrary constant."""
    ci, x, y = _synthetic()
    rng = np.random.default_rng(7)
    offsets = rng.normal(0.0, 30.0, ci.max() + 1)
    a = censored_fit(y, x[:, None], ci, threshold=-1e9)
    b = censored_fit(y + offsets[ci], x[:, None], ci, threshold=-1e9)
    assert b["beta"][0] == pytest.approx(a["beta"][0], abs=1e-4)


def test_decay_fit_reports_range_and_depth_separately():
    frame = pd.DataFrame({
        "call": np.repeat([f"c{i}" for i in range(12)], 20),
        "range_m": np.tile(np.linspace(4e3, 5e4, 20), 12),
        "receiver_z_m": np.tile(np.linspace(100.0, 1400.0, 20), 12),
        "snr": np.tile(np.linspace(20.0, 8.0, 20), 12),
        "axial_response": 1.0,
    })
    out = decay_fit_table(frame, threshold=0.0, n_bootstrap=20, seed=0)
    assert {"term", "estimate", "ci_lo", "ci_hi"} <= set(out.columns)
    assert {"log10_range", "receiver_depth_km"} <= set(out.term)


def test_bootstrap_resamples_whole_calls_not_individual_picks():
    """Picks within a call are correlated, so resampling picks would understate
    the interval. Two runs with the same seed must agree exactly, and the
    interval must widen when there are fewer calls to draw from."""
    def frame_with(n_calls):
        # An earlier draft of this frame gave every call an IDENTICAL,
        # noise-free (range_m, receiver_z_m) grid with the per-call SNR
        # offset absorbed entirely by the free per-call intercept. That left
        # zero genuine residual variance, so bootstrap draws differed only by
        # optimizer-level numerical noise and the interval did not reliably
        # widen with fewer calls -- a test with no teeth. Real picks have
        # noise and real calls do not share an identical depth-vs-range
        # track, so this version adds per-pick SNR noise and per-call jitter
        # on receiver_z_m (decorrelating it a little from range_m, as on a
        # real cable) so the bootstrap has real sampling variability to
        # detect shrinking with more calls.
        rng = np.random.default_rng(123)
        base_range = np.linspace(4e3, 5e4, 20)
        base_depth = np.linspace(100.0, 1400.0, 20)
        rows = []
        for c in range(n_calls):
            depth = base_depth + rng.normal(0.0, 150.0, 20)
            snr = (np.linspace(20.0, 8.0, 20)
                   + (c / max(n_calls - 1, 1)) * 5.0
                   + rng.normal(0.0, 1.5, 20))
            rows.append(pd.DataFrame({
                "call": f"c{c}", "range_m": base_range,
                "receiver_z_m": depth, "snr": snr, "axial_response": 1.0,
            }))
        return pd.concat(rows, ignore_index=True)

    a = decay_fit_table(frame_with(30), threshold=0.0, n_bootstrap=40, seed=1)
    b = decay_fit_table(frame_with(30), threshold=0.0, n_bootstrap=40, seed=1)
    pd.testing.assert_frame_equal(a, b)

    few = decay_fit_table(frame_with(6), threshold=0.0, n_bootstrap=40, seed=1)
    wide = (few.ci_hi - few.ci_lo).mean()
    narrow = (a.ci_hi - a.ci_lo).mean()
    assert wide > narrow


def test_zero_snr_rule_rejects_an_unknown_value():
    frame = pd.DataFrame({
        "call": np.repeat([f"c{i}" for i in range(12)], 20),
        "range_m": np.tile(np.linspace(4e3, 5e4, 20), 12),
        "receiver_z_m": np.tile(np.linspace(100.0, 1400.0, 20), 12),
        "snr": np.tile(np.linspace(20.0, 8.0, 20), 12),
        "axial_response": 1.0,
    })
    with pytest.raises(ValueError, match="zero_snr_rule"):
        decay_fit_table(frame, threshold=0.0, n_bootstrap=5, seed=0,
                        zero_snr_rule="bogus")


def test_zero_snr_rule_dropped_actually_removes_the_zero_picks():
    """'dropped' must change n_picks by exactly the number of exact-zero SNR
    rows relative to 'censored', which keeps them -- otherwise the two rules
    would silently do the same thing and the both-ways report would have no
    teeth."""
    calls = [f"c{i}" for i in range(12)]
    rows = []
    for i, c in enumerate(calls):
        snr = np.linspace(20.0, 8.0, 20)
        if i == 0:
            snr = snr.copy()
            snr[0] = 0.0
            snr[1] = 0.0
        rows.append(pd.DataFrame({
            "call": c, "range_m": np.linspace(4e3, 5e4, 20),
            "receiver_z_m": np.linspace(100.0, 1400.0, 20),
            "snr": snr, "axial_response": 1.0,
        }))
    frame = pd.concat(rows, ignore_index=True)

    censored = decay_fit_table(frame, threshold=0.0, n_bootstrap=5, seed=0,
                               zero_snr_rule="censored")
    dropped = decay_fit_table(frame, threshold=0.0, n_bootstrap=5, seed=0,
                              zero_snr_rule="dropped")
    assert (censored.n_picks - dropped.n_picks == 2).all()
    assert set(censored.zero_snr_rule) == {"censored"}
    assert set(dropped.zero_snr_rule) == {"dropped"}


def test_warm_start_matches_dense_ols_on_uncensored_data():
    """The Frisch-Waugh-Lovell within-call-demeaning warm start must give the
    same beta as the previous dense one-hot lstsq warm start, on the same
    uncensored data -- this is claimed as an exact identity, not merely a
    fast approximation, so it is checked directly against the dense
    reference computation, not just against convergence of the final fit."""
    ci, x, y = _synthetic(n_calls=25, per_call=15, seed=3)
    X = x[:, None]
    codes, uniq = pd.factorize(ci)
    n_calls = len(uniq)
    n_beta = X.shape[1]

    # The old, dense reference computation.
    design = np.column_stack([X, np.eye(n_calls)[codes]])
    dense_coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    dense_beta = dense_coef[:n_beta]

    # The new, demeaned computation (mirrors censored_fit's warm start with
    # no censoring, i.e. obs == everything).
    counts = np.bincount(codes, minlength=n_calls).astype(float)
    call_mean_y = np.bincount(codes, weights=y, minlength=n_calls) / counts
    call_mean_X = np.zeros((n_calls, n_beta))
    for j in range(n_beta):
        call_mean_X[:, j] = np.bincount(codes, weights=X[:, j], minlength=n_calls) / counts
    y_dm = y - call_mean_y[codes]
    X_dm = X - call_mean_X[codes]
    fast_beta, *_ = np.linalg.lstsq(X_dm, y_dm, rcond=None)

    np.testing.assert_allclose(fast_beta, dense_beta, atol=1e-8, rtol=1e-6)


def test_analytic_gradient_matches_finite_differences():
    """The analytic gradient (`neg_loglik_grad`, passed to L-BFGS-B as `jac`)
    must agree with a finite-difference approximation at both a fully
    uncensored point and a partially censored one -- a wrong analytic
    gradient converges confidently to the wrong place, which is worse than
    the finite-difference-only optimizer's honest non-convergence it
    replaced. Checked via `scipy.optimize.minimize`'s own spy hook rather
    than reaching into `censored_fit`'s closures directly, since those are
    not part of its public interface."""
    from scipy import optimize as scipy_optimize
    import dclde_soundspeed.experiments.propagation as prop_mod

    ci, x, y = _synthetic(n_calls=15, per_call=10, seed=11)
    X = x[:, None]

    captured = {}
    real_minimize = scipy_optimize.minimize

    def spy_minimize(fun, x0, jac=None, **kw):
        captured["fun"], captured["jac"], captured["x0"] = fun, jac, x0
        return real_minimize(fun, x0, jac=jac, **kw)

    rng = np.random.default_rng(1)
    try:
        scipy_optimize.minimize = spy_minimize

        # Fully uncensored.
        prop_mod.censored_fit(y, X, ci, threshold=-1e9)
        fun, jac, x0 = captured["fun"], captured["jac"], captured["x0"]
        err_uncensored = scipy_optimize.check_grad(fun, jac, x0)
        assert err_uncensored < 1e-3, err_uncensored
        for _ in range(3):
            pt = x0 + rng.normal(0.0, 0.5, size=x0.shape)
            assert scipy_optimize.check_grad(fun, jac, pt) < 1e-3

        # Partially censored (~50%).
        thresh = float(np.quantile(y, 0.5))
        mask = y <= thresh
        prop_mod.censored_fit(y, X, ci, threshold=thresh, censored_mask=mask)
        fun, jac, x0 = captured["fun"], captured["jac"], captured["x0"]
        err_censored = scipy_optimize.check_grad(fun, jac, x0)
        assert err_censored < 1e-3, err_censored
        for _ in range(3):
            pt = x0 + rng.normal(0.0, 0.5, size=x0.shape)
            assert scipy_optimize.check_grad(fun, jac, pt) < 1e-3
    finally:
        scipy_optimize.minimize = real_minimize


def test_censored_fit_converges_with_many_call_intercepts():
    """Regression guard for the real-data failure mode the analytic gradient
    fixed: with no `jac`, L-BFGS-B's finite-difference gradient costs
    ~(n_beta + n_calls) evaluations per iteration and exhausted the default
    evaluation budget on the real ~381-call, ~50%-censored data before
    reaching convergence. This synthetic case is deliberately much smaller
    (still cheap for a unit test) but exercises the same shape of problem --
    many call intercepts, partial censoring -- and must converge."""
    ci, x, y = _synthetic(n_calls=80, per_call=15, seed=13)
    X = x[:, None]
    thresh = float(np.quantile(y, 0.4))
    mask = y <= thresh
    out = censored_fit(y, X, ci, threshold=thresh, censored_mask=mask)
    assert out["converged"], out
    assert out["beta"][0] == pytest.approx(-8.0, abs=0.6)


def test_warm_start_change_does_not_move_the_converged_fit():
    """The warm-start replacement must not change what censored_fit reports:
    same converged beta on the same data, regardless of which warm start
    fed the optimizer (this exercises the actual code path, including
    censoring, not just the isolated warm-start formula above)."""
    ci, x, y = _synthetic(n_calls=30, per_call=20, seed=5)
    thresh = float(np.quantile(y, 0.4))
    seen = y >= thresh
    y_c = np.where(seen, y, thresh)
    out = censored_fit(y_c, x[:, None], ci, threshold=thresh, censored_mask=~seen)
    assert out["converged"]
    assert out["beta"][0] == pytest.approx(-8.0, abs=0.6)


def test_censored_fit_offset_is_a_fixed_shape_not_a_fitted_one():
    """`offset` must enter the likelihood with its coefficient pinned at 1
    (never re-estimated): a fit against the TRUE shape must reach a strictly
    higher log-likelihood than the identical fit against a distorted
    version of that shape, since nothing in the model can compensate for
    the distortion by adjusting a coefficient on it."""
    rng = np.random.default_rng(0)
    n_calls, per_call = 30, 20
    call_index, offset, y = [], [], []
    for c in range(n_calls):
        a = rng.normal(60.0, 5.0)
        r = rng.uniform(np.log10(3e3), np.log10(6e4), per_call)
        shape = -8.0 * r
        call_index += [c] * per_call
        offset.append(shape)
        y.append(a + shape + rng.normal(0.0, 1.5, per_call))
    call_index = np.array(call_index)
    offset = np.concatenate(offset)
    y = np.concatenate(y)
    X = np.zeros((len(y), 0))

    correct = censored_fit(y, X, call_index, threshold=-1e9, offset=offset)
    distorted = censored_fit(y, X, call_index, threshold=-1e9, offset=offset * 0.5)

    assert correct["converged"] and distorted["converged"]
    assert correct["beta"].shape == (0,)
    assert correct["loglik"] > distorted["loglik"]


def test_censored_fit_offset_defaults_to_zero():
    """Omitting `offset` (the pre-existing call signature) must behave
    identically to passing an explicit all-zero offset -- a regression guard
    for every caller written before `offset` existed."""
    ci, x, y = _synthetic(n_calls=15, per_call=10, seed=2)
    X = x[:, None]
    a = censored_fit(y, X, ci, threshold=-1e9)
    b = censored_fit(y, X, ci, threshold=-1e9, offset=np.zeros_like(y))
    assert a["beta"] == pytest.approx(b["beta"])
    assert a["loglik"] == pytest.approx(b["loglik"])


def test_decay_fit_table_reports_convergence_draw_count_and_loglik():
    """A reader must be able to tell whether a CI came from 200 draws or 12,
    and whether the point fit converged -- these were computed and silently
    discarded before this test existed."""
    frame = pd.DataFrame({
        "call": np.repeat([f"c{i}" for i in range(12)], 20),
        "range_m": np.tile(np.linspace(4e3, 5e4, 20), 12),
        "receiver_z_m": np.tile(np.linspace(100.0, 1400.0, 20), 12),
        "snr": np.tile(np.linspace(20.0, 8.0, 20), 12),
        "axial_response": 1.0,
    })
    out = decay_fit_table(frame, threshold=0.0, n_bootstrap=10, seed=0)
    for col in ("converged", "n_draws", "loglik"):
        assert col in out.columns, col
    assert out["converged"].dtype == bool
    assert (out["n_draws"] <= 10).all()
    assert (out["n_draws"] == out["n_draws"].iloc[0]).all()  # same fit, one row per term


def test_decay_fit_table_persists_the_joint_bootstrap_draws():
    """The marginal per-term 2.5/97.5 percentiles cannot support a linear
    combination of terms (e.g. beta_range/beta_proj); the joint draws must
    be recoverable from the table so that calculation is possible."""
    frame = pd.DataFrame({
        "call": np.repeat([f"c{i}" for i in range(12)], 20),
        "range_m": np.tile(np.linspace(4e3, 5e4, 20), 12),
        "receiver_z_m": np.tile(np.linspace(100.0, 1400.0, 20), 12),
        "snr": np.tile(np.linspace(20.0, 8.0, 20), 12),
        "axial_response": np.tile(np.linspace(0.9, 0.3, 20), 12),
    })
    out = decay_fit_table(frame, threshold=0.0, n_bootstrap=15, seed=0,
                          projection="fitted")
    draws = out.attrs["draws"]
    terms = out.attrs["terms"]
    assert terms == ["log10_range", "receiver_depth_km", "axial_projection_db"]
    assert draws.shape[1] == len(terms) == 3
    assert draws.shape[0] <= 15


def test_projection_omitted_and_fitted_are_fit_on_the_same_sample():
    """A NaN axial_response must drop that pick from BOTH arms, not just
    'fitted' -- otherwise the two would be non-comparable for a reason
    having nothing to do with whether the term is fitted."""
    calls = [f"c{i}" for i in range(12)]
    rows = []
    for i, c in enumerate(calls):
        axial = np.linspace(0.9, 0.3, 20)
        if i == 0:
            axial = axial.copy()
            axial[0] = np.nan
        rows.append(pd.DataFrame({
            "call": c, "range_m": np.linspace(4e3, 5e4, 20),
            "receiver_z_m": np.linspace(100.0, 1400.0, 20),
            "snr": np.linspace(20.0, 8.0, 20), "axial_response": axial,
        }))
    frame = pd.concat(rows, ignore_index=True)

    fitted = decay_fit_table(frame, threshold=0.0, n_bootstrap=5, seed=0,
                             projection="fitted")
    omitted = decay_fit_table(frame, threshold=0.0, n_bootstrap=5, seed=0,
                              projection="omitted")
    assert fitted.n_picks.iloc[0] == omitted.n_picks.iloc[0] == 12 * 20 - 1


def test_hypothesis_shape_geometric_is_negative_of_geometric_tl():
    from dclde_soundspeed.experiments.propagation import (
        geometric_tl, hypothesis_shape_geometric)

    r = np.array([1e3, 1e4, 3e4])
    np.testing.assert_allclose(hypothesis_shape_geometric(r), -geometric_tl(r))


def test_hypothesis_comparison_table_favors_the_true_shape():
    """The pre-registered statistic (spec section 6.4): each hypothesis's
    shape is a FIXED offset, and the true shape must win by delta_loglik
    with a bootstrap CI that excludes zero, on data generated from that
    exact shape."""
    from dclde_soundspeed.experiments.propagation import hypothesis_comparison_table

    rng = np.random.default_rng(0)
    n_calls, per_call = 40, 15
    rows = []
    for c in range(n_calls):
        a = rng.normal(60.0, 5.0)
        r = rng.uniform(3e3, 6e4, per_call)
        true_shape = -8.0 * np.log10(r)
        wrong_shape = -3.0 * np.log10(r)   # a visibly different decay
        snr = a + true_shape + rng.normal(0.0, 1.5, per_call)
        rows.append(pd.DataFrame({
            "call": f"c{c}", "range_m": r, "receiver_z_m": 500.0, "snr": snr,
            "shape_true": true_shape, "shape_wrong": wrong_shape,
        }))
    frame = pd.concat(rows, ignore_index=True)

    summary, deltas = hypothesis_comparison_table(
        frame, {"true": "shape_true", "wrong": "shape_wrong"},
        threshold=-1e9, n_bootstrap=40, seed=0)

    assert set(summary["hypothesis"]) == {"true", "wrong"}
    assert summary["converged"].all()
    true_ll = summary.loc[summary.hypothesis == "true", "loglik"].iloc[0]
    wrong_ll = summary.loc[summary.hypothesis == "wrong", "loglik"].iloc[0]
    assert true_ll > wrong_ll

    row = deltas[deltas.comparison == "true - wrong"].iloc[0]
    assert row.delta_loglik > 0
    assert row.ci_lo > 0, "the CI should exclude zero when the shapes differ this much"


def test_hypothesis_comparison_table_requires_at_least_two_hypotheses():
    from dclde_soundspeed.experiments.propagation import hypothesis_comparison_table

    frame = pd.DataFrame({"call": ["a"], "range_m": [1e4], "receiver_z_m": [500.0],
                          "snr": [10.0], "shape_a": [-5.0]})
    with pytest.raises(ValueError, match="at least 2"):
        hypothesis_comparison_table(frame, {"a": "shape_a"}, threshold=0.0)


def test_hypothesis_comparison_table_delta_ci_is_paired_not_independent():
    """The Delta CI in `hypothesis_comparison_table` (propagation.py ~975-988)
    pairs each bootstrap draw's two hypothesis fits on the SAME resampled
    calls -- both `loglik_draws[a]` and `loglik_draws[b]` come from index i
    of the SAME draw's resample. That pairing is what makes the interval
    tight when the two hypotheses' fits are correlated across resamples
    (both driven by the same per-call intercept noise): pairing subtracts
    that common noise out of the delta, an independent (unpaired) draw does
    not. All 349 tests in this suite passed under a mutation that reverses
    one side's draw order (`loglik_draws[b][:n_paired][::-1]`), so this test
    exists specifically to give that property teeth.

    Built from first principles rather than a magic number: two hypotheses
    with a NEARLY IDENTICAL predicted shape (differing by a small constant),
    fit against data with LARGE per-call intercept variance. Because the two
    shapes are almost the same, most of each fit's loglik variation across
    resamples comes from which calls (and their random intercepts) got
    drawn -- the same source for both hypotheses in a given draw. Pairing
    removes that shared variation from the delta; matching each hypothesis's
    OWN independent resample (computed here by hand, not via the pairing
    `hypothesis_comparison_table` provides) does not, and should give a
    materially wider interval.
    """
    from dclde_soundspeed.experiments.propagation import (
        censored_fit, hypothesis_comparison_table)

    rng = np.random.default_rng(0)
    n_calls, per_call = 60, 10
    rows = []
    for c in range(n_calls):
        a = rng.normal(0.0, 15.0)  # large per-call intercept variance: the
                                    # shared noise source that correlates
                                    # the two hypotheses' loglik across draws
        r = rng.uniform(3e3, 6e4, per_call)
        shape_a = -8.0 * np.log10(r)
        shape_b = shape_a + 0.05    # nearly identical shape
        snr = a + shape_a + rng.normal(0.0, 1.0, per_call)
        rows.append(pd.DataFrame({
            "call": f"c{c}", "range_m": r, "receiver_z_m": 500.0, "snr": snr,
            "shape_a": shape_a, "shape_b": shape_b,
        }))
    frame = pd.concat(rows, ignore_index=True)

    n_bootstrap = 150
    threshold = -1e9
    summary, deltas = hypothesis_comparison_table(
        frame, {"a": "shape_a", "b": "shape_b"},
        threshold=threshold, n_bootstrap=n_bootstrap, seed=0)
    row = deltas[deltas.comparison == "a - b"].iloc[0]
    paired_width = float(row.ci_hi - row.ci_lo)

    def fit_one(sub, shape_col):
        y = sub["snr"].to_numpy(float)
        offset = sub[shape_col].to_numpy(float)
        X = np.zeros((len(sub), 0))
        ci = sub["call"].to_numpy()
        mask = y <= threshold
        return censored_fit(y, X, ci, threshold=threshold, censored_mask=mask,
                            offset=offset)

    def bootstrap_one(shape_col, seed):
        rng2 = np.random.default_rng(seed)
        calls = np.array(sorted(frame["call"].unique()))
        draws = []
        for _ in range(n_bootstrap):
            pick = rng2.choice(calls, size=calls.size, replace=True)
            parts = []
            for k, c in enumerate(pick):
                part = frame[frame["call"] == c].copy()
                part["call"] = f"{c}__{k}"
                parts.append(part)
            resampled = pd.concat(parts, ignore_index=True)
            r = fit_one(resampled, shape_col)
            if r["converged"]:
                draws.append(r["loglik"])
        return np.array(draws)

    # Independent (unpaired) draws: each hypothesis resampled with its OWN
    # RNG stream, so a draw's two logliks are no longer forced to share a
    # resample.
    draws_a = bootstrap_one("shape_a", seed=1)
    draws_b = bootstrap_one("shape_b", seed=2)
    n_unpaired = min(len(draws_a), len(draws_b))
    unpaired_deltas = draws_a[:n_unpaired] - draws_b[:n_unpaired]
    lo, hi = np.percentile(unpaired_deltas, [2.5, 97.5])
    unpaired_width = float(hi - lo)

    assert paired_width < 0.5 * unpaired_width, (
        f"paired CI width ({paired_width}) should be materially narrower "
        f"than the unpaired CI width ({unpaired_width}) when the two "
        f"hypotheses' fits are strongly correlated across resamples -- "
        f"pairing is supposed to cancel that shared noise out of the delta")


def test_modal_shape_columns_reuses_paths_across_seabeds_and_counts_failures(monkeypatch):
    """Path lookup (`depth_along_path`) must run ONCE per pick and be reused
    across every seabed -- mode-solving is the only seabed-dependent step --
    and a refused path must show up as a counted NaN, never a silently
    dropped row."""
    import dclde_localization.physics.bathymetry as bathymetry_mod
    from dclde_localization.physics.bathymetry import PathProfile
    from dclde_soundspeed.experiments.propagation import modal_shape_columns
    from dclde_soundspeed.soundspeed.modes import Seabed
    from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile

    call_count = {"n": 0}

    def fake_depth_along_path(grid, sx, sy, px, py, n_steps=32):
        call_count["n"] += 1
        if call_count["n"] == 2:   # the second pick's path is refused
            return PathProfile(range_m=np.array([]), depth_m=np.array([]),
                               ok=False, reason="refused_for_test")
        depth = np.linspace(50.0, 500.0, n_steps)
        return PathProfile(range_m=np.linspace(0.0, 10_000.0, n_steps),
                           depth_m=depth, ok=True)

    monkeypatch.setattr(bathymetry_mod, "depth_along_path", fake_depth_along_path)

    frame = pd.DataFrame({
        "range_m": [10_000.0, 10_000.0, 10_000.0],
        "receiver_z_m": [400.0, 400.0, 400.0],
        "sx": [0.0] * 3, "sy": [0.0] * 3, "px": [0.0] * 3, "py": [0.0] * 3,
    })
    profile = SoundSpeedProfile(z_down=np.linspace(0.0, 1500.0, 64),
                                c=np.full(64, 1480.0), source="test")
    seabeds = {"soft": Seabed(c=1550.0, rho=1.5, alpha=0.05),
              "hard": Seabed(c=2100.0, rho=2.3, alpha=0.80)}

    results = modal_shape_columns(frame, grid=None, profile=profile,
                                  seabeds=seabeds, source_z=30.0, nz=64)

    assert call_count["n"] == 3, (
        "path should be computed once per pick (3), not once per "
        "pick-per-seabed (6) -- reuse across seabeds is the whole point")
    assert set(results) == {"soft", "hard"}
    for shape, counts in results.values():
        assert counts["n_picks"] == 3
        assert counts["n_path_refused"] == 1
        assert np.isnan(shape[1]), "the refused-path pick must be NaN, not dropped"
        assert counts["reasons"].get("refused_for_test") == 1
        # Every pick is accounted for: predicted, refused, or a non-finite
        # adiabatic solve -- none silently missing from the bookkeeping.
        assert (counts["n_predicted"] + counts["n_path_refused"]
               + counts["n_tl_nonfinite"] == counts["n_picks"])


def test_modal_shape_for_seabed_matches_modal_shape_columns(monkeypatch):
    """The memory-conscious split (compute_pick_paths once, then
    modal_shape_for_seabed per seabed, processed and released one at a
    time) must give EXACTLY the same result as the convenience wrapper that
    holds every seabed at once -- this is a refactor for peak memory, not a
    change in what gets computed."""
    import dclde_localization.physics.bathymetry as bathymetry_mod
    from dclde_localization.physics.bathymetry import PathProfile
    from dclde_soundspeed.experiments.propagation import (
        compute_pick_paths, modal_shape_columns, modal_shape_for_seabed)
    from dclde_soundspeed.soundspeed.modes import Seabed
    from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile

    def fake_depth_along_path(grid, sx, sy, px, py, n_steps=32):
        depth = np.linspace(50.0, 500.0, n_steps)
        return PathProfile(range_m=np.linspace(0.0, 10_000.0, n_steps),
                           depth_m=depth, ok=True)

    monkeypatch.setattr(bathymetry_mod, "depth_along_path", fake_depth_along_path)

    frame = pd.DataFrame({
        "range_m": [10_000.0, 12_000.0, 8_000.0],
        "receiver_z_m": [400.0, 420.0, 380.0],
        "sx": [0.0] * 3, "sy": [0.0] * 3, "px": [0.0] * 3, "py": [0.0] * 3,
    })
    profile = SoundSpeedProfile(z_down=np.linspace(0.0, 1500.0, 64),
                                c=np.full(64, 1480.0), source="test")
    seabeds = {"soft": Seabed(c=1550.0, rho=1.5, alpha=0.05),
              "hard": Seabed(c=2100.0, rho=2.3, alpha=0.80)}

    wrapped = modal_shape_columns(frame, grid=None, profile=profile,
                                  seabeds=seabeds, source_z=30.0, nz=64)

    paths, reasons = compute_pick_paths(frame, grid=None)
    receiver_z = frame["receiver_z_m"].to_numpy(float)
    split = {name: modal_shape_for_seabed(paths, reasons, receiver_z, profile,
                                          seabed, source_z=30.0, nz=64)
            for name, seabed in seabeds.items()}

    for name in seabeds:
        np.testing.assert_array_equal(wrapped[name][0], split[name][0])
        assert wrapped[name][1] == split[name][1]


def test_modal_shape_columns_counts_nan_or_nonpositive_range_without_crashing():
    """A NaN or non-positive `range_m` must be caught BEFORE calling
    `depth_along_path` at all (the grid is never even touched for such a
    row), so this needs no real grid -- both rows here are invalid inputs on
    purpose, to keep the test independent of the grid fixture."""
    from dclde_soundspeed.experiments.propagation import modal_shape_columns
    from dclde_soundspeed.soundspeed.modes import Seabed
    from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile

    frame = pd.DataFrame({
        "range_m": [np.nan, 0.0], "receiver_z_m": [400.0, 400.0],
        "sx": [0.0, 0.0], "sy": [0.0, 0.0], "px": [0.0, 0.0], "py": [0.0, 0.0],
    })
    profile = SoundSpeedProfile(z_down=np.linspace(0.0, 1500.0, 32),
                                c=np.full(32, 1480.0), source="test")
    results = modal_shape_columns(
        frame, grid=object(), profile=profile,
        seabeds={"mid": Seabed(c=1700.0, rho=1.8, alpha=0.15)},
        source_z=30.0, nz=32)
    shape, counts = results["mid"]
    assert np.isnan(shape).all()
    assert counts["n_path_refused"] == 2


def test_hypothesis_comparison_table_drops_nan_shape_rows_for_every_hypothesis():
    from dclde_soundspeed.experiments.propagation import hypothesis_comparison_table

    frame = pd.DataFrame({
        "call": ["a", "a", "b", "b"], "range_m": [1e4, 2e4, 1e4, 2e4],
        "receiver_z_m": [500.0] * 4, "snr": [10.0, 8.0, 12.0, 6.0],
        "shape_a": [-5.0, np.nan, -6.0, -7.0],
        "shape_b": [-4.0, -4.5, -5.0, -5.5],
    })
    summary, deltas = hypothesis_comparison_table(
        frame, {"a": "shape_a", "b": "shape_b"}, threshold=-1e9, n_bootstrap=2, seed=0)
    assert (summary["n_picks"] == 3).all()
