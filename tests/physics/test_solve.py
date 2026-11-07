import numpy as np
import pytest

from dclde_localization.physics import forward
from dclde_localization.physics.forward import DEFAULT_C
from dclde_localization.physics.solve import Basin, Solution, _grid_misfit, compare, solve
from dclde_localization.physics.sound_speed_models import ConstantC, PathAverage
from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile

MODEL = ConstantC(DEFAULT_C)

OREGON_PROFILE = SoundSpeedProfile(
    np.array([0.0, 100.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0]),
    np.array([1491.0, 1480.0, 1478.0, 1477.0, 1479.0, 1484.0, 1490.0]),
    "oregon-nov")


def _straight_cable(n=40, x0=10000.0, x1=60000.0, y=20000.0, depth=-300.0):
    return (np.linspace(x0, x1, n), np.full(n, y), np.full(n, depth))


def _synthesise(src, cable, model=MODEL, t0=7.5):
    px, py, pz = cable
    return model.travel_time(*src, px, py, pz) + t0


def test_recovers_a_known_source_from_noiseless_arrivals():
    cable = _straight_cable()
    truth = (35000.0, 26000.0, -30.0)
    sol = solve(*cable, _synthesise(truth, cable), MODEL, sz=-30.0)
    assert np.hypot(sol.best.x - truth[0], sol.best.y - truth[1]) < 60.0


def test_returns_both_lobes_for_a_single_straight_cable():
    # A straight cable cannot tell which side the source is on. An estimator
    # that returns one point has thrown that away, and every ablation built on
    # it would report basin flips as small position changes.
    cable = _straight_cable()
    truth = (35000.0, 26000.0, -30.0)
    sol = solve(*cable, _synthesise(truth, cable), MODEL, sz=-30.0)
    assert len(sol.basins) >= 2
    sep = sol.separation_m
    assert sep is not None and sep > 8000.0
    mirrored_y = 2 * 20000.0 - truth[1]
    ys = sorted(b.y for b in sol.basins[:2])
    assert min(abs(ys[0] - mirrored_y), abs(ys[1] - mirrored_y)) < 600.0


def test_the_two_lobes_fit_almost_equally_well():
    # Both lobes' misfit is genuine sub-1e-9 optimizer noise here (noiseless
    # synthetic picks at an exact geometric tie -- see task-4-report.md), not
    # a real difference in fit, and Basin.misfit reports that raw number
    # rather than a fabricated floor (review round 2, Important-1). A ratio
    # of two near-zero floats is itself noisy: 1.19 measured on the
    # reviewer's environment, comfortably inside 1.25 but not 1.15. The
    # threshold is widened, not the reported value.
    cable = _straight_cable()
    sol = solve(*cable, _synthesise((35000.0, 26000.0, -30.0), cable), MODEL, sz=-30.0)
    a, b = sol.basins[0].misfit, sol.basins[1].misfit
    assert b / max(a, 1e-12) < 1.25


def test_two_cables_resolve_the_ambiguity():
    # With a second cable off-axis the mirror lobe stops fitting, so the gap
    # between the best two basins should widen sharply.
    px1, py1, pz1 = _straight_cable()
    px2 = np.linspace(20000.0, 50000.0, 30)
    py2 = np.linspace(38000.0, 41000.0, 30)
    pz2 = np.full(30, -700.0)
    px, py, pz = (np.r_[px1, px2], np.r_[py1, py2], np.r_[pz1, pz2])
    obs = MODEL.travel_time(35000.0, 26000.0, -30.0, px, py, pz) + 7.5
    sol = solve(px, py, pz, obs, MODEL, sz=-30.0)
    assert sol.basins[1].misfit / max(sol.basins[0].misfit, 1e-12) > 3.0


def test_t0_is_recovered_and_is_shared_across_all_picks():
    cable = _straight_cable()
    sol = solve(*cable, _synthesise((35000.0, 26000.0, -30.0), cable, t0=12.25), MODEL, sz=-30.0)
    assert sol.best.t0 == pytest.approx(12.25, abs=0.05)


def test_compare_separates_basin_flips_from_position_deltas():
    # The reporting contract: a call that jumped to the mirror lobe must never
    # be averaged into a median position change.
    def sol_at(x, y):
        return Solution(basins=(Basin(x=x, y=y, z=-30.0, c=DEFAULT_C, t0=0.0, misfit=1.0),),
                        n_picks=20, model_name="test")

    a = {"c1": sol_at(0.0, 0.0), "c2": sol_at(0.0, 0.0), "c3": sol_at(0.0, 0.0)}
    b = {"c1": sol_at(80.0, 0.0), "c2": sol_at(120.0, 0.0), "c3": sol_at(9000.0, 0.0)}
    cmp = compare(a, b, flip_threshold_m=500.0)
    assert cmp.n == 3
    assert cmp.n_flipped == 1
    assert cmp.n_stable == 2
    assert np.median(cmp.stable_delta_m) == pytest.approx(100.0)
    assert 9000.0 not in cmp.stable_delta_m


def test_compare_ignores_calls_missing_from_either_side():
    def sol_at(x):
        return Solution(basins=(Basin(x=x, y=0.0, z=-30.0, c=DEFAULT_C, t0=0.0, misfit=1.0),),
                        n_picks=20, model_name="test")

    cmp = compare({"a": sol_at(0.0), "b": sol_at(0.0)}, {"a": sol_at(10.0)})
    assert cmp.n == 1


# --- Beyond the brief: pin the vectorised grid to the frozen scalar definitions,
# and check the answer is a property of the data, not the grid resolution. ---

@pytest.mark.parametrize("objective", ["l1", "l2"])
@pytest.mark.parametrize("model", [ConstantC(DEFAULT_C), PathAverage(OREGON_PROFILE)])
def test_vectorised_grid_matches_scalar_misfit_at_several_cells(objective, model):
    # Ruling R4: the coarse grid must be vectorised (one model call, not a
    # Python double loop). This test pins that vectorised path to
    # forward.misfit_value called scalar-wise, cell by cell, so the two
    # definitions of the objective cannot silently drift apart.
    px, py, pz = _straight_cable()
    obs = model.travel_time(35000.0, 26000.0, -30.0, px, py, pz) + 7.5

    xs = np.linspace(10000.0, 60000.0, 9)
    ys = np.linspace(5000.0, 35000.0, 7)
    gx, gy = np.meshgrid(xs, ys)
    field, t0_field = _grid_misfit(gx, gy, -30.0, px, py, pz, obs, model, objective)

    rng = np.random.default_rng(0)
    sample_cells = rng.choice(field.size, size=6, replace=False)
    for flat in sample_cells:
        iy, ix = np.unravel_index(flat, field.shape)
        tt = model.travel_time(gx[iy, ix], gy[iy, ix], -30.0, px, py, pz)
        want_misfit = forward.misfit_value(obs, tt, objective)
        want_t0 = forward.best_t0(obs, tt, objective)
        assert field[iy, ix] == pytest.approx(want_misfit, rel=1e-10, abs=1e-12)
        assert t0_field[iy, ix] == pytest.approx(want_t0, rel=1e-10, abs=1e-12)


def test_solve_best_position_is_independent_of_coarse_grid_resolution():
    # The estimate should be a property of the data, not an artifact of how
    # finely the coarse grid happened to sample it.
    cable = _straight_cable()
    truth = (35000.0, 26000.0, -30.0)
    obs = _synthesise(truth, cable)

    sol_coarse = solve(*cable, obs, MODEL, sz=-30.0, n_coarse=81)
    sol_fine = solve(*cable, obs, MODEL, sz=-30.0, n_coarse=161)

    assert np.hypot(sol_coarse.best.x - sol_fine.best.x,
                     sol_coarse.best.y - sol_fine.best.y) < 10.0


def test_coarse_grid_evaluation_calls_the_model_once_for_the_whole_grid():
    # Sanity guard against re-introducing a per-cell Python loop. A wall-clock
    # bound (the original version of this test: "< 2.0 s") is flaky on shared
    # CI and only an indirect proxy for the actual invariant Ruling R4
    # promises -- prefer counting, the same pattern test_raytrace.py's
    # table_build_count already uses (Task 14 review). This counts the size
    # of every travel_time call the coarse-grid solve makes and checks that
    # exactly one of them covers the whole (n_coarse, n_coarse, n_picks) grid
    # in a single shot; a reintroduced per-cell loop would instead show
    # n_coarse*n_coarse separate calls of size n_picks each, and this would
    # fail regardless of how fast the box happens to run that day.
    cable = _straight_cable()
    px, py, pz = cable
    n_coarse = 41  # small enough to keep the test itself fast
    call_sizes = []

    class _CountingModel:
        name = "counting"

        def travel_time(self, sx, sy, sz, qpx, qpy, qpz):
            call_sizes.append(int(np.broadcast(np.asarray(sx), np.asarray(qpx)).size))
            return MODEL.travel_time(sx, sy, sz, qpx, qpy, qpz)

    solve(*cable, _synthesise((35000.0, 26000.0, -30.0), cable), _CountingModel(),
         sz=-30.0, n_coarse=n_coarse)

    grid_query_size = n_coarse * n_coarse * px.size
    assert call_sizes.count(grid_query_size) == 1, (
        f"expected exactly one travel_time call covering the whole "
        f"{n_coarse}x{n_coarse} grid in one shot; call sizes were {call_sizes}")


# --- Review round 2 regressions: both of these were found by measurement,
# not by the suite above, which is exactly why they are pinned here now. ---

def test_off_grid_source_is_not_biased_by_a_coarse_refinement_box():
    # Important-2: an earlier version capped the local-minimum refinement
    # radius at MIN_BASIN_SEPARATION_M / 2 (750 m) regardless of grid
    # resolution. At n_coarse=41 the grid's own half-cell quantisation error
    # (~1750 m here) exceeds that cap, so the refinement box could not reach
    # an off-grid true source and `.best` came back 1141 m off with no
    # indication anything had gone wrong. The radius must scale with the
    # grid's own cell size with no such cap.
    cable = _straight_cable()
    truth = (37123.0, 27456.0, -30.0)  # deliberately off any coarse grid node
    sol = solve(*cable, _synthesise(truth, cable), MODEL, sz=-30.0, n_coarse=41)
    err = np.hypot(sol.best.x - truth[0], sol.best.y - truth[1])
    assert err < 60.0


def test_a_genuinely_close_mirror_lobe_is_not_replaced_by_a_fabricated_one():
    # Important-3: a fixed basin-selection separation (previously 3000 m)
    # masked away a *real* nearby mirror lobe and reported some other,
    # unrelated grid point as basin 2 instead -- a fabricated stand-in that
    # looked exactly like a located basin. For a source only 800 m off the
    # cable (true mirror separation 1600 m, comfortably under the old
    # 3000 m mask), `.separation_m` must reflect the true separation, not an
    # arbitrary replacement position.
    cable = _straight_cable()
    offset = 800.0
    truth = (35000.0, 20000.0 + offset, -30.0)
    true_mirror_separation = 2.0 * offset
    sol = solve(*cable, _synthesise(truth, cable), MODEL, sz=-30.0)

    assert len(sol.basins) >= 2
    assert np.hypot(sol.best.x - truth[0], sol.best.y - truth[1]) < 60.0
    assert sol.separation_m == pytest.approx(true_mirror_separation, abs=100.0)


def test_close_mirror_lobes_are_not_collapsed_into_a_fabricated_replacement():
    # Review round 3, Finding A: at 400 m offset (true mirror separation
    # 800 m), Tier 1's grid search *does* resolve two distinct local minima
    # (1125 m apart on this grid) -- so this is not the sub-grid-cell wall
    # documented for the 800 m case. The bug was that each seed's refinement
    # box, sized only from the grid's cell size with no reference to the
    # other seed, was large enough to cross the (real but shallow, at this
    # small an offset) ridge between them: one seed's Nelder-Mead search
    # walked onto the *other* lobe's optimum, both basins then reported the
    # same position, the duplicate was dropped, Tier 1 fell to one basin,
    # and Tier 2 fabricated an unrelated ~3600 m-away substitute --
    # `separation_m` = 3587.5 against a true 800.
    #
    # The estimator must not repeat that: either it reports both real lobes
    # (separation close to the true 800 m) or it honestly declines to report
    # a resolved separation at all (`separation_resolved` False / `None`).
    # A fabricated intermediate number is the one outcome that is never
    # acceptable.
    #
    # This fixture is deterministic (a fixed synthetic geometry, no
    # randomness), so an `if sol.separation_resolved: ... else: assert
    # sol.separation_m is None` here does not test two live possibilities --
    # it tests one real outcome (whichever solve() actually produces on this
    # input) plus, in the other branch, a fact `Solution.separation_m`
    # already guarantees by its own definition (`return None` whenever
    # `not separation_resolved`) regardless of what solve() computed. That
    # branch cannot fail no matter how badly solve() is broken, so it was
    # not the safety net it looked like (Task 14 review). Asserted directly
    # instead, against the behaviour this exact input is known to produce:
    # two genuinely resolved local minima, separation close to the true
    # 800 m -- if a regression ever makes this geometry fall back to the
    # unresolved case, this test must fail, not silently accept it.
    cable = _straight_cable()
    offset = 400.0
    truth = (35000.0, 20000.0 + offset, -30.0)
    true_mirror_separation = 2.0 * offset
    sol = solve(*cable, _synthesise(truth, cable), MODEL, sz=-30.0)

    assert sol.separation_resolved is True
    assert sol.separation_m == pytest.approx(true_mirror_separation, abs=150.0)
    # The number must never be the fabricated ~3587.5 m this exact regression
    # previously produced.
    assert abs(sol.separation_m - 3587.5) > 500.0
