import numpy as np
import pytest

from dclde_localization.physics.forward import DEFAULT_C
from dclde_localization.physics.sound_speed_models import PathAverage
from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile, to_repo
from dclde_soundspeed.soundspeed.raytrace import (
    RayTrace, direct_time, max_direct_range, trace)

OREGON = SoundSpeedProfile(
    np.array([0.0, 100.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0]),
    np.array([1491.0, 1480.0, 1478.0, 1477.0, 1479.0, 1484.0, 1490.0]),
    "oregon-nov")
FLAT = SoundSpeedProfile.constant(DEFAULT_C)


@pytest.fixture(scope="module")
def oregon_raytrace():
    # Building the table is the expensive part (tens of thousands of brentq
    # solves, see task-3-report.md); every test below that only needs a
    # source depth of -30 (repo frame) shares this one instance and its one
    # cached table, rather than each triggering its own multi-second rebuild.
    return RayTrace(OREGON)


# --- The brief's nine tests, unchanged -------------------------------------

@pytest.mark.parametrize("z1,z2,rng", [(30.0, 100.0, 5000.0), (30.0, 1575.0, 30000.0),
                                       (100.0, 500.0, 10000.0)])
def test_constant_profile_reproduces_the_straight_line_time(z1, z2, rng):
    # The tracer must collapse onto simple geometry when there is no gradient.
    # Without this, any difference it reports later could be its own
    # quadrature error.
    got = direct_time(FLAT, z1, z2, rng)
    want = float(np.hypot(rng, z2 - z1) / DEFAULT_C)
    assert got is not None
    assert got == pytest.approx(want, rel=1e-9)


def test_downward_refraction_limits_how_far_a_direct_ray_reaches():
    # The finding this whole module exists to express: with c falling from the
    # surface, a shallow source cannot reach a shallow receiver at range.
    assert max_direct_range(OREGON, 30.0, 100.0) == pytest.approx(1400.0, abs=300.0)
    assert max_direct_range(OREGON, 30.0, 1575.0) == pytest.approx(15900.0, abs=1500.0)


def test_no_direct_ray_beyond_the_maximum_range():
    mx = max_direct_range(OREGON, 30.0, 500.0)
    assert direct_time(OREGON, 30.0, 500.0, mx * 0.8) is not None
    assert direct_time(OREGON, 30.0, 500.0, mx * 1.5) is None


def test_constant_profile_has_no_range_limit_at_these_scales():
    # An iso-speed ocean does not refract, so nothing is shadowed. This guards
    # against the range limit being an artefact of the tracer rather than
    # physics.
    assert max_direct_range(FLAT, 30.0, 500.0) > 1e6


def test_trace_returns_none_when_the_ray_turns():
    assert trace(OREGON, 30.0, 1575.0, 1.0 / 1000.0) is None


def test_raytrace_model_reduces_to_constant_c_on_a_flat_profile():
    # Deliberately checked against independent geometry, not
    # ConstantC(DEFAULT_C).travel_time(...): RayTrace's flat-profile bypass
    # and ConstantC both delegate straight through to the same
    # forward.travel_time call with the same speed, so comparing the two
    # model wrappers to each other is "f(x) == f(x)" -- it cannot fail
    # regardless of what either implementation actually computes (Task 14
    # review). A hand-written straight-line formula has no such shared
    # implementation to hide a bug behind.
    sx, sy, sz = 20000.0, 19000.0, -30.0
    px = np.array([12000.0, 35000.0])
    py = np.array([18000.0, 21000.0])
    pz = np.array([-220.0, -640.0])
    got = RayTrace(FLAT).travel_time(sx, sy, sz, px, py, pz)
    want = np.sqrt((px - sx) ** 2 + (py - sy) ** 2 + (pz - sz) ** 2) / DEFAULT_C
    assert np.allclose(got, want, rtol=1e-8)


def test_raytrace_records_what_fraction_of_picks_had_no_direct_ray(oregon_raytrace):
    # A model that silently substitutes a different physics for some picks
    # would make its own results uninterpretable. The fraction must be
    # observable.
    px = np.array([12000.0, 60000.0])
    py = np.array([18000.0, 24000.0])
    pz = np.array([-100.0, -1500.0])
    oregon_raytrace.travel_time(37000.0, 22000.0, -30.0, px, py, pz)
    assert 0.0 <= oregon_raytrace.last_fallback_fraction <= 1.0
    assert oregon_raytrace.last_fallback_fraction > 0.0


# --- Task 3 / Ruling R9: tabulated interpolator, beyond the brief's nine ---

@pytest.mark.parametrize("range_m", [5000.0, 20000.0, 40000.0])
def test_raytrace_falls_back_rather_than_returning_a_wrong_answer_near_the_duct(
        oregon_raytrace, range_m):
    # Finding 1 (review round 1): a channel 1 m above the profile's shallowest
    # sample, adjacent to a source depth near the profile's maximum sound
    # speed, forms an idealised "surface duct" where max_direct_range diverges
    # (see the module docstring). Before this fix, a query here returned an
    # ordinary finite number several seconds off, with last_fallback_fraction
    # at 0 -- invisible to any downstream consumer. It must now either match
    # the tracer's own ground truth tightly, or be reported as a fallback;
    # what it must never do is return a confident-looking wrong number.
    z1_down, z2_down = 30.0, -11.0
    want = direct_time(OREGON, z1_down, z2_down, range_m)
    assert want is not None, "test range must be one the direct ray actually reaches"

    sx, sy = 0.0, 0.0
    sz = to_repo(z1_down)
    px = np.array([range_m])
    py = np.array([0.0])
    pz = to_repo(np.array([z2_down]))

    # Deterministic on this fixed geometry (checked directly: all three
    # parametrized range_m values fall back at the table's shipped
    # resolution), so an `if last_fallback_fraction == 0.0: assert ... # else:
    # (nothing)` here has an if-branch that never runs and an else-branch
    # with no assertion at all -- the test passed regardless of what
    # travel_time actually returned (Task 14 review). Asserted directly
    # instead, against both known outcomes: the fallback fires (so this
    # duct case is honestly reported, not silently mis-answered), and the
    # returned value is exactly what PathAverage alone would give -- not
    # some other, unreported number -- confirming the fallback path was
    # actually taken, not a coincidentally-close wrong answer.
    got = oregon_raytrace.travel_time(sx, sy, float(sz), px, py, pz)[0]
    assert oregon_raytrace.last_fallback_fraction == 1.0, \
        "this duct geometry is expected to fall back at the shipped table " \
        "resolution -- if this fires, re-check whether it now resolves " \
        "directly and this test needs the other branch instead"
    fallback_only = PathAverage(OREGON).travel_time(sx, sy, float(sz), px, py, pz)[0]
    assert got == pytest.approx(fallback_only, rel=1e-12)


@pytest.mark.parametrize("z2_down,frac", [
    (100.0, 0.3), (100.0, 0.7),
    (300.0, 0.2), (300.0, 0.6), (300.0, 0.9),
    (800.0, 0.2), (800.0, 0.5), (800.0, 0.85),
    (1500.0, 0.1), (1500.0, 0.4), (1500.0, 0.8),
    (1900.0, 0.3), (1900.0, 0.6),
    # Finding 1 (review round 1): pinned 5 m below the profile's 1500 m knot,
    # where c(z) has a real gradient kink -- exactly the kind of point whose
    # error could regrow silently under a resolution or margin change.
    # Measured 5.18e-5 s at the shipped resolution.
    (1505.0, 0.1),
])
def test_tabulated_travel_time_agrees_with_direct_scalar_calls(oregon_raytrace, z2_down, frac):
    # The whole point of the table: it must reproduce direct_time's own answer
    # (its ground truth) to well under the ~96 ms pick noise floor -- the
    # brief's target is 1e-4 s. Measured worst case across these points at the
    # shipped resolution is ~5.2e-5 s (see task-3-report.md).
    z1_down = 30.0
    mx = max_direct_range(OREGON, z1_down, z2_down)
    r = frac * mx
    want = direct_time(OREGON, z1_down, z2_down, r)
    assert want is not None, "test range must be one the direct ray actually reaches"

    sx, sy = 0.0, 0.0
    sz = to_repo(z1_down)
    px = np.array([r])
    py = np.array([0.0])
    pz = to_repo(np.array([z2_down]))

    got = oregon_raytrace.travel_time(sx, sy, float(sz), px, py, pz)[0]
    assert oregon_raytrace.last_fallback_fraction == 0.0, \
        "this point must come from the table, not the fallback"
    assert got == pytest.approx(want, abs=1e-4)


@pytest.mark.parametrize("z2_down,frac", [(1500.0, 0.995), (800.0, 0.99), (100.0, 0.99)])
def test_tabulated_travel_time_agrees_with_direct_scalar_calls_close_to_grazing(
        oregon_raytrace, z2_down, frac):
    # Second review round, split from a single grazing-limit test that
    # conflated two different failure modes (see the fallback-boundary test
    # below). This half covers cells the table DOES hold a value for, very
    # close to (but not past) the quantised existence boundary -- confirmed
    # non-fallback at these exact points. Interpolation accuracy here is not
    # meaningfully worse than the away-from-grazing case: measured error at
    # these three points is 4.85e-9 s, 1.36e-6 s and 3.27e-6 s respectively,
    # consistent with the ~1e-5 to 6e-5 s ceiling found by several
    # independent sweep strategies (task-3-report.md, fix round 2) -- an
    # earlier, larger reported figure for this regime turned out to be
    # unrelated to interpolation (see the fallback test's comment).
    z1_down = 30.0
    mx = max_direct_range(OREGON, z1_down, z2_down)
    r = frac * mx
    want = direct_time(OREGON, z1_down, z2_down, r)
    assert want is not None, "test range must be one the direct ray actually reaches"

    sx, sy = 0.0, 0.0
    sz = to_repo(z1_down)
    px = np.array([r])
    py = np.array([0.0])
    pz = to_repo(np.array([z2_down]))

    got = oregon_raytrace.travel_time(sx, sy, float(sz), px, py, pz)[0]
    assert oregon_raytrace.last_fallback_fraction == 0.0, \
        "this point is expected to come from the table, not the fallback -- if " \
        "this fires, the quantised existence boundary moved and the test needs " \
        "a new close-but-still-valid point"
    assert got == pytest.approx(want, abs=1e-4)


@pytest.mark.parametrize("z2_down,frac", [(1575.0, 0.999), (800.0, 0.999), (300.0, 0.995)])
def test_raytrace_falls_back_near_the_quantised_grazing_boundary(oregon_raytrace, z2_down, frac):
    # Second review round: an independent review re-measurement reported up
    # to ~1.4e-3 s of "interpolation error" near grazing. It wasn't
    # interpolation error -- at those exact geometries the model had fallen
    # back to PathAverage (confirmed: last_fallback_fraction was 100%), so
    # the reviewer was measuring the physical gap between PathAverage and a
    # refracted ray, not a table defect.
    #
    # The real finding underneath that confusion: the table's existence
    # boundary is QUANTISED by the range axis, so it can call "no direct
    # ray" for a range slightly inside the true, exact max_direct_range.
    # Measured gap at these points: 11-720 m depending on depth, i.e. a
    # small fraction of max_direct_range but not always tiny in absolute
    # terms (see task-3-report.md's fix round 3 for the per-depth table).
    # This is safe, not silent: RayTrace correctly falls back rather than
    # extrapolating past the table's last valid cell, and reports it via
    # last_fallback_fraction, which this test checks explicitly.
    #
    # PathAverage's own divergence from the true refracted ray in this band
    # varies by geometry -- measured 4e-4 s to ~9e-3 s across the depths
    # probed, not a flat ~1 ms. This test's three points measure 7.8e-4 s,
    # 8.0e-4 s and 1.9e-3 s; the tolerance below has real margin over that
    # for these three specific points, but is not a claim about the whole
    # table (deeper channels showed larger gaps -- see the report).
    z1_down = 30.0
    mx = max_direct_range(OREGON, z1_down, z2_down)
    r = frac * mx
    want = direct_time(OREGON, z1_down, z2_down, r)
    assert want is not None, "test range must be one the direct ray actually reaches"

    sx, sy = 0.0, 0.0
    sz = to_repo(z1_down)
    px = np.array([r])
    py = np.array([0.0])
    pz = to_repo(np.array([z2_down]))

    got = oregon_raytrace.travel_time(sx, sy, float(sz), px, py, pz)[0]
    assert oregon_raytrace.last_fallback_fraction == 1.0, \
        "this point is expected to have fallen back -- if this fires, the " \
        "quantised boundary moved and this is no longer testing the fallback band"
    assert got == pytest.approx(want, abs=3e-3)


def test_travel_time_broadcasts_a_grid_shaped_source_against_flat_picks():
    # This test exercises control flow (shapes, broadcasting), not table
    # accuracy, so it uses its own tiny (5x5) table rather than the shared
    # production-resolution fixture -- a full build costs ~23 s for no benefit
    # here (Finding 2, task-3-report.md).
    m = RayTrace(OREGON, n_z=5, n_r=5)

    ny, nx = 3, 4
    xs = np.linspace(10000.0, 40000.0, nx)
    ys = np.linspace(15000.0, 25000.0, ny)
    gx, gy = np.meshgrid(xs, ys)
    sx = gx[:, :, None]
    sy = gy[:, :, None]
    sz = -30.0

    px = np.array([12000.0, 20000.0, 35000.0, 60000.0, 25000.0])
    py = np.array([18000.0, 19000.0, 21000.0, 24000.0, 20000.0])
    pz = np.array([-80.0, -220.0, -640.0, -1500.0, -300.0])

    got = m.travel_time(sx, sy, sz, px, py, pz)
    assert got.shape == (ny, nx, px.size)

    # And it must agree with looping the scalar call, same as the other
    # models -- reusing the same instance, since the table is immutable once
    # built and last_fallback_fraction reflects whichever call ran most
    # recently (see fallback_fraction_overall for a whole-loop figure).
    expected = np.empty((ny, nx, px.size))
    for j in range(ny):
        for i in range(nx):
            expected[j, i] = m.travel_time(gx[j, i], gy[j, i], sz, px, py, pz)
    assert np.allclose(got, expected, rtol=1e-10, atol=0.0)


def test_trace_agrees_with_an_independent_quadrature_of_the_ray_equations():
    # No independent oracle for trace()/direct_time() (Task 14 review): every
    # existing table test compares against direct_time/trace themselves, so a
    # shared algebra bug in the log-ratio range/time expressions (module
    # docstring, `trace()`) would pass every test in this file. A
    # constant-gradient layer's ray equations reduce to plain scalar
    # integrals -- dx/dz = p*c(z) / sqrt(1-(p*c(z))**2), dt/dz = 1 /
    # (c(z)*sqrt(1-(p*c(z))**2)), from Snell's invariant p = cos(theta)/c(z)
    # holding pointwise -- and `scipy.integrate.quad` evaluates those with no
    # dependency on trace()'s own closed form, giving a genuinely independent
    # check of both the range and the time trace() returns.
    from scipy.integrate import quad

    z0, z1 = 0.0, 1000.0
    c0, c1 = 1500.0, 1480.0
    profile = SoundSpeedProfile(np.array([z0, z1]), np.array([c0, c1]), "const-grad")
    g = (c1 - c0) / (z1 - z0)

    def c_of(z):
        return c0 + g * (z - z0)

    theta_a = np.radians(30.0)  # grazing angle from horizontal at z0
    p = np.cos(theta_a) / c0

    def dx_dz(z):
        pc = p * c_of(z)
        return pc / np.sqrt(1.0 - pc ** 2)

    def dt_dz(z):
        cz = c_of(z)
        pc = p * cz
        return 1.0 / (cz * np.sqrt(1.0 - pc ** 2))

    want_rng, _ = quad(dx_dz, z0, z1, limit=200)
    want_t, _ = quad(dt_dz, z0, z1, limit=200)

    got_rng, got_t = trace(profile, z0, z1, p)
    assert got_rng == pytest.approx(want_rng, rel=1e-9)
    assert got_t == pytest.approx(want_t, rel=1e-9)

    # direct_time/max_direct_range must agree with the same independent
    # target once routed through the public range-inversion interface, not
    # just the internal trace() this ray parameter came from directly.
    assert max_direct_range(profile, z0, z1) > want_rng  # this ray is not yet grazing
    got_direct_t = direct_time(profile, z0, z1, want_rng)
    assert got_direct_t == pytest.approx(want_t, rel=1e-6)


def test_table_is_cached_and_not_rebuilt_for_a_repeated_source_depth():
    # Rebuilding per call would reintroduce the per-pick brentq cost this
    # design exists to avoid. Assert on the build counter, not wall-clock
    # time, which is flaky. Control flow only, so a tiny table (Finding 2).
    px = np.array([12000.0, 35000.0])
    py = np.array([18000.0, 21000.0])
    pz = np.array([-220.0, -640.0])

    m = RayTrace(OREGON, n_z=5, n_r=5)
    m.travel_time(20000.0, 19000.0, -30.0, px, py, pz)
    assert m.table_build_count == 1

    m.travel_time(25000.0, 21000.0, -30.0, px, py, pz)  # same sz, different sx/sy
    assert m.table_build_count == 1, "same source depth must reuse the cached table"

    m.travel_time(20000.0, 19000.0, -45.0, px, py, pz)  # different sz
    assert m.table_build_count == 2, "a new source depth must build a new table"


def test_last_fallback_fraction_is_biased_toward_zero_after_a_multi_call_loop():
    # Task 14 review, Important 4: a solve() loop calls travel_time many times
    # per call -- once for the whole coarse grid (mostly fallback, since most
    # of a 45km-padded grid sits beyond the profile's refraction limit), then
    # several small Nelder-Mead refinement calls, ending on a tiny few-point
    # final misfit call that is far more likely to land inside the table. A
    # single overwritten scalar therefore reports whichever call ran last --
    # here, deliberately, a call with NO fallback -- even though most of the
    # loop's queries did fall back. fallback_fraction_overall pools every
    # recorded call and is not fooled by this.
    m = RayTrace(OREGON, n_z=5, n_r=5)
    far_px = np.array([200_000.0])  # far beyond any direct-ray range: all fallback
    far_py = np.array([0.0])
    far_pz = to_repo(np.array([1500.0]))
    m.travel_time(0.0, 0.0, to_repo(30.0), far_px, far_py, far_pz)
    assert m.last_fallback_fraction == 1.0

    near_px = np.array([10.0])  # a query the table resolves directly: no fallback
    near_py = np.array([0.0])
    near_pz = to_repo(np.array([-11.0]))
    m.travel_time(0.0, 0.0, to_repo(30.0), near_px, near_py, near_pz)
    assert m.last_fallback_fraction == 0.0, \
        "the most recent call alone must not reflect the earlier all-fallback call"
    assert m.fallback_fraction_overall == pytest.approx(0.5), \
        "pooled over both recorded calls, half the queries fell back"


def test_reset_fallback_record_clears_only_earlier_calls():
    m = RayTrace(OREGON, n_z=5, n_r=5)
    far_px = np.array([200_000.0])
    far_py = np.array([0.0])
    far_pz = to_repo(np.array([1500.0]))
    m.travel_time(0.0, 0.0, to_repo(30.0), far_px, far_py, far_pz)
    assert m.fallback_fraction_overall == 1.0

    m.reset_fallback_record()
    assert m.fallback_calls == []

    near_px = np.array([10.0])
    near_py = np.array([0.0])
    near_pz = to_repo(np.array([-11.0]))
    m.travel_time(0.0, 0.0, to_repo(30.0), near_px, near_py, near_pz)
    assert m.fallback_fraction_overall == 0.0, \
        "the pre-reset all-fallback call must not still count"


def test_raytrace_raises_on_non_scalar_source_depth():
    # Finding 4: source depth keys the cached table (it is fixed per
    # localization, not part of the x/y grid search), so a source depth that
    # varied per query would need a different table per element -- reject it
    # explicitly rather than silently keying on, say, the first element.
    m = RayTrace(OREGON)
    with pytest.raises(ValueError):
        m.travel_time(0.0, 0.0, np.array([-30.0, -45.0]),
                       np.array([1000.0]), np.array([0.0]), np.array([-50.0]))


def test_tabulated_travel_time_matches_straight_line_for_a_negligible_gradient():
    # Finding 3: the flat-profile test above exercises the table's BYPASS,
    # never the interpolator itself, and the only other table check compares
    # against direct_time -- its own ground truth. This profile has a real,
    # nonzero gradient (small enough that refractive bending is negligible),
    # so RayTrace goes through the table, and the check is against plain
    # straight-line geometry instead of the tracer.
    #
    # Justifying the tolerance: over 3000 m the sound speed changes by 1 mm/s
    # (gradient ~3.3e-7 /s). Fermat's principle makes travel time stationary
    # to path perturbation, so the correction from bending is second order in
    # the gradient and is unmeasurable at this scale (path-averaging alone,
    # which is first order, would already differ from the endpoint speed by
    # only ~1e-9 relative -- see PathAverage's docstring). The tolerance below
    # is therefore attributable entirely to table interpolation error, and is
    # well under the 1e-4 s target used for the (much more sharply refracting)
    # Oregon profile.
    tiny_gradient = SoundSpeedProfile(np.array([0.0, 3000.0]), np.array([1480.0, 1480.001]),
                                       "tiny-gradient")
    assert np.ptp(tiny_gradient.c) > 0.0  # must NOT hit the constant-profile bypass

    sx, sy, sz = 0.0, 0.0, -30.0
    px = np.array([20000.0])
    py = np.array([0.0])
    pz = np.array([-800.0])

    # This test only needs a straight-line comparison, not accuracy-grade
    # table resolution -- a full-resolution build (production defaults
    # TABLE_N_Z=121, TABLE_N_R=361) costs real seconds for no benefit here
    # (Task 14 review; measured ~0.31 s at n_z=15/n_r=40 vs. several seconds
    # at the default, with the same 1e-3 s tolerance still comfortably met).
    got = RayTrace(tiny_gradient, n_z=15, n_r=40).travel_time(sx, sy, sz, px, py, pz)[0]
    want = float(np.hypot(px[0] - sx, pz[0] - sz) / 1480.0)
    assert got == pytest.approx(want, abs=1e-3)
