import numpy as np
import pytest

from dclde_localization.physics import forward
from dclde_localization.physics.forward import DEFAULT_C
from dclde_localization.physics.sound_speed_models import ConstantC, PathAverage
from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile

PX = np.array([12000.0, 20000.0, 35000.0, 60000.0])
PY = np.array([18000.0, 19000.0, 21000.0, 24000.0])
PZ = np.array([-80.0, -220.0, -640.0, -1500.0])
SRC = (37000.0, 22000.0, -30.0)

OREGON_PROFILE = SoundSpeedProfile(
    np.array([0.0, 100.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0]),
    np.array([1491.0, 1480.0, 1478.0, 1477.0, 1479.0, 1484.0, 1490.0]),
    "oregon-nov")


def _independent_straight_line_time(sx, sy, sz, px, py, pz, c):
    # Deliberately NOT forward.travel_time (or anything that calls it): the
    # point of this helper is an arithmetic path independent of both
    # ConstantC's implementation and forward.py's, so the two geometry tests
    # below check real physics instead of "f(x) == f(x)" (Task 14 review --
    # an earlier version of this file called forward.travel_time as both
    # `got` and `want` here, since ConstantC.travel_time delegates to it
    # directly, and so could never fail regardless of what either
    # implementation computed). The delegation contract itself -- that
    # ConstantC really does call forward.travel_time, not a parallel
    # reimplementation that could drift from the frozen module -- is a
    # separate, legitimate thing to test and is checked explicitly below,
    # by name, against forward.travel_time directly.
    dx = np.asarray(px, float) - sx
    dy = np.asarray(py, float) - sy
    dz = np.asarray(pz, float) - sz
    return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) / c


def test_constant_c_matches_independent_straight_line_geometry():
    got = ConstantC(DEFAULT_C).travel_time(*SRC, PX, PY, PZ)
    want = _independent_straight_line_time(*SRC, PX, PY, PZ, DEFAULT_C)
    assert np.allclose(got, want, rtol=1e-12, atol=0.0)


def test_constant_c_matches_independent_geometry_at_a_non_default_speed():
    got = ConstantC(1440.0).travel_time(*SRC, PX, PY, PZ)
    want = _independent_straight_line_time(*SRC, PX, PY, PZ, 1440.0)
    assert np.allclose(got, want, rtol=1e-12, atol=0.0)


def test_constant_c_delegates_exactly_to_the_frozen_forward_model():
    # This is the contract that keeps one definition of the physics: ConstantC
    # must not reimplement travel-time arithmetic of its own, even one that
    # happens to agree with forward.py to the last bit -- it must call
    # forward.travel_time itself, so the two can never independently drift.
    # Checked with exact equality (not allclose): both sides run the identical
    # frozen numpy expression, so anything less than bit-for-bit agreement
    # would mean ConstantC stopped delegating. If this ever fails, either
    # forward.py changed (it is frozen -- see git diff --exit-code
    # src/dclde_localization/physics/forward.py) or ConstantC drifted from it.
    for c in (DEFAULT_C, 1440.0):
        got = ConstantC(c).travel_time(*SRC, PX, PY, PZ)
        want = forward.travel_time(*SRC, PX, PY, PZ, c)
        assert np.array_equal(got, want)


def test_path_average_with_a_constant_profile_reduces_to_constant_c():
    # PathAverage adds a depth-dependent speed; with no depth dependence it must
    # collapse exactly onto the null model, or comparisons between them are
    # measuring the implementation rather than the physics.
    pa = PathAverage(SoundSpeedProfile.constant(DEFAULT_C))
    got = pa.travel_time(*SRC, PX, PY, PZ)
    want = ConstantC(DEFAULT_C).travel_time(*SRC, PX, PY, PZ)
    assert np.allclose(got, want, rtol=1e-12, atol=0.0)


def test_path_average_differs_per_channel_under_a_real_profile():
    # The whole point: c_eff depends on BOTH endpoint depths, so two channels at
    # different depths get different effective speeds from the same source.
    prof = SoundSpeedProfile(
        np.array([0.0, 100.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0]),
        np.array([1491.0, 1480.0, 1478.0, 1477.0, 1479.0, 1484.0, 1490.0]),
        "oregon-nov")
    tt = PathAverage(prof).travel_time(*SRC, PX, PY, PZ)
    ref = ConstantC(DEFAULT_C).travel_time(*SRC, PX, PY, PZ)
    delta = tt - ref
    assert np.ptp(delta) > 1e-3, "effective speed should vary across channel depths"


def test_path_average_depends_on_source_depth():
    # c_eff(z_s, z_c) cannot be precomputed per channel. This test is why the
    # interface takes positions rather than an array of speeds.
    prof = SoundSpeedProfile(np.array([0.0, 1000.0]), np.array([1400.0, 1600.0]), "steep")
    pa = PathAverage(prof)
    shallow = pa.travel_time(37000.0, 22000.0, -10.0, PX, PY, PZ)
    deep = pa.travel_time(37000.0, 22000.0, -400.0, PX, PY, PZ)
    geom_only = (forward.travel_time(37000.0, 22000.0, -400.0, PX, PY, PZ, DEFAULT_C)
                 - forward.travel_time(37000.0, 22000.0, -10.0, PX, PY, PZ, DEFAULT_C))
    assert not np.allclose(deep - shallow, geom_only, rtol=1e-6)


def test_names_identify_the_model_and_its_profile():
    assert ConstantC(1480.0).name == "constant_c:1480"
    assert PathAverage(SoundSpeedProfile.constant(1480.0)).name == "path_average:constant:1480"


# --- Ruling R4: models must accept broadcastable array source coordinates. ---
# A later task evaluates the misfit over a full grid of candidate source
# positions (e.g. 161x161). A scalar-only travel_time contract would force a
# Python-level loop per grid cell per pick, which does not terminate in usable
# time. These two tests pin the broadcasting contract for both models.

def test_travel_time_broadcasts_over_a_grid_of_candidate_sources():
    ny, nx = 4, 5
    xs = np.linspace(30000.0, 40000.0, nx)
    ys = np.linspace(18000.0, 24000.0, ny)
    gx, gy = np.meshgrid(xs, ys)  # each (ny, nx)
    sx = gx[:, :, None]
    sy = gy[:, :, None]
    sz = -30.0

    for model in (ConstantC(DEFAULT_C), PathAverage(OREGON_PROFILE)):
        got = model.travel_time(sx, sy, sz, PX, PY, PZ)
        assert got.shape == (ny, nx, PX.size)


def test_grid_broadcast_agrees_with_looping_the_scalar_call():
    ny, nx = 4, 5
    xs = np.linspace(30000.0, 40000.0, nx)
    ys = np.linspace(18000.0, 24000.0, ny)
    gx, gy = np.meshgrid(xs, ys)  # each (ny, nx)
    sx = gx[:, :, None]
    sy = gy[:, :, None]
    sz = -30.0

    for model in (ConstantC(DEFAULT_C), PathAverage(OREGON_PROFILE)):
        got = model.travel_time(sx, sy, sz, PX, PY, PZ)

        expected = np.empty((ny, nx, PX.size))
        for j in range(ny):
            for i in range(nx):
                expected[j, i] = model.travel_time(gx[j, i], gy[j, i], sz, PX, PY, PZ)

        assert np.allclose(got, expected, rtol=1e-12, atol=0.0)
