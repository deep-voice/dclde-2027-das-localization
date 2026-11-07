import numpy as np
import pytest

from dclde_localization.physics.forward import DEFAULT_C
from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile, to_down, to_repo

OREGON_Z = np.array([0.0, 100.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0])
OREGON_C = np.array([1491.0, 1480.0, 1478.0, 1477.0, 1479.0, 1484.0, 1490.0])


def test_sign_conversion_round_trips_and_flips():
    # The repo frame is negative-down; the profile is positive-down. A silent
    # flip here would corrupt every downstream number without raising.
    pz = np.array([11.0, -30.0, -589.0, -1575.0])
    assert np.allclose(to_down(pz), [-11.0, 30.0, 589.0, 1575.0])
    assert np.allclose(to_repo(to_down(pz)), pz)


def test_constant_profile_path_average_is_that_constant():
    p = SoundSpeedProfile.constant(DEFAULT_C)
    assert p.path_average(30.0, 1575.0) == pytest.approx(DEFAULT_C, rel=1e-12)
    assert p.path_average(500.0, 500.0) == pytest.approx(DEFAULT_C, rel=1e-12)


def test_path_average_is_harmonic_not_arithmetic():
    # Travel time is distance / c, so the average that preserves travel time is
    # the mean of SLOWNESS. With a strong gradient the two differ measurably and
    # using the arithmetic mean would bias every range.
    p = SoundSpeedProfile(np.array([0.0, 1000.0]), np.array([1400.0, 1600.0]), "synthetic")
    harmonic = p.path_average(0.0, 1000.0)
    arithmetic = 1500.0
    assert harmonic < arithmetic
    assert harmonic == pytest.approx(1000.0 / np.trapezoid(
        1.0 / np.linspace(1400.0, 1600.0, 20001), np.linspace(0.0, 1000.0, 20001)), rel=1e-6)


def test_path_average_is_symmetric_in_its_endpoints():
    p = SoundSpeedProfile(OREGON_Z, OREGON_C, "oregon-nov")
    assert p.path_average(30.0, 1200.0) == pytest.approx(p.path_average(1200.0, 30.0), rel=1e-15)


def test_c_at_clamps_outside_the_profile_rather_than_extrapolating():
    # Extrapolating a sound-speed profile past its deepest sample invents data.
    # Clamping is wrong too, but it is wrong in a bounded, visible way.
    p = SoundSpeedProfile(OREGON_Z, OREGON_C, "oregon-nov")
    assert p.c_at(5000.0) == pytest.approx(OREGON_C[-1])
    assert p.c_at(-10.0) == pytest.approx(OREGON_C[0])


def test_from_ts_reproduces_a_known_sound_speed():
    # TEOS-10 at S=35 psu, T=10 C, ~1000 dbar is close to 1500 m/s. This is a
    # coarse anchor: its job is to catch swapped T/S arguments, which would give
    # a wildly wrong answer, not to validate gsw.
    p = SoundSpeedProfile.from_ts(
        z_down=np.array([1000.0]), temperature_c=np.array([10.0]),
        salinity_psu=np.array([35.0]), lat=45.0, lon=-125.0, source="anchor")
    assert 1490.0 < float(p.c[0]) < 1520.0


def test_rejects_unsorted_or_mismatched_input():
    with pytest.raises(ValueError, match="ascending"):
        SoundSpeedProfile(np.array([100.0, 0.0]), np.array([1480.0, 1490.0]), "bad")
    with pytest.raises(ValueError, match="same length"):
        SoundSpeedProfile(np.array([0.0, 100.0]), np.array([1480.0]), "bad")


# --- Regression tests added after code review (Findings 1-3) --------------------


def test_path_average_treats_out_of_range_depth_as_travelling_at_the_clamped_speed():
    # Finding 1: an endpoint beyond the sampled range must be treated as
    # travelled at the clamped boundary speed (matching c_at), not as if it
    # were covered at infinite speed. Independent ground truth: split the
    # path into an out-of-range leg (constant boundary speed) and an in-range
    # leg (a fine independent trapezoid quadrature, not the class's own cache).
    p = SoundSpeedProfile(OREGON_Z, OREGON_C, "oregon-nov")
    z_mid = np.linspace(0.0, 2000.0, 20001)
    c_mid = np.interp(z_mid, OREGON_Z, OREGON_C)
    in_range_time = np.trapezoid(1.0 / c_mid, z_mid)
    out_of_range_time = (5000.0 - 2000.0) / OREGON_C[-1]
    expected = 5000.0 / (in_range_time + out_of_range_time)
    assert p.path_average(0.0, 5000.0) == pytest.approx(expected, rel=1e-6)


def test_path_average_out_of_range_on_the_instrument_shaped_profile():
    # The shape we will actually have: a single in-situ instrument spanning
    # 97-575 m, queried across a cable spanning far outside that range. This
    # is the exact case that regressed to 4778.83 m/s (should be ~1477.6).
    p = SoundSpeedProfile(np.array([97.0, 575.0]), np.array([1480.0, 1477.0]), "instrument")
    z_mid = np.linspace(97.0, 575.0, 20001)
    c_mid = np.interp(z_mid, [97.0, 575.0], [1480.0, 1477.0])
    in_range_time = np.trapezoid(1.0 / c_mid, z_mid)
    total_time = (97.0 - 30.0) / 1480.0 + in_range_time + (1575.0 - 575.0) / 1477.0
    expected = 1545.0 / total_time
    result = p.path_average(30.0, 1575.0)
    assert result == pytest.approx(expected, rel=1e-6)
    # Sanity bound: it must be a physically plausible sound speed, not 3x reality.
    assert 1470.0 < result < 1490.0


def test_path_average_fully_outside_the_profile_is_exactly_the_boundary_speed():
    p = SoundSpeedProfile(OREGON_Z, OREGON_C, "oregon-nov")
    # Both endpoints shallower than the shallowest sample (z=0).
    assert p.path_average(-100.0, -50.0) == pytest.approx(OREGON_C[0], rel=1e-9)
    # Both endpoints deeper than the deepest sample (z=2000).
    assert p.path_average(3000.0, 5000.0) == pytest.approx(OREGON_C[-1], rel=1e-9)


def test_path_average_rejects_a_singleton_profile():
    # Finding 2: a one-sample profile is coherent for c_at (it clamps to that
    # one value everywhere) but defines no path, so path_average must raise
    # rather than silently return inf (division by a zero cumulative-slowness
    # span).
    p = SoundSpeedProfile(np.array([1000.0]), np.array([1500.0]), "single")
    with pytest.raises(ValueError, match="at least two"):
        p.path_average(0.0, 500.0)


def test_path_average_is_vectorised_over_array_endpoints():
    # Finding 3: the localization search calls path_average once per pick per
    # grid cell, so it must accept array endpoints and return a matching shape.
    p = SoundSpeedProfile(OREGON_Z, OREGON_C, "oregon-nov")
    z1 = np.array([0.0, 100.0, 500.0])
    z2 = np.array([1000.0, 1500.0, 500.0])
    result = p.path_average(z1, z2)
    assert isinstance(result, np.ndarray)
    assert result.shape == z1.shape
    expected = np.array([p.path_average(float(a), float(b)) for a, b in zip(z1, z2)])
    assert np.allclose(result, expected, rtol=1e-12)


def test_path_average_broadcasts_a_scalar_against_an_array():
    p = SoundSpeedProfile(OREGON_Z, OREGON_C, "oregon-nov")
    z2 = np.array([200.0, 800.0, 1900.0])
    result = p.path_average(30.0, z2)
    assert isinstance(result, np.ndarray)
    assert result.shape == z2.shape
    expected = np.array([p.path_average(30.0, float(z)) for z in z2])
    assert np.allclose(result, expected, rtol=1e-12)


def test_path_average_returns_a_plain_float_for_scalar_inputs():
    p = SoundSpeedProfile(OREGON_Z, OREGON_C, "oregon-nov")
    result = p.path_average(30.0, 1200.0)
    assert isinstance(result, float)
    assert not isinstance(result, np.ndarray)
