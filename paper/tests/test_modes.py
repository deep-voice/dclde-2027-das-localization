"""Normal-mode solver tests.

The load-bearing test is the Pekeris mode count: it compares against a closed
form, so it is a real check rather than a restatement of the implementation.
The brief allowed +-1 mode because its sketched discretisation imposed a
pressure-release bottom, which systematically drops the mode nearest cutoff.
The implementation models the halfspace explicitly instead, so the count is
EXACT -- `test_pekeris_mode_count_is_exact_across_frequency_and_depth` pins
that down, and the +-1 test below is kept as the brief wrote it.
"""

import warnings

import numpy as np
import pytest

from dclde_soundspeed.soundspeed.modes import Seabed, modes
from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile

PEKERIS_D = 200.0
PEKERIS_CW = 1500.0
PEKERIS_CB = 1800.0
FREQ = 20.0


def _pekeris_profile():
    return SoundSpeedProfile(np.array([0.0, PEKERIS_D]), np.array([PEKERIS_CW, PEKERIS_CW]),
                             "pekeris")


def _analytic_count(freq_hz, depth_m, cw, cb):
    """Trapped-mode count for a Pekeris waveguide, closed form."""
    return int(2 * freq_hz * depth_m * np.sqrt(1 / cw**2 - 1 / cb**2) + 0.5)


# --------------------------------------------------------------------------
# The brief's tests, unchanged.
# --------------------------------------------------------------------------

def test_pekeris_wavenumbers_lie_between_the_two_sound_speeds():
    # Trapped modes must have phase speed between water and seabed speed; this is
    # the defining property of a trapped mode and catches an eigensolver that has
    # picked up leaky or spurious roots.
    w = 2 * np.pi * FREQ
    ms = modes(_pekeris_profile(), Seabed(c=PEKERIS_CB, rho=1.8, alpha=0.0),
               FREQ, PEKERIS_D)
    assert ms.n_modes >= 1
    assert np.all(ms.kr < w / PEKERIS_CW)
    assert np.all(ms.kr > w / PEKERIS_CB)


def test_pekeris_mode_count_matches_the_analytic_estimate():
    # For a Pekeris waveguide the number of trapped modes is
    # floor( 2 f D sqrt(1/cw^2 - 1/cb^2) + 1/2 ) -- a closed form, so this is a
    # real check rather than a restatement of the implementation.
    expected = int(2 * FREQ * PEKERIS_D * np.sqrt(1 / PEKERIS_CW**2 - 1 / PEKERIS_CB**2) + 0.5)
    ms = modes(_pekeris_profile(), Seabed(c=PEKERIS_CB, rho=1.8, alpha=0.0), FREQ, PEKERIS_D)
    assert abs(ms.n_modes - expected) <= 1


def test_mode_shapes_vanish_at_the_pressure_release_surface():
    ms = modes(_pekeris_profile(), Seabed(c=PEKERIS_CB, rho=1.8, alpha=0.0), FREQ, PEKERIS_D)
    assert np.allclose(ms.phi[0, :], 0.0, atol=1e-9)


def test_mode_shapes_are_orthonormal_which_only_checks_the_symmetrisation():
    # NOTE ON WHAT THIS DOES NOT SHOW. `eigh_tridiagonal` returns orthonormal
    # eigenvectors of any symmetric operator, and the implementation divides by
    # sqrt(dz), so this passes for a symmetric operator that is physically
    # wrong. It checks the phi/sqrt(rho) symmetrisation bookkeeping and the
    # grid/normalisation contract -- worth checking, but near-tautological.
    # `test_pekeris_wavenumbers_match_the_analytic_dispersion_roots` and
    # `test_pekeris_mode_count_is_exact_across_frequency_and_depth` are the
    # tests that carry physical content. (Kept from the brief, renamed.)
    ms = modes(_pekeris_profile(), Seabed(c=PEKERIS_CB, rho=1.8, alpha=0.0), FREQ, PEKERIS_D)
    dz = ms.z_down[1] - ms.z_down[0]
    gram = ms.phi.T @ ms.phi * dz
    assert np.allclose(gram, np.eye(ms.n_modes), atol=5e-2)


def test_deeper_water_traps_more_modes():
    sb = Seabed(c=PEKERIS_CB, rho=1.8, alpha=0.0)
    shallow = modes(_pekeris_profile(), sb, FREQ, PEKERIS_D)
    deep_prof = SoundSpeedProfile(np.array([0.0, 800.0]), np.array([PEKERIS_CW, PEKERIS_CW]),
                                  "pekeris-deep")
    deep = modes(deep_prof, sb, FREQ, 800.0)
    assert deep.n_modes > shallow.n_modes


def test_twenty_hertz_in_shallow_water_traps_very_few_modes():
    # The regime claim in the spec: at 20 Hz in 100-500 m of water the guide is
    # 1.3-7 wavelengths deep, so only a handful of modes exist and ray theory,
    # which needs many, is marginal.
    sb = Seabed(c=1700.0, rho=1.8, alpha=0.0)
    prof = SoundSpeedProfile(np.array([0.0, 150.0]), np.array([1480.0, 1480.0]), "shelf")
    assert modes(prof, sb, 20.0, 150.0).n_modes <= 4


# --------------------------------------------------------------------------
# Task 12 will draw a physical conclusion from the mode count and the
# trapped/leaky split, so both are pinned harder than +-1.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cw,cb", [(1500.0, 1800.0), (1480.0, 1700.0), (1490.0, 2000.0)])
@pytest.mark.parametrize("depth_m", [100.0, 200.0, 500.0])
@pytest.mark.parametrize("freq_hz", [10.0, 20.0, 40.0])
def test_pekeris_mode_count_is_exact_across_frequency_and_depth(cw, cb, depth_m, freq_hz):
    # Exact, not +-1: the count is what Task 12 reasons from, and an off-by-one
    # at these mode counts (1-13) is a 10-100% error.
    prof = SoundSpeedProfile(np.array([0.0, depth_m]), np.array([cw, cw]), "pekeris")
    ms = modes(prof, Seabed(c=cb, rho=1.8, alpha=0.0), freq_hz, depth_m)
    assert ms.n_modes == _analytic_count(freq_hz, depth_m, cw, cb)


def test_pekeris_wavenumbers_match_the_analytic_dispersion_roots():
    # The closed-form count only constrains how many roots there are. This pins
    # WHERE they are, against tan(kz D) = -(rho_b/rho_w) kz / gamma solved
    # independently by bisection -- so a plausible-but-wrong operator (a
    # mishandled density jump, say) cannot slip through on the count alone.
    from scipy.optimize import brentq

    cw, cb, rho_b, depth_m, freq_hz = 1500.0, 1800.0, 1.8, 200.0, 25.0
    w = 2 * np.pi * freq_hz

    def residual(kr):
        kz = np.sqrt(max((w / cw) ** 2 - kr**2, 0.0))
        gamma = np.sqrt(max(kr**2 - (w / cb) ** 2, 0.0))
        return np.cos(kz * depth_m) * kz + gamma * np.sin(kz * depth_m) / rho_b

    grid = np.linspace(w / cb + 1e-12, w / cw - 1e-12, 200_001)
    vals = np.array([residual(k) for k in grid])
    roots = np.array([brentq(residual, grid[i], grid[i + 1], xtol=1e-14)
                      for i in np.where(np.diff(np.sign(vals)) != 0)[0]])
    roots = np.sort(roots)[::-1]

    prof = SoundSpeedProfile(np.array([0.0, depth_m]), np.array([cw, cw]), "pekeris")
    ms = modes(prof, Seabed(c=cb, rho=rho_b, alpha=0.0), freq_hz, depth_m)
    assert ms.n_modes == roots.size
    assert np.allclose(ms.kr, roots, atol=1e-6)


def test_modes_are_ordered_with_mode_one_first_and_are_distinct():
    ms = modes(_pekeris_profile(), Seabed(c=PEKERIS_CB, rho=1.8, alpha=0.0), FREQ, PEKERIS_D)
    assert np.all(np.diff(ms.kr) < 0)


def test_mode_m_has_m_minus_one_zero_crossings_in_the_water():
    # Sturm oscillation theorem. Independent of the eigenvalues, so it catches a
    # duplicated or spurious root that the count test alone would not.
    prof = SoundSpeedProfile(np.array([0.0, 500.0]), np.array([1500.0, 1500.0]), "pekeris")
    ms = modes(prof, Seabed(c=1800.0, rho=1.8, alpha=0.0), 25.0, 500.0)
    assert ms.n_modes >= 5
    for m in range(ms.n_modes):
        signs = np.sign(ms.phi[ms.in_water, m])
        signs = signs[signs != 0]
        assert int(np.sum(np.diff(signs) != 0)) == m


def test_trapped_modes_decay_into_the_seabed_and_are_alive_at_the_seafloor():
    # The distinction the whole module exists to make. A mode that radiated into
    # the halfspace would not decay; a pressure-release bottom (the brief's
    # sketch) would zero the mode exactly where the DAS fibre lies.
    ms = modes(_pekeris_profile(), Seabed(c=PEKERIS_CB, rho=1.8, alpha=0.0), FREQ, PEKERIS_D)
    assert ms.seabed_extension_m > 0
    seafloor = np.searchsorted(ms.z_down, PEKERIS_D)
    peak = np.max(np.abs(ms.phi), axis=0)
    assert np.all(np.abs(ms.phi[seafloor, :]) > 1e-3 * peak)
    assert np.all(np.abs(ms.phi[-1, :]) < 1e-6 * peak)
    below = np.abs(ms.phi[seafloor:, :])
    assert np.all(np.diff(below, axis=0) <= 1e-12)


# --------------------------------------------------------------------------
# Near the trapping threshold. A truncated seabed fakes BOTH a missing mode and
# a confident zero, and a zero count is affirmative evidence for the shadow
# zone -- so this is the regime where a numerical artefact turns straight into
# a wrong physical conclusion.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cb,depth_m,freq_hz", [
    (1700.0, 30.0, 25.0),    # threshold argument 0.503: found reporting 0 modes
    (1700.0, 50.0, 15.0),    # threshold argument 0.503: found reporting 0 modes
    (1700.0, 112.02, 20.0),  # threshold argument 1.502: found reporting 1 of 2
    (1700.0, 186.59, 20.0),  # threshold argument 2.502: found reporting 2 of 3
])
def test_a_mode_just_inside_the_trapping_threshold_is_not_lost(cb, depth_m, freq_hz):
    # Each of these sits a few thousandths above a half-integer threshold, so a
    # mode exists but with an evanescent tail far longer than the water column.
    cw = 1477.0
    prof = SoundSpeedProfile(np.array([0.0, depth_m]), np.array([cw, cw]), "near-cutoff")
    ms = modes(prof, Seabed(c=cb, rho=1.8), freq_hz, depth_m)
    expected = _analytic_count(freq_hz, depth_m, cw, cb)
    assert expected >= 1
    assert ms.n_modes_expected == expected
    assert ms.n_modes == expected


def test_the_mode_count_holds_just_above_every_threshold_it_can_reach():
    # Sweep the threshold argument across the first three mode thresholds from
    # just-above to comfortably-above. The count must track the closed form the
    # whole way; the old count-stability loop broke at the +0.002 offsets.
    cw, cb, freq_hz = 1477.0, 1700.0, 20.0
    k = np.sqrt(1 / cw**2 - 1 / cb**2)
    checked = 0
    for threshold in (0.5, 1.5, 2.5):
        for offset in (0.002, 0.01, 0.05, 0.2):
            depth_m = (threshold + offset) / (2 * freq_hz * k)
            prof = SoundSpeedProfile(np.array([0.0, depth_m]), np.array([cw, cw]), "near-cutoff")
            ms = modes(prof, Seabed(c=cb, rho=1.8), freq_hz, depth_m)
            assert ms.n_modes_expected == _analytic_count(freq_hz, depth_m, cw, cb), depth_m
            assert ms.n_modes == ms.n_modes_expected, depth_m
            checked += 1
    assert checked == 12


def test_the_mode_count_holds_just_below_every_threshold_too():
    # Approaching each threshold from BELOW. Everything else in this file
    # approaches from above, which cannot catch a SPURIOUS mode appearing just
    # before its threshold -- the overcount direction. Just below threshold m
    # there must be exactly m-1 modes, not m.
    cw, cb, freq_hz = 1477.0, 1700.0, 20.0
    k = np.sqrt(1 / cw**2 - 1 / cb**2)
    checked = 0
    for threshold, count_below in ((0.5, 0), (1.5, 1), (2.5, 2)):
        for offset in (0.002, 0.01, 0.05, 0.2):
            depth_m = (threshold - offset) / (2 * freq_hz * k)
            prof = SoundSpeedProfile(np.array([0.0, depth_m]), np.array([cw, cw]), "near-cutoff")
            ms = modes(prof, Seabed(c=cb, rho=1.8), freq_hz, depth_m)
            assert ms.n_modes_expected == count_below, (depth_m, threshold)
            assert ms.n_modes_expected == _analytic_count(freq_hz, depth_m, cw, cb)
            assert ms.n_modes == ms.n_modes_expected, depth_m
            checked += 1
    assert checked == 12


def test_a_count_below_threshold_that_is_zero_is_still_certified():
    # The just-below-first-threshold case is a certified empty result: no
    # trapped mode exists at all, and that must be converged, not a shrug.
    cw, cb, freq_hz = 1477.0, 1700.0, 20.0
    k = np.sqrt(1 / cw**2 - 1 / cb**2)
    depth_m = 0.498 / (2 * freq_hz * k)
    prof = SoundSpeedProfile(np.array([0.0, depth_m]), np.array([cw, cw]), "just-below-cutoff")
    ms = modes(prof, Seabed(c=cb, rho=1.8), freq_hz, depth_m)
    assert ms.n_modes_expected == 0
    assert ms.n_modes == 0
    assert ms.converged


def test_more_roots_than_the_cutoff_count_allows_is_warned_and_never_converged(monkeypatch):
    # The overcount branch is prospective -- no real configuration has produced
    # it in ~2200 tested cases -- so it is exercised by understating the cutoff
    # count. A spurious root must not be able to inflate n_modes silently.
    # NB: the package re-exports the `modes` FUNCTION, which shadows the
    # same-named submodule as an attribute, so reach it through sys.modules.
    import sys

    mod = sys.modules["dclde_soundspeed.soundspeed.modes"]
    real = mod._threshold_count
    monkeypatch.setattr(mod, "_threshold_count",
                        lambda *a, **k: max(real(*a, **k) - 1, 0))
    with pytest.warns(UserWarning, match="cutoff-limit count"):
        ms = modes(_pekeris_profile(), Seabed(c=PEKERIS_CB, rho=1.8), FREQ, PEKERIS_D)
    assert not ms.converged
    assert ms.n_modes > ms.n_modes_expected


def test_converged_false_near_threshold_still_means_the_count_is_exact():
    # The docstring contract a Task 12 implementer will rely on: converged=False
    # with n_modes == n_modes_expected means only the SHAPES are under-resolved,
    # and the count is still trustworthy. Discarding these counts would bias
    # away the marginal regime the shadow-zone question lives in.
    cw, cb, freq_hz = 1477.0, 1700.0, 20.0
    k = np.sqrt(1 / cw**2 - 1 / cb**2)
    saw_unconverged = False
    for threshold in (0.5, 1.5, 2.5):
        depth_m = (threshold + 0.002) / (2 * freq_hz * k)
        prof = SoundSpeedProfile(np.array([0.0, depth_m]), np.array([cw, cw]), "near-cutoff")
        ms = modes(prof, Seabed(c=cb, rho=1.8), freq_hz, depth_m)
        assert ms.n_modes == ms.n_modes_expected == _analytic_count(freq_hz, depth_m, cw, cb)
        saw_unconverged |= not ms.converged
    assert saw_unconverged, "expected the near-threshold cases to report unresolved tails"


def test_an_empty_result_is_certified_rather_than_assumed():
    # The distinction the `converged` flag exists to make. A genuinely
    # untrapping waveguide reports zero AND converged; a zero that came from
    # the solver running out of grid must not look the same.
    ms = modes(_pekeris_profile(), Seabed(c=1400.0, rho=1.8), FREQ, PEKERIS_D)
    assert ms.n_modes == 0
    assert ms.n_modes_expected == 0
    assert ms.converged


def test_the_expected_count_never_undershoots_the_resolved_count():
    # n_modes_expected is the truth the growth loop aims at; resolving more
    # shapes than exist would mean spurious roots.
    for depth_m, freq_hz in [(200.0, 20.0), (800.0, 20.0), (500.0, 25.0), (150.0, 10.0)]:
        prof = SoundSpeedProfile(np.array([0.0, depth_m]), np.array([1500.0, 1500.0]), "p")
        ms = modes(prof, Seabed(c=1800.0, rho=1.8), freq_hz, depth_m)
        assert ms.n_modes == ms.n_modes_expected
        assert ms.converged


def test_a_profile_shallower_than_the_water_column_warns_about_clamping():
    # c_at clamps rather than extrapolating, so a 0-500 m profile under 1575 m
    # of water silently makes the lower 1075 m isovelocity. A docstring will not
    # stop Task 12 reading that mode count as physics.
    z = np.linspace(0.0, 500.0, 51)
    prof = SoundSpeedProfile(z, 1491.0 - 14.0 * z / 500.0, "oregon-november-like")
    with pytest.warns(UserWarning, match="CLAMPED"):
        modes(prof, Seabed(c=1700.0, rho=1.8), 20.0, 1575.0)


def test_a_profile_covering_the_water_column_does_not_warn():
    z = np.linspace(0.0, 500.0, 51)
    prof = SoundSpeedProfile(z, 1491.0 - 14.0 * z / 500.0, "oregon-november-like")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        modes(prof, Seabed(c=1700.0, rho=1.8), 20.0, 500.0)


def test_a_seabed_slower_than_the_water_traps_nothing():
    # Physically there is no trapping without a faster halfspace to turn the
    # energy back. This must be an empty answer, not a crash.
    ms = modes(_pekeris_profile(), Seabed(c=1400.0, rho=1.8, alpha=0.0), FREQ, PEKERIS_D)
    assert ms.n_modes == 0
    assert ms.phi.shape[1] == 0
    assert ms.kr.size == 0
    assert ms.converged


def test_seabed_attenuation_defaults_to_zero():
    # A non-zero default would read as "attenuation is accounted for" to a
    # caller who never thought about it. modes() does not use it at all.
    assert Seabed().alpha == 0.0


def test_seabed_attenuation_does_not_change_the_real_eigenproblem():
    # Documented behaviour, asserted so nobody later assumes alpha was used.
    quiet = modes(_pekeris_profile(), Seabed(c=PEKERIS_CB, rho=1.8, alpha=0.0), FREQ, PEKERIS_D)
    lossy = modes(_pekeris_profile(), Seabed(c=PEKERIS_CB, rho=1.8, alpha=0.9), FREQ, PEKERIS_D)
    assert np.array_equal(quiet.kr, lossy.kr)


def test_a_downward_refracting_profile_still_traps_modes_at_twenty_hertz():
    # The Oregon November case that motivates the module: c falls 1491 -> 1477
    # over 0-500 m, which is what ray theory turns into a shadow zone. The claim
    # under test here is only the weak one -- that trapped modes exist at all.
    z = np.linspace(0.0, 500.0, 51)
    c = 1491.0 - 14.0 * z / 500.0
    prof = SoundSpeedProfile(z, c, "oregon-november-like")
    ms = modes(prof, Seabed(c=1700.0, rho=1.8, alpha=0.1), 20.0, 500.0)
    assert ms.n_modes >= 1
    w = 2 * np.pi * 20.0
    assert np.all(ms.kr > w / 1700.0)
    assert np.all(ms.kr < w / c.min())


def test_the_grid_resolves_the_water_column_and_the_seabed_at_one_spacing():
    ms = modes(_pekeris_profile(), Seabed(c=PEKERIS_CB, rho=1.8, alpha=0.0), FREQ, PEKERIS_D)
    steps = np.diff(ms.z_down)
    assert np.allclose(steps, steps[0])
    assert ms.z_down[-1] > PEKERIS_D
    assert ms.phi.shape == (ms.z_down.size, ms.n_modes)


@pytest.mark.parametrize("kwargs,match", [
    (dict(freq_hz=0.0), "freq_hz must be positive"),
    (dict(freq_hz=-1.0), "freq_hz must be positive"),
    (dict(water_depth_m=0.0), "water_depth_m must be positive"),
    (dict(nz=1), "nz must be at least 2"),
    (dict(nz=4), "grid too coarse"),
])
def test_unusable_inputs_raise_rather_than_returning_a_wrong_mode_count(kwargs, match):
    call = dict(profile=_pekeris_profile(), seabed=Seabed(c=PEKERIS_CB, rho=1.8, alpha=0.0),
                freq_hz=FREQ, water_depth_m=PEKERIS_D)
    nz = kwargs.pop("nz", None)
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        modes(**call) if nz is None else modes(**call, nz=nz)


@pytest.mark.parametrize("kwargs", [dict(c=0.0), dict(c=-1.0), dict(rho=0.0), dict(alpha=-0.1)])
def test_an_unphysical_seabed_raises(kwargs):
    with pytest.raises(ValueError):
        Seabed(**{**dict(c=1700.0, rho=1.8, alpha=0.1), **kwargs})
