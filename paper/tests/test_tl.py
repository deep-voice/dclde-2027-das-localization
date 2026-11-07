import numpy as np
import pytest

from dclde_soundspeed.soundspeed.modes import Seabed, modes
from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile
from dclde_localization.physics.bathymetry import PathProfile
from dclde_soundspeed.soundspeed.tl import (
    adiabatic_tl,
    alpha_nepers_per_m,
    excitation,
    modal_attenuation,
    mode_sum_tl,
)

C_WATER = 1480.0


def isovelocity(depth_m=1000.0, c=C_WATER, n=64):
    z = np.linspace(0.0, depth_m, n)
    return SoundSpeedProfile(z_down=z, c=np.full(n, c), source="test isovelocity")


def test_db_per_wavelength_converts_to_nepers_per_metre():
    # 1 dB/wavelength at 20 Hz in a 1700 m/s sediment: wavelength 85 m,
    # so 1/85 dB/m, and a neper is 20/ln(10) = 8.685889... dB.
    got = alpha_nepers_per_m(1.0, 20.0, 1700.0)
    expected = (1.0 * 20.0 / 1700.0) / (20.0 / np.log(10.0))
    assert got == pytest.approx(expected, rel=1e-12)
    # (20/1700) / (20/ln(10)) = 0.00135446..., not 0.0013540 -- the brief's
    # hand-rounded literal has a typo in the 5th significant figure. Corrected
    # to the value the formula above actually produces.
    assert got == pytest.approx(0.0013544618, abs=1e-7)


def test_a_second_independent_hand_computed_literal_pins_the_same_constant():
    # A second, differently-chosen (alpha, freq, c) so the DB_PER_NEPER
    # constant is pinned by more than the one literal above -- otherwise
    # every other test in this file would pass for any wrong constant, since
    # they only restate or scale the implementation's own expression.
    #
    # alpha=0.5 dB/wavelength, freq=50 Hz, c=1600 m/s.
    # dB/m = alpha * freq / c = 0.5 * 50 / 1600 = 25 / 1600 = 0.015625
    # 1 neper = 20/ln(10) dB = 8.685889638... dB
    # nepers/m = 0.015625 / 8.685889638... = 0.0017988946...
    got = alpha_nepers_per_m(0.5, 50.0, 1600.0)
    assert got == pytest.approx(0.0017988946, abs=1e-9)


def test_conversion_scales_linearly_in_alpha_and_frequency():
    a = alpha_nepers_per_m(0.2, 20.0, 1700.0)
    assert alpha_nepers_per_m(0.4, 20.0, 1700.0) == pytest.approx(2 * a)
    assert alpha_nepers_per_m(0.2, 40.0, 1700.0) == pytest.approx(2 * a)


def test_a_lossless_seabed_gives_exactly_zero_attenuation_for_every_mode():
    ms = modes(isovelocity(), Seabed(c=1700.0, rho=1.8, alpha=0.0), 20.0, 1000.0)
    assert ms.n_modes > 0, "need trapped modes for this test to mean anything"
    att = modal_attenuation(ms, Seabed(c=1700.0, rho=1.8, alpha=0.0))
    assert att.shape == (ms.n_modes,)
    assert np.all(att == 0.0), "a non-attenuating seabed must not acquire loss"


def test_attenuation_is_positive_and_linear_in_alpha():
    ms = modes(isovelocity(), Seabed(c=1700.0, rho=1.8, alpha=0.15), 20.0, 1000.0)
    a1 = modal_attenuation(ms, Seabed(c=1700.0, rho=1.8, alpha=0.15))
    a2 = modal_attenuation(ms, Seabed(c=1700.0, rho=1.8, alpha=0.30))
    assert np.all(a1 > 0)
    # The perturbation is first-order, so doubling alpha doubles the loss.
    assert np.allclose(a2, 2 * a1, rtol=1e-10)


def test_higher_modes_attenuate_more_because_they_penetrate_the_sediment_further():
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    ms = modes(isovelocity(), seabed, 20.0, 1000.0)
    if ms.n_modes < 2:
        pytest.skip("needs at least two trapped modes")
    att = modal_attenuation(ms, seabed)
    # Mode 1 is the shallowest-grazing (largest kr) and the best trapped.
    assert att[0] < att[-1]


def test_matches_an_independently_evaluated_closed_form_on_the_sediment_tail():
    """Independent oracle.

    In the sediment a trapped mode is phi(z) = A*exp(-gamma*(z-D)), with
    gamma = sqrt(kr^2 - (omega/c_sed)^2). The perturbation integral over the
    tail then has the closed form

        (omega / kr) * (alpha_n / c_sed) * A^2 / (2*gamma)

    which is evaluated here from the mode's own kr and its amplitude at the
    seafloor, independently of how `modal_attenuation` does its quadrature.

    Reading A off the grid: `ModeSet.phi` is the density-REDUCED shape
    u = phi_pressure / sqrt(rho), not the physical pressure mode. rho jumps
    from 1 (water) to `seabed.rho` at z = D, so u itself is discontinuous
    there by a factor sqrt(seabed.rho) even though the physical pressure mode
    is continuous. The exponential-tail ansatz above describes u INSIDE the
    sediment, so A must be read from a sediment-side node and extrapolated
    back to D -- never from the water side, and never from `argmin(|z - D|)`,
    which ties (the seafloor sits exactly midway between the two nearest grid
    points by construction of `modes.py`'s half-point placement) and resolves
    to the water-side node. Using that node instead of the sediment-side one
    was verified numerically to inflate A by ~sqrt(seabed.rho) = 1.34x here,
    i.e. amp**2 by ~1.8x -- a ~45% error, not a quadrature bug.
    """
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    depth = 1000.0
    ms = modes(isovelocity(depth_m=depth), seabed, 20.0, depth)
    assert ms.n_modes > 0
    got = modal_attenuation(ms, seabed)

    omega = 2 * np.pi * ms.freq_hz
    a_n = alpha_nepers_per_m(seabed.alpha, ms.freq_hz, seabed.c)
    sed_idx = np.where(~ms.in_water)[0][0]
    z_sed = ms.z_down[sed_idx]
    expected = []
    for m in range(ms.n_modes):
        kr = ms.kr[m]
        gamma = np.sqrt(kr**2 - (omega / seabed.c) ** 2)
        # Extrapolate the sediment-side node back to the true interface z = D.
        amp = ms.phi[sed_idx, m] * np.exp(gamma * (z_sed - depth))
        expected.append((omega / kr) * (a_n / seabed.c) * amp**2 / (2 * gamma))

    # Achieved agreement is ~5e-5 (ratios 0.99994-0.99998 across modes), so
    # rtol=0.05 was only a sanity check. Tightened to 1e-3 -- a real
    # regression guard -- while staying ~20x looser than what is actually
    # achieved, so it will not be flaky. Not tighter than 1e-3 because the
    # mode shapes themselves come from a second-order finite-difference
    # discretisation (`modes.py`), which leaves a residual of that rough
    # order between the discrete quadrature and the continuous closed form.
    assert np.allclose(got, expected, rtol=1e-3), (
        f"numeric quadrature {got} disagrees with the closed form {expected}")


def test_a_seabed_that_does_not_match_the_modes_raises():
    modes_seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    ms = modes(isovelocity(), modes_seabed, 20.0, 1000.0)
    assert ms.n_modes > 0

    # A much SLOWER seabed than the one used to compute `ms`: every mode's kr
    # is bounded above by omega/min(c_water) ~= omega/1480, but
    # omega/seabed.c for seabed.c=1000 is omega/1000, which exceeds that
    # bound -- so no mode's kr can exceed omega/seabed.c, and this seabed is
    # provably not the one these modes were computed with (a seabed this slow
    # cannot even trap anything -- omega/c > omega/min(c_water)).
    mismatched = Seabed(c=1000.0, rho=1.8, alpha=0.15)
    with pytest.raises(ValueError, match="seabed does not match the modes"):
        modal_attenuation(ms, mismatched)


def test_the_seabed_the_modes_were_actually_computed_with_does_not_raise():
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    ms = modes(isovelocity(), seabed, 20.0, 1000.0)
    assert ms.n_modes > 0
    att = modal_attenuation(ms, seabed)
    assert att.shape == (ms.n_modes,)


def test_a_source_at_the_surface_excites_nothing_because_phi_is_pinned_to_zero():
    ms = modes(isovelocity(), Seabed(c=1700.0, rho=1.8, alpha=0.15), 20.0, 1000.0)
    assert np.allclose(excitation(ms, 0.0), 0.0)


def test_excitation_is_the_mode_shape_at_source_depth():
    ms = modes(isovelocity(), Seabed(c=1700.0, rho=1.8, alpha=0.15), 20.0, 1000.0)
    z_s = 300.0
    got = excitation(ms, z_s)
    k = np.argmin(np.abs(ms.z_down - z_s))
    assert np.allclose(got, ms.phi[k, :], rtol=1e-6)


def test_a_source_at_a_node_is_not_coupled_and_at_an_antinode_is_maximally_coupled():
    """Mode 1 in an isovelocity guide is a half-sine: node at the surface,
    antinode near mid-depth. A source at a node drives that mode far less than
    one at its antinode -- this is the term that decides whether a whale at
    -30 m can drive a mode trapped in a deep guide at all."""
    depth = 1000.0
    ms = modes(isovelocity(depth_m=depth), Seabed(c=1700.0, rho=1.8, alpha=0.15),
               20.0, depth)
    phi1 = ms.phi[ms.in_water, 0]
    z_in = ms.z_down[ms.in_water]
    antinode_z = float(z_in[np.argmax(np.abs(phi1))])
    near_node_z = 5.0
    assert abs(excitation(ms, near_node_z)[0]) < 0.2 * abs(excitation(ms, antinode_z)[0])


def test_tl_grows_with_range():
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    ms = modes(isovelocity(), seabed, 20.0, 1000.0)
    r = np.array([1e3, 5e3, 20e3, 40e3])
    tl = mode_sum_tl(ms, seabed, 30.0, 500.0, r, coherent=False)
    assert np.all(np.diff(tl) > 0), f"TL must increase with range, got {tl}"


def test_incoherent_sum_has_no_interference_nulls_and_coherent_one_may():
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    ms = modes(isovelocity(), seabed, 20.0, 1000.0)
    if ms.n_modes < 2:
        pytest.skip("interference needs at least two modes")
    r = np.linspace(2e3, 20e3, 400)
    inc = mode_sum_tl(ms, seabed, 30.0, 500.0, r, coherent=False)
    coh = mode_sum_tl(ms, seabed, 30.0, 500.0, r, coherent=True)
    # The incoherent sum is smooth; the coherent one is not.
    assert np.abs(np.diff(inc, 2)).max() < np.abs(np.diff(coh, 2)).max()


def test_attenuation_strictly_increases_tl():
    lossless = Seabed(c=1700.0, rho=1.8, alpha=0.0)
    lossy = Seabed(c=1700.0, rho=1.8, alpha=0.30)
    ms = modes(isovelocity(), lossless, 20.0, 1000.0)
    r = np.array([20e3])
    assert (mode_sum_tl(ms, lossy, 30.0, 500.0, r, coherent=False)[0]
            > mode_sum_tl(ms, lossless, 30.0, 500.0, r, coherent=False)[0])


def test_no_trapped_modes_gives_infinite_tl_rather_than_a_misleading_number():
    """A seabed slower than the water traps nothing. That is a physical answer,
    and it must not come back as a finite level."""
    slow = Seabed(c=1400.0, rho=1.8, alpha=0.15)
    ms = modes(isovelocity(), slow, 20.0, 1000.0)
    assert ms.n_modes == 0
    tl = mode_sum_tl(ms, slow, 30.0, 500.0, np.array([10e3]))
    assert np.all(np.isinf(tl))


# --- Depth-convention guard -------------------------------------------------
# `z_down` starts at 0, so a negative depth silently snaps to the surface row
# where phi is pinned to zero. That reads as "the source excites nothing" --
# a legitimate-looking physical answer for what is actually a caller bug, and
# the rest of this repo uses negative-down depths outside `soundspeed/`.


def test_excitation_refuses_a_negative_depth_rather_than_snapping_to_the_surface():
    ms = modes(isovelocity(), Seabed(c=1700.0, rho=1.8, alpha=0.15), 20.0, 1000.0)
    with pytest.raises(ValueError, match="positive down"):
        excitation(ms, -30.0)


def test_mode_sum_tl_refuses_a_negative_source_or_receiver_depth():
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    ms = modes(isovelocity(), seabed, 20.0, 1000.0)
    r = np.array([10e3])
    with pytest.raises(ValueError, match="positive down"):
        mode_sum_tl(ms, seabed, -30.0, 500.0, r)
    with pytest.raises(ValueError, match="positive down"):
        mode_sum_tl(ms, seabed, 30.0, -500.0, r)


def test_the_rejected_value_is_named_so_the_caller_can_see_what_it_passed():
    ms = modes(isovelocity(), Seabed(c=1700.0, rho=1.8, alpha=0.15), 20.0, 1000.0)
    with pytest.raises(ValueError, match="-30.0"):
        excitation(ms, -30.0)


def test_a_positive_depth_is_unaffected_by_the_guard():
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    ms = modes(isovelocity(), seabed, 20.0, 1000.0)
    k = int(np.argmin(np.abs(ms.z_down - 300.0)))
    assert np.allclose(excitation(ms, 300.0), ms.phi[k, :])
    assert np.isfinite(mode_sum_tl(ms, seabed, 30.0, 500.0, np.array([10e3]))).all()
    # Zero is a legal depth: the surface itself, not a sign error.
    assert np.allclose(excitation(ms, 0.0), 0.0)


# --- Adiabatic range stepping ----------------------------------------------


def flat_path(depth_m, total_m, n=24):
    return PathProfile(range_m=np.linspace(0.0, total_m, n),
                       depth_m=np.full(n, depth_m), ok=True)


def test_a_constant_depth_path_reduces_exactly_to_the_range_independent_sum():
    """The strongest check on the range stepping: with nothing varying, the
    adiabatic machinery must reproduce the plain mode sum."""
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    depth, rng = 1000.0, 20_000.0
    prof = isovelocity(depth_m=depth)
    ms = modes(prof, seabed, 20.0, depth)

    direct = mode_sum_tl(ms, seabed, 30.0, 500.0, np.array([rng]), coherent=False)[0]
    got = adiabatic_tl(prof, seabed, 20.0, flat_path(depth, rng), 30.0, 500.0,
                       coherent=False)

    assert got.ok
    assert got.n_stripped == 0
    assert got.tl_db == pytest.approx(direct, rel=1e-3)


def test_the_coherent_constant_depth_path_also_reduces_to_the_plain_sum():
    """Separately pinned because the coherent branch is where a phase-quadrature
    error shows up -- an incoherent sum discards the phase entirely."""
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    depth, rng = 1000.0, 20_000.0
    prof = isovelocity(depth_m=depth)
    ms = modes(prof, seabed, 20.0, depth)
    direct = mode_sum_tl(ms, seabed, 30.0, 500.0, np.array([rng]), coherent=True)[0]
    got = adiabatic_tl(prof, seabed, 20.0, flat_path(depth, rng), 30.0, 500.0,
                       coherent=True)
    assert got.tl_db == pytest.approx(direct, rel=1e-3)


def test_a_shallowing_path_strips_modes_and_never_regains_one():
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    n = 20
    path = PathProfile(range_m=np.linspace(0.0, 30_000.0, n),
                       depth_m=np.linspace(1400.0, 120.0, n), ok=True)
    got = adiabatic_tl(isovelocity(depth_m=1400.0), seabed, 20.0, path, 30.0, 100.0,
                       coherent=False)
    assert got.ok
    assert got.n_stripped > 0
    assert got.n_modes_end <= got.n_modes_start
    assert got.n_modes_end == got.n_modes_start - got.n_stripped


def test_a_mode_stripped_in_a_shallow_notch_does_not_come_back_when_it_deepens():
    """Adiabatic bookkeeping is irreversible here: a mode that stopped being
    trapped where the guide pinched radiated into the seabed, and a deeper
    guide downrange does not recover it."""
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    n = 21
    depth = np.full(n, 1400.0)
    depth[n // 2] = 150.0                     # a single shallow notch mid-path
    notched = PathProfile(range_m=np.linspace(0.0, 30_000.0, n),
                          depth_m=depth, ok=True)
    got = adiabatic_tl(isovelocity(depth_m=1400.0), seabed, 20.0, notched, 30.0,
                       500.0, coherent=False)
    assert got.ok
    assert got.n_stripped > 0
    # Ends deep, but only the modes that survived the notch are still carried.
    assert got.n_modes_end == got.n_modes_start - got.n_stripped


def test_stripping_a_mode_can_only_increase_transmission_loss():
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    prof = isovelocity(depth_m=1400.0)
    n = 20
    flat = PathProfile(range_m=np.linspace(0.0, 30_000.0, n),
                       depth_m=np.full(n, 1400.0), ok=True)
    shoaling = PathProfile(range_m=np.linspace(0.0, 30_000.0, n),
                           depth_m=np.linspace(1400.0, 200.0, n), ok=True)
    a = adiabatic_tl(prof, seabed, 20.0, flat, 30.0, 200.0, coherent=False)
    b = adiabatic_tl(prof, seabed, 20.0, shoaling, 30.0, 200.0, coherent=False)
    assert b.n_stripped > 0
    assert b.tl_db > a.tl_db


def test_a_failed_path_is_refused_rather_than_silently_treated_as_flat():
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    bad = PathProfile(range_m=np.empty(0), depth_m=np.empty(0), ok=False,
                      reason="path crosses land: 3 of 9 samples")
    got = adiabatic_tl(isovelocity(), seabed, 20.0, bad, 30.0, 500.0)
    assert not got.ok
    assert "land" in got.reason
    assert not np.isfinite(got.tl_db)


def test_losing_every_mode_gives_infinite_loss():
    slow_guide = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    n = 12
    # Shoal to a few metres: nothing stays trapped at 20 Hz.
    path = PathProfile(range_m=np.linspace(0.0, 30_000.0, n),
                       depth_m=np.linspace(1400.0, 3.0, n), ok=True)
    got = adiabatic_tl(isovelocity(depth_m=1400.0), slow_guide, 20.0, path, 30.0, 3.0,
                       coherent=False)
    assert got.ok
    assert got.n_modes_end == 0
    assert np.isinf(got.tl_db)


def test_a_guide_that_traps_nothing_anywhere_reports_zero_modes_at_both_ends():
    """No mode is ever stripped here because none was ever there -- a different
    finding from 'they existed and were lost', and it must read differently."""
    slow = Seabed(c=1400.0, rho=1.8, alpha=0.15)      # slower than the water
    got = adiabatic_tl(isovelocity(), slow, 20.0, flat_path(1000.0, 20_000.0),
                       30.0, 500.0, coherent=False)
    assert got.ok
    assert got.n_modes_start == 0
    assert got.n_modes_end == 0
    assert got.n_stripped == 0
    assert np.isinf(got.tl_db)


def test_a_single_sample_path_is_refused_because_there_is_nothing_to_step_over():
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    one = PathProfile(range_m=np.array([0.0]), depth_m=np.array([1000.0]), ok=True)
    got = adiabatic_tl(isovelocity(), seabed, 20.0, one, 30.0, 500.0)
    assert not got.ok
    assert not np.isfinite(got.tl_db)


def test_adiabatic_tl_refuses_a_negative_source_or_receiver_depth():
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    path = flat_path(1000.0, 20_000.0)
    with pytest.raises(ValueError, match="positive down"):
        adiabatic_tl(isovelocity(), seabed, 20.0, path, -30.0, 500.0)
    with pytest.raises(ValueError, match="positive down"):
        adiabatic_tl(isovelocity(), seabed, 20.0, path, 30.0, -500.0)


def test_the_amplitude_factor_is_applied_once_not_twice():
    """sqrt(kr(0)/kr(r)) is the whole adiabatic amplitude correction. Applying
    it twice still reduces to 1 on a constant-depth path, so the reduction test
    cannot see that error; this compares against an independently evaluated
    single-step sum where the two ends genuinely differ."""
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.0)     # isolate the amplitude
    prof = isovelocity(depth_m=1000.0)
    d0, d1, rng = 1000.0, 900.0, 10_000.0
    path = PathProfile(range_m=np.array([0.0, rng]),
                       depth_m=np.array([d0, d1]), ok=True)
    got = adiabatic_tl(prof, seabed, 20.0, path, 30.0, 400.0, coherent=False)

    m0 = modes(prof, seabed, 20.0, d0)
    m1 = modes(prof, seabed, 20.0, d1)
    n = min(m0.n_modes, m1.n_modes)
    assert n > 1
    a = excitation(m0, 30.0)[:n]
    k = int(np.argmin(np.abs(m1.z_down - 400.0)))
    b = m1.phi[k, :n]
    kr0, kr1 = m0.kr[:n], m1.kr[:n]
    # 1/sqrt(kr(0)) * sqrt(kr(0)/kr(r)) == 1/sqrt(kr(r)); written the long way
    # so the factor under test is visible.
    amp = a * b / np.sqrt(kr0) * np.sqrt(kr0 / kr1)
    p = np.sqrt((amp**2).sum()) / np.sqrt(8.0 * np.pi * rng)
    assert got.tl_db == pytest.approx(-20.0 * np.log10(p), rel=1e-9)


def test_a_coherent_shoaling_path_matches_a_hand_built_three_sample_oracle():
    """The coherent reduction test above runs on a CONSTANT-depth path, where
    no mode is ever stripped and the per-mode arrays are never truncated -- so
    it cannot see a truncation that takes the WRONG END of the accumulated
    phase or attenuation. This path holds depth for one step, so `phase` and
    `atten` are non-zero and unequal across modes, then shoals on the second,
    forcing a real truncation. Slicing `phase[-n_live:]` instead of
    `phase[:n_live]` here moves the answer by ~0.7 dB.
    """
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    prof = isovelocity(depth_m=1400.0)
    freq, z_s, z_r = 20.0, 30.0, 250.0
    rs = np.array([0.0, 10_000.0, 20_000.0])
    ds = np.array([1400.0, 1400.0, 300.0])
    got = adiabatic_tl(prof, seabed, freq,
                       PathProfile(range_m=rs, depth_m=ds, ok=True),
                       z_s, z_r, coherent=True)

    m0 = modes(prof, seabed, freq, ds[0])
    m1 = modes(prof, seabed, freq, ds[1])
    m2 = modes(prof, seabed, freq, ds[2])
    n = min(m0.n_modes, m1.n_modes, m2.n_modes)
    assert n >= 2, "need several modes for the truncation to be visible"
    assert n < m0.n_modes, "the shoaling end must actually strip modes"

    # Trapezoid over the two steps, carried on the modes that survive to the
    # end -- i.e. the LOW-index ones, which is what the truncation must keep.
    kr0, kr1, kr2 = m0.kr[:n], m1.kr[:n], m2.kr[:n]
    at0 = modal_attenuation(m0, seabed)[:n]
    at1 = modal_attenuation(m1, seabed)[:n]
    at2 = modal_attenuation(m2, seabed)[:n]
    dr1, dr2 = rs[1] - rs[0], rs[2] - rs[1]
    phase = 0.5 * (kr0 + kr1) * dr1 + 0.5 * (kr1 + kr2) * dr2
    atten = 0.5 * (at0 + at1) * dr1 + 0.5 * (at1 + at2) * dr2

    a = excitation(m0, z_s)[:n]
    k = int(np.argmin(np.abs(m2.z_down - z_r)))
    b = m2.phi[k, :n]
    amp = a * b / np.sqrt(kr0) * np.sqrt(kr0 / kr2) * np.exp(-atten)
    p = abs((amp * np.exp(1j * phase)).sum()) / np.sqrt(8.0 * np.pi * rs[-1])

    assert got.ok
    assert got.n_modes_end == n
    assert got.tl_db == pytest.approx(-20.0 * np.log10(p), rel=1e-9)


def test_a_non_finite_path_sample_is_refused_rather_than_returning_a_nan_level():
    """`nan < 0` and `nan <= 0` are both False, so a NaN slips past every
    ordering guard and comes back as tl_db=nan with ok=True -- a failure
    wearing an answer's clothes."""
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    bad_r = PathProfile(range_m=np.array([0.0, np.nan]),
                        depth_m=np.array([1000.0, 900.0]), ok=True)
    got = adiabatic_tl(isovelocity(), seabed, 20.0, bad_r, 30.0, 500.0)
    assert not got.ok
    assert "finite" in got.reason
    assert not np.isfinite(got.tl_db)

    bad_d = PathProfile(range_m=np.array([0.0, 10_000.0]),
                        depth_m=np.array([1000.0, np.nan]), ok=True)
    got = adiabatic_tl(isovelocity(), seabed, 20.0, bad_d, 30.0, 500.0)
    assert not got.ok
    assert "finite" in got.reason
    assert not np.isfinite(got.tl_db)


def test_a_non_positive_depth_sample_is_refused():
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    dry = PathProfile(range_m=np.array([0.0, 10_000.0]),
                      depth_m=np.array([1000.0, 0.0]), ok=True)
    got = adiabatic_tl(isovelocity(), seabed, 20.0, dry, 30.0, 500.0)
    assert not got.ok
    assert not np.isfinite(got.tl_db)


def test_a_receiver_below_the_local_seafloor_is_refused_not_buried_in_sediment():
    """The mode grid extends into the sediment, so the nearest-index lookup has
    no bounds check: a 1200 m receiver on a path that shoals to 200 m snaps to
    the deepest seabed node and returns a finite ~250 dB, which reads as 'very
    lossy' rather than 'wrong'. Real cable channels reach 1554.9 m on paths
    that shoal toward the shelf, so this is a data condition to count, not a
    caller bug to raise on."""
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    n = 20
    shoaling = PathProfile(range_m=np.linspace(0.0, 30_000.0, n),
                           depth_m=np.linspace(1400.0, 200.0, n), ok=True)
    got = adiabatic_tl(isovelocity(depth_m=1400.0), seabed, 20.0, shoaling,
                       30.0, 1200.0, coherent=False)
    assert not got.ok
    assert not np.isfinite(got.tl_db)
    assert "1200" in got.reason and "200" in got.reason
    # A receiver exactly at the local seafloor is legal -- that is where the
    # fibre actually lies.
    at_floor = adiabatic_tl(isovelocity(depth_m=1400.0), seabed, 20.0, shoaling,
                            30.0, 200.0, coherent=False)
    assert at_floor.ok


def test_a_source_below_the_seafloor_at_the_start_of_the_path_is_refused():
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    n = 20
    deepening = PathProfile(range_m=np.linspace(0.0, 30_000.0, n),
                            depth_m=np.linspace(200.0, 1400.0, n), ok=True)
    got = adiabatic_tl(isovelocity(depth_m=1400.0), seabed, 20.0, deepening,
                       900.0, 500.0, coherent=False)
    assert not got.ok
    assert not np.isfinite(got.tl_db)
    assert "900" in got.reason and "200" in got.reason


def test_an_unresolved_shape_is_counted_but_a_long_evanescent_tail_is_not():
    """`n_unresolved_steps` must count only `n_modes < n_modes_expected` -- a
    shape missing entirely. `converged=False` with an EXACT count is the common,
    expected state near the trapping threshold (a mode whose evanescent tail is
    longer than the seabed extension), and counting it would flag exactly the
    marginal regime this experiment is about as untrustworthy.

    Depths chosen just above the 5th-mode cutoff for this guide: 338.61 m
    resolves all 5 with an unresolved tail, 338.41 m loses the 5th shape.
    """
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    prof = isovelocity(depth_m=1400.0)

    near = modes(prof, seabed, 20.0, 338.61)
    assert near.n_modes == near.n_modes_expected, "premise: the count is exact"
    assert not near.converged, "premise: the tail is under-resolved"
    got = adiabatic_tl(prof, seabed, 20.0, flat_path(338.61, 20_000.0, n=4),
                       30.0, 300.0, coherent=False)
    assert got.ok
    assert got.n_unresolved_steps == 0, "a long tail is not a missing shape"

    missing = modes(prof, seabed, 20.0, 338.41)
    assert missing.n_modes < missing.n_modes_expected, "premise: a shape is gone"
    got = adiabatic_tl(prof, seabed, 20.0, flat_path(338.41, 20_000.0, n=4),
                       30.0, 300.0, coherent=False)
    assert got.ok
    assert got.n_unresolved_steps == 4, "every step of this path is unresolved"


# --- DAS axial response ---

from dclde_soundspeed.soundspeed.tl import das_axial_response


def test_endfire_arrival_along_the_fibre_is_fully_sensed():
	# Source due west, cable running east-west: propagation is along the fibre.
	assert das_axial_response((-10_000.0, 0.0), (0.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)


def test_broadside_arrival_is_rejected():
	# Source due south, cable east-west: propagation is perpendicular.
	assert das_axial_response((0.0, -10_000.0), (0.0, 0.0), (1.0, 0.0)) == pytest.approx(0.0, abs=1e-12)


def test_forty_five_degrees_gives_one_half():
	assert das_axial_response((-1000.0, -1000.0), (0.0, 0.0), (1.0, 0.0)) == pytest.approx(0.5)


def test_response_does_not_depend_on_the_sign_of_the_tangent():
	a = das_axial_response((-1000.0, -300.0), (0.0, 0.0), (1.0, 0.0))
	b = das_axial_response((-1000.0, -300.0), (0.0, 0.0), (-1.0, 0.0))
	assert a == pytest.approx(b)


def test_response_is_bounded_in_zero_one():
	rng = np.random.default_rng(0)
	for _ in range(200):
		s = rng.normal(size=2) * 1e4
		t = rng.normal(size=2)
		v = das_axial_response(tuple(s), (0.0, 0.0), tuple(t))
		assert 0.0 <= v <= 1.0


def test_a_source_on_top_of_the_channel_is_refused_rather_than_returning_nan():
	with pytest.raises(ValueError, match="zero-length"):
		das_axial_response((0.0, 0.0), (0.0, 0.0), (1.0, 0.0))
