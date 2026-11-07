"""Transmission loss from the trapped modes `modes.py` computes.

`modes.py` answers EXISTENCE: are there trapped modes, and how many. That is
not arrival. A mode can exist and still be irrelevant if a source at -30 m does
not excite it, or if it does not survive 5-40 km of a shelf-to-abyssal margin.
This module supplies the missing three: per-mode sediment attenuation (the
`Seabed.alpha` that `modes()` deliberately carries without consuming), source
excitation from the mode shape at source depth, and the mode sum that turns
those into a received level.

Depths are POSITIVE DOWN throughout, as everywhere in this package.

What this module does NOT do, and what a reader must not assume from it:
  * Adiabatic range stepping bounds mode coupling, it does not resolve it. The
    Oregon margin is steep and the adiabatic approximation assumes gentle
    slopes.
  * There is no gauge-length response. At 20 Hz the wavelength is ~74 m against
    a gauge length of order 10 m, so it is second order -- but it is absent,
    not included.
  * Absolute level is not meaningful against this dataset's denoised SNR
    product. Use shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from dclde_soundspeed.soundspeed.modes import ModeSet, Seabed, modes
from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile

if TYPE_CHECKING:  # `bathymetry` pulls in requests/xarray; this module needs
    from dclde_localization.physics.bathymetry import PathProfile  # only the type name.

# A neper is 20/ln(10) dB. Named because getting this factor wrong is the
# classic silent error in attenuation code, and a bare 8.686 is unreadable.
DB_PER_NEPER = 20.0 / np.log(10.0)


def _depth_down(value, name: str) -> float:
    """Validate a POSITIVE-DOWN depth argument.

    Every depth here is positive down, but the rest of this package -- and the
    way the spec itself talks ("a source at -30 m") -- is negative down. A
    negative depth is not caught by anything downstream: `ModeSet.z_down`
    starts at 0, so `argmin(|z - (-30)|)` resolves to the surface row, where
    `phi` is pinned to zero by the pressure-release condition. The caller then
    gets "this source excites nothing", which reads as a physical finding about
    a shallow source rather than a sign error. Refuse instead.
    """
    v = float(value)
    if not np.isfinite(v) or v < 0.0:
        raise ValueError(
            f"{name} must be a depth measured positive down from the sea "
            f"surface (0 is the surface, 1000 is 1000 m below it), got {v}; "
            "if this came from a negative-down convention, negate it"
        )
    return v


def alpha_nepers_per_m(alpha_db_per_wavelength: float, freq_hz: float,
                       c_m_s: float) -> float:
    """Convert `Seabed.alpha` (dB per WAVELENGTH) to nepers per metre.

    dB/m = alpha_dB_per_wavelength * freq / c, since one wavelength is c/freq.
    """
    alpha_db_per_wavelength = float(alpha_db_per_wavelength)
    freq_hz = float(freq_hz)
    c_m_s = float(c_m_s)
    if alpha_db_per_wavelength < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha_db_per_wavelength}")
    if freq_hz <= 0 or c_m_s <= 0:
        raise ValueError(f"freq_hz and c must be positive, got {freq_hz}, {c_m_s}")
    return (alpha_db_per_wavelength * freq_hz / c_m_s) / DB_PER_NEPER


def modal_attenuation(modeset: ModeSet, seabed: Seabed) -> np.ndarray:
    """Per-mode attenuation Im(kr), nepers/m, from sediment absorption.

    First-order perturbation of the eigenvalue by a complex sediment sound
    speed (see the plan's Task 3 for the derivation):

        Im(kr_m) = (omega / kr_m) * INTEGRAL (alpha_n(z) / c(z)) phi_m(z)^2 dz

    with `phi` the orthonormal, density-reduced shapes `ModeSet` already holds,
    and the integrand non-zero only in the sediment. A seabed with `alpha = 0`
    returns exactly zero for every mode.

    `seabed` is NOT read off `modeset` -- `ModeSet` does not carry the seabed
    it was built with, so nothing stops a caller passing one that does not
    match. Every trapped mode must satisfy `kr > omega / seabed.c` (that is
    what "trapped by this seabed" means: the mode decays in it, i.e.
    `gamma = sqrt(kr^2 - (omega/c)^2)` is real); a `seabed` for which that
    fails cannot be the one these modes were computed with, so this raises
    rather than silently producing an imaginary `gamma`.

    This guard catches a GROSS seabed swap -- one that violates the trapping
    inequality outright -- not a near-but-wrong one: modes built with
    `c=1700` and this function called with `c=1750` for the same `modeset`
    will pass the `kr > omega/seabed.c` check and return a silently wrong
    attenuation. It is a consistency check, not a match check.
    """
    if modeset.n_modes == 0:
        return np.empty(0, dtype=float)

    omega = 2.0 * np.pi * modeset.freq_hz
    kr_min_allowed = omega / seabed.c
    bad = modeset.kr <= kr_min_allowed
    if np.any(bad):
        raise ValueError(
            "seabed does not match the modes: mode(s) at index "
            f"{np.flatnonzero(bad).tolist()} have kr={modeset.kr[bad].tolist()} "
            f"<= omega/seabed.c={kr_min_allowed!r} (omega={omega!r}, "
            f"seabed.c={seabed.c!r}); a mode trapped by this seabed must have "
            "kr strictly greater than omega/seabed.c, so this seabed cannot be "
            "the one these modes were computed with"
        )

    if seabed.alpha == 0.0:
        return np.zeros(modeset.n_modes, dtype=float)

    a_n = alpha_nepers_per_m(seabed.alpha, modeset.freq_hz, seabed.c)

    z = modeset.z_down
    dz = float(z[1] - z[0])
    in_sediment = ~modeset.in_water

    # phi is orthonormal under sum(phi^2)*dz, so the quadrature matches the
    # normalisation the shapes were built with -- do not swap in a fancier rule
    # without renormalising, or the two disagree at the few-percent level.
    weight = np.where(in_sediment, a_n / seabed.c, 0.0)
    integral = (weight[:, None] * modeset.phi**2).sum(axis=0) * dz
    return omega * integral / modeset.kr


def excitation(modeset: ModeSet, source_z_down: float) -> np.ndarray:
    """Mode shape at source depth -- how strongly a source at `source_z_down`
    drives each mode.

    This is the term that separates "the modes are not there" from "the modes
    are there but the whale does not excite them". They are different findings
    with different consequences, so this is reported per mode rather than
    folded into a total.
    """
    source_z_down = _depth_down(source_z_down, "source_z_down")
    if modeset.n_modes == 0:
        return np.empty(0, dtype=float)
    z = modeset.z_down
    k = int(np.argmin(np.abs(z - source_z_down)))
    return modeset.phi[k, :].astype(float)


def mode_sum_tl(modeset: ModeSet, seabed: Seabed, source_z_down: float,
                receiver_z_down: float, range_m, *, coherent: bool = True) -> np.ndarray:
    """Transmission loss, dB re 1 m, at each range in `range_m`.

    Far-field normal-mode sum:

        p(r,z) ~ (i e^{-i pi/4} / (rho(z_s) sqrt(8 pi r)))
                 * SUM_m phi_m(z_s) phi_m(z) e^{i kr_m r} / sqrt(kr_m)

    with per-mode attenuation applied as e^{-Im(kr_m) r}. The density prefactor
    is a constant here (it does not vary with range or mode) and TL is used
    only as a SHAPE against this dataset's uncalibrated denoised SNR, so it is
    omitted rather than invented from an unknown source-depth density.

    `coherent=False` sums mode powers instead of amplitudes. Both are produced
    and carried together deliberately: the coherent sum's interference nulls
    are far sharper than this geometry's uncertainty justifies, so a verdict
    must never rest on where a null falls.

    Returns +inf where no mode is trapped -- a physical answer, not a failure,
    and one that must not come back looking like a finite level.
    """
    source_z_down = _depth_down(source_z_down, "source_z_down")
    receiver_z_down = _depth_down(receiver_z_down, "receiver_z_down")
    r = np.atleast_1d(np.asarray(range_m, dtype=float))
    if np.any(r <= 0):
        raise ValueError("range_m must be positive; the far-field mode sum "
                         "diverges at r = 0")
    if modeset.n_modes == 0:
        return np.full(r.shape, np.inf)

    a_m = excitation(modeset, source_z_down)
    z = modeset.z_down
    k_r = int(np.argmin(np.abs(z - receiver_z_down)))
    b_m = modeset.phi[k_r, :].astype(float)

    kr = modeset.kr.astype(float)
    att = modal_attenuation(modeset, seabed)

    # (n_ranges, n_modes)
    decay = np.exp(-att[None, :] * r[:, None])
    amp = (a_m * b_m / np.sqrt(kr))[None, :] * decay

    if coherent:
        phase = np.exp(1j * kr[None, :] * r[:, None])
        field = np.abs((amp * phase).sum(axis=1))
    else:
        field = np.sqrt((amp**2).sum(axis=1))

    p = field / np.sqrt(8.0 * np.pi * r)
    with np.errstate(divide="ignore"):
        return -20.0 * np.log10(np.where(p > 0, p, 0.0))


@dataclass(frozen=True)
class AdiabaticResult:
    """One adiabatic mode-sum transmission loss along one path.

    tl_db:              transmission loss, dB. +inf if every mode was stripped,
                        or if none was ever trapped.
    n_modes_start:      modes trapped in the guide at the source end.
    n_modes_end:        modes still trapped at the receiver end.
    n_stripped:         how many were lost along the way. Reported because mode
                        stripping, not sediment absorption, is the dominant
                        loss on a shelf-to-abyssal path -- it is the number that
                        explains the answer.
                        Read it precisely: it counts modes the SOLVER could no
                        longer carry, which is physics PLUS any solver artifact.
                        Stripping is decided by `ModeSet.n_modes` (the shapes
                        resolved) and not by `n_modes_expected` (the exact
                        physical count), because the sum needs a `kr` and a
                        `phi` for every mode it carries and there is none for a
                        mode that was not resolved. The two agree except where
                        the seabed extension ran out, and `n_unresolved_steps`
                        is THE ONLY THING that separates the two contributions.
                        A non-zero `n_unresolved_steps` means part of this
                        count may be the solver rather than the ocean.
    n_unresolved_steps: steps where the local mode solve returned
                        `n_modes < n_modes_expected`, i.e. a shape was missing
                        entirely. Counted rather than absorbed. Steps that were
                        merely `converged=False` with an exact count are NOT
                        counted here -- per `ModeSet`'s own documentation that
                        is the common, expected case near the trapping
                        threshold, and discarding it would bias away exactly
                        the marginal regime this experiment is about.
                        Only steps actually taken are counted: once every mode
                        is gone the walk stops, so this is a count over the
                        prefix of the path that was evaluated, not the whole
                        path.
    ok / reason:        false when the path itself was unusable. `reason` is
                        also set on some `ok` results, to say why the loss is
                        infinite.
    """

    tl_db: float
    n_modes_start: int
    n_modes_end: int
    n_stripped: int
    n_unresolved_steps: int
    ok: bool
    reason: str = ""


def _refused(reason: str) -> AdiabaticResult:
    return AdiabaticResult(np.inf, 0, 0, 0, 0, ok=False, reason=reason)


def adiabatic_tl(profile: SoundSpeedProfile, seabed: Seabed, freq_hz: float,
                 path: "PathProfile", source_z_down: float, receiver_z_down: float,
                 *, coherent: bool = True, nz: int = 2000) -> AdiabaticResult:
    """Adiabatic mode-sum TL over a range-varying waveguide.

    The field is the same far-field mode sum `mode_sum_tl` evaluates, with each
    mode carried along the path by its own local wavenumber::

        p(r,z) ~ SUM_m phi_m(z_s; 0) phi_m(z; r)
                 * (1 / sqrt(kr_m(0))) * sqrt(kr_m(0) / kr_m(r))
                 * exp(i INTEGRAL_0^r kr_m dr') * exp(-INTEGRAL_0^r Im(kr_m) dr')
                 / sqrt(8 pi r)

    The amplitude factor `sqrt(kr_m(0)/kr_m(r))` is the WKB/energy-flux
    correction for a mode whose wavenumber changes along the path; it multiplies
    the range-independent amplitude `1/sqrt(kr_m(0))` ONCE, and the product is
    just `1/sqrt(kr_m(r))`. It is written the long way here because that is the
    form the physics is stated in, and because applying it twice -- easy to do,
    since it is often quoted as the whole amplitude rather than a correction --
    still reduces to 1 on a constant-depth path and so survives the reduction
    test. `test_the_amplitude_factor_is_applied_once_not_twice` is what catches
    it.

    Both integrals are trapezoid quadratures over the path's own samples, which
    is exact for a constant-depth path (constant integrand), so the whole
    calculation collapses onto `mode_sum_tl` when nothing varies. That
    reduction is the real check on this routine.

    A mode is counted as STRIPPED when it stops being trapped in the local
    guide. Tracking is by index -- mode 1 is the best-trapped everywhere -- so
    stripping removes from the high-index end, which is exact for a guide that
    only shoals and is the standard adiabatic bookkeeping otherwise. Stripping
    is irreversible: a mode that stopped being trapped radiated into the
    seabed, and a deeper guide downrange does not recover it.

    A negative source or receiver depth RAISES (a caller using the wrong sign
    convention), but a source or receiver below the local seafloor is REFUSED
    with `ok=False` -- that is a data condition, a cable channel deeper than the
    water at the shelf end of its own path, and a sweep needs to count those
    rather than die on them. So are a non-finite or non-monotonic path and a
    path with a non-positive depth sample.

    Which count decides stripping: `ModeSet.n_modes`, the number of shapes the
    local solve resolved, because the sum needs a `kr` and a `phi` for every
    mode it carries and there is none for a mode that was not resolved.
    `n_modes_expected` is the exact physical count and `n_modes` never exceeds
    it, so the two agree except where the seabed extension ran out -- and every
    such step is reported in `n_unresolved_steps` rather than being quietly
    folded into `n_stripped`. Read a non-zero `n_unresolved_steps` as "part of
    this stripping may be the solver, not the ocean".

    ADIABATIC BOUNDS MODE COUPLING, IT DOES NOT RESOLVE IT. This approximation
    assumes gentle slopes and the Oregon margin is steep: energy stripped from
    a mode here is simply lost, where in reality some of it couples into the
    modes that remain. Read the result as a bound, not a prediction.

    Source/receiver-below-seafloor is checked ONLY at the two path endpoints
    (above), not at every intermediate step. That is correct only because the
    intermediate `b_m` (the receiver-row mode shape at each step) is
    overwritten and discarded on every iteration of the loop below -- only the
    final one survives to be used. A future change that accumulated `b_m`
    along the path (rather than just carrying the latest one forward) would
    need a per-step seafloor check, not just the endpoint ones.
    """
    source_z_down = _depth_down(source_z_down, "source_z_down")
    receiver_z_down = _depth_down(receiver_z_down, "receiver_z_down")

    if not path.ok:
        return _refused(path.reason)

    r = np.asarray(path.range_m, dtype=float)
    d = np.asarray(path.depth_m, dtype=float)
    if r.size != d.size:
        return _refused(f"path range_m and depth_m have different lengths: "
                        f"{r.size} and {d.size}")
    if r.size < 2:
        return _refused(f"path has fewer than two samples ({r.size}); there is "
                        "nothing to step over")
    # Finiteness FIRST: every comparison below is False for a NaN, so a NaN
    # sails through the monotonicity and zero-length guards and comes back as
    # `tl_db=nan, ok=True` -- a refusal disguised as an answer.
    if not np.all(np.isfinite(r)):
        return _refused(f"path range_m is not all finite: "
                        f"{int((~np.isfinite(r)).sum())} of {r.size} samples")
    if not np.all(np.isfinite(d)):
        return _refused(f"path depth_m is not all finite: "
                        f"{int((~np.isfinite(d)).sum())} of {d.size} samples")
    if np.any(d <= 0.0):
        return _refused(f"path depth_m must be positive down and non-zero: "
                        f"{int((d <= 0.0).sum())} of {d.size} samples are at or "
                        "above sea level")
    if np.any(np.diff(r) < 0.0):
        return _refused("path ranges are not monotonically increasing, so the "
                        "quadrature along it would double back")
    total_r = float(r[-1] - r[0])
    if total_r <= 0.0:
        return _refused("path has zero length; the far-field mode sum diverges "
                        "at r = 0")

    # A source or receiver below the LOCAL seafloor is not in the water. The
    # grid extends into the sediment, so `argmin(|z_down - z|)` has no bounds
    # check and would quietly snap such a receiver to the deepest seabed node,
    # returning a finite, plausible-looking, very large TL. On a path that
    # shoals toward the shelf with a cable channel at 1500 m this is a real
    # data condition, not a caller bug -- hence `ok=False` rather than a raise,
    # so it can be counted rather than crashing a sweep.
    if source_z_down > float(d[0]):
        return _refused(f"source at {source_z_down} m is below the seafloor "
                        f"where the path starts ({float(d[0])} m of water)")
    if receiver_z_down > float(d[-1]):
        return _refused(f"receiver at {receiver_z_down} m is below the seafloor "
                        f"where the path ends ({float(d[-1])} m of water)")

    def _receiver_row(ms: ModeSet, n: int) -> np.ndarray:
        k = int(np.argmin(np.abs(ms.z_down - receiver_z_down)))
        return ms.phi[k, :n].astype(float)

    ms = modes(profile, seabed, freq_hz, float(d[0]), nz=nz)
    n_unresolved = int(ms.n_modes < ms.n_modes_expected)
    n_start = n_live = int(ms.n_modes)
    if n_live == 0:
        return AdiabaticResult(np.inf, 0, 0, 0, n_unresolved, ok=True,
                               reason="no mode is trapped where the path starts")

    phase = np.zeros(n_live)          # accumulated INTEGRAL kr dr, per live mode
    atten = np.zeros(n_live)          # accumulated INTEGRAL Im(kr) dr, per live mode
    kr_first = ms.kr[:n_live].astype(float)
    a_m = excitation(ms, source_z_down)
    # `.copy()` so `prev_kr` never aliases the source-end wavenumbers: they are
    # needed again at the very end for the amplitude factor.
    prev_kr = kr_first.copy()
    prev_att = modal_attenuation(ms, seabed)
    # Overwritten on every step below; the loop always runs at least once.
    kr_last = kr_first
    b_m = _receiver_row(ms, n_live)

    for i in range(1, r.size):
        ms = modes(profile, seabed, freq_hz, float(d[i]), nz=nz)
        if ms.n_modes < ms.n_modes_expected:
            n_unresolved += 1

        n_live = min(n_live, int(ms.n_modes))
        if n_live == 0:
            return AdiabaticResult(
                np.inf, n_start, 0, n_start, n_unresolved, ok=True,
                reason=f"every mode was stripped by {float(d[i])} m of water at "
                       f"{float(r[i])} m along the path")

        kr_here = ms.kr[:n_live].astype(float)
        att_here = modal_attenuation(ms, seabed)[:n_live]
        dr = float(r[i] - r[i - 1])
        # Trapezoid along the path, truncated to the modes still alive.
        phase = phase[:n_live] + 0.5 * (prev_kr[:n_live] + kr_here) * dr
        atten = atten[:n_live] + 0.5 * (prev_att[:n_live] + att_here) * dr

        prev_kr, prev_att = kr_here, att_here
        kr_last = kr_here
        b_m = _receiver_row(ms, n_live)

    n_stripped = n_start - n_live
    kr0 = kr_first[:n_live]
    amp = (a_m[:n_live] * b_m
           / np.sqrt(kr0)                      # range-independent amplitude
           * np.sqrt(kr0 / kr_last)            # adiabatic correction, applied once
           * np.exp(-atten))

    if coherent:
        field = float(np.abs((amp * np.exp(1j * phase)).sum()))
    else:
        field = float(np.sqrt((amp**2).sum()))

    p = field / np.sqrt(8.0 * np.pi * total_r)
    tl = np.inf if p <= 0.0 else float(-20.0 * np.log10(p))
    return AdiabaticResult(tl, n_start, n_live, n_stripped, n_unresolved, ok=True)


class DegenerateGeometryError(ValueError):
    """A source/channel/tangent geometry that `das_axial_response` refuses to
    score: the source coincides horizontally with the channel, or the cable
    tangent has zero length.

    A `ValueError` subclass, not a bare `ValueError`, so a caller that wants
    to treat THIS specific, recoverable condition differently from any other
    `ValueError` (e.g. `experiments/propagation.py` counting it rather than
    aborting) can catch it by type. Catching by message-substring instead --
    tried in an earlier revision of this module's caller -- couples the two
    modules through prose: a reworded exception message silently stops being
    caught, with no test to catch the drift, exactly the failure mode
    the original deep-voice/dclde_2027 `dclde.das_finwhale`'s profile-source string-matching bug (commit
    babbf50) already cost this repo once. `pytest.raises(ValueError, ...)`
    call sites are unaffected, since this remains a `ValueError`.
    """


def das_axial_response(source_xy, channel_xy, tangent_xy) -> float:
	"""Fraction of the arrival a DAS channel senses: cos^2(theta), with theta
	the angle between the horizontal propagation direction and the local cable
	tangent.

	DAS measures axial strain, so a broadside arrival is rejected and an endfire
	one is fully sensed. This is not a refinement: it varies strongly and
	systematically with geometry along these cables, and leaving it out would
	put a large range-correlated error into the prediction and invite it to be
	read as propagation physics.

	The gauge-length response is NOT included. At 20 Hz the wavelength is ~74 m
	against a gauge length of order 10 m, so it is second order -- but it is
	absent, not accounted for.
	"""
	sx, sy = (float(v) for v in source_xy)
	cx, cy = (float(v) for v in channel_xy)
	tx, ty = (float(v) for v in tangent_xy)

	dx, dy = cx - sx, cy - sy
	dn = np.hypot(dx, dy)
	tn = np.hypot(tx, ty)
	if dn == 0.0:
		raise DegenerateGeometryError("zero-length source-to-channel vector: the source is "
						 "at the channel, so there is no propagation direction")
	if tn == 0.0:
		raise DegenerateGeometryError("zero-length cable tangent")

	cos_theta = (dx * tx + dy * ty) / (dn * tn)
	return float(min(1.0, cos_theta**2))
