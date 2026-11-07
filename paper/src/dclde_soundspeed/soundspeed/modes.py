"""Range-independent normal modes for a fluid waveguide over a fluid halfspace.

At 20 Hz, lambda ~ 74 m, and in 100-500 m of water the guide is only 1.3-7
wavelengths deep. Ray theory -- including the sharp shadow zone it predicts for
the downward-refracting Oregon profile -- is marginal there, and normal modes
are what adjudicate it.

`arlpy` and `pyat` both shell out to Acoustics Toolbox Fortran binaries that are
not pip-installable and are hostile on Windows, so the Sturm-Liouville
eigenproblem is solved directly here and validated against the closed-form
Pekeris waveguide (`paper/tests/test_modes.py`).

The problem
-----------
Depth-separated wave equation with a pressure-release surface and a penetrable
fluid halfspace below the seafloor at z = D (depth POSITIVE DOWN throughout,
matching `SoundSpeedProfile`)::

    d/dz[ (1/rho) dphi/dz ] + (omega^2 / (rho c(z)^2)) phi = kr^2 (1/rho) phi
    phi(0) = 0                                   (pressure release)
    phi, (1/rho) dphi/dz continuous at z = D     (fluid-fluid interface)
    phi -> 0 as z -> infinity                    (trapped modes only)

A mode is *trapped* exactly when it oscillates in the water and decays in the
seabed, i.e. when omega/c_seabed < kr < omega/min(c_water). Outside that window
the field radiates into the halfspace (leaky) and is not returned.

Why this does not follow the task brief's sketch
------------------------------------------------
The brief sketched imposing the halfspace as a *cutoff on the eigenvalue
interval* while discretising the water column alone, with the grid implicitly
terminated by a Dirichlet condition at the seafloor. That is a pressure-release
bottom, and it is wrong in two ways that matter here:

1. It forces phi(D) = 0. Real trapped modes have a substantial amplitude at the
   seafloor and an evanescent tail into the sediment. The DAS fibre lies ON the
   seafloor, so a bottom condition that zeroes the mode there would predict the
   receiver senses nothing -- the opposite of the physics, and fatal for any
   conclusion drawn about the shadow zone.
2. It systematically undercounts by one mode. A pressure-release bottom places
   eigenvalues at kz = m*pi/D; the true interface condition,
   tan(kz D) = -(rho_b/rho_w) kz/gamma, places them between (m-1/2)*pi/D and
   m*pi/D. The mode nearest cutoff is the one dropped -- and the marginal modes
   are precisely what the shadow-zone question turns on.

So the halfspace is modelled explicitly instead: the grid is extended into the
seabed and truncated where the slowest-decaying trapped mode has died away, with
the extension grown adaptively until both the mode count and the decay length
have converged. Away from the trapping threshold this reproduces the analytic
Pekeris mode count exactly -- 224/224 over a sweep of (c_water, c_seabed, D, f)
-- and the wavenumbers to ~4e-8 rad/m, rather than to the +-1 mode the brief's
sketch needed.

NEAR the threshold, resolving mode SHAPES is not free: a mode whose grazing
angle is just inside the critical angle has an arbitrarily long evanescent tail,
and any truncation eventually loses it. So the COUNT is not obtained by
truncation at all. It comes from the cutoff limit, where the halfspace condition
degenerates to a Neumann condition on the water column alone and no seabed grid
exists to truncate (`_threshold_count`), and the seabed extension is then grown
until it has resolved that many shapes. Growing until the count merely stops
changing is a proxy driven by the modes already found, and it fails silently and
confidently near the threshold in both directions -- a fabricated zero, and a
stable-looking undercount. `ModeSet.n_modes_expected` is the exact count;
`ModeSet.converged` says whether every shape was resolved.

Discretisation
--------------
Second-order conservative finite differences on a uniform grid. The seafloor is
placed at a half-point so that every node lies unambiguously in water or in
sediment, and the interface enters through the flux coefficient (the harmonic
mean of 1/rho), which is how a density jump is handled correctly.

The resulting generalised problem A phi = kr^2 B phi (B = diag(1/rho)) is
symmetrised by the substitution u = phi / sqrt(rho), giving a symmetric
tridiagonal standard eigenproblem. `ModeSet.phi` holds that density-reduced
function u, which
  * equals the pressure mode shape in the water column, where rho = 1, and
  * is orthonormal in the plain (unweighted) L2 sense, because
    integral u_m u_n dz = integral phi_m phi_n / rho dz = delta_mn, which is the
    physical modal orthogonality relation.

What this module does NOT answer
--------------------------------
It computes EXISTENCE, not ARRIVAL. "Trapped modes exist" is not "energy arrives
at 5-40 km". Absent here: modal attenuation (see below), source excitation (a
mode can exist and carry nothing from a whale at -30 m, depending on the mode
shape at that depth), and transmission loss. A mode count refutes a claimed
shadow zone only in the weak sense that ray theory's "no path exists" is not the
right description of a guide a few wavelengths deep; establishing that a
receiver 20 km away actually hears the call needs a transmission-loss
calculation that is not in this module.

The count being exact and a zero being certified is what makes this worth
stating loudly: the rigour of the counting invites the result to be read as more
than it is.

Attenuation
-----------
`Seabed.alpha` is NOT used by `modes()`. Sediment attenuation makes kr complex
and turns every mode slightly leaky; this solver deliberately poses a real
eigenproblem, because the question it exists to answer is whether trapped modes
exist and how many, not how strongly they are damped. The field is kept on
`Seabed` because it is part of any honest seabed description and a later task
computing transmission loss will need it. Passing a non-zero alpha changes
nothing about the returned modes -- that is documented, not accidental.

Group slowness and modal arrival time are deliberately absent (Ruling R7):
nothing downstream consumes them yet, and untested numerical code in the module
that adjudicates a physics question would invite misplaced trust.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal

from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile

# Truncate the seabed where the slowest-decaying trapped mode has fallen by
# exp(-_TAIL_DECAYS): amplitude 3e-4, energy 1e-7. Verified sufficient against
# the analytic Pekeris roots.
_TAIL_DECAYS = 8.0
# Grid must resolve the shortest water wavelength this well, or the eigenvalues
# (and hence the mode count) are not trustworthy.
_MIN_POINTS_PER_WAVELENGTH = 8.0
_MAX_GROWTH_STEPS = 8
_MAX_SEABED_NODES = 500_000


@dataclass(frozen=True)
class Seabed:
    """Fluid halfspace below the seafloor.

    c:     sediment sound speed, m/s. Trapped modes exist only if this exceeds
           the minimum water sound speed; a slower seabed traps nothing.
    rho:   sediment density RATIO to water (dimensionless, ~1.5-2.0 for sand).
    alpha: attenuation in dB per wavelength. Carried for downstream use and
           documented as NOT consumed by `modes()` -- see the module docstring.
           DEFAULTS TO ZERO deliberately: a non-zero default would read as
           "attenuation is being accounted for" to a caller who never thought
           about it, when in fact it changes nothing here.
    """

    c: float = 1700.0
    rho: float = 1.8
    alpha: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.c) or self.c <= 0:
            raise ValueError(f"seabed sound speed must be positive and finite, got {self.c}")
        if not np.isfinite(self.rho) or self.rho <= 0:
            raise ValueError(f"seabed density ratio must be positive and finite, got {self.rho}")
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError(f"seabed attenuation must be non-negative and finite, "
                             f"got {self.alpha}")


@dataclass(frozen=True)
class ModeSet:
    """Trapped modes at one frequency.

    kr:        horizontal wavenumbers, rad/m, DESCENDING -- mode 1 (largest
               kr, lowest phase speed, shallowest-grazing) first. Every entry
               satisfies omega/c_seabed < kr < omega/min(c_water).
    phi:       (n_z, n_modes) density-reduced mode shapes u = phi/sqrt(rho),
               orthonormal under sum(phi_m * phi_n) * dz. Equals the pressure
               mode shape in the water column. phi[0] = 0 at the surface.
    z_down:    uniform depth grid, POSITIVE DOWN, starting at 0. It EXTENDS
               BELOW `water_depth_m` by `seabed_extension_m`, because the modes
               have a real evanescent tail in the sediment and excluding it
               would break both the physics and the orthonormality. Use
               `in_water` to mask the water column.
    freq_hz:   frequency these modes were computed at.
    water_depth_m:      seafloor depth; the water/sediment interface.
    seabed_extension_m: how far below the seafloor the grid runs.
    n_modes_expected: how many trapped modes EXIST. Comes from the
               truncation-free cutoff-limit count (`_threshold_count`), so it is
               exact whatever the grid did. **FOR A COUNTING QUESTION -- "are
               there trapped modes, and how many" -- READ THIS FIELD.** A zero
               here is a certified physical statement that no trapped mode
               exists, never the solver giving up.
    n_modes:   how many mode SHAPES the finite grid resolved. Equals
               `n_modes_expected` unless the seabed extension ran out.
    converged: `n_modes == n_modes_expected` AND every returned mode's
               evanescent tail is resolved to `_TAIL_DECAYS` e-foldings within
               the seabed-extension cap.

    READING `converged=False` CORRECTLY -- it has two quite different causes,
    and only one of them makes the count untrustworthy:

      * `n_modes == n_modes_expected` (the COMMON case, and the expected one
        near the trapping threshold, where a mode's evanescent tail is very
        long): the count is exact and fully trustworthy. Only the mode SHAPES
        are under-resolved in the sediment, because the tail was cut before
        `_TAIL_DECAYS` e-foldings. Do not discard the count here -- discarding
        near-threshold counts would bias away exactly the marginal regime the
        shadow-zone question lives in.
      * `n_modes < n_modes_expected`: a shape is missing entirely. `n_modes` is
        a LOWER BOUND; `n_modes_expected` is still exact.
      * `n_modes > n_modes_expected`: should not happen. It means the
        eigensolver returned a root the cutoff count says does not exist, and a
        `UserWarning` is raised. Trust neither `n_modes` nor the shapes.
    """

    kr: np.ndarray
    phi: np.ndarray
    z_down: np.ndarray
    freq_hz: float
    water_depth_m: float
    seabed_extension_m: float = 0.0
    n_modes_expected: int = 0
    converged: bool = True

    @property
    def n_modes(self) -> int:
        return int(self.kr.size)

    @property
    def in_water(self) -> np.ndarray:
        """Boolean mask selecting the water-column rows of `z_down` / `phi`."""
        return self.z_down <= self.water_depth_m


def _assemble(c_water: np.ndarray, dz: float, omega: float, seabed: Seabed,
              n_seabed: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric tridiagonal operator for nodes 1..N (node 0 is pinned to zero).

    `c_water` is sampled at the interior water nodes 1..nz-1; the seafloor sits
    at a half-point just past the last of them, so every node is unambiguously
    in one medium and the density jump lives on the flux coefficient between
    the two nodes straddling it.
    """
    c = np.concatenate([c_water, np.full(n_seabed, float(seabed.c))])
    rho = np.concatenate([np.ones(c_water.size), np.full(n_seabed, float(seabed.rho))])

    # Flux coefficient (1/rho) on the half-points between consecutive nodes.
    # Where the two nodes share a medium this is just 1/rho; where they straddle
    # the interface the exact series combination of the two half-cells is the
    # harmonic mean of the 1/rho values, i.e. 2/(rho_i + rho_{i+1}).
    r1, r2 = rho[:-1], rho[1:]
    t_mid = np.where(r1 == r2, 1.0 / r1, 2.0 / (r1 + r2))
    # Half-point above node 1 is inside the water; beyond the last node the grid
    # is terminated by a Dirichlet condition, so its coefficient is unused for
    # coupling but still contributes to that node's diagonal.
    t_left = np.concatenate([[1.0 / rho[0]], t_mid])
    t_right = np.concatenate([t_mid, [t_mid[-1]]])

    a_diag = -(t_left + t_right) / dz**2 + omega**2 / (rho * c**2)
    # Symmetrise A phi = kr^2 diag(1/rho) phi via u = phi/sqrt(rho).
    diag = rho * a_diag
    off = np.sqrt(r1 * r2) * t_mid / dz**2
    return diag, off


def _threshold_count(c_water: np.ndarray, dz: float, omega: float,
                     kr2_lo: float, kr2_hi: float) -> int:
    """Trapped-mode count, free of any seabed truncation.

    At the cutoff kr^2 = (omega/c_seabed)^2 the halfspace has gamma = 0, so the
    interface condition phi'(D) = -(rho_w/rho_b) gamma phi(D) degenerates to
    phi'(D) = 0 -- a Neumann condition, independent of density (which is why the
    analytic Pekeris count carries no rho). Every trapped mode is born at that
    threshold and moves up in kr as gamma grows, so the number of trapped modes
    is exactly the number of eigenvalues of the pressure-release/Neumann problem
    on the WATER COLUMN ALONE lying strictly above the cutoff.

    That problem has no seabed grid, so it cannot be truncated -- which is the
    point. Counting modes by growing the seabed until the count stops changing
    is a proxy driven by the modes already found, and it says nothing about a
    marginal mode still hidden below the truncation: it reports a stable,
    confident, wrong count. This is the target the growth loop aims at instead.

    Reproduces floor(2 f D sqrt(1/cw^2 - 1/cb^2) + 1/2) on 576/576 Pekeris cases,
    including depths tuned to sit 0.002 above a mode threshold.
    """
    n = c_water.size
    t_right = np.ones(n)
    t_right[-1] = 0.0                     # zero flux across the seafloor half-point
    diag = -(np.ones(n) + t_right) / dz**2 + omega**2 / c_water**2
    off = np.ones(n - 1) / dz**2
    vals = eigh_tridiagonal(diag, off, select="v", select_range=(kr2_lo, kr2_hi),
                            eigvals_only=True)
    return int(np.sum(vals > kr2_lo))


def modes(profile: SoundSpeedProfile, seabed: Seabed, freq_hz: float,
          water_depth_m: float, *, nz: int = 2000) -> ModeSet:
    """Trapped normal modes of the water column over a fluid halfspace.

    profile:       sound speed vs depth, POSITIVE DOWN. Sampled over
                   [0, water_depth_m]; `SoundSpeedProfile.c_at` CLAMPS rather
                   than extrapolates, so if the profile is shallower than the
                   water depth the deep part of the guide is treated as
                   isovelocity at the profile's deepest value. That is the
                   profile's documented contract, but it is a real modelling
                   assumption when the water is deeper than the profile.
    seabed:        fluid halfspace below the seafloor (`alpha` is not used).
    freq_hz:       frequency, Hz.
    water_depth_m: seafloor depth, m.
    nz:            number of grid points across the WATER COLUMN. The seabed
                   extension adds further points at the same spacing.

    Returns a `ModeSet` holding only trapped modes, ordered mode 1 first. If the
    seabed is no faster than the slowest water, nothing is trapped and the
    result is empty -- that is a physical answer, not a failure.
    """
    freq_hz = float(freq_hz)
    water_depth_m = float(water_depth_m)
    nz = int(nz)
    if not np.isfinite(freq_hz) or freq_hz <= 0:
        raise ValueError(f"freq_hz must be positive and finite, got {freq_hz}")
    if not np.isfinite(water_depth_m) or water_depth_m <= 0:
        raise ValueError(f"water_depth_m must be positive and finite, got {water_depth_m}")
    if nz < 2:
        raise ValueError(f"nz must be at least 2 to have an interior node, got {nz}")

    omega = 2.0 * np.pi * freq_hz

    # Seafloor at a half-point: water nodes are 0..nz-1, the last at D - dz/2,
    # and the first seabed node sits at D + dz/2.
    dz = water_depth_m / (nz - 0.5)
    z_water = np.arange(1, nz) * dz                     # interior water nodes

    # `c_at` clamps outside the sampled range rather than extrapolating, so a
    # profile shallower than the water turns the rest of the guide isovelocity
    # without saying so. A docstring will not stop a caller passing a 0-500 m
    # profile with a 1575 m water depth and reading the mode count as physics.
    z_top, z_bot = float(profile.z_down[0]), float(profile.z_down[-1])
    if water_depth_m > z_bot or z_top > 0.0:
        warnings.warn(
            f"sound-speed profile '{profile.source}' spans {z_top:g}-{z_bot:g} m "
            f"but the water column is 0-{water_depth_m:g} m; c is CLAMPED (not "
            f"extrapolated) outside the sampled range, so "
            f"{max(water_depth_m - z_bot, 0.0):g} m at the bottom and "
            f"{max(z_top, 0.0):g} m at the top are treated as isovelocity. Mode "
            f"counts from this call depend on that assumption.",
            UserWarning, stacklevel=2)

    c_water = np.asarray(profile.c_at(z_water), dtype=float)
    c_min = float(min(c_water.min(), float(profile.c_at(0.0))))

    lam_min = c_min / freq_hz
    if dz * _MIN_POINTS_PER_WAVELENGTH > lam_min:
        raise ValueError(
            f"grid too coarse to resolve the field: dz={dz:.3f} m over a "
            f"{water_depth_m:g} m column with nz={nz}, but the shortest water "
            f"wavelength at {freq_hz:g} Hz is {lam_min:.1f} m; need at least "
            f"{_MIN_POINTS_PER_WAVELENGTH:g} points per wavelength "
            f"(nz >= {int(np.ceil(_MIN_POINTS_PER_WAVELENGTH * water_depth_m / lam_min)) + 1})"
        )

    kr2_lo = (omega / float(seabed.c)) ** 2      # below this the mode radiates
    kr2_hi = (omega / c_min) ** 2                # above this it is evanescent everywhere
    empty = ModeSet(kr=np.zeros(0), phi=np.zeros((nz, 0)),
                    z_down=np.arange(nz) * dz, freq_hz=freq_hz,
                    water_depth_m=water_depth_m, seabed_extension_m=0.0)
    if kr2_lo >= kr2_hi:
        # Seabed no faster than the slowest water: the halfspace cannot turn
        # anything back, so no mode is trapped.
        return empty

    def solve(extension_m: float):
        n_seabed = min(int(np.ceil(extension_m / dz)), _MAX_SEABED_NODES)
        n_seabed = max(n_seabed, 1)
        diag, off = _assemble(c_water, dz, omega, seabed, n_seabed)
        vals, vecs = eigh_tridiagonal(diag, off, select="v",
                                      select_range=(kr2_lo, kr2_hi))
        # select='v' is half-open at the top; keep the window strictly open so
        # that every returned kr lies strictly between the two sound speeds.
        keep = (vals > kr2_lo) & (vals < kr2_hi)
        vals, vecs = vals[keep], vecs[:, keep]
        order = np.argsort(vals)[::-1]           # mode 1 first (largest kr)
        return vals[order], vecs[:, order], n_seabed * dz

    # How many trapped modes there ARE, computed without a seabed grid and so
    # without any truncation. Everything below is about resolving their shapes.
    n_expected = _threshold_count(c_water, dz, omega, kr2_lo, kr2_hi)
    if n_expected == 0:
        # Certified empty: no seabed extension could produce a mode. This is a
        # physical statement, not the solver giving up, so it is `converged`.
        return ModeSet(kr=np.zeros(0), phi=np.zeros((nz, 0)),
                       z_down=np.arange(nz) * dz, freq_hz=freq_hz,
                       water_depth_m=water_depth_m, seabed_extension_m=0.0,
                       n_modes_expected=0, converged=True)

    # Grow the seabed until every mode that exists has been resolved AND the
    # slowest evanescent tail is covered.
    #
    # The target must come from `n_expected`, never from the modes found so far.
    # Requiring only that the count stop changing across a doubling is a proxy
    # driven by what is already visible: the found modes' tails are short, so
    # the tail requirement is easily met, the count repeats, and the loop
    # declares success while a marginal mode is still hidden below the
    # truncation. Measured: a 112.02 m guide at 20 Hz over a 1700 m/s seabed
    # (threshold argument 1.502) returned 1 mode against an analytic 2 and
    # flagged itself converged.
    extension = max(0.5 * water_depth_m, 4.0 * float(seabed.c) / freq_hz)
    ext_max = max(4.0 * water_depth_m, 64.0 * float(seabed.c) / freq_hz)
    vals = vecs = None
    used = 0.0
    converged = False
    for _ in range(_MAX_GROWTH_STEPS):
        vals, vecs, used = solve(extension)
        gamma = np.sqrt(np.maximum(vals - kr2_lo, 0.0))
        gamma_min = float(gamma.min()) if gamma.size else 0.0
        # gamma_min == 0 means a mode sitting exactly at cutoff, whose tail no
        # finite grid resolves; inf sends the growth straight to the cap.
        needed = _TAIL_DECAYS / gamma_min if gamma_min > 0 else np.inf

        if vals.size > n_expected:
            # More roots than modes exist. The cutoff count is rigorous, so this
            # is the eigensolver at fault -- most likely a seabed quasi-continuum
            # ("box") mode from a long extension landing just inside the trapped
            # window. Growing further cannot remove it. Never seen in ~2200
            # tested configurations, but it must be loud rather than silent: a
            # spurious mode inflates the count Task 12 reasons from.
            warnings.warn(
                f"eigensolver returned {vals.size} roots in the trapped window "
                f"but the cutoff-limit count says {n_expected} exist "
                f"({freq_hz:g} Hz, {water_depth_m:g} m water, seabed "
                f"{seabed.c:g} m/s, seabed extension {used:.0f} m). Treat both "
                f"n_modes and the mode shapes as unreliable; n_modes_expected "
                f"is still exact.", UserWarning, stacklevel=3)
            break                                   # `converged` stays False
        if vals.size == n_expected and used >= needed:
            converged = True
            break
        if extension >= ext_max:
            break
        if vals.size == n_expected:
            # Only the tails are short: grow to exactly what they need.
            extension = min(max(2.0 * extension, 1.25 * needed), ext_max)
        else:
            # A mode is missing and its decay length is unknowable from here,
            # so go straight to the cap rather than creep up to it.
            extension = ext_max

    if vals.size == 0:
        return ModeSet(kr=np.zeros(0), phi=np.zeros((nz, 0)),
                       z_down=np.arange(nz) * dz, freq_hz=freq_hz,
                       water_depth_m=water_depth_m,
                       seabed_extension_m=float(used),
                       n_modes_expected=n_expected, converged=converged)

    n_total = vecs.shape[0]
    phi = np.vstack([np.zeros((1, vals.size)), vecs]) / np.sqrt(dz)
    # Eigenvector signs are arbitrary; fix them so results are reproducible.
    # Every mode leaves the pressure-release surface upward.
    phi *= np.where(phi[1, :] < 0, -1.0, 1.0)
    return ModeSet(kr=np.sqrt(vals), phi=phi,
                   z_down=np.arange(n_total + 1) * dz,
                   freq_hz=freq_hz, water_depth_m=water_depth_m,
                   seabed_extension_m=float(used),
                   n_modes_expected=n_expected, converged=converged)
