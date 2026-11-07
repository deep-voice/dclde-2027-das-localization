"""Depth-dependent sound speed.

Depth convention: this module is POSITIVE DOWN, the oceanographic convention and
the one gsw expects. The rest of the repo's geometry frame is NEGATIVE DOWN
(cable depths run +11 to -1575 m). `to_down` and `to_repo` are the only
sanctioned conversions; a sign error here is silent and corrupts everything.

Performance note (Ruling R3): `path_average` is called once per pick per grid
cell by the localization search, so it cannot afford to build a fresh quadrature
on every call. Instead, `__post_init__` precomputes a fine cumulative-slowness
curve S(z) = integral of 1/c from the profile's shallowest depth to z, once, at
construction time. `path_average(z1, z2)` then reduces to
(z2 - z1) / (S(z2) - S(z1)) via two O(log n) interpolations, independent of how
many times it is called afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid

_QUAD_POINTS = 4001


def to_down(pz):
    """Repo frame (negative down) -> profile frame (positive down)."""
    return -np.asarray(pz, dtype=float)


def to_repo(z_down):
    """Profile frame (positive down) -> repo frame (negative down)."""
    return -np.asarray(z_down, dtype=float)


@dataclass(frozen=True)
class SoundSpeedProfile:
    z_down: np.ndarray
    c: np.ndarray
    source: str
    # Human-readable validation status, set explicitly by whoever constructs
    # (or `dataclasses.replace`s) a profile that a result table needs to
    # report provenance for -- e.g. experiments.runner.PROFILES's three named
    # profiles. `None` means "not classified" (most profiles built in tests
    # never need one); a consumer wanting a display string for an unclassified
    # profile should show "unknown", not treat `None` as an error.
    #
    # This exists instead of parsing `source` (Task 14 review, Important 8):
    # `experiments.regime._profile_validation_note` used to string-match
    # substrings of the formatted `source` field (itself built by
    # `reanalysis.fetch_hycom_profile` as
    # f"hycom:{date}:{lat:.2f},{lon:.2f}:..."), and that fragility already
    # shipped one broken classifier (a dead check for "45.1492", a substring
    # that occurs in none of the real `:.2f`-formatted coordinates). A
    # structured field set once, at the one place a profile's provenance is
    # actually known, cannot silently drift out of sync with a display string
    # built for a different purpose.
    validated: str | None = None

    def __post_init__(self):
        z = np.asarray(self.z_down, dtype=float)
        c = np.asarray(self.c, dtype=float)
        if z.shape != c.shape:
            raise ValueError(f"z_down and c must be the same length: {z.shape} vs {c.shape}")
        if z.size < 1:
            raise ValueError("a profile needs at least one sample")
        if z.size >= 2 and np.any(np.diff(z) <= 0):
            raise ValueError("z_down must be strictly ascending (positive down)")
        object.__setattr__(self, "z_down", z)
        object.__setattr__(self, "c", c)

        # Ruling R3: precompute cumulative slowness on a fine grid so
        # path_average is O(1) per call instead of a fresh quadrature each time.
        z_fine = np.linspace(z.min(), z.max(), _QUAD_POINTS)
        c_fine = np.interp(z_fine, z, c)
        cumslow = cumulative_trapezoid(1.0 / c_fine, z_fine, initial=0.0)
        object.__setattr__(self, "_z_fine", z_fine)
        object.__setattr__(self, "_cumslow", cumslow)

    def c_at(self, z_down):
        """Sound speed at depth. Clamped outside the profile, never extrapolated."""
        return np.interp(np.asarray(z_down, dtype=float), self.z_down, self.c)

    def _cumulative_slowness(self, z_down):
        """S(z): cumulative integral of 1/c from the profile top to z.

        Beyond the sampled range this extends analytically at the clamped
        boundary speed (c[0] above the shallowest sample, c[-1] below the
        deepest one) so that S agrees with c_at's clamping instead of treating
        out-of-range distance as travelled infinitely fast (Finding 1).
        """
        z = np.asarray(z_down, dtype=float)
        z_min = self.z_down[0]
        z_max = self.z_down[-1]
        s = np.interp(np.clip(z, z_min, z_max), self._z_fine, self._cumslow)
        below = np.clip(z_min - z, 0.0, None) / self.c[0]
        above = np.clip(z - z_max, 0.0, None) / self.c[-1]
        return s - below + above

    def path_average(self, z1_down, z2_down):
        """Harmonic mean of c between two depths: the mean of slowness.

        Travel time is distance / c, so the average preserving travel time
        averages 1/c. The arithmetic mean would bias every range. Depth
        outside the sampled range is treated as travelled at the clamped
        boundary speed, matching c_at's documented clamping behaviour.

        Vectorised: z1_down/z2_down may be scalars or arrays (broadcastable).
        Returns a plain float when both inputs are scalars, otherwise an array
        matching the broadcast shape.
        """
        if self.z_down.size < 2:
            raise ValueError(
                "path_average requires a profile with at least two depth "
                "samples; this profile has only one and defines no path"
            )
        z1 = np.asarray(z1_down, dtype=float)
        z2 = np.asarray(z2_down, dtype=float)
        scalar_input = z1.ndim == 0 and z2.ndim == 0

        z1b, z2b = np.broadcast_arrays(z1, z2)
        lo = np.minimum(z1b, z2b)
        hi = np.maximum(z1b, z2b)
        diff = hi - lo

        s_lo = self._cumulative_slowness(lo)
        s_hi = self._cumulative_slowness(hi)
        denom = s_hi - s_lo

        near_zero = diff < 1e-9
        safe_denom = np.where(near_zero, 1.0, denom)
        result = np.where(near_zero, self.c_at(lo), diff / safe_denom)

        if scalar_input:
            return float(result)
        return result

    @classmethod
    def constant(cls, c: float, z_max: float = 3000.0) -> "SoundSpeedProfile":
        return cls(np.array([0.0, float(z_max)]), np.array([float(c), float(c)]),
                   f"constant:{c:g}")

    @classmethod
    def from_ts(cls, z_down, temperature_c, salinity_psu, lat, lon, source):
        """Build c(z) from in-situ temperature and practical salinity (TEOS-10)."""
        import gsw

        z = np.asarray(z_down, dtype=float)
        t = np.asarray(temperature_c, dtype=float)
        sp = np.asarray(salinity_psu, dtype=float)
        p = gsw.p_from_z(-z, lat)                    # gsw wants height, negative down
        sa = gsw.SA_from_SP(sp, p, lon, lat)
        ct = gsw.CT_from_t(sa, t, p)
        return cls(z, np.asarray(gsw.sound_speed(sa, ct, p), dtype=float), source)
