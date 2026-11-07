"""Refracted rays through a depth-dependent profile.

Within a layer of constant gradient the ray is a circular arc, so range and
travel time are closed-form and the grazing limit stays finite. A quadrature in
depth diverges there and silently reports "no path" for rays that do exist --
this cost an hour during reconnaissance, so the closed form is not an
optimisation, it is the correctness fix.

Depths here are POSITIVE DOWN (profile frame). The model wrapper at the bottom
converts from the repo frame.

Ruling R9: `RayTrace.travel_time` does NOT call `direct_time` per pick. A
`brentq` root-find per pick cannot vectorise, and a later task evaluates a
161x161 grid of candidate sources for each of 524 calls -- looping would be on
the order of 10**9 root-finds. Instead, for a given source depth, `RayTrace`
lazily builds a 2-D table of direct travel time over (channel depth, horizontal
range) using the scalar `direct_time`/`max_direct_range` below (unchanged --
they are the ground truth the table approximates), caches it per source depth,
and answers `travel_time` by vectorised bilinear interpolation. Where the table
has no direct ray (or a query falls outside the table's covered domain), it
falls back to `PathAverage`, and the fraction that fell back is published --
per call via `last_fallback_fraction`, and pooled over every call recorded
since construction (the number a solve loop or ablation should read, not the
per-call scalar -- see the class docstring) via `fallback_fraction_overall` --
rather than hidden.

Table parameterisation: range is stored on a single ABSOLUTE range axis shared
by every depth row, geometrically spaced (dense near 0, coarse near the outer
cap) rather than linear. A first version of this table normalised range
per-row as u = r / max_direct_range(z), which independently made every row's
table build cheap -- but that broke bilinear interpolation's own precondition,
that its two axes are independent physical coordinates. Two adjacent depth
rows can have wildly different reach (a channel 30 m from the source's depth
had max_direct_range ~4.5e8 m on the Oregon fixture two grid rows away from a
channel with max_direct_range ~740 m -- a real, if extreme, "surface duct" in
this ray model, see `max_direct_range`'s docstring), so the same u meant
physically different ranges in each row; interpolating "time at u=0.13" across
such rows blended travel times for a 20 km ray in one row with a 100 m ray in
the other and produced errors of several seconds. A shared absolute axis has
no such mismatch: interpolating time(z, r) always blends times at the same
real range. Geometric spacing is what keeps it accurate at both ends of the
scale: refraction-limited ranges are hundreds to tens of thousands of metres
(`test_downward_refraction_limits_how_far_a_direct_ray_reaches`), while a
profile with little gradient has essentially no limit at all
(`test_constant_profile_has_no_range_limit_at_these_scales` asserts > 1e6 m);
a linear axis fine enough for the former wastes nearly all its points on
ranges no real deployment reaches, and one coarse enough to span to the cap
barely samples the former at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

from dclde_localization.physics.forward import travel_time as _straight_ray
from dclde_localization.physics.sound_speed_models import PathAverage
from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile, to_down

# --------------------------------------------------------------------------
# Scalar tracer arithmetic -- promoted unchanged (module constants replaced by
# a `profile` argument) from paper/scripts/recon_propagation_regime.py.
# This is the ground truth the table below approximates; do not tune it for
# table-building performance.
# --------------------------------------------------------------------------


def _layers(profile: SoundSpeedProfile, z1: float, z2: float):
    edges = sorted({z1, z2} | {float(e) for e in profile.z_down if z1 < e < z2})
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        ca, cb = float(profile.c_at(a)), float(profile.c_at(b))
        out.append((a, b, ca, cb, (cb - ca) / (b - a)))
    return out


def trace(profile, z1_down, z2_down, p):
    """Range and travel time for ray parameter p = cos(theta)/c, or None if it turns."""
    lo, hi = sorted((float(z1_down), float(z2_down)))
    rng = t = 0.0
    for za, zb, ca, cb, g in _layers(profile, lo, hi):
        sa2, sb2 = 1.0 - (p * ca) ** 2, 1.0 - (p * cb) ** 2
        if sa2 < 0.0 or sb2 < 0.0:
            return None
        sa, sb = np.sqrt(sa2), np.sqrt(sb2)
        if abs(g) < 1e-12:
            if sa < 1e-15:
                return None
            rng += p * ca * (zb - za) / sa
            t += (zb - za) / (ca * sa)
        else:
            rng += abs((sa - sb) / (p * g))
            t += abs(np.log((cb * (1.0 + sa)) / (ca * (1.0 + sb))) / g)
    return rng, t


def _p_grazing(profile, z1_down, z2_down) -> float:
    lo, hi = sorted((float(z1_down), float(z2_down)))
    return 1.0 / float(np.max(profile.c_at(np.linspace(lo, hi, 2001))))


def max_direct_range(profile, z1_down, z2_down) -> float:
    """Longest range a direct, water-borne ray can span between two depths."""
    got = trace(profile, z1_down, z2_down, _p_grazing(profile, z1_down, z2_down) * (1 - 1e-15))
    return float(got[0]) if got else 0.0


def direct_time(profile, z1_down, z2_down, target_range):
    """Travel time of the direct eigenray, or None if none reaches target_range."""
    target_range = float(target_range)
    if max_direct_range(profile, z1_down, z2_down) < target_range:
        return None
    pmax = _p_grazing(profile, z1_down, z2_down)

    def miss(p):
        got = trace(profile, z1_down, z2_down, p)
        return np.inf if got is None else got[0] - target_range

    lo = pmax * 1e-9
    if miss(lo) > 0:
        return None
    # Range is violently sensitive to p near grazing; interpolating across the
    # bracket costs ~1 ms, the size of the effect being measured. Bisect.
    p = brentq(miss, lo, pmax * (1 - 1e-15), xtol=1e-20, rtol=8.9e-16, maxiter=300)
    return float(trace(profile, z1_down, z2_down, p)[1])


# --------------------------------------------------------------------------
# Tabulated interpolator (Ruling R9).
# --------------------------------------------------------------------------

# Resolution over (channel depth, horizontal range). Chosen so bilinear
# interpolation error stays well under the ~96 ms pick noise floor, while
# keeping a table build (tens of thousands of `direct_time` brentq solves,
# each roughly sub-millisecond on the interior, up to a few ms near the
# grazing edge) to a matter of seconds rather than minutes.
#
# Where the table holds a value at all (see the existence-boundary note
# below), bilinear interpolation error is consistently small: order 1e-5 s,
# including immediately adjacent to the existence boundary -- four
# independent sweep strategies (uniform random, grazing-biased random, a
# dense grid, and a direct ray-parameter sweep down to eps=1e-6 from the
# true grazing p) all found a ~1e-5 to 6e-5 s ceiling with no growth toward
# the grazing limit, and the committed near-grazing accuracy test measures
# 4.85e-9 s, 1.36e-6 s and 3.27e-6 s at three specific non-fallback points.
# (An earlier draft of this comment claimed error degrades to ~1.4e-3 s near
# grazing, based on an independent review measurement; that number turned
# out to be confounded -- see below and task-3-report.md's fix round 3.)
#
# What is NOT exact: whether a direct ray exists at all. `max_direct_range`
# itself IS computed exactly (no root-find, no interpolation) whenever it is
# called directly. But the TABLE's existence boundary -- which is what
# `RayTrace.travel_time` actually consults per query -- is QUANTISED by the
# range axis: a row's last filled cell is the largest r_axis node at or
# below that row's true max_direct_range, so the table can report "no direct
# ray" for a range that is, in exact arithmetic, still slightly inside the
# true grazing limit. Measured gap: 11-720 m depending on channel depth (a
# small fraction of max_direct_range, but not always small in absolute
# terms -- see task-3-report.md). Bounded, in general, by the local r_axis
# spacing. This is not silent: those queries fall back to `PathAverage`
# (which, in this narrow band, differs from the true refracted ray by
# something between ~4e-4 s and ~9e-3 s depending on geometry -- not a flat
# figure, see the report), and `last_fallback_fraction` reports exactly this,
# which is what that field exists for.
# See test_tabulated_travel_time_agrees_with_direct_scalar_calls_close_to_
# grazing (interpolation accuracy where the table holds a value) and
# test_raytrace_falls_back_near_the_quantised_grazing_boundary (the
# existence-boundary gap, and that it is reported rather than silent).
TABLE_N_Z = 121
TABLE_N_R = 361
# Depth padding added beyond the profile's sampled range and the source depth,
# so the table's z-axis comfortably covers real channel depths without needing
# to know the cable geometry here.
TABLE_Z_MARGIN = 200.0
# Outer range cap and geometric spacing (see module docstring): 150 km
# comfortably covers this dataset's picks (typically 5-40 km, up to ~95 km
# along cable) plus the localization search's coarse-grid excursions, and
# ranges no real profile is expected to shadow-zone-limit below in practice.
TABLE_R_CAP = 150_000.0
TABLE_R_MIN = 1.0
# r=0 (a purely vertical ray) makes direct_time's own brentq bracket
# degenerate (its lower bound already overshoots target_range=0) and returns
# None even though the vertical ray is physically real; TABLE_R_MIN keeps
# every table node off that one pathological point. Real picks are never
# within a metre of directly above/below the source, so this costs nothing.


@dataclass
class _DirectTimeTable:
    """A 2-D (channel depth, horizontal range) grid of direct travel time.

    `time` holds NaN wherever no direct ray reaches that (depth, range) cell,
    or wherever `direct_time` failed to converge; `interp` propagates that NaN
    through bilinear interpolation (any query whose four surrounding corners
    include a NaN, or that falls outside the axes entirely, comes back NaN),
    which is exactly the fallback signal `RayTrace.travel_time` needs.

    Both axes are real physical coordinates -- shared across every row -- so
    bilinear interpolation is always blending travel times at the same actual
    range, never at a per-row-rescaled fraction of it (see module docstring).
    """

    z_axis: np.ndarray
    r_axis: np.ndarray
    time: np.ndarray

    def __post_init__(self):
        self.interp = RegularGridInterpolator(
            (self.z_axis, self.r_axis), self.time,
            method="linear", bounds_error=False, fill_value=np.nan)

    def query(self, z, r):
        """Vectorised bilinear lookup. z, r broadcast against each other."""
        z = np.asarray(z, dtype=float)
        r = np.asarray(r, dtype=float)
        zb, rb = np.broadcast_arrays(z, r)
        pts = np.stack([zb, rb], axis=-1)
        return self.interp(pts)


def _z_axis_with_knots(z_lo: float, z_hi: float, n_z: int,
                        profile: SoundSpeedProfile) -> np.ndarray:
    """A base linspace, plus the profile's own sample depths inside [z_lo, z_hi].

    `c(z)` is piecewise-linear between the profile's samples (see `_layers`),
    so its gradient has a real kink at every one of them: `direct_time(zs_down,
    z2, r)`, as a function of `z2`, inherits a kink at every profile depth
    between `zs_down` and `z2`. A z-axis row that straddles a kink without a
    grid line exactly on it linearly interpolates across a genuine
    discontinuity in slope, not just ordinary curvature -- this produced the
    multi-millisecond errors traced in task-3-report.md (Finding 1), worst at
    the profile's shallow clamp boundary but not exclusive to it. Aligning a
    grid line to every knot keeps each interpolation cell inside a single
    constant-gradient segment, matching the tracer's own layer decomposition.
    """
    base = np.linspace(z_lo, z_hi, n_z)
    knots = profile.z_down[(profile.z_down > z_lo) & (profile.z_down < z_hi)]
    if knots.size:
        # Drop base points that would sit almost on top of a knot: harmless
        # numerically, just a wasted near-zero-spacing pair.
        too_close = np.any(np.abs(base[:, None] - knots[None, :]) < 1e-6, axis=1)
        base = base[~too_close]
    return np.unique(np.concatenate([base, knots]))


def _build_table(profile: SoundSpeedProfile, zs_down: float,
                  n_z: int = TABLE_N_Z, n_r: int = TABLE_N_R) -> _DirectTimeTable:
    zp_lo, zp_hi = float(profile.z_down[0]), float(profile.z_down[-1])
    margin = max(TABLE_Z_MARGIN, 0.1 * (zp_hi - zp_lo))
    z_lo = min(zp_lo, zs_down, 0.0) - margin
    z_hi = max(zp_hi, zs_down, 0.0) + margin
    z_axis = _z_axis_with_knots(z_lo, z_hi, n_z, profile)
    r_axis = np.geomspace(TABLE_R_MIN, TABLE_R_CAP, n_r)

    # z_axis may be longer than n_z: knot alignment adds a grid line at every
    # profile sample depth inside [z_lo, z_hi] on top of the base linspace.
    time = np.full((z_axis.size, n_r), np.nan)
    for iz, z in enumerate(z_axis):
        zf = float(z)
        mx = max_direct_range(profile, zs_down, zf)
        if mx <= 0.0:
            continue
        # Only the in-reach cells need the expensive brentq solve; cells
        # beyond this row's reach stay NaN without calling direct_time.
        ir_max = int(np.searchsorted(r_axis, mx, side="right"))
        for ir in range(ir_max):
            try:
                t = direct_time(profile, zs_down, zf, float(r_axis[ir]))
            except Exception:
                t = None
            if t is not None:
                time[iz, ir] = t
    return _DirectTimeTable(z_axis, r_axis, time)


@dataclass
class RayTrace:
    """Refracted direct ray, falling back to path-average where none exists.

    The fallback is not a detail to hide: where it fires, this model is not
    describing the propagation that actually happened, and the fraction is
    reported so results can say so.

    Per Ruling R9, `travel_time` never root-finds per pick. It looks up a
    tabulated approximation of `direct_time`/`max_direct_range`, built lazily
    and cached per source depth (fixed within one localization), and falls
    back to `PathAverage` -- vectorised, via `np.where` -- wherever the table
    has no entry (no direct ray, or the query fell outside the table's
    covered domain). A profile with zero variation in c is a special case
    (see `travel_time`): it skips the table entirely, since there is nothing
    to tabulate an approximation of.

    `n_z`/`n_r` default to the production resolution (`TABLE_N_Z`/`TABLE_N_R`)
    but are overridable so tests that only exercise control flow -- caching,
    broadcasting -- can use a tiny grid instead of paying for a full,
    accuracy-grade table build every time.
    """

    profile: SoundSpeedProfile
    fallback: str = "path_average"
    n_z: int = TABLE_N_Z
    n_r: int = TABLE_N_R
    # Per-call fallback bookkeeping (Task 14 review, Important 4). Each
    # `travel_time()` call appends its own (n_queries, n_fallback) here rather
    # than overwriting a single scalar -- a solve() loop calls `travel_time`
    # many times (once for the whole coarse grid, then again and again during
    # Nelder-Mead refinement of each basin, ending on a tiny few-point call for
    # the final misfit), and a plain "last call" scalar silently reports
    # whichever of those happened to run last -- usually the smallest,
    # least-representative one, biased toward zero. Measured concretely: 0.719
    # at a call's own best-fit position versus 0.994 pooled over the coarse
    # search grid that actually drove the localization (see
    # `fallback_fraction_overall` below and experiments/ablation.py's
    # `raytrace_fallback_fraction_over_search_grid`). `last_fallback_fraction`
    # is kept, unchanged in meaning (the most recent call only), for the
    # existing per-query tests below that check one `travel_time()` call at a
    # time; anything summarising a whole solve (or ablation) must use
    # `fallback_fraction_overall` / `reset_fallback_record`, not this scalar.
    fallback_calls: list = field(default_factory=list, init=False, repr=False)
    _tables: dict = field(default_factory=dict, init=False, repr=False)
    table_build_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        if self.fallback != "path_average":
            raise ValueError(f"unknown fallback: {self.fallback!r}; expected 'path_average'")

    @property
    def name(self) -> str:
        return f"raytrace:{self.profile.source}"

    @property
    def last_fallback_fraction(self) -> float:
        """Fallback fraction of the MOST RECENT `travel_time()` call only.

        Not representative of a whole `solve()` loop or ablation -- see the
        `fallback_calls` field comment. Use `fallback_fraction_overall` after
        recording every call of interest (reset first with
        `reset_fallback_record()` if earlier calls should not count).
        """
        if not self.fallback_calls:
            return 0.0
        n_total, n_fallback = self.fallback_calls[-1]
        return (n_fallback / n_total) if n_total else 0.0

    @property
    def fallback_fraction_overall(self) -> float:
        """Fallback fraction pooled over every `travel_time()` call recorded
        since construction (or the last `reset_fallback_record()`) -- the
        number to read after a solve loop or a batch of calls, not
        `last_fallback_fraction`."""
        n_total = sum(t for t, _ in self.fallback_calls)
        n_fallback = sum(f for _, f in self.fallback_calls)
        return (n_fallback / n_total) if n_total else 0.0

    def reset_fallback_record(self) -> None:
        """Clear the accumulated `fallback_calls` record so a subsequent
        `fallback_fraction_overall` reflects only calls made after this
        point (e.g. isolating one call's worth of queries, or one ablation's
        worth, from whatever ran before it on a shared instance)."""
        self.fallback_calls = []

    def _table_for(self, zs_down: float) -> _DirectTimeTable:
        key = round(float(zs_down), 6)
        table = self._tables.get(key)
        if table is None:
            table = _build_table(self.profile, zs_down, n_z=self.n_z, n_r=self.n_r)
            self._tables[key] = table
            self.table_build_count += 1
        return table

    def travel_time(self, sx, sy, sz, px, py, pz) -> np.ndarray:
        # A profile with literally no variation in c has no gradient anywhere,
        # so no ray can refract: the exact answer is the same straight-line
        # formula ConstantC uses, available in closed form. Tabulating an
        # approximation to it would only add interpolation error a table
        # cannot avoid for a curved (non-piecewise-linear) function of range --
        # see test_raytrace_model_reduces_to_constant_c_on_a_flat_profile,
        # which checks agreement with ConstantC to rtol=1e-8, well past what
        # bilinear interpolation over any practical grid can guarantee.
        if float(np.ptp(self.profile.c)) == 0.0:
            result = _straight_ray(sx, sy, sz, px, py, pz, float(self.profile.c[0]))
            self.fallback_calls.append((int(np.asarray(result).size), 0))
            return result

        px = np.asarray(px, dtype=float)
        py = np.asarray(py, dtype=float)
        pz = np.asarray(pz, dtype=float)
        sx_a = np.asarray(sx, dtype=float)
        sy_a = np.asarray(sy, dtype=float)
        horiz = np.sqrt((px - sx_a) ** 2 + (py - sy_a) ** 2)

        # Ruling R9: source depth is fixed within a localization (it is the
        # assumed depth, not part of the grid search over x/y), so the table
        # is keyed on a single scalar source depth. sx/sy may still vary over
        # a grid; only sz must be scalar.
        sz_arr = np.asarray(sz, dtype=float)
        if sz_arr.size != 1:
            raise ValueError(
                "RayTrace.travel_time requires a scalar sz (source depth is "
                "fixed per localization and keys the cached table); got "
                f"shape {sz_arr.shape}"
            )
        zs_down = float(to_down(sz_arr))
        zc_down = to_down(pz)

        table = self._table_for(zs_down)
        tabulated = table.query(zc_down, horiz)
        missing = np.isnan(tabulated)

        fallback_model = PathAverage(self.profile)
        fallback_time = fallback_model.travel_time(sx, sy, sz, px, py, pz)

        out = np.where(missing, fallback_time, tabulated)
        self.fallback_calls.append((int(missing.size), int(np.sum(missing))))
        return out
