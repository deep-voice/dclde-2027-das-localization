"""Localization estimator that returns every misfit basin, not a point.

The objective is bimodal for a single cable: the source could be on either side.
A local optimiser seeded from a coarse grid lands in whichever lobe the grid
happened to favour, and a small parameter change can flip it. If an ablation
reports "positions moved a median of 80 m" while some calls jumped kilometres to
the mirror lobe, the median has hidden the only interesting thing that happened.

So solve() returns ranked basins, and compare() reports basin-stable deltas and
flip counts as two separate numbers. Merging them is prohibited by this
contract, not by convention.

Ruling R4: the coarse grid is evaluated in ONE model call, not a Python double
loop. `model.travel_time` honours the broadcasting contract (see
physics/sound_speed_models.py): sx/sy shaped (ny, nx, 1) against picks shaped (npicks,)
returns travel times of shape (ny, nx, npicks). `_grid_misfit` then computes the
analytic t0 and the misfit along the last axis with numpy reductions, mirroring
forward.best_t0 / forward.misfit_value's own definitions rather than calling
them per cell -- test_vectorised_grid_matches_scalar_misfit_at_several_cells
pins the two definitions together so they cannot drift apart.

Ruling R2: Basin.c is `float | None`. A PathAverage or RayTrace solve never
used a single scalar sound speed, so writing DEFAULT_C into that row would put
a plausible but wrong number into a committed results table -- worse than a
blank. `getattr(model, "c", None)` leaves it blank for those models and
populates it only for ConstantC (and anything else that genuinely has `.c`).

Beyond R2/R4, basin-finding is a two-tier search, not the brief's single
"lower than all eight neighbours" grid test (review round 1 found that this
under-reports, and round 2 found the fix that followed had its own bugs --
see task-4-report.md for every worked numeric example referenced below):

  - Tier 1, `_local_minima`: genuine turning points in the discretised misfit
    surface (grid cells <= all eight neighbours). These are real basins --
    Nelder-Mead started from one stays inside it with no artificial bound,
    because there is an actual ridge between it and any neighbour (verified
    directly for two lobes 1600 m apart). This is also naturally
    scale-adaptive: it resolves two basins however close together the grid
    can tell them apart, with no arbitrary distance threshold to tune -- the
    fixed-radius masking that used to do this job could mask away a real,
    nearby second lobe and report a fabricated stand-in position instead.
  - Tier 2, the fallback loop in `_select_basins`: only used when Tier 1
    finds fewer than `max_basins`. A second cable that fully resolves the
    single-cable ambiguity turns the old mirror lobe into a shoulder on an
    unbroken downhill slope to the global optimum, not a real turning point,
    so Tier 1 alone would report just one basin. The fallback greedily takes
    the best remaining grid cell at least `fallback_separation_m` from every
    basin already found, so "how much worse is the wrong side now" is still
    reported, just understood as an approximate regional estimate rather
    than a precisely located basin.

Refinement radius (review round 3, Finding A): a Tier-1 seed's box is capped
at half its distance to the *nearest other Tier-1 seed*, not just scaled to
cell size with no upper limit. Round 2's "real basins are self-limiting"
claim was verified only for well-separated lobes (1600 m); at 400 m offset
two genuine Tier-1 minima sit 1125 m apart on this grid, and an oversized,
cell-only-scaled box let Nelder-Mead cross the (real, but shallow at small
offsets) ridge between them and converge onto its neighbour's optimum
instead of its own -- silently collapsing two basins into one, which then
triggered Tier 2 to fabricate a replacement "basin 2" nowhere near the true
mirror. Bounding the box to half the nearest-seed distance makes that
crossing geometrically impossible (the box cannot reach the neighbour's
territory), independent of ridge height. Tier-2/fallback seeds are excluded
from this neighbour distance: they sit at least `fallback_separation_m` away
by construction of their own masking, and letting an unrelated, weaker
fallback candidate shrink a real basin's box below its required precision
would reintroduce Important-2's bug through a different door.

When the neighbour-capped radius would fall *below* half a grid cell --
i.e. the two seeds are closer together than about one cell -- Important-2's
precision requirement and this collapse-avoidance requirement genuinely
conflict, meaning the grid cannot separately resolve them at all. Rather
than pick one arbitrarily and refine it into a possibly-wrong position, that
seed is tagged `"unresolved"` and reported at its unrefined coarse-grid
position: an honest "there is a basin near here" rather than a fabricated
precise one.

The box fix alone is not sufficient: the post-refinement duplicate check
used a single `MIN_BASIN_SEPARATION_M` (1500 m) for every comparison, which
would itself discard two correctly, independently resolved "local_min"
basins as young as 800 m apart -- reintroducing the exact same fabrication
failure one step later (a real basin dropped here falls through to Tier 2,
which invents a stand-in for it). Two `"local_min"` basins are already
guaranteed genuinely distinct by the box capping above, so their duplicate
check uses a tight, numerical-duplicate-only tolerance instead; any
comparison touching a `"fallback"` or `"unresolved"` basin keeps the wider
`MIN_BASIN_SEPARATION_M`, since those positions are approximate by nature.

Tier-2 seeds get a radius capped well inside MIN_BASIN_SEPARATION_M, because
nothing stops Nelder-Mead from walking a Tier-2 seed downhill onto a
stronger basin if the box lets it reach that far. A Tier-1 refinement that
still lands on its own box boundary is discarded (the box should never need
its edge for a real, interior optimum). Tier-2 refinements are *expected* to
land on their boundary -- there is no real optimum in that masked-off
region by construction -- so they are kept, not discarded.

Every `Basin` carries `provenance` (`"local_min"`, `"fallback"`, or
`"unresolved"`) so a consumer can tell a confidently, independently located
basin from an approximate regional estimate or an unrefined stand-in
(review round 3, Finding B). `Solution.separation_m` returns `None`, and
`Solution.separation_resolved` is `False`, whenever either of the top two
basins is not `"local_min"` -- a fabricated or unresolved number must never
be indistinguishable from a measured one in a results table, which is the
whole reason this module reports basins instead of a point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import minimum_filter
from scipy.optimize import minimize

from dclde_localization.physics.forward import OBJECTIVES, best_t0, misfit_value

DEFAULT_SOURCE_Z = -30.0
MIN_BASIN_SEPARATION_M = 1500.0
MAX_BASINS = 4
# Padding added around the pick extent to build solve()'s own default coarse
# search grid. Named (rather than left as a literal inside solve()) so a
# caller reconstructing solve()'s exact grid outside of solve() itself --
# e.g. experiments/ablation.py's raytrace-fallback-over-the-search-grid
# measurement -- uses the same number by construction, not a second literal
# that could silently drift out of sync with this one.
COARSE_GRID_PAD_M = 45_000.0
# Misfits below this are indistinguishable from zero (well past the noise
# floor of the l1/l2 arithmetic itself) and are never distinguishable from
# real pick-timing noise (tens of milliseconds per pick on real data), so
# comparing them to sub-nanosecond precision compares optimizer artifacts,
# not fit quality. Used only as a *sort key* floor -- never written into a
# reported Basin.misfit, which must stay the number the estimator actually
# computed. See the final ranking step in `solve()`.
NOISE_FLOOR = 1e-6
# Tolerance for deciding a refined position is "on" its box boundary (and
# therefore untrustworthy -- see module docstring). Generous enough to catch
# a simplex that stalled just inside the edge without also discarding an
# interior optimum that happens to land near, but not on, the boundary.
_BOUNDARY_TOL_M = 1.0


@dataclass(frozen=True)
class Basin:
    x: float
    y: float
    z: float
    c: float | None
    t0: float
    misfit: float
    # How this basin was obtained -- see module docstring:
    #   "local_min":  a genuine, grid-resolved turning point, independently
    #                 refined. Confidently, precisely located.
    #   "fallback":   Tier 2's greedy, masked-off approximate estimate of
    #                 the best fit in a region with no real local minimum.
    #   "unresolved": a genuine local minimum whose nearest neighbour was
    #                 too close for this grid to separate; reported at its
    #                 unrefined coarse-grid position rather than risk
    #                 collapsing onto (or fabricating a stand-in for) it.
    provenance: str = "local_min"


@dataclass(frozen=True)
class Solution:
    basins: tuple[Basin, ...]
    n_picks: int
    model_name: str

    @property
    def best(self) -> Basin:
        return self.basins[0]

    @property
    def separation_resolved(self) -> bool:
        """Whether `separation_m` reflects two independently, confidently
        located basins, rather than a Tier-2/fallback estimate or an
        unresolved (too-close-to-separate) one standing in for it. A
        fabricated or unrefined number must never look like a measured one
        (review round 3, Finding B)."""
        if len(self.basins) < 2:
            return False
        return self.basins[0].provenance == "local_min" and self.basins[1].provenance == "local_min"

    @property
    def separation_m(self) -> float | None:
        if not self.separation_resolved:
            return None
        a, b = self.basins[0], self.basins[1]
        return float(np.hypot(a.x - b.x, a.y - b.y))


def _misfit_at(x, y, sz, px, py, pz, obs_t, model, objective):
    """Scalar-candidate misfit, used only for local refinement (a few hundred
    calls per basin, not the 26k-cell coarse grid -- that path is
    `_grid_misfit` below). Delegates to the frozen forward.py so refinement and
    the grid share one definition of the objective."""
    tt = model.travel_time(x, y, sz, px, py, pz)
    return misfit_value(obs_t, tt, objective), float(best_t0(obs_t, tt, objective))


def _grid_misfit(gx, gy, sz, px, py, pz, obs_t, model, objective):
    """Vectorised misfit over a whole (ny, nx) grid of candidate sources, in a
    single model call (Ruling R4).

    gx, gy: (ny, nx) coordinate grids (as from np.meshgrid).
    Returns (misfit, t0), each shape (ny, nx).

    Reproduces forward.best_t0 / forward.misfit_value exactly (residual =
    obs_t - tt; l1 -> median residual and sum of absolute deviations; l2 ->
    mean residual and RMS deviation) but along the grid's last axis instead of
    per scalar call, so it cannot silently drift from the frozen definitions --
    see test_vectorised_grid_matches_scalar_misfit_at_several_cells.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective: {objective!r}; expected one of {OBJECTIVES}")

    sx = gx[:, :, None]
    sy = gy[:, :, None]
    tt = model.travel_time(sx, sy, sz, px, py, pz)  # (ny, nx, npicks)
    r = np.asarray(obs_t, dtype=float) - tt

    if objective == "l1":
        t0 = np.median(r, axis=-1)
        misfit = np.abs(r - t0[..., None]).sum(axis=-1)
    else:  # "l2"
        t0 = np.mean(r, axis=-1)
        d = r - t0[..., None]
        misfit = np.sqrt(np.mean(d * d, axis=-1))
    return misfit, t0


def _local_minima(field, gx, gy):
    """Grid cells lower than or equal to all eight neighbours, best first, as
    (misfit, x, y, "local_min").

    Vectorised via a 3x3 minimum filter. Tie-break: among exactly-tied cells
    (bit-identical misfit -- happens for a perfectly straight,
    depth-and-offset-constant cable, where the mirror lobe is not just
    *almost* as good but numerically indistinguishable) numpy's argmin would
    silently prefer the first cell in raster order purely because of array
    memory layout. That is not a real preference for one side over the
    other, so ties are instead broken by preferring the *last* raster-order
    occurrence -- a fixed, documented convention rather than undocumented
    layout noise. An exact tie has no principled winner either way; fixing
    the convention just makes the choice reproducible.
    """
    mf = minimum_filter(field, size=3, mode="nearest")
    interior = np.zeros_like(field, dtype=bool)
    interior[1:-1, 1:-1] = True
    is_min = interior & (field <= mf)

    iy_idx, ix_idx = np.nonzero(is_min)
    vals = field[iy_idx, ix_idx]
    flat_idx = iy_idx * field.shape[1] + ix_idx
    order = np.lexsort((-flat_idx, vals))  # primary key: vals; tie-break: later raster position first
    return [(float(vals[i]), float(gx[iy_idx[i], ix_idx[i]]), float(gy[iy_idx[i], ix_idx[i]]), "local_min")
            for i in order]


def _select_basins(field, gx, gy, max_basins, fallback_separation_m):
    """Up to `max_basins` ranked candidates: real local minima first (Tier
    1), then, only if there are not enough of those, an approximate greedy
    fallback (Tier 2). See the module docstring for why the two tiers exist
    and use different refinement radii downstream.
    """
    out = _local_minima(field, gx, gy)[:max_basins]
    if len(out) >= max_basins:
        return out

    masked = field.copy()
    for _, cx, cy, _ in out:
        masked = np.where(np.hypot(gx - cx, gy - cy) < fallback_separation_m, np.inf, masked)

    while len(out) < max_basins:
        flat = masked.ravel()
        if not np.isfinite(flat).any():
            break
        last_occurrence = flat.size - 1 - int(np.argmin(flat[::-1]))
        iy, ix = np.unravel_index(last_occurrence, masked.shape)
        val = float(masked[iy, ix])
        if not np.isfinite(val):
            break
        cx, cy = float(gx[iy, ix]), float(gy[iy, ix])
        out.append((val, cx, cy, "fallback"))
        dist = np.hypot(gx - cx, gy - cy)
        masked = np.where(dist < fallback_separation_m, np.inf, masked)
    return out


def solve(px, py, pz, obs_t, model, *, sz=DEFAULT_SOURCE_Z, objective="l1",
          bounds=None, n_coarse=161) -> Solution:
    px, py, pz, obs_t = (np.asarray(v, dtype=float) for v in (px, py, pz, obs_t))
    if bounds is None:
        pad = COARSE_GRID_PAD_M
        bounds = (px.min() - pad, px.max() + pad, py.min() - pad, py.max() + pad)
    x0, x1, y0, y1 = bounds

    xs = np.linspace(x0, x1, n_coarse)
    ys = np.linspace(y0, y1, n_coarse)
    gx, gy = np.meshgrid(xs, ys)
    field, _ = _grid_misfit(gx, gy, sz, px, py, pz, obs_t, model, objective)

    # Tier 1 (real local minima) refinement radius upper bound: generous and
    # scaled to the grid's own cell size, so it can reach anywhere within the
    # seed's own quantisation error (up to half a cell in each axis) --
    # Important-2. The *actual* per-seed radius below is also capped by half
    # the distance to that seed's nearest Tier-1 neighbour (Finding A), since
    # an oversized box is only safe when there is nothing nearby to collapse
    # onto.
    cell = max(xs[1] - xs[0] if n_coarse > 1 else 0.0,
               ys[1] - ys[0] if n_coarse > 1 else 0.0)
    local_min_radius_cap = max(3.0 * cell, 10.0)

    # Tier 2 (fallback) refinement radius: capped well inside half of
    # MIN_BASIN_SEPARATION_M, because a fallback seed has no ridge protecting
    # it -- an oversized box lets Nelder-Mead walk it straight downhill onto
    # a stronger basin (observed directly). Tier-2 seeds are themselves
    # picked at least `fallback_separation_m` apart (padded so that even if
    # both slide the full radius towards each other they stay
    # >= MIN_BASIN_SEPARATION_M apart), so this box can never reach a
    # neighbouring Tier-2 (or Tier-1) basin's own territory.
    fallback_radius = min(local_min_radius_cap, MIN_BASIN_SEPARATION_M / 2.0)
    fallback_separation = MIN_BASIN_SEPARATION_M + 2.0 * fallback_radius

    seeds = _select_basins(field, gx, gy, MAX_BASINS, fallback_separation)

    # Finding A: each Tier-1 seed's box is capped by its distance to the
    # *nearest other Tier-1 seed only* -- computed once, up front, from the
    # untouched seed list, so tagging one seed "unresolved" below cannot
    # change another seed's own radius calculation. Tier-2/fallback seeds are
    # excluded deliberately: they sit >= fallback_separation_m away by
    # construction of their own masking (usually >> local_min_radius_cap at
    # coarse n_coarse), so letting them shrink a real basin's box would
    # reintroduce Important-2's bug through a different door.
    local_min_idx = [i for i, s in enumerate(seeds) if s[3] == "local_min"]
    nearest_local_min = {}
    for i in local_min_idx:
        xi, yi = seeds[i][1], seeds[i][2]
        ds = [np.hypot(xi - seeds[j][1], yi - seeds[j][2]) for j in local_min_idx if j != i]
        nearest_local_min[i] = min(ds) if ds else np.inf

    basins = []
    for i, (_, sx, sy, kind) in enumerate(seeds):
        if kind == "local_min":
            radius = min(local_min_radius_cap, 0.5 * nearest_local_min[i])
            if radius < 0.5 * cell:
                # Genuinely conflicting requirements: a box wide enough to
                # cover this grid's own half-cell quantisation error would
                # reach into the nearest neighbour's territory. The two are
                # not separately resolvable at this resolution -- report the
                # unrefined coarse-grid position rather than pick one and
                # either clip its precision or risk it collapsing onto the
                # other (see module docstring and Solution.separation_m).
                kind = "unresolved"
                rx, ry = sx, sy
                mis, t0 = _misfit_at(rx, ry, sz, px, py, pz, obs_t, model, objective)
            else:
                box = [(sx - radius, sx + radius), (sy - radius, sy + radius)]

                def obj(p, sz=sz):
                    return _misfit_at(p[0], p[1], sz, px, py, pz, obs_t, model, objective)[0]

                res = minimize(obj, [sx, sy], method="Nelder-Mead", bounds=box,
                               options={"xatol": 0.5, "fatol": 1e-9, "maxiter": 4000})
                rx, ry = float(res.x[0]), float(res.x[1])

                # A Tier-1 refinement landing on its own box boundary means
                # the box -- sized to comfortably contain an interior optimum
                # -- did not: a sign something is wrong, so this basin is
                # discarded rather than reported.
                on_boundary = (
                    abs(rx - box[0][0]) < _BOUNDARY_TOL_M or abs(rx - box[0][1]) < _BOUNDARY_TOL_M or
                    abs(ry - box[1][0]) < _BOUNDARY_TOL_M or abs(ry - box[1][1]) < _BOUNDARY_TOL_M
                )
                if on_boundary:
                    continue
                mis, t0 = _misfit_at(rx, ry, sz, px, py, pz, obs_t, model, objective)
        else:  # "fallback"
            radius = fallback_radius
            box = [(sx - radius, sx + radius), (sy - radius, sy + radius)]

            def obj(p, sz=sz):
                return _misfit_at(p[0], p[1], sz, px, py, pz, obs_t, model, objective)[0]

            # Tier-2 refinements are *expected* to land on their box
            # boundary -- there is no real optimum in this masked-off region
            # by construction (that is exactly why Tier 1 did not find one
            # here), so the box edge nearest the global optimum is the
            # honest "best point achievable on the wrong side" answer, kept
            # rather than discarded.
            res = minimize(obj, [sx, sy], method="Nelder-Mead", bounds=box,
                           options={"xatol": 0.5, "fatol": 1e-9, "maxiter": 4000})
            rx, ry = float(res.x[0]), float(res.x[1])
            mis, t0 = _misfit_at(rx, ry, sz, px, py, pz, obs_t, model, objective)

        # Duplicate check. Two "local_min" basins are already guaranteed
        # genuinely distinct by construction (their boxes are capped to
        # never reach each other's territory -- Finding A), so two real,
        # independently-resolved basins as close as 800 m apart must not be
        # discarded here as if they were duplicates: that was exactly how
        # Important-3/Finding A's fabrication bug reappeared (a real basin
        # dropped here falls through to Tier 2, which invents a stand-in).
        # A tight tolerance still catches genuine numerical duplicates (two
        # seeds converging to literally the same point). Any comparison
        # involving a "fallback" or "unresolved" basin keeps the wider
        # MIN_BASIN_SEPARATION_M, since those positions are approximate by
        # nature and a tight tolerance would under-deduplicate them.
        is_duplicate = False
        for b in basins:
            both_local_min = kind == "local_min" and b.provenance == "local_min"
            threshold = (2.0 * _BOUNDARY_TOL_M) if both_local_min else MIN_BASIN_SEPARATION_M
            if np.hypot(rx - b.x, ry - b.y) < threshold:
                is_duplicate = True
                break
        if is_duplicate:
            continue

        basins.append(Basin(x=rx, y=ry, z=sz, c=getattr(model, "c", None),
                            t0=t0, misfit=float(mis), provenance=kind))
        if len(basins) >= MAX_BASINS:
            break

    # Rank by misfit, but treat misfits below NOISE_FLOOR as tied on the sort
    # key only -- the *reported* Basin.misfit is always the raw value the
    # estimator computed, never the floor. Nelder-Mead's default initial
    # simplex step is relative to each seed's absolute coordinate value, so
    # two basins that are physically equally good (e.g. the two lobes of a
    # perfectly straight, depth-and-offset-constant cable) converge to
    # slightly different sub-1e-9 residuals depending on how far their seed
    # sits from the coordinate origin -- an optimizer artifact, not a real
    # difference in fit. Below NOISE_FLOOR that difference is not signal, so
    # ties are broken by `_select_basins`' ranking instead (a single
    # vectorised grid evaluation with no per-seed optimizer noise), via
    # Python's stable sort preserving `basins`' insertion order.
    basins.sort(key=lambda b: max(b.misfit, NOISE_FLOOR))
    return Solution(basins=tuple(basins), n_picks=int(px.size), model_name=model.name)


@dataclass(frozen=True)
class Comparison:
    n: int
    n_stable: int
    n_flipped: int
    flip_fraction: float
    stable_delta_m: np.ndarray


def compare(a: dict, b: dict, *, flip_threshold_m: float = 500.0) -> Comparison:
    """Position change between two solution sets, flips reported separately."""
    shared = sorted(set(a) & set(b))
    deltas, flips = [], 0
    for k in shared:
        pa, pb = a[k].best, b[k].best
        d = float(np.hypot(pa.x - pb.x, pa.y - pb.y))
        if d > flip_threshold_m:
            flips += 1
        else:
            deltas.append(d)
    n = len(shared)
    return Comparison(n=n, n_stable=n - flips, n_flipped=flips,
                      flip_fraction=(flips / n if n else 0.0),
                      stable_delta_m=np.asarray(deltas, dtype=float))
