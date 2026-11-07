"""The Plan B gate: does moveout carry source-depth information?

Plan A's findings closed the amplitude route to depth -- the transmission-loss
study measured the receiver-depth coefficient of received level with an
interval spanning zero in all 16 configurations. So depth, if it is reachable
at all, is reachable through travel-time moveout. This module measures whether
it is, before any depth-observability machinery is built.

Two questions, deliberately separated:

  * DEPTH SENSITIVITY. With `x`, `y` and `t0` free to compensate, how sharply
    does the misfit bound source depth? That marginal width -- not the width of
    a slice at a fixed horizontal position -- is what a depth solve would
    actually achieve.
  * PATH. Direct versus bottom-bounce, compared on travel-time CURVATURE. The
    transmission-loss study answered this from amplitude, on an observable whose
    dB scale failed its own calibration check. This answers it from timing,
    which that problem does not touch.

OBJECTIVE. This module profiles under L2, not the repo's default L1, because a
Delta-chi-squared reading of a misfit width needs a Gaussian likelihood. The L1
solve still supplies the starting position. Mixing the two silently would make
`sigma_z_m` uninterpretable, so the choice is stated rather than assumed.

Depths are NEGATIVE-DOWN here, the repo convention outside `soundspeed/`. This
module's depth-sign conventions are consistent with `dclde_localization.physics.sound_speed_models.ConstantC`,
which delegates to `forward.travel_time` with no depth-sign conversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from dclde_localization.physics.forward import misfit_value, travel_time
from dclde_localization.physics.solve import solve
from dclde_localization.physics.sound_speed_models import ConstantC

DEFAULT_C = 1480.0
NULL_SOURCE_Z = -30.0

# Delta-chi-squared = 1 is the standard 1-sigma interval for one parameter with
# the others profiled out.
_DELTA_CHI2_1SIGMA = 1.0


def robust_sigma(residuals) -> float:
    """1.4826 x median absolute deviation.

    Robust rather than the plain standard deviation because this dataset's
    residuals carry a gross mis-association tail out to tens of seconds (Plan A
    section 3.2), which would inflate an ordinary sigma by an order of magnitude
    and make every depth interval look falsely wide.
    """
    r = np.asarray(residuals, dtype=float)
    if r.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def _misfit_at_depth(z, px, py, pz, obs_t, c, x0, y0):
    """L2 misfit at source depth `z`, minimised over x and y from a warm
    start `(x0, y0)`. Returns `(misfit, x_opt, y_opt)` so the caller can walk
    the optimum forward into the next depth's start.

    A single Nelder-Mead call is cheap and reliable here because the caller
    supplies a good starting point: either the previous depth step's optimum
    (adjacent depths have adjacent optima, so each step is a short local
    move) or, for the first step, a one-off `solve.solve` seed. Calling
    `solve.solve`'s full coarse-grid search at every depth step -- the
    module's first approach -- was correct but far too expensive for a
    profile of O(100) depths (a 161x161 grid per step is ~3e6 model
    evaluations per depth).

    `t0` needs no separate minimisation -- `forward.misfit_value` eliminates
    it in closed form for every candidate `(x, y, z)`.
    """
    def obj(p):
        tt = travel_time(p[0], p[1], z, px, py, pz, c)
        return misfit_value(obs_t, tt, "l2")

    # Same tolerances as `_fit_xy` below, deliberately (fix round 1, MEDIUM 4):
    # both are the same Nelder-Mead x/y refit of an L2 travel-time misfit, so
    # there is no reason for them to converge to different precisions -- and
    # the direction matters here, not just the match. `xatol=0.1` m is
    # 0.1/1480 ~= 68 us of positional slop, which EXCEEDS the ~23 us of
    # unresolvable depth signal `depth_profile`'s own docstring measures for
    # a monotonic single-arc (single-cable) geometry -- exactly the geometry
    # of every direction-inconsistent call in the real run. Loosening this to
    # match `_fit_xy`'s old 0.1/1e-12 (as a first pass at this module did)
    # would put the optimiser's own convergence slop at the same order as the
    # quantity being measured for precisely the calls that misbehave, making
    # "local-minimum artifact" and "numerical noise floor" indistinguishable.
    # `_fit_xy` is tightened to match THIS, not the other way around.
    res = optimize.minimize(obj, np.array([x0, y0], dtype=float),
                            method="Nelder-Mead",
                            options={"xatol": 0.05, "fatol": 1e-13, "maxiter": 5000})
    return float(res.fun), float(res.x[0]), float(res.x[1])


def depth_profile(px, py, pz, obs_t, *, z_grid, c=DEFAULT_C, x0=None, y0=None):
    """Profiled misfit versus source depth.

    At every candidate depth, x and y are RE-MINIMISED and the optimum is
    carried forward as the next depth's starting point (a warm-started walk
    down `z_grid`, in the order given). That is the whole point: depth trades
    off against horizontal range and emission time, so a profile taken at a
    fixed horizontal position is spuriously sharp and would overstate what a
    depth solve can achieve.

    If `x0`/`y0` are not supplied, the walk is seeded by one call to
    `solve.solve` at `NULL_SOURCE_Z` (a depth-agnostic starting guess -- see
    `solve.DEFAULT_SOURCE_Z`, the same value). In the real pipeline (Task 3)
    the caller already has a solved `(x, y)` from its own localisation and
    passes it as `x0`/`y0`, so this extra solve only fires for tests and
    ad-hoc callers.

    IMPORTANT -- depth/range degeneracy. A cable whose receiver depth varies
    MONOTONICALLY with along-cable position is nearly depth/range-degenerate
    under this re-minimisation: a source-depth change is almost exactly
    mimicked by a source-position shift along the cable, because the two
    perturbations point the same "direction" in travel-time space for every
    receiver at once. Measured on this module's own synthetic fixture, 400 m
    of depth error left only:
      single arc, depth monotonic along the cable:    ~23 us  of unresolvable
                                                                RMS residual
      single arc, depth V-shaped along the cable:     ~918 us
      two arcs at clearly different bearings:         ~3044 us
    A single monotonic arc is the worst case and the two-cable case is not
    close to it -- receivers approaching the source from different bearings
    cannot be reconciled by one common horizontal shift, so depth is
    identifiable in a way a single arc's geometry does not allow. This is why
    the two-cable geometry matters for depth, not just for the mirror-side
    ambiguity `solve.py` already handles in (x, y).
    """
    px, py, pz, obs_t = (np.asarray(v, dtype=float) for v in (px, py, pz, obs_t))
    z_grid = np.asarray(z_grid, dtype=float)

    if x0 is None or y0 is None:
        seed = solve(px, py, pz, obs_t, ConstantC(c), sz=NULL_SOURCE_Z, objective="l2")
        gx = float(x0) if x0 is not None else float(seed.best.x)
        gy = float(y0) if y0 is not None else float(seed.best.y)
    else:
        gx, gy = float(x0), float(y0)

    out = np.empty(z_grid.size, dtype=float)
    for i, z in enumerate(z_grid):
        out[i], gx, gy = _misfit_at_depth(z, px, py, pz, obs_t, c, gx, gy)
    return z_grid, out


def _nearest_crossing(z, chi2, k, step, threshold):
    """Walk away from the minimum index `k` in direction `step` (+-1) and
    return the z of the FIRST point where `chi2` crosses `threshold`,
    linearly interpolated between the bracketing pair.

    Fix round 1, MEDIUM 5: the previous implementation fed the whole
    half-array to `np.interp`, which silently assumes `chi2` is monotone on
    that side -- wrong on a multi-modal profile (precisely the failure the
    bidirectional walk check exists to catch, so not hypothetical here).
    Walking outward one step at a time and stopping at the FIRST crossing
    finds the crossing NEAREST the minimum regardless of what the profile
    does further out, which is the physically meaningful 1-sigma bound.
    Returns None if `threshold` is never reached before the grid edge.
    """
    i = k
    while True:
        j = i + step
        if j < 0 or j >= z.size:
            return None
        if chi2[i] < threshold <= chi2[j]:
            t = (threshold - chi2[i]) / (chi2[j] - chi2[i])
            return float(z[i] + t * (z[j] - z[i]))
        i = j


def marginal_sigma_z(z_grid, profiled_misfit, n_picks, resid_sigma) -> tuple[float, str]:
    """Half-width of the profiled depth minimum at Delta-chi-squared = 1.

    Returns `(sigma_z, reason)`. `sigma_z` is `inf` -- meaning "this call
    does not bound depth" -- exactly when `reason` is non-empty; `reason` is
    `""` when `sigma_z` is finite. Fix round 1, MAJOR 2: an undifferentiated
    `inf` conflated "the grid was too shallow to reach" with "the data
    themselves never bound depth", which a reader could not tell apart
    without re-deriving it from the raw profile. The reasons:

      "invalid"      -- degenerate input (profile too short, non-finite
                        values, or a non-positive noise estimate).
      "edge_deep"     -- the misfit minimum sits at the DEEPER end of
                        `z_grid` (the more negative value): the grid's own
                        depth floor, not the data, is what stopped the walk.
      "edge_surface"  -- the minimum sits at the SHALLOWER end (closest to
                        0 m). This one is NOT a grid-extension artifact --
                        0 m is the physical sea surface and cannot be
                        extended past.
      "not_crossed"   -- the minimum is interior, but Delta-chi-squared = 1
                        is not reached on both sides inside the grid.
    """
    z = np.asarray(z_grid, dtype=float)
    m = np.asarray(profiled_misfit, dtype=float)
    if z.size < 3 or not np.all(np.isfinite(m)) or resid_sigma <= 0:
        return float("inf"), "invalid"

    k = int(np.argmin(m))
    if k == 0 or k == z.size - 1:
        deep_end_is_first = z[0] < z[-1]  # ascending z_grid: index 0 is deepest
        at_first = k == 0
        is_deep = at_first == deep_end_is_first
        return float("inf"), ("edge_deep" if is_deep else "edge_surface")

    # L2 misfit is an RMS residual, so n * (m^2 - m_min^2) / sigma^2 is the
    # chi-squared increase.
    chi2 = n_picks * (m**2 - m[k] ** 2) / (resid_sigma**2)
    left = _nearest_crossing(z, chi2, k, -1, _DELTA_CHI2_1SIGMA)
    right = _nearest_crossing(z, chi2, k, +1, _DELTA_CHI2_1SIGMA)
    if left is None or right is None:
        return float("inf"), "not_crossed"

    return float(abs(right - left) / 2.0), ""


def bounce_travel_time(sx, sy, sz, px, py, pz, c, bottom_z):
    """Travel time for one flat-seafloor reflection, by image source.

    The source is mirrored in a flat seafloor at `bottom_z` (NEGATIVE-DOWN, as
    everywhere outside `soundspeed/`), putting the image at `2*bottom_z - sz`.

    Flat-bottom is an approximation on a shelf-to-abyssal margin. It is used
    here because the gate needs the two paths' CURVATURE to differ measurably,
    not because the real seafloor is flat -- and a sloping bottom changes the
    curvature more, not less, so a flat bottom is the conservative choice for a
    discrimination test.
    """
    px, py, pz = (np.asarray(v, dtype=float) for v in (px, py, pz))
    z_img = 2.0 * float(bottom_z) - float(sz)
    return np.sqrt((px - sx) ** 2 + (py - sy) ** 2 + (pz - z_img) ** 2) / c


def _fit_xy(tt_fn, px, py, obs_t, x0, y0):
    def obj(p):
        return misfit_value(obs_t, tt_fn(p[0], p[1]), "l2")

    # Tightened to match `_misfit_at_depth`'s tolerances (fix round 1,
    # MEDIUM 4) -- see that function's comment for why the tighter value,
    # not the looser one, is the correct match.
    res = optimize.minimize(obj, np.array([x0, y0], dtype=float),
                            method="Nelder-Mead",
                            options={"xatol": 0.05, "fatol": 1e-13, "maxiter": 5000})
    return float(res.fun), float(res.x[0]), float(res.x[1])


def path_comparison(px, py, pz, obs_t, *, bottom_z, c=DEFAULT_C,
                    source_z=NULL_SOURCE_Z) -> dict:
    """Fit the direct and the one-bounce model to the same picks and compare.

    `delta_misfit` is bounce minus direct, so a NEGATIVE value favours the
    bounce path. Source depth is held at the null's -30 m on both sides: this
    asks which PATH fits better, not which depth.
    """
    px, py, pz, obs_t = (np.asarray(v, dtype=float) for v in (px, py, pz, obs_t))
    x0, y0 = float(px.mean()), float(py.mean())

    md, xd, yd = _fit_xy(
        lambda x, y: travel_time(x, y, source_z, px, py, pz, c), px, py, obs_t, x0, y0)
    mb, xb, yb = _fit_xy(
        lambda x, y: bounce_travel_time(x, y, source_z, px, py, pz, c, bottom_z),
        px, py, obs_t, x0, y0)

    return {"misfit_direct": md, "misfit_bounce": mb, "delta_misfit": mb - md,
            "x_direct": xd, "y_direct": yd, "x_bounce": xb, "y_bounce": yb}


GATE_COLUMNS = ["call", "n_picks", "depth_span_m", "n_cables", "in_stratum", "solved",
                "flipped", "sigma_z_m", "sigma_z_reason", "z_at_min_m", "resid_sigma_ms",
                "bottom_z_m", "misfit_direct", "misfit_bounce", "delta_misfit",
                "delta_misfit_over_resid_sigma", "delta_chi2",
                "sigma_z_m_reverse", "profile_max_discrepancy_s", "walk_consistent"]


def _walk_consistent(sigma_forward: float, sigma_reverse: float, tol: float = 0.10,
                     grid_spacing_m: float | None = None) -> bool:
    """Do the deep-to-shallow and shallow-to-deep warm-started walks agree?

    A local-minimum-tracking walk is direction-dependent (it can follow a
    mirror lobe across depths on this multi-basin dataset); an honest one is
    not. Both non-finite (neither walk bounded depth) counts as agreement --
    that is the same answer reached twice. Exactly one non-finite, or a NaN
    input, is disagreement: a NaN is not a value this comparison can trust
    silently.

    `grid_spacing_m`, if given, makes the tolerance `max(10% relative, one
    grid spacing)` (fix round 1, MINORS): a purely relative tolerance is not
    defensible for two small sigma_z values a few metres apart, where 10%
    can be sub-metre -- finer than `z_grid` itself resolves. On this run's
    real data this did not change any verdict (smallest finite disagreement
    measured was 7.1 m against a 5 m grid spacing), but the leniency is
    right in general, not just here.
    """
    a, b = float(sigma_forward), float(sigma_reverse)
    if np.isnan(a) or np.isnan(b):
        return False
    fa, fb = np.isfinite(a), np.isfinite(b)
    if not fa and not fb:
        return True
    if fa != fb:
        return False
    denom = max(abs(a), abs(b))
    if denom == 0.0:
        return True
    if abs(a - b) / denom <= tol:
        return True
    if grid_spacing_m is not None and abs(a - b) <= grid_spacing_m:
        return True
    return False


def gate_table(picks, grid, *, min_picks=8, span_min_m=200.0, z_grid=None,
               c=DEFAULT_C, flip_threshold_m=500.0,
               bidirectional_sample=None, bidirectional_seed=0):
    """Per-call depth sensitivity and path comparison, plus the denominators.

    The stratum is pre-declared (see the config's notes): n_picks >= min_picks,
    receiver-depth span >= span_min_m, basin-stable, solved under the null.
    Calls outside it are still rowed, with `in_stratum` False, so a reader can
    see what was excluded rather than inferring it from a missing row.

    `n_cables` counts the distinct cables a call's own picks span (from the
    picks' `cable` column) -- recorded for every call regardless of stratum
    membership, and pre-declared (see the config's notes) as a dimension to
    report, not a term in the pass/fail criterion.

    THE PROFILE IS WALKED IN BOTH DIRECTIONS for in-stratum, resolvable
    calls: once deep-to-shallow through `z_grid` (z_grid's own ascending
    order -- z_grid = linspace(-600, 0, ...) runs deep to shallow) and once
    the reverse. Each
    walk warm-starts (x, y) from the previous depth step's optimum, so a walk
    that happens to follow a mirror lobe (this dataset has explicit
    multi-basin structure -- see `solve.py`) rather than the true global
    optimum at some depths would trace a smooth, believable, WRONG profile
    that a single-direction walk cannot catch. `sigma_z_m` is always the
    FORWARD (as-given `z_grid` order) walk's result -- the direction the
    pre-declared criterion is applied to -- `sigma_z_m_reverse` and
    `profile_max_discrepancy_s` (max abs difference between the two
    profiled-misfit curves, same units as the L2 misfit -- SECONDS, not
    decibels despite an earlier draft's column name) are reported
    alongside it, and `walk_consistent` (the two sigma_z agree within 10%) is
    False whenever this check was run and disagreed. `bidirectional_sample`
    caps how many in-stratum, resolvable calls pay for the second walk (None
    means all of them); calls outside that sample keep `walk_consistent` as
    <NA> (not computed, not assumed consistent) rather than a silently
    optimistic default -- see the run report for the measured budget and
    which policy was used.

    `sigma_z_reason` records why `sigma_z_m` is `inf`, when it is (fix round
    1, MAJOR 2): `"edge_deep"` (the minimum sits at the grid's deep floor --
    the GRID, not the data, stopped the walk), `"edge_surface"` (the minimum
    sits at 0 m, the physical sea surface -- not a grid-extension artifact),
    or `"not_crossed"` (an interior minimum where Delta-chi-squared = 1 is
    never reached on both sides). Empty when `sigma_z_m` is finite.

    `delta_misfit_over_resid_sigma` divides `delta_misfit` by that SAME
    call's own null-model residual noise (`resid_sigma_ms`/1000), so the
    direct-versus-bounce path comparison can be read against each call's own
    noise floor rather than as a bare, denominator-free count -- see
    `path_comparison_summary`. IMPORTANT -- this is a SCALE comparison, not a
    significance test: it is a difference of two whole-fit RMS misfits
    divided by a PER-PICK sigma, with no `n_picks` in it, so it does not
    answer "does this call individually resolve the path question" the way a
    chi-squared statistic would. Use `delta_chi2` for that question.

    `delta_chi2` is the module's own chi-squared convention (the same one
    `marginal_sigma_z` uses for `sigma_z_m`, `n_picks * (m**2 - m_min**2) /
    sigma**2`) applied to the two path misfits instead of a depth profile:
    `n_picks * (misfit_bounce**2 - misfit_direct**2) / resid_sigma**2`
    (`resid_sigma` in seconds, i.e. `resid_sigma_ms`/1000). A positive value
    favours direct, a negative value favours bounce, and `|delta_chi2| >= 1,
    4, 9` are the usual 1/2/3-sigma significance thresholds for ONE degree of
    freedom -- unlike `delta_misfit_over_resid_sigma`, this is the statistic
    that answers whether an individual call's path preference exceeds its own
    noise, and `path_comparison_summary` reports it that way.
    """
    from dclde_localization.physics.bathymetry import depth_along_path

    if z_grid is None:
        z_grid = np.linspace(-600.0, 0.0, 121)
    z_grid = np.asarray(z_grid, dtype=float)
    model = ConstantC(c)

    rows, flipped_n, unsolved_n = [], 0, 0
    # Calls eligible for the second (reverse) walk, with everything the
    # reverse pass needs -- collected during the single main loop so `solve`
    # is never paid for twice per call.
    eligible = []
    for call, sub in picks.groupby("call"):
        px, py, pz = (sub[k].to_numpy(float) for k in ("px", "py", "pz"))
        obs_t = sub["time"].to_numpy(float)
        n_cables = int(sub["cable"].nunique()) if "cable" in sub.columns else 1
        span = float(abs(pz.max() - pz.min()))
        base = {"call": call, "n_picks": int(len(sub)), "depth_span_m": span,
                "n_cables": n_cables,
                "in_stratum": False, "solved": False, "flipped": False,
                "sigma_z_m": np.inf, "sigma_z_reason": "", "z_at_min_m": np.nan,
                "resid_sigma_ms": np.nan, "bottom_z_m": np.nan,
                "misfit_direct": np.nan, "misfit_bounce": np.nan,
                "delta_misfit": np.nan, "delta_misfit_over_resid_sigma": np.nan,
                "delta_chi2": np.nan,
                "sigma_z_m_reverse": np.inf, "profile_max_discrepancy_s": np.nan,
                "walk_consistent": pd.NA}

        if len(sub) < min_picks:
            rows.append(base)
            continue
        try:
            sol = solve(px, py, pz, obs_t, model, sz=NULL_SOURCE_Z)
        except (ValueError, FloatingPointError):
            unsolved_n += 1
            rows.append(base)
            continue
        if not sol.basins:
            unsolved_n += 1
            rows.append(base)
            continue

        base["solved"] = True
        b = sol.best
        if sol.separation_m is not None and sol.separation_m > flip_threshold_m:
            base["flipped"] = True
            flipped_n += 1
            rows.append(base)
            continue

        base["in_stratum"] = span >= span_min_m
        resid = obs_t - (travel_time(b.x, b.y, NULL_SOURCE_Z, px, py, pz, c) + b.t0)
        sigma_r = robust_sigma(resid)
        base["resid_sigma_ms"] = sigma_r * 1e3

        if base["in_stratum"] and sigma_r > 0:
            zs, mis = depth_profile(px, py, pz, obs_t, z_grid=z_grid, c=c,
                                    x0=b.x, y0=b.y)
            sigma_z, reason = marginal_sigma_z(zs, mis, len(sub), sigma_r)
            base["sigma_z_m"] = sigma_z
            base["sigma_z_reason"] = reason
            base["z_at_min_m"] = float(zs[int(np.argmin(mis))])

            # Bottom depth is taken along the path from the solved source to
            # the CENTROID of this call's own picks -- for a two-cable call
            # that centroid can sit between the cables, on no real receiver
            # path; findings.md 8.1 names this alongside the flat-bottom
            # approximation it already flags.
            prof = depth_along_path(grid, b.x, b.y, float(px.mean()), float(py.mean()),
                                    n_steps=16)
            if prof.ok:
                bottom = -float(np.max(prof.depth_m))
                base["bottom_z_m"] = bottom
                base.update({k: v for k, v in
                             path_comparison(px, py, pz, obs_t, bottom_z=bottom,
                                             c=c).items()
                             if k in ("misfit_direct", "misfit_bounce", "delta_misfit")})
                # Fix round 1, MEDIUM 3: `delta_misfit` alone is a bare count
                # waiting to happen -- this ratio puts it against the SAME
                # call's own noise floor (`resid_sigma_ms`, already computed
                # above from the null-model residuals) so a reader can judge
                # whether a given delta_misfit is a meaningful path signal
                # or noise-scale for that specific call, not just its sign.
                # IMPORTANT -- this is a SCALE comparison, not a significance
                # test (no n_picks in it); see the docstring above and use
                # delta_chi2 below for "does this call individually resolve
                # the path question".
                if sigma_r > 0:
                    base["delta_misfit_over_resid_sigma"] = base["delta_misfit"] / sigma_r
                    # The module's own chi-squared convention (marginal_sigma_z
                    # uses n_picks * (m**2 - m_min**2) / sigma**2 for depth;
                    # this is the same form applied to the two path misfits).
                    # Positive favours direct, negative favours bounce.
                    base["delta_chi2"] = (len(sub) * (base["misfit_bounce"] ** 2
                                                       - base["misfit_direct"] ** 2)
                                          / (sigma_r ** 2))

            eligible.append({
                "idx": len(rows), "px": px, "py": py, "pz": pz, "obs_t": obs_t,
                "n_picks": len(sub), "sigma_r": sigma_r,
                "x0": b.x, "y0": b.y, "mis_forward": mis,
            })
        rows.append(base)

    n_eligible = len(eligible)
    if bidirectional_sample is None:
        selected = eligible
    else:
        k = min(int(bidirectional_sample), n_eligible)
        rng = np.random.default_rng(bidirectional_seed)
        sel_idx = rng.choice(n_eligible, size=k, replace=False) if k > 0 else np.array([], int)
        selected = [eligible[i] for i in sorted(sel_idx.tolist())]

    grid_spacing_m = float(np.median(np.abs(np.diff(z_grid)))) if z_grid.size > 1 else None
    for e in selected:
        _, mis_rev = depth_profile(e["px"], e["py"], e["pz"], e["obs_t"],
                                   z_grid=z_grid[::-1], c=c, x0=e["x0"], y0=e["y0"])
        mis_rev_aligned = mis_rev[::-1]
        sigma_rev, _reason_rev = marginal_sigma_z(z_grid, mis_rev_aligned, e["n_picks"], e["sigma_r"])
        discrepancy = float(np.max(np.abs(e["mis_forward"] - mis_rev_aligned)))
        row = rows[e["idx"]]
        row["sigma_z_m_reverse"] = sigma_rev
        row["profile_max_discrepancy_s"] = discrepancy
        row["walk_consistent"] = _walk_consistent(row["sigma_z_m"], sigma_rev,
                                                   grid_spacing_m=grid_spacing_m)

    table = pd.DataFrame(rows, columns=GATE_COLUMNS)
    den = {"calls_total": int(picks["call"].nunique()),
           "calls_below_min_picks": int((table["n_picks"] < min_picks).sum()),
           "calls_unsolved": int(unsolved_n),
           "calls_flipped": int(flipped_n),
           "calls_in_stratum": int(table["in_stratum"].sum()),
           "calls_bidirectional_checked": int(len(selected)),
           "calls_bidirectional_eligible": int(n_eligible),
           "calls_two_cable": int((table["n_cables"] >= 2).sum()),
           "picks_total": int(len(picks))}
    return table, den


def gate_summary(table, denominators, *, sigma_target_m=100.0, min_fraction=0.10,
                 min_calls=50) -> pd.DataFrame:
    """Apply the pre-declared gate criterion and show every denominator.

    BOTH legs must pass: a fraction of the stratum AND an absolute count. The
    fraction alone would let a tiny stratum pass on a handful of favourable
    geometries; the count alone would punish a small but genuinely resolvable
    subset.

    A non-finite `sigma_z_m` is UNRESOLVED, not missing -- it means the call did
    not bound depth, which is an answer.

    Addition (A): a call whose `walk_consistent` column is explicitly False
    is NEVER counted as resolved, regardless of its (forward) `sigma_z_m` --
    an inconsistent forward/reverse walk means the profile may have tracked a
    local, not global, (x, y) optimum at some depth, so its sigma_z is not
    trustworthy evidence either way. A missing/NaN `walk_consistent` (the
    bidirectional check was not run for that call, e.g. it fell outside a
    sampled subset) does not disqualify a call -- only an explicit False
    does, so the pre-declared sigma_z-only criterion still applies unchanged
    to a table that never ran the bidirectional check at all (i.e. this
    function's behaviour on the original brief's three tests, which carry no
    `walk_consistent` column, is unaffected).

    Fix round 1, MAJOR 1: `n_resolved` is therefore a COMPOSITE of the
    pre-declared criterion and the walk-consistency addition, and the config
    promises the pre-declared criterion is "applied unchanged" -- so both
    numbers are reported, not just the composite. `n_resolved_forward_only`
    is the pre-declared criterion ALONE (sigma_z_m finite and <=
    sigma_target_m, nothing else); `walk_check_applied` says whether this
    table carries a `walk_consistent` column at all (False for a table that
    never ran the bidirectional check, matching the note above). `n_resolved`
    and the pass/fail columns below still use the walk-consistency-filtered
    count -- the honest number given what the bidirectional check found --
    but a reader can now recover the pre-declared statistic directly from
    this summary instead of cross-tabulating the per-call table by hand.
    """
    strat = table[table["in_stratum"].astype(bool)]
    n_stratum = int(len(strat))
    finite = np.isfinite(strat["sigma_z_m"].to_numpy(float))
    resolved_forward_only = finite & (strat["sigma_z_m"].to_numpy(float) <= sigma_target_m)
    n_resolved_forward_only = int(resolved_forward_only.sum())
    walk_check_applied = "walk_consistent" in strat.columns
    resolved = resolved_forward_only
    if walk_check_applied:
        # Only an EXPLICIT False disqualifies. A missing/NaN/<NA> entry (the
        # bidirectional check was never run for that call) must not silently
        # read as "consistent" or "inconsistent" -- `pd.notna` guards both
        # NaN and pandas' nullable <NA> before the equality check, so this
        # works whether the column holds plain bool/np.nan or the "boolean"
        # extension dtype.
        disqualified = strat["walk_consistent"].map(
            lambda v: bool(pd.notna(v)) and (v == False)).to_numpy(bool)  # noqa: E712
        resolved = resolved_forward_only & ~disqualified
    n_resolved = int(resolved.sum())
    frac = float(n_resolved / n_stratum) if n_stratum else 0.0
    passes_fraction = bool(frac >= min_fraction)
    passes_min_calls = bool(n_resolved >= min_calls)

    row = {"sigma_target_m": sigma_target_m, "min_fraction": min_fraction,
           "min_calls": min_calls, "n_stratum": n_stratum,
           "n_resolved_forward_only": n_resolved_forward_only,
           "walk_check_applied": bool(walk_check_applied),
           "n_resolved": n_resolved,
           "fraction_resolved": frac, "passes_fraction": passes_fraction,
           "passes_min_calls": passes_min_calls,
           "gate_passes": bool(passes_fraction and passes_min_calls)}
    row.update({k: v for k, v in denominators.items()})
    return pd.DataFrame([row])


def path_comparison_summary(table) -> pd.DataFrame:
    """Denominator-bearing summary of the direct-versus-bounce comparison
    (design note section 3's required reporting, fix round 1 MEDIUM 3).

    A bare count of "N favour direct, M favour bounce" is exactly the kind
    of denominator-free rate CLAUDE.md forbids: `delta_misfit` magnitudes
    here are ~1e-3 s against misfits of ~0.1 s (about 1%), so the count needs
    context to say whether that 1% is a real path signal or noise for the
    calls it is drawn from. This reports the `delta_misfit` distribution and
    the `delta_misfit_over_resid_sigma` distribution (delta_misfit divided by
    THAT SAME call's own null-model residual noise, `resid_sigma_ms`/1000) --
    but that ratio is a SCALE comparison, not a significance test (see
    `gate_table`'s docstring): it has no `n_picks` in it, so it cannot say
    whether an individual call's path preference exceeds its own noise.
    `delta_chi2` (`n_picks * (misfit_bounce**2 - misfit_direct**2) /
    resid_sigma**2`, the module's own chi-squared convention) is that
    statistic, and the `n_direct_sig_*`/`n_bounce_sig_*` counts below report
    it at the usual 1/2/3-sigma-for-one-dof thresholds (|delta_chi2| >= 1, 4,
    9) on BOTH the direct-favouring (positive) and bounce-favouring
    (negative) sides, so a bounce-favouring tail is visible rather than
    collapsed into an "exceeds own noise" count that never distinguishes
    direction.
    """
    rows = table[np.isfinite(table["delta_misfit"].to_numpy(float))]
    n = int(len(rows))
    dm = rows["delta_misfit"].to_numpy(float)
    ratio = rows["delta_misfit_over_resid_sigma"].to_numpy(float)
    ratio_finite = ratio[np.isfinite(ratio)]
    chi2 = rows["delta_chi2"].to_numpy(float) if "delta_chi2" in rows.columns \
        else np.full(n, np.nan)
    chi2_finite = chi2[np.isfinite(chi2)]

    def _stats(x, prefix):
        if x.size == 0:
            return {f"{prefix}_mean": np.nan, f"{prefix}_median": np.nan,
                    f"{prefix}_std": np.nan, f"{prefix}_min": np.nan,
                    f"{prefix}_max": np.nan}
        return {f"{prefix}_mean": float(np.mean(x)), f"{prefix}_median": float(np.median(x)),
                f"{prefix}_std": float(np.std(x)), f"{prefix}_min": float(np.min(x)),
                f"{prefix}_max": float(np.max(x))}

    row = {
        "n_compared": n,
        "n_favor_direct": int(np.sum(dm > 0)),
        "n_favor_bounce": int(np.sum(dm < 0)),
        "n_tied": int(np.sum(dm == 0)),
        "n_ratio_finite": int(ratio_finite.size),
        "n_exceeds_own_noise": int(np.sum(np.abs(ratio_finite) >= 1.0)),
        "n_chi2_finite": int(chi2_finite.size),
        "n_direct_sig_1sigma": int(np.sum(chi2_finite >= 1.0)),
        "n_direct_sig_2sigma": int(np.sum(chi2_finite >= 4.0)),
        "n_direct_sig_3sigma": int(np.sum(chi2_finite >= 9.0)),
        "n_bounce_sig_1sigma": int(np.sum(chi2_finite <= -1.0)),
        "n_bounce_sig_2sigma": int(np.sum(chi2_finite <= -4.0)),
        "n_bounce_sig_3sigma": int(np.sum(chi2_finite <= -9.0)),
    }
    row.update(_stats(dm, "delta_misfit"))
    row.update(_stats(ratio_finite, "delta_misfit_over_resid_sigma"))
    row.update(_stats(chi2_finite, "delta_chi2"))
    return pd.DataFrame([row])
