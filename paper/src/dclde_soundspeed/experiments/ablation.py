"""Sections 7.2 and 7.3 -- free-c and effective-c ablations.

REVISED CONCLUSION FOR 7.2 (review round 1 of task-13; see task-13-report.md
"Review response" section for the numbers behind this): the original
docstring here claimed a free scalar c "absorbs most of the effect ... closing
that branch" and stands as "the null model the depth experiment has to beat".
Running against the real 524-call dataset falsifies the clean version of that
claim. Of the 498 calls the search could fit, only **23 (4.6%)** land inside
the actual measured sound-speed range of the profile used elsewhere in this
plan (hycom_south_deep: ~1477.5-1497.5 m/s); **208 (41.8%)** pin at the outer
search-bracket edge (1430 or 1530 m/s -- water that does not exist at this
site), and restricted to the basin-stable subset alone, edge-pinned calls
(n=113) shift a median of 250 m while physically-plausible calls (n=18) shift
only 59 m. A free `c` is absorbing real position-affecting error, but most of
that absorption is unphysical -- it is a nuisance sink for model and
association error (of which true sound-speed variation is at most a minor
component), not a genuine physical fit. **A free scalar c is therefore not a
usable null model for the depth experiment to beat**; that comparison would be
against a parameter that is mostly fitting something else. See
`classify_c_plausibility` below, which makes this classification a tested,
reusable function rather than a one-off report calculation.

Every report emits basin-stable deltas and basin-flip counts separately. A call
that jumped to the mirror lobe must never be averaged into a median position
change -- see spec section 5 and solve.py's own module docstring.

Ruling R8 (overrides the original brief for `fit_free_c`): a single-pass grid
of ~40 candidate c values times ~500 calls is 20,000 full `solve()` calls, so
instead of one flat grid this profiles c with three coarse-to-fine passes of
`_N_PER_PASS` (9) values each -- 27 `solve()` calls per call, not ~40, at a
substantially finer final resolution than a single 40-point pass over the same
bracket ever could. Each pass's best sample becomes the centre of the next
pass's bracket, clipped to never leave the original outer bracket.

CAVEAT (review round 1): this ruling was originally justified by "misfit is
smooth and near-unimodal in c". That justification does NOT hold in general
and the real run contradicts the clean version of it: a 37.5% basin-flip rate
(the free-c best position lands in a different basin than the fixed-c one)
means the top-ranked-basin identity changes with c for a large share of
calls, which puts a kink in the (c -> best-basin's misfit) curve at whatever
c the ranking flips -- not a smooth, unimodal function. Edge-pinned calls are
safe regardless (pass 1 already samples the whole outer bracket at coarse,
~12.5 m/s spacing, so any of those 9 samples nearest an edge drives the
narrowing straight to it), but for an *interior* optimum this search has no
global-optimality guarantee against a nearby competing basin. What is
actually checked, not assumed: `test_r8_coarse_to_fine_search_agrees_with_a_
dense_brute_force_grid` re-fits a sample call against an 81-point brute-force
grid over the same bracket and asserts agreement -- a spot check, not a proof
of unimodality.

`c_grid`'s role changes with it: it supplies the outer bracket (min/max of
whatever is passed, so both a raw grid array -- as the unit tests use -- and a
config's `{start, stop}` work unchanged), and a config's `step` is no longer a
grid spacing at all; it is read only as the stopping tolerance to compare the
*achieved* resolution against when reporting the fit (see
`achieved_resolution_m_s` in the output and task-13-report.md for the number
that came out: 0.2-0.8 m/s, 3-13x finer than the 2.5 m/s tolerance named in
the committed config).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dclde_localization.physics.forward import DEFAULT_C
from dclde_localization.physics.solve import COARSE_GRID_PAD_M, compare, solve
from dclde_localization.physics.sound_speed_models import ConstantC, PathAverage
from dclde_soundspeed.soundspeed.raytrace import RayTrace

# Ruling R8: 3 coarse-to-fine passes of 9 samples each = 27 solve() calls per
# call, versus one flat ~40-point pass over the same bracket in the original
# brief.
_N_PASSES = 3
_N_PER_PASS = 9

# Same edge case regime.py's `_solve_and_residualize` and experiments/repro.py
# both guard against: a pick geometry can leave solve() with no usable basin
# at all (every Tier-1 refinement lands on its own box boundary and is
# discarded -- solve.py's own "on_boundary" rule), and solve() itself can
# raise ValueError for a degenerate geometry (e.g. collapsed grid bounds).
# Neither is a bug in this module; both are counted and the call is dropped
# from that table rather than crashing the whole ablation on one bad call.
_EXPECTED_SOLVE_ERRORS = (ValueError,)


def _solve_or_none(px, py, pz, t, model, source_z):
    """`solve()`, returning None instead of a `Solution` with no usable basin
    or an expected solver error -- see `_EXPECTED_SOLVE_ERRORS`."""
    try:
        sol = solve(px, py, pz, t, model, sz=source_z)
    except _EXPECTED_SOLVE_ERRORS:
        return None
    return sol if sol.basins else None


def solve_all(picks: pd.DataFrame, model, *, source_z: float,
              min_picks: int = 8) -> dict:
    """Solve every call with >= min_picks picks against `model`. Calls below
    the threshold are skipped entirely (not solved on too little data), as are
    calls for which `solve()` raises an expected error or returns no usable
    basin (see `_solve_or_none`) -- `compare()` then only ever compares calls
    both sides actually solved."""
    out = {}
    for call, sub in picks.groupby("call"):
        if len(sub) < min_picks:
            continue
        sol = _solve_or_none(sub["px"].to_numpy(float), sub["py"].to_numpy(float),
                             sub["pz"].to_numpy(float), sub["time"].to_numpy(float),
                             model, source_z)
        if sol is not None:
            out[call] = sol
    return out


def fit_free_c(picks: pd.DataFrame, *, source_z: float, c_grid,
              min_picks: int = 8, flip_threshold_m: float = 500.0) -> pd.DataFrame:
    """Add c as a fifth unknown, profiled with three coarse-to-fine 9-point
    passes per call (Ruling R8), not one flat grid.

    `shift_m` is the raw distance between the free-c and fixed-1480 *best*
    positions. Like `compare()`, this can be a genuine small shift or a jump
    to the mirror lobe -- fitting c as a free scalar does not just rescale a
    position, it can pick a different top-ranked basin than the fixed-c
    solve did (measured directly: 37.5% of calls in the committed
    `ablation_free_c.csv` have `shift_m` > 500 m, several tens of km). Per
    solve.py's own reporting contract, this column is therefore never
    reported as a single median; `flipped` (`shift_m > flip_threshold_m`)
    is carried alongside it so a caller can split stable shifts from flips
    exactly as `ablation_report` does for `compare()`, and must always
    report both, never one merged statistic.

    Neither `shift_m` nor `flipped` says whether `best_c` itself is a
    physically plausible sound speed -- see `classify_c_plausibility` for
    that, and the module docstring's "REVISED CONCLUSION" above for why that
    distinction, not the raw shift, is the headline finding: most of the
    basin-stable shift comes from calls whose fitted c sits at the search
    bracket's edge, not from calls whose fitted c is real water.

    `c_grid` supplies only the outer bracket: `(min(c_grid), max(c_grid))`.
    This accepts a raw array (as the unit tests pass) or a bracket built from
    a config's `{start, stop}` equally well; whatever fine spacing the caller
    might have implied by an array's own step is irrelevant -- resolution is
    determined entirely by the three-pass narrowing described in the module
    docstring, not by how densely `c_grid` itself was sampled.

    Every row also reports `fixed_misfit`/`fixed_x`/`fixed_y`: the misfit and
    position solving with `c` pinned at `forward.DEFAULT_C` (1480 m/s, the
    published baseline speed) -- the null model this free fit must match or
    beat (see `test_free_c_reduces_or_matches_the_fixed_c_misfit`).

    `achieved_resolution_m_s` is the finest pass's grid spacing actually
    reached for that call (it can differ slightly call-to-call when a pass's
    best sample sits at, or is clipped to, the outer bracket edge, shifting
    that pass's bracket asymmetrically) -- report it, don't assume a single
    fixed number for the whole table.

    A call is dropped from the returned table (not crashed on) whenever every
    sampled c in a pass -- or the fixed-1480 solve -- leaves `solve()` with no
    usable basin or an expected solver error (`_EXPECTED_SOLVE_ERRORS`, the
    same edge case `solve_all` guards against). `fit_free_c` itself has no
    denominator to report that against (it only sees calls already filtered
    by `min_picks`); a caller wanting `n_dropped_no_basin` should compare
    `picks["call"].nunique()` restricted to `>= min_picks` against
    `len(result)`.
    """
    c_grid = np.asarray(c_grid, dtype=float)
    if c_grid.size == 0:
        raise ValueError("c_grid must not be empty")
    lo, hi = float(c_grid.min()), float(c_grid.max())
    if hi <= lo:
        raise ValueError(f"c_grid outer bracket must have stop > start; got [{lo}, {hi}]")

    rows = []
    for call, sub in picks.groupby("call"):
        if len(sub) < min_picks:
            continue
        px, py, pz, t = (sub[k].to_numpy(float) for k in ("px", "py", "pz", "time"))

        best_misfit, best_c, best_x, best_y = np.inf, np.nan, np.nan, np.nan
        bracket_lo, bracket_hi = lo, hi
        resolution = np.nan
        n_solves = 0

        for _ in range(_N_PASSES):
            cs = np.linspace(bracket_lo, bracket_hi, _N_PER_PASS)
            spacing = (bracket_hi - bracket_lo) / (_N_PER_PASS - 1)
            resolution = spacing
            pass_best_misfit, pass_best_c = np.inf, cs[_N_PER_PASS // 2]
            for c in cs:
                sol = _solve_or_none(px, py, pz, t, ConstantC(float(c)), source_z)
                n_solves += 1
                if sol is None:
                    continue
                m = sol.best.misfit
                if m < pass_best_misfit:
                    pass_best_misfit, pass_best_c = m, float(c)
                if m < best_misfit:
                    best_misfit, best_c = m, float(c)
                    best_x, best_y = sol.best.x, sol.best.y
            # Next pass's bracket is centred on this pass's own best sample
            # (not the running global best -- the two coincide whenever the
            # surface is genuinely unimodal, which is the assumption this
            # ruling makes explicit), one grid spacing either side, clipped to
            # the original outer bracket so narrowing can never wander outside
            # the range the caller actually asked to search. If every sample
            # in this pass had no usable basin, pass_best_c stays at its
            # initialised midpoint, which still keeps the bracket valid and
            # inside the outer bracket for the next pass to try again.
            bracket_lo = max(lo, pass_best_c - spacing)
            bracket_hi = min(hi, pass_best_c + spacing)
            if bracket_hi <= bracket_lo:
                # Degenerate only if the outer bracket itself is a single
                # point away from an edge; keep a tiny, valid span so the
                # next pass's linspace doesn't divide by zero.
                bracket_hi = bracket_lo + max(spacing, 1e-6)

        if not np.isfinite(best_misfit):
            # No sampled c, across all three passes, produced a usable basin
            # for this call's pick geometry -- an estimator failure for this
            # call (see solve.py's "on_boundary" rule), not a bug in this
            # search. Drop it rather than report a fabricated best_c.
            continue

        fixed = _solve_or_none(px, py, pz, t, ConstantC(DEFAULT_C), source_z)
        n_solves += 1
        if fixed is None:
            # The free search found a basin but the fixed-1480 null model
            # could not -- there is no fixed_misfit to compare against, so
            # this call cannot honestly populate shift_m/fixed_misfit either.
            continue

        shift_m = float(np.hypot(best_x - fixed.best.x, best_y - fixed.best.y))
        rows.append({"call": call, "n_picks": int(len(sub)),
                     "best_c": best_c, "best_misfit": best_misfit,
                     "best_x": best_x, "best_y": best_y,
                     "fixed_misfit": fixed.best.misfit,
                     "fixed_x": fixed.best.x, "fixed_y": fixed.best.y,
                     "shift_m": shift_m,
                     "flipped": bool(shift_m > flip_threshold_m),
                     "achieved_resolution_m_s": resolution,
                     "outer_bracket_lo": lo, "outer_bracket_hi": hi,
                     "n_solves": n_solves})
    return pd.DataFrame(rows)


def classify_c_plausibility(df: pd.DataFrame, profile, *, outer_lo_col: str = "outer_bracket_lo",
                            outer_hi_col: str = "outer_bracket_hi", c_col: str = "best_c") -> pd.DataFrame:
    """Label each `fit_free_c` row's fitted c as edge-pinned, physically
    plausible, or neither, against a real measured sound-speed profile.

    Added in review round 1 of task-13: the headline number for section 7.2
    is not the raw median shift (see the module docstring) but *how much of
    that shift comes from physically implausible fits*. Making this a tested
    function, rather than a one-off pandas snippet in a report script, is
    the point -- a repeated ad hoc calculation is exactly how "two committed
    artifacts disagree about the same measurement" happens (this plan has
    already hit that failure once).

    - `edge_pinned`: `best_c` sits at (or numerically on) the outer search
      bracket's own edge -- the search ran out of room, not out of slope.
    - `physically_plausible`: `best_c` falls within `[profile.c.min(),
      profile.c.max()]`, the actual measured range of sound speed in
      `profile` (e.g. hycom_south_deep: ~1477.5-1497.5 m/s). A profile with
      no real spatial/temporal coverage at this site would make this too
      permissive a bar; it is still the most defensible reference available
      per this plan's PROFILES commentary (see experiments/runner.py).
    - A row can be neither (an interior value still outside the measured
      range -- unphysical but not edge-pinned) but never both, since the
      profile's own range sits strictly inside the outer bracket for every
      config this plan has committed.
    """
    lo = df[outer_lo_col].astype(float)
    hi = df[outer_hi_col].astype(float)
    c = df[c_col].astype(float)
    edge_pinned = np.isclose(c, lo, atol=1e-6) | np.isclose(c, hi, atol=1e-6)
    c_lo, c_hi = float(profile.c.min()), float(profile.c.max())
    physically_plausible = (c >= c_lo) & (c <= c_hi)
    out = df.copy()
    out["edge_pinned"] = edge_pinned
    out["physically_plausible"] = physically_plausible
    out["profile_c_min"] = c_lo
    out["profile_c_max"] = c_hi
    return out


def ablation_report(base: dict, variant: dict, label: str,
                    *, flip_threshold_m: float = 500.0) -> dict:
    """Wrap `compare()` and emit basin-stable deltas and basin-flip counts as
    two separate figures, never merged into one statistic (solve.py's own
    reporting contract -- a median over a set containing mirror-lobe flips
    hides the only interesting failure)."""
    c = compare(base, variant, flip_threshold_m=flip_threshold_m)
    d = c.stable_delta_m
    return {"label": label, "n": c.n, "n_stable": c.n_stable, "n_flipped": c.n_flipped,
            "flip_fraction": c.flip_fraction,
            "stable_delta_median_m": float(np.median(d)) if d.size else float("nan"),
            "stable_delta_p90_m": float(np.percentile(d, 90)) if d.size else float("nan"),
            "stable_delta_max_m": float(d.max()) if d.size else float("nan")}


def free_c_flip_diagnostics(fit_df: pd.DataFrame, picks: pd.DataFrame, *,
                            source_z: float, min_picks: int = 8) -> pd.DataFrame:
    """For every `fit_free_c` row flagged `flipped`, re-solve that call at
    both its fixed-1480 and best-fit `c` and record whether each side's
    top-two basins are a confidently, independently resolved pair
    (`separation_resolved`) or an approximate/fallback stand-in.

    This is what section 3.4's "the 37.6% flip rate overstates real
    ambiguity" finding is built on (module docstring, review round 1 of
    task-13): most flips turn out to be "a confident winner replaces a
    weakly-characterised runner-up," not two well-characterised lobes trading
    rank, and that distinction is only visible by looking at
    `separation_resolved`, not at `shift_m` alone. Promoted from a one-off
    `uv run python -c "..."` snippet (task-13-report.md's review response)
    into a tested, reusable function -- exactly the kind of ad hoc
    recalculation that let two committed artifacts disagree about the same
    measurement once already in this plan (see `classify_c_plausibility`'s
    own docstring for the earlier instance).

    Only re-solves the flipped calls (typically a small fraction of the full
    table), not every row -- the point of this diagnostic is to characterise
    the flips specifically, not to redo the whole ablation.
    """
    rows = []
    for _, r in fit_df.loc[fit_df["flipped"]].iterrows():
        sub = picks[picks["call"] == r["call"]]
        if len(sub) < min_picks:
            continue
        px, py, pz, t = (sub[k].to_numpy(float) for k in ("px", "py", "pz", "time"))
        fixed = solve(px, py, pz, t, ConstantC(DEFAULT_C), sz=source_z)
        best = solve(px, py, pz, t, ConstantC(float(r["best_c"])), sz=source_z)
        rows.append({
            "call": r["call"], "best_c": float(r["best_c"]), "shift_m": float(r["shift_m"]),
            "fixed_separation_resolved": fixed.separation_resolved,
            "fixed_separation_m": fixed.separation_m,
            "best_c_separation_resolved": best.separation_resolved,
            "best_c_separation_m": best.separation_m,
            "fixed_n_basins": len(fixed.basins), "best_c_n_basins": len(best.basins),
            "fixed_best_provenance": fixed.best.provenance,
            "best_c_best_provenance": best.best.provenance,
        })
    return pd.DataFrame(rows)


def raytrace_fallback_fraction_at_best_fit(raytrace_model: RayTrace, solutions: dict,
                                           picks: pd.DataFrame) -> tuple[float, int]:
    """Fallback fraction of `raytrace_model`, evaluated once per solved call at
    that call's own best-fit position -- a lower bound on how much of the
    RayTrace ablation never described a refracted ray (see
    `raytrace_over_search_grid` for the larger, more representative figure
    over the actual search grid, and `raytrace.py`'s module docstring for why
    a single scalar read after the fact undercounts this).

    `raytrace_model.reset_fallback_record()` is called first, so this reads
    only the calls made by this function, not whatever ran on the same
    instance before it (e.g. the `solve_all` pass that produced `solutions`).
    Returns `(fraction, n_picks)` -- the denominator, stated per the project's
    reporting rule.
    """
    raytrace_model.reset_fallback_record()
    for call, sol in solutions.items():
        sub = picks[picks["call"] == call]
        b = sol.best
        raytrace_model.travel_time(b.x, b.y, b.z, sub["px"].to_numpy(float),
                                   sub["py"].to_numpy(float), sub["pz"].to_numpy(float))
    n_total = sum(t for t, _ in raytrace_model.fallback_calls)
    return raytrace_model.fallback_fraction_overall, n_total


def raytrace_fallback_over_search_grid(raytrace_model: RayTrace, solutions: dict,
                                       picks: pd.DataFrame, *, n_coarse: int = 161,
                                       pad: float = COARSE_GRID_PAD_M) -> tuple[float, int]:
    """Fallback fraction of `raytrace_model` pooled over the SAME coarse
    search grid `solve()` itself builds for each call (same bounds -- pick
    extent padded by `pad`, matching `solve.COARSE_GRID_PAD_M` -- and the same
    `n_coarse`), i.e. every (grid-cell, pick) query that actually drove the
    localization, not just the winning position.

    This is the number that matters for judging whether a RayTrace ablation
    tested refracted-ray physics at all: at the best-fit position alone, a
    call's own picks are the ones closest to the source, where the profile is
    most likely to support a direct ray; the search grid also covers
    positions far from the best fit, most of which fall outside this
    profile's refraction limits and fall back to `PathAverage`. Measured on
    the committed `hycom_south_deep` ablation: 71.85% at the best-fit position
    versus 99.38% over the search grid (see `experiments/ablation.py`'s
    `effective_c_report` and `paper/docs/das-finwhale-soundspeed-plan-a-findings.md`
    S3.5).

    Rebuilding the grid here (rather than calling `solve()` again) is
    deliberate: re-solving would cost another full optimisation per call for
    a measurement that only needs the grid's `travel_time` queries, not its
    minimisation.
    """
    raytrace_model.reset_fallback_record()
    for call, sol in solutions.items():
        sub = picks[picks["call"] == call]
        px = sub["px"].to_numpy(float)
        py = sub["py"].to_numpy(float)
        pz = sub["pz"].to_numpy(float)
        xs = np.linspace(px.min() - pad, px.max() + pad, n_coarse)
        ys = np.linspace(py.min() - pad, py.max() + pad, n_coarse)
        gx, gy = np.meshgrid(xs, ys)
        sx = gx[:, :, None]
        sy = gy[:, :, None]
        raytrace_model.travel_time(sx, sy, sol.best.z, px, py, pz)
    n_total = sum(t for t, _ in raytrace_model.fallback_calls)
    return raytrace_model.fallback_fraction_overall, n_total


def effective_c_report(picks: pd.DataFrame, profile, *, source_z: float, min_picks: int = 8,
                       flip_threshold_m: float = 500.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end driver for section 7.3: reproduces
    `ablation_effective_c.csv` and `ablation_effective_c_flip_detail.csv` from
    picks and a profile alone, so the table is regenerable from committed
    inputs via a real function rather than only from the one-off
    `uv run python -c "..."` snippets `task-13-report.md`'s review response
    ran by hand (Task 14 review, Critical 1).

    Solves `ConstantC(DEFAULT_C)` (the null model), `PathAverage(profile)` and
    `RayTrace(profile)`, reports each depth-dependent model against the
    constant one via `ablation_report` (basin-stable deltas and flip counts,
    never merged), and adds the profile's own provenance/validation status
    and -- for the RayTrace row only -- how much of it is actually
    `PathAverage` arithmetic under a different name, via
    `raytrace_fallback_fraction_at_best_fit` and
    `raytrace_fallback_over_search_grid`.

    Returns `(summary, flip_detail)`: `summary` has one row per depth-dependent
    model (matching `ablation_effective_c.csv`); `flip_detail` has one row per
    call shared by all three solves, with each arm's own delta and flip flag
    (matching `ablation_effective_c_flip_detail.csv`) -- this is what lets a
    reader verify the "both models flip exactly the same calls" claim
    (SS3.5) per call, not just trust the summary counts agree.
    """
    from dclde_soundspeed.experiments.regime import _profile_validation_note

    base = solve_all(picks, ConstantC(DEFAULT_C), source_z=source_z, min_picks=min_picks)
    pa_model = PathAverage(profile)
    rt_model = RayTrace(profile)
    pa = solve_all(picks, pa_model, source_z=source_z, min_picks=min_picks)
    rt = solve_all(picks, rt_model, source_z=source_z, min_picks=min_picks)

    shared = sorted(set(base) & set(pa) & set(rt))
    profile_validated = _profile_validation_note(profile)

    pa_rep = ablation_report(base, pa, "path_average vs constant", flip_threshold_m=flip_threshold_m)
    rt_rep = ablation_report(base, rt, "raytrace vs constant", flip_threshold_m=flip_threshold_m)

    rt_frac_best, rt_n_best = raytrace_fallback_fraction_at_best_fit(rt_model, rt, picks)
    rt_frac_grid, rt_n_grid = raytrace_fallback_over_search_grid(rt_model, rt, picks)

    for rep in (pa_rep, rt_rep):
        rep["profile_source"] = profile.source
        rep["profile_validated"] = profile_validated
    pa_rep.update({
        "raytrace_fallback_fraction_at_best_fit": np.nan,
        "raytrace_fallback_fraction_at_best_fit_n_picks": np.nan,
        "raytrace_fallback_fraction_over_search_grid": np.nan,
        "raytrace_fallback_fraction_over_search_grid_n_queries": np.nan,
        "dependency_note": "independent of the raytrace row",
    })
    rt_rep.update({
        "raytrace_fallback_fraction_at_best_fit": rt_frac_best,
        "raytrace_fallback_fraction_at_best_fit_n_picks": float(rt_n_best),
        "raytrace_fallback_fraction_over_search_grid": rt_frac_grid,
        "raytrace_fallback_fraction_over_search_grid_n_queries": float(rt_n_grid),
        "dependency_note": (
            f"NOT independent of the path_average row: raytrace falls back to PathAverage "
            f"physics for {rt_frac_best * 100:.1f}% of picks at the best-fit position "
            f"(higher, {rt_frac_grid * 100:.1f}%, over the actual 161x161 search grid) -- "
            "for most of its queries this row and the path_average row use the same "
            "underlying arithmetic."),
    })
    summary = pd.DataFrame([pa_rep, rt_rep])

    detail_rows = []
    for call in shared:
        d_pa = float(np.hypot(base[call].best.x - pa[call].best.x, base[call].best.y - pa[call].best.y))
        d_rt = float(np.hypot(base[call].best.x - rt[call].best.x, base[call].best.y - rt[call].best.y))
        detail_rows.append({
            "call": call,
            "delta_m_path_average": d_pa, "flipped_path_average": d_pa > flip_threshold_m,
            "delta_m_raytrace": d_rt, "flipped_raytrace": d_rt > flip_threshold_m,
        })
    flip_detail = pd.DataFrame(detail_rows)
    return summary, flip_detail
