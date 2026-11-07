"""Section 7.0.1 -- is the range trend in residuals path error or depth error?

Section 2.5C measured a residual trend from -34 ms at short range to +54 ms
beyond 20 km that survives inside every SNR quartile and is strongest at the
HIGHEST SNR, which rules out low-SNR late picks. But it is confounded with the
fixed -30 m source depth, which produces range structure of its own.

Sweep the assumed source depth. If the trend flattens, the signal is depth and
section 7.4 can read it. If it does not, the depth hypothesis is falsified,
but that alone does not identify the actual cause -- mis-association,
bathymetry error, timing/clock error, and propagation-path error are all
still-live candidates that this experiment does not distinguish between. This
experiment is allowed to conclude either "depth explains it" or "depth does
not explain it"; it is not equipped to conclude *which* alternative does.

Outlier policy (review round 1, Finding 1): this dataset's residuals run as
high as 43.5 s from gross mis-associations -- the same failure mode
`paper/scripts/recon_propagation_regime.py` guards against with its own
`d[d.res.abs() < 1.0]` cut before computing any mean. `stratify` applies the
identical `|residual_s| < clip_s` policy (default `clip_s=1.0`) before
aggregating, and records `clip_s` and the per-cell dropped count
(`n_dropped_outlier`) in its own output so a reader of the table can see
what was excluded without reading this docstring. `residual_table` itself
stays uncut -- it is the raw per-pick record -- so a caller who wants the raw
statistic (or a different cut) can still get it via `stratify(..., clip_s=None)`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dclde_localization.physics.forward import best_t0
from dclde_soundspeed.soundspeed.modes import modes
from dclde_soundspeed.soundspeed.raytrace import max_direct_range
from dclde_localization.physics.solve import solve

# Exceptions solve() can plausibly raise for a degenerate pick geometry (e.g.
# a coarse grid whose bounds collapse, or a Nelder-Mead step landing outside
# a model's valid domain) surface through numpy/scipy as ValueError. Anything
# else (AttributeError, TypeError, ...) is far more likely a genuine bug in
# this module or its caller than a per-call data problem, so it is left to
# propagate rather than being swallowed here (review round 1, Finding 3).
_EXPECTED_SOLVE_ERRORS = (ValueError,)


def _solve_and_residualize(picks: pd.DataFrame, model, source_z: float, min_picks: int):
    """Shared solve loop for `residual_table` and `depth_sweep`.

    Returns `(rows, counts)`: `rows` is a list of per-call residual
    DataFrames (see `residual_table`), and `counts` is a dict with keys
    `too_few_picks`, `no_basin`, `solver_error`, `solved` -- every call is
    counted in exactly one bucket, so `sum(counts.values()) ==
    picks["call"].nunique()` always holds. This is what lets `depth_sweep`
    report *why* a call was dropped, not just that the solved total is lower
    than the number of annotated calls (review round 1, Finding 3).
    """
    rows = []
    counts = {"too_few_picks": 0, "no_basin": 0, "solver_error": 0, "solved": 0}
    for call, sub in picks.groupby("call"):
        if len(sub) < min_picks:
            counts["too_few_picks"] += 1
            continue
        px, py, pz, t = (sub[k].to_numpy(float) for k in ("px", "py", "pz", "time"))
        try:
            sol = solve(px, py, pz, t, model, sz=source_z)
        except _EXPECTED_SOLVE_ERRORS:
            counts["solver_error"] += 1
            continue
        if not sol.basins:
            # Same edge case documented in experiments/repro.py: a pick
            # geometry that leaves no usable basin is a genuine estimator
            # failure for this call, not a bug in this table, so the call is
            # dropped from the residual table rather than crashing the sweep.
            counts["no_basin"] += 1
            continue
        counts["solved"] += 1
        b = sol.best
        tt = model.travel_time(b.x, b.y, source_z, px, py, pz)
        res = t - tt - best_t0(t, tt)
        rows.append(pd.DataFrame({
            "call": call, "residual_s": res,
            "range_m": np.hypot(px - b.x, py - b.y),
            "channel_z": pz, "snr": sub["snr"].to_numpy(float)}))
    return rows, counts


def residual_table(picks: pd.DataFrame, model, *, source_z: float,
                   min_picks: int = 8) -> pd.DataFrame:
    """One row per pick: the raw (uncut) misfit left over after solving each
    call at a *fixed, assumed* `source_z`, using `model` for the propagation
    physics.

    Calls with fewer than `min_picks` picks are excluded entirely (not
    silently solved on too little data), as are calls for which `solve()`
    raises or returns no usable basin -- callers needing the exclusion
    breakdown should use `depth_sweep`, which surfaces it as columns, or call
    `_solve_and_residualize` directly.

    No outlier cut is applied here -- this is the raw per-pick record. Gross
    mis-associations (residuals up to 43.5 s in this dataset) are still
    present; `stratify` is where they are excluded from any reported mean.
    """
    rows, _ = _solve_and_residualize(picks, model, source_z, min_picks)
    return (pd.concat(rows, ignore_index=True) if rows
            else pd.DataFrame(columns=["call", "residual_s", "range_m", "channel_z", "snr"]))


def stratify(df: pd.DataFrame, range_bins, *, snr_quartiles: bool = False,
            clip_s: float | None = 1.0) -> pd.DataFrame:
    """Mean/median/std residual per range bin (optionally crossed with SNR
    quartile), with the denominator and the outlier policy both carried
    alongside every number rather than left implicit.

    `clip_s`: rows with `|residual_s| >= clip_s` are excluded from
    `mean_ms`/`median_ms`/`std_ms` (but still counted, in `n_dropped_outlier`)
    -- gross mis-associations must not be allowed to carry a mean, matching
    the committed policy in `recon_propagation_regime.py`
    (`d[d.res.abs() < 1.0]`). Pass `clip_s=None` to aggregate the raw,
    uncut residuals instead. Whichever is used is recorded in the output's
    own `clip_s` column, and `n` is the count of rows actually behind
    `mean_ms` (i.e. after the cut) -- the true denominator for that number,
    not the raw cell population.
    """
    d = df.copy()
    if clip_s is not None:
        outlier_mask = d["residual_s"].abs() >= clip_s
    else:
        outlier_mask = pd.Series(False, index=d.index)
    d["_kept"] = d["residual_s"].where(~outlier_mask)
    d["range_bin"] = pd.cut(d["range_m"], range_bins)
    keys = ["range_bin"]
    if snr_quartiles:
        d["snr_q"] = pd.qcut(d["snr"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
        keys.insert(0, "snr_q")
    g = d.groupby(keys, observed=True)
    out = g.agg(
        n=("_kept", lambda s: int(s.notna().sum())),
        n_dropped_outlier=("_kept", lambda s: int(s.isna().sum())),
        mean_ms=("_kept", lambda s: s.mean() * 1e3),
        median_ms=("_kept", lambda s: s.median() * 1e3),
        std_ms=("_kept", lambda s: s.std() * 1e3),
    ).reset_index()
    out["clip_s"] = clip_s if clip_s is not None else np.nan
    return out


def depth_sweep(picks: pd.DataFrame, model, z_values, *, min_picks: int = 8,
                range_bins=(0, 5_000, 10_000, 20_000, 30_000, 200_000),
                clip_s: float | None = 1.0) -> pd.DataFrame:
    """Range trend at each assumed source depth. Flat trend => source depth
    does not explain it (see module docstring for what that does and does
    not establish about the actual cause).

    `model` is held fixed across the whole sweep (matching section 2.5C's own
    setup, where the trend was first measured) so that source depth is the
    only thing varied -- the confound this experiment exists to break apart.

    Every row states, in columns rather than only in this docstring:
    - `n_calls_total`: distinct annotated calls presented to this depth's
      solve loop (constant across depths -- pick membership doesn't depend
      on assumed source depth).
    - `n_too_few_picks`, `n_no_basin`, `n_solver_error`, `n_calls_solved`:
      where every one of `n_calls_total` went (review round 1, Finding 3) --
      these four always sum to `n_calls_total`.
    - `clip_s`, `n_near_outliers_dropped`, `n_far_outliers_dropped`: the
      outlier policy applied before `near_mean_ms`/`far_mean_ms` were
      computed, and how many picks in each of those two bins it removed
      (review round 1, Finding 1).
    - `near_mean_ms`, `far_mean_ms`, `range_trend_ms`: the primary,
      outlier-cut statistic.
    - `near_median_ms`, `far_median_ms`, `range_trend_median_ms`: the same
      trend computed via the median instead of the mean -- naturally
      resistant to the outliers `clip_s` removes, kept alongside the mean as
      an independent check rather than a replacement for it.
    - `near_mean_raw_ms`, `far_mean_raw_ms`, `range_trend_raw_ms`: the
      uncut mean (`clip_s=None`), kept for transparency so a reader can see
      how much the outlier cut moved the number, not just trust that it did.
    - `near_n`, `far_n`, `near_std_ms`, `far_std_ms`, `range_trend_se_ms`: the
      cut bin's own population, spread, and a naive standard error on
      `range_trend_ms` (`sqrt(SE_near**2 + SE_far**2)`), so a change in
      `range_trend_ms` across rows of this table can be weighed against
      sampling noise rather than eyeballed. This treats the near and far
      bins as independent samples, which is only approximate (both draw from
      the same solved calls) -- a rough gauge, not a rigorous paired test.
    - `model_name` and, when the model carries a sound-speed profile,
      `profile_source` / `profile_validated`: which physics produced this
      row and whether that physics was checked against measurement -- see
      this module's own module docstring and experiments/runner.py's
      PROFILES commentary for why that distinction must survive into result
      tables, not just code comments.
    """
    profile = getattr(model, "profile", None)
    model_name = getattr(model, "name", str(model))
    profile_source = getattr(profile, "source", "") if profile is not None else ""
    # Delegates to `_profile_validation_note` (defined below in this module),
    # which since Task 14 reads `SoundSpeedProfile.validated` -- a structured
    # field set once, in experiments.runner, where each profile's provenance
    # is actually known -- rather than parsing `source`. An earlier version of
    # this classifier matched on "hycom_south_deep" / "45.1492", substrings
    # that occur in NONE of the three `experiments.runner.PROFILES` source
    # strings (they look like "hycom:2021-11-04:45.15,-125.11:..."), so it
    # silently fell through to "unknown" for every profile-carrying model --
    # including `hycom_oregon_nov`, the one profile that genuinely IS
    # validated (Task 12 review, Finding 2). A second round of string-matching
    # fixed that but remained fragile to the same class of bug; the
    # structured field removes the parsing step entirely.
    profile_validated = _profile_validation_note(profile)

    range_bins = list(range_bins)
    out = []
    for z in z_values:
        rows, counts = _solve_and_residualize(picks, model, float(z), min_picks)
        n_total = sum(counts.values())
        if not rows:
            out.append({"source_z": float(z), "n_calls_total": n_total,
                        "n_too_few_picks": counts["too_few_picks"],
                        "n_no_basin": counts["no_basin"],
                        "n_solver_error": counts["solver_error"],
                        "n_calls_solved": counts["solved"],
                        "n_picks": 0, "clip_s": clip_s,
                        "near_mean_ms": np.nan, "far_mean_ms": np.nan, "range_trend_ms": np.nan,
                        "near_median_ms": np.nan, "far_median_ms": np.nan, "range_trend_median_ms": np.nan,
                        "near_mean_raw_ms": np.nan, "far_mean_raw_ms": np.nan, "range_trend_raw_ms": np.nan,
                        "n_near_outliers_dropped": 0, "n_far_outliers_dropped": 0,
                        "near_n": 0, "far_n": 0, "near_std_ms": np.nan, "far_std_ms": np.nan,
                        "range_trend_se_ms": np.nan,
                        "model_name": model_name, "profile_source": profile_source,
                        "profile_validated": profile_validated})
            continue
        df = pd.concat(rows, ignore_index=True)

        s_cut = stratify(df, range_bins, clip_s=clip_s)
        s_raw = stratify(df, range_bins, clip_s=None)
        near_cut, far_cut = s_cut.iloc[0], s_cut.iloc[-1]
        near_raw, far_raw = s_raw.iloc[0], s_raw.iloc[-1]

        # Naive standard error of the (cut) trend, from each bin's own std/n
        # via ordinary error propagation (SE_trend = sqrt(SE_near^2 + SE_far^2)).
        # This treats the near and far bins as independent samples, which is
        # only approximately true (both come from the same solved calls,
        # correlated through each call's own picks) -- a rough gauge of
        # whether an observed change in the trend is bigger than sampling
        # noise, not a rigorous paired-sample test.
        near_se = near_cut["std_ms"] / np.sqrt(near_cut["n"]) if near_cut["n"] > 0 else np.nan
        far_se = far_cut["std_ms"] / np.sqrt(far_cut["n"]) if far_cut["n"] > 0 else np.nan
        trend_se = float(np.hypot(near_se, far_se))

        out.append({
            "source_z": float(z),
            "n_calls_total": n_total,
            "n_too_few_picks": counts["too_few_picks"],
            "n_no_basin": counts["no_basin"],
            "n_solver_error": counts["solver_error"],
            "n_calls_solved": counts["solved"],
            "n_picks": int(df.shape[0]),
            "clip_s": clip_s if clip_s is not None else np.nan,
            "near_mean_ms": near_cut["mean_ms"], "far_mean_ms": far_cut["mean_ms"],
            "range_trend_ms": far_cut["mean_ms"] - near_cut["mean_ms"],
            "near_median_ms": near_cut["median_ms"], "far_median_ms": far_cut["median_ms"],
            "range_trend_median_ms": far_cut["median_ms"] - near_cut["median_ms"],
            "near_mean_raw_ms": near_raw["mean_ms"], "far_mean_raw_ms": far_raw["mean_ms"],
            "range_trend_raw_ms": far_raw["mean_ms"] - near_raw["mean_ms"],
            "n_near_outliers_dropped": int(near_cut["n_dropped_outlier"]),
            "n_far_outliers_dropped": int(far_cut["n_dropped_outlier"]),
            "near_n": int(near_cut["n"]), "far_n": int(far_cut["n"]),
            "near_std_ms": near_cut["std_ms"], "far_std_ms": far_cut["std_ms"],
            "range_trend_se_ms": trend_se,
            "model_name": model_name, "profile_source": profile_source,
            "profile_validated": profile_validated,
        })
    return pd.DataFrame(out)


def _profile_validation_note(profile) -> str:
    """Validation status for a `SoundSpeedProfile` (or `None`, for a model
    with no profile at all, e.g. `ConstantC`).

    Same standing requirement `depth_sweep` carries above (Task 9): a result
    row must say not just which profile produced it but whether that physics
    was ever checked against measurement, so a reader holding only the CSV
    can judge it without re-deriving provenance from code comments.

    Task 14 review, Important 8: this used to string-match substrings of the
    profile's own `source` field (built by `reanalysis.fetch_hycom_profile`
    for display, not for parsing) -- first a dead "45.1492" check that never
    matched any real source string, then a fixed-but-still-fragile check on
    "45.15,-125.11". Any future profile whose formatted coordinates happened
    to collide, or whose `source` format changed, would silently misclassify
    again. Reading the structured `validated` field set once in
    `experiments.runner` (the one place a profile's real provenance is known)
    removes the parsing step -- and the whole class of bug -- entirely.
    """
    if profile is None:
        return "n/a (no profile; constant speed)"
    return profile.validated if profile.validated is not None else "unknown"


def shadow_zone_table(profile, seabed_sweep, freq_hz, geometries) -> pd.DataFrame:
    """Section 7.0.3 -- does ray theory's shadow zone survive a modal treatment
    at 20 Hz?

    A downward-refracting profile puts a source out of *direct-ray* reach of
    some receiver depths past `max_direct_range` (Task 3). At 20 Hz, in a
    guide only 1.3-7 wavelengths deep, that ray prediction is exactly the
    kind normal modes can overturn (see `soundspeed.modes`'s module
    docstring): a mode being trapped at that depth and range means a
    propagating path exists in principle even where the ray picture finds
    none.

    Mode existence is read from `ModeSet.n_modes_expected`, never the raw
    `n_modes`: Task 10 found that truncation-based counting can silently
    return `n_modes == 0` with a trapped mode still present near the
    threshold, which here would misread a numerical artefact as evidence
    *for* the shadow zone. `n_modes_expected` is the exact, truncation-free
    count and does not depend on `ModeSet.converged` -- a near-threshold
    `converged=False` usually means only the mode SHAPES are under-resolved,
    not the count (see `ModeSet`'s own docstring for the three-state
    breakdown), so rows are never dropped or reweighted on `converged` here.
    `all_converged` is carried through only as a diagnostic, not a filter.

    Seabed geoacoustics (`c`, `rho`, `alpha`) are unknown at this site, so
    `seabed_sweep` must supply more than one `Seabed` and the mode count is
    reported as the range [`n_modes_min`, `n_modes_max`] across the sweep --
    a spread standing in for the uncertainty, never a single value invented
    by fixing the seabed at one guess.

    BOUNDARY (see `soundspeed.modes`'s "What this module does NOT answer"):
    a nonzero `n_modes_min` means a propagating path exists in principle. It
    is NOT a statement that a detectable arrival reaches the receiver --
    there is no attenuation, source excitation, or transmission-loss model
    behind this number. Read `n_modes_min >= 1` as "ray theory's 'no path
    exists' claim is falsified", never as "energy arrives".

    MODELLING CAVEAT (review round 1, Finding 3): each row models a
    RANGE-INDEPENDENT guide at the RECEIVER's own depth (`water_depth_m =
    max(z_rcv, 1.0)`), because the DAS fibre lies on the seafloor and so the
    channel depth is the local water depth there. The real cable spans a
    shoaling path (roughly 100-1575 m here), and adiabatically the
    propagating mode set over a whole path is bounded by its SHALLOWEST
    point -- so a deep-receiver row's count is an upper bound on what the
    actual path supports, not the path count itself, and reads too high if
    the path crosses shallower water. This does not change the EXISTENCE
    conclusion (a shallower guide traps fewer modes, but on this profile at
    20 Hz `n_modes_expected` does not reach zero until well below the
    cable's ~100 m shallow end -- see the module's report for the measured
    floor), but the per-row mode counts above should be read as
    receiver-depth counts, not as the true along-path count.
    """
    if not seabed_sweep:
        raise ValueError(
            "seabed_sweep must not be empty: seabed geoacoustics are unknown at "
            "this site and must be swept, never fixed at one invented value")

    columns = ["z_src", "z_rcv", "range_m", "freq_hz", "ray_max_direct_m",
               "ray_says_shadowed", "n_modes_min", "n_modes_max", "n_seabed",
               "all_converged", "seabed_c_min", "seabed_c_max",
               "profile_source", "profile_validated"]
    if not geometries:
        return pd.DataFrame(columns=columns)

    profile_source = getattr(profile, "source", "")
    profile_validated = _profile_validation_note(profile)
    seabed_c_min = float(min(sb.c for sb in seabed_sweep))
    seabed_c_max = float(max(sb.c for sb in seabed_sweep))
    n_seabed = len(seabed_sweep)

    rows = []
    for z_src, z_rcv, range_m in geometries:
        z_src, z_rcv, range_m = float(z_src), float(z_rcv), float(range_m)
        # `max_direct_range` clamps the profile past its sampled range exactly
        # like `modes` does (`SoundSpeedProfile.c_at`), but does so silently --
        # it emits no warning of its own. A geometry whose receiver depth
        # exceeds the profile's range is clamp-dependent here too.
        ray_max = max_direct_range(profile, z_src, z_rcv)
        # The DAS fibre lies on the seafloor, so the receiver depth in a
        # (z_src, z_rcv, range_m) geometry doubles as the local waveguide
        # depth for the modal treatment -- see the MODELLING CAVEAT above for
        # what that does and does not bias.
        water_depth_m = max(z_rcv, 1.0)
        mode_sets = [modes(profile, sb, freq_hz, water_depth_m) for sb in seabed_sweep]
        counts = [ms.n_modes_expected for ms in mode_sets]
        rows.append({
            "z_src": z_src, "z_rcv": z_rcv, "range_m": range_m,
            "freq_hz": float(freq_hz),
            "ray_max_direct_m": float(ray_max),
            "ray_says_shadowed": bool(ray_max < range_m),
            "n_modes_min": int(min(counts)), "n_modes_max": int(max(counts)),
            "n_seabed": n_seabed,
            "all_converged": bool(all(ms.converged for ms in mode_sets)),
            "seabed_c_min": seabed_c_min, "seabed_c_max": seabed_c_max,
            "profile_source": profile_source,
            "profile_validated": profile_validated,
        })
    return pd.DataFrame(rows, columns=columns)
