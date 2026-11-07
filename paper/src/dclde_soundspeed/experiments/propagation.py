"""Does the first arrival travel a direct water path, or a trapped one?

Plan A established that trapped modes EXIST wherever ray theory called the
geometry shadowed (findings section 3.3). Existence is not arrival. This module
asks the next question: does such a mode survive 5-40 km across a
shelf-to-abyssal margin, and does a source at -30 m excite it -- and can this
dataset tell the difference.

Two hypotheses, compared by the SHAPE of received level against range and
against receiver depth. Never by absolute level: the pick SNR comes from a
denoised strain product with unknown gain -- and this applies to a
MODEL-VS-MODEL comparison exactly as it does to model-vs-data. The two TL
columns this module produces sit on different reference scales for a
structural reason, not a propagation reason: the modal columns
(`tl_modal_*_db`) carry the real source excitation and receiver mode-shape
factors (a 20 Hz source at 30 m couples weakly to these guides), while
`tl_geometric_db` is a bare spreading law against an implicit unit source at
1 m with none of that structure. `tl_modal - tl_geometric` is NOT excess loss
from mode stripping or bottom interaction -- it is dominated by that
reference-scale mismatch, and reading it as excess loss is a confident wrong
conclusion this docstring exists to head off.

  H-geometric  cylindrical spreading plus water absorption. No bottom loss, no
               mode stripping. This is what the straight-ray ConstantC /
               PathAverage family implies about amplitude, and it is the null.
  H-modal      adiabatic mode sum over the real bathymetry, with per-mode
               sediment attenuation and mode stripping.

Neither is 'the truth'. The question is which is less inconsistent with what
was measured, and whether the difference clears the noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from dclde_localization.physics.bathymetry import PathProfile
from dclde_soundspeed.soundspeed.modes import Seabed, modes
from dclde_localization.physics.sound_speed_profile import to_down
from dclde_soundspeed.soundspeed.tl import (
    DegenerateGeometryError, adiabatic_tl, das_axial_response, excitation)

# Thorp-scale water absorption at 20 Hz is ~0.001 dB/km -- utterly negligible
# over 40 km (0.04 dB). It is carried as an explicit parameter rather than
# dropped so that the null is a stated model, not an unexamined 10*log10(r).
DEFAULT_WATER_ALPHA_DB_PER_KM = 0.001

# The two self-description gaps a reader of the CSV -- not this source file --
# needs stated plainly, so they are written into `write_results`'s provenance
# header via `run_propagation_tl` rather than living only here.
CAVEAT_REFERENCE_SCALE = (
    "tl_modal_coherent_db / tl_modal_incoherent_db and tl_geometric_db are NOT "
    "on the same reference scale: the modal columns carry real source "
    "excitation and receiver mode-shape factors (a 20 Hz source at 30 m "
    "couples weakly to these guides), while tl_geometric_db is a bare "
    "spreading law against an implicit unit source at 1 m with none of that "
    "structure. The 50-70 dB gap between them is NOT excess loss or evidence "
    "about propagation -- do not compute tl_modal - tl_geometric and call it "
    "excess loss from mode stripping.")
CAVEAT_RANGE_DEPTH_COVARY = (
    "range_m and the water depths at each geometry's two ends co-vary by "
    "construction (the geometries trace a realistic shelf-to-abyssal path): "
    "this is not a TL-versus-range curve at fixed geometry. Grouping by "
    "range_m alone conflates a range trend with a depth trend.")

# The fixed-threshold left-censoring model is an APPROXIMATION swept over a
# grid, not a description of a known instrument setting -- stated here so it
# travels with the CSV, not just this module's source (Task 11, item C).
# Measured over all 11,117 real picks: 31 (0.3%) sit at or below threshold=0,
# 172 (1.5%) at threshold=4, 384 (3.5%) at threshold=6, 1015 (9.1%) at
# threshold=8. The SNR distribution tails smoothly to zero across that whole
# range -- there is NO sharp detection cliff visible in it -- so a single
# fixed threshold cannot be "the" detector setting; it stands in for
# whatever the annotators actually did, and the correction it produces is
# small on this data. A reader must not infer a known instrument threshold
# from this table.
CAVEAT_THRESHOLD_IS_APPROXIMATION = (
    "The censoring threshold is SWEPT, not a known instrument setting: the "
    "annotation snapshot does not record a detection threshold, and the "
    "observed pick SNR tails smoothly to zero with no sharp cliff (measured "
    "over all 11,117 picks: 31 [0.3%] at or below threshold=0, 172 [1.5%] at "
    "4, 384 [3.5%] at 6, 1015 [9.1%] at 8). The fixed-threshold left-"
    "censoring (Tobit) model is therefore an APPROXIMATION to whatever the "
    "annotators actually did, and the correction it makes is small on this "
    "data -- do not read the swept values as a documented sensor property.")
# The floor `10*log10(clip(axial_response, AXIAL_RESPONSE_FLOOR, None))` uses
# to avoid log(0): axial_response is a squared cosine and reaches exactly 0
# at true broadside incidence, so an unclipped log is -inf. -30 dB
# (AXIAL_RESPONSE_FLOOR = 1e-3) affects 5 of the 8,660 real eligible picks
# (0.06%) -- verified directly against the real frame, not asserted.
AXIAL_RESPONSE_FLOOR = 1e-3

CAVEAT_AXIAL_PROJECTION_CONFOUND = (
    "The regression carries an explicit projection term "
    "(10*log10(clip(axial_response, 1e-3, None)), floored -30 dB, affecting "
    "5/8,660 real picks) because DROPPING it is a confound, not a "
    "simplification: pooled corr(10*log10(axial_response), log10(range_m)) "
    "is only -0.09 across all picks, but the fit has PER-CALL INTERCEPTS, so "
    "only the WITHIN-CALL variation identifies the range and depth slopes -- "
    "and the within-call correlation between the two is much stronger (mean "
    "0.36, median 0.45, 78% of calls |corr| > 0.3, measured over the 381 "
    "real eligible calls). Within a call, longer-range picks are "
    "systematically more endfire and therefore sensed better, which pushes "
    "the extra long-range SNR into the range coefficient and biases the "
    "fitted decay SHALLOW if the term is omitted. The `projection` column "
    "('fitted' includes this term as a free coefficient; 'omitted' drops it, "
    "reproducing the earlier, confounded regression) lets a reader see the "
    "difference directly rather than trust either alone.")
CAVEAT_MAX_RANGE_ASYMMETRY = (
    "Spec section 6.3's falsification asymmetry, reported as one rather than "
    "left in a docstring the CSV does not carry: maximum observed range is "
    "bounded by cable extent and by which channels were examined, not by "
    "physics alone. This table can therefore FALSIFY a hypothesis that "
    "predicts detection ranges SHORTER than what was observed here, but it "
    "CANNOT distinguish two hypotheses that both predict detection beyond "
    "the cable's own extent. Do not read a long max_range_m as confirming "
    "either hypothesis; it can only rule one out.")
CAVEAT_DENOMINATOR_BUCKETS = (
    "Every call in the source annotation snapshot is accounted for by "
    "exactly one of calls_eligible, calls_below_min_picks, calls_unsolved, "
    "calls_flipped, calls_path_failed or calls_geometry_error, and those six "
    "sum to calls_total; picks_eligible is the pick-level count within "
    "calls_eligible only. No rate elsewhere in this experiment (e.g. a "
    "censored fraction or a max-range statistic) may be quoted without "
    "naming which of these denominators it is over.")
CAVEAT_ZERO_SNR_RULE = (
    "31 of the 11,117 real annotation-snapshot picks carry exactly zero SNR; "
    "of those, 16 fall within the 8,660-pick eligible fitting sample used "
    "here (the other 15 belong to calls excluded upstream -- below min "
    "picks, unsolved, flipped, path-failed, or geometry-error -- see "
    "propagation_denominators.csv). Spec section 6.4 treats the 16 as "
    "censored AT the threshold rather than dropped (see decay_fit_table's "
    "docstring), but this table reports the fit BOTH ways -- see the "
    "zero_snr_rule column ('censored' keeps them in the sample folded into "
    "the ordinary below-threshold censoring mask; 'dropped' removes them "
    "before fitting) -- so the rule's own influence on the answer is "
    "visible rather than asserted away.")
CAVEAT_DESCRIPTIVE_NOT_TEST = (
    "THIS TABLE IS DESCRIPTIVE, NOT THE PRE-REGISTERED HYPOTHESIS TEST. It "
    "fits FREE slopes for range, depth and (optionally) axial projection, "
    "and a reader is naturally invited to eyeball those coefficients "
    "against a physics benchmark (e.g. -10 for spherical spreading) -- spec "
    "section 6.4 forbids exactly that substitution for the actual "
    "hypothesis test. The pre-registered statistic is the DIFFERENCE IN "
    "FITTED LOG-LIKELIHOOD between H-geometric and H-modal, each offered as "
    "a FIXED shape (not a free slope), reported in "
    "propagation_hypothesis_test.csv -- that is where the actual "
    "pre-registered comparison lives, not here.")
CAVEAT_CENSORED_VS_TRUNCATED = (
    "This is a left-CENSORED (Type-I Tobit) likelihood, not a left-TRUNCATED "
    "one, and the two are not the same correction for the same problem. "
    "Censoring means a row is PRESENT with a value known only as a bound; "
    "truncation means sub-threshold rows are ABSENT from the sample "
    "entirely. The real DAS picks are truncated -- rows below whatever "
    "threshold the annotators used were never recorded at all -- but this "
    "fit is censored: `snr <= threshold` observations that ARE present in "
    "the data are treated as bounded-below observations, which tells the "
    "estimator both how many such picks exist and at what range/depth they "
    "occurred, information a genuinely truncated sample does not contain. "
    "The threshold sweep therefore measures robustness to DISCARDING "
    "low-SNR information (whether keeping it as a bound changes the answer "
    "much), not sensitivity to a detection threshold in the truncation "
    "sense the spec describes. The correct estimator for a genuinely "
    "truncated sample is a truncated-normal likelihood (subtracting "
    "log(1 - Phi((threshold - mu)/sigma)) per RETAINED observation, with no "
    "bounded-observation term at all, since there are no such rows) -- not "
    "implemented here; recorded as a follow-up, not undertaken as part of "
    "this task.")


def geometric_tl(range_m, *, alpha_db_per_km: float = DEFAULT_WATER_ALPHA_DB_PER_KM):
    """H-geometric: cylindrical spreading plus water absorption.

    Cylindrical (10 log r), not spherical (20 log r), because beyond a few water
    depths a waveguide spreads energy in two dimensions, and the comparison
    against H-modal would otherwise be rigged by giving the null the wrong
    spreading law.
    """
    r = np.asarray(range_m, dtype=float)
    if np.any(r <= 0):
        raise ValueError("range_m must be positive")
    return 10.0 * np.log10(r) + alpha_db_per_km * r / 1000.0


def hypothesis_shape_geometric(range_m, *,
                               alpha_db_per_km: float = DEFAULT_WATER_ALPHA_DB_PER_KM):
    """H-geometric's predicted per-pick SNR SHAPE in dB, for use as a fixed
    offset (spec section 6.4's pre-registered hypothesis test).

    Predicted received level falls as transmission loss rises, so the shape
    is simply `-geometric_tl`; the per-call intercept in
    `hypothesis_comparison_table` absorbs the unknown source level and
    denoiser gain, so only this shape -- not any additive constant -- is
    compared against the data.
    """
    return -geometric_tl(range_m, alpha_db_per_km=alpha_db_per_km)


def _linear_path(range_m: float, d_source: float, d_receiver: float,
                 n_steps: int) -> PathProfile:
    return PathProfile(range_m=np.linspace(0.0, float(range_m), int(n_steps)),
                       depth_m=np.linspace(float(d_source), float(d_receiver),
                                           int(n_steps)),
                       ok=True)


def predicted_tl_table(profile, seabed_sweep, freq_hz: float, geometries,
                       *, n_steps: int = 32,
                       water_alpha_db_per_km: float = DEFAULT_WATER_ALPHA_DB_PER_KM,
                       nz: int = 400) -> pd.DataFrame:
    """Result 1 (spec section 6.2): predicted TL for both hypotheses over the
    seabed sweep, with no data involved.

    `geometries` is a list of
    `(source_z_m, receiver_z_m, range_m, water_depth_source_m, water_depth_receiver_m)`,
    all positive-down metres. The path between the two water depths is linear:
    this is a forward calculation over idealised geometries, and real
    bathymetry enters in the data comparison, not here.

    Every row reports modes existing, modes stripped, and how strongly a source
    at that depth excites them, because 'no modes', 'modes stripped' and 'modes
    present but not excited' are three different answers.

    Two things this table does NOT self-evidently show, stated here and
    carried into the committed CSV's header by `run_propagation_tl` (see
    `CAVEAT_REFERENCE_SCALE` and `CAVEAT_RANGE_DEPTH_COVARY`):

    1. `tl_modal_*_db` and `tl_geometric_db` sit on different reference
       scales (real excitation/mode-shape factors vs. a bare unit-source
       spreading law), so their 50-70 dB gap is not excess loss and must
       never be read as evidence about propagation.
    2. `range_m` and the water depths at each geometry's two ends co-vary by
       construction, so this is not a TL-versus-range curve at fixed
       geometry -- grouping by `range_m` alone conflates range and depth.
    """
    rows = []
    for (sz, rz, rng, d_src, d_rcv) in geometries:
        path = _linear_path(rng, d_src, d_rcv, n_steps)
        for sb in seabed_sweep:
            seabed = Seabed(c=float(sb["c"]), rho=float(sb["rho"]),
                            alpha=float(sb["alpha"]))

            coh = adiabatic_tl(profile, seabed, freq_hz, path, sz, rz,
                               coherent=True, nz=nz)
            inc = adiabatic_tl(profile, seabed, freq_hz, path, sz, rz,
                               coherent=False, nz=nz)
            ms0 = modes(profile, seabed, freq_hz, float(d_src), nz=nz)

            exc = excitation(ms0, sz)
            exc_rms = float(np.sqrt((exc**2).mean())) if exc.size else 0.0

            rows.append({
                "source_z_m": float(sz), "receiver_z_m": float(rz),
                "range_m": float(rng), "water_depth_m": float(d_rcv),
                "seabed_c": seabed.c, "seabed_rho": seabed.rho,
                "seabed_alpha": seabed.alpha,
                "n_modes_start": inc.n_modes_start,
                "n_modes_end": inc.n_modes_end,
                "n_stripped": inc.n_stripped,
                "n_unresolved_steps": inc.n_unresolved_steps,
                "excitation_rms": exc_rms,
                "tl_modal_coherent_db": coh.tl_db,
                "tl_modal_incoherent_db": inc.tl_db,
                "tl_geometric_db": float(geometric_tl(
                    rng, alpha_db_per_km=water_alpha_db_per_km)),
                "path_ok": bool(inc.ok),
            })
    return pd.DataFrame(rows)


_FRAME_COLUMNS = ["call", "cable", "dist", "snr", "px", "py", "pz", "sx", "sy",
                 "range_m", "receiver_z_m", "axial_response",
                 "water_depth_receiver_m", "path_ok", "path_reason"]


def _range_to_channels(sx: float, sy: float, source_z_repo: float,
                       px, py, pz) -> np.ndarray:
    """Slant source-to-channel range, metres.

    `picks.dist` is distance ALONG THE FIBRE, not range. Confusing the two is
    wrong by tens of kilometres and is the easiest mistake in this module, so
    range is computed here, once, from positions.

    `source_z_repo` and `pz` are both in the repo's negative-down convention
    (see module callers); the difference is convention-agnostic, so this
    works regardless of sign as long as both arguments share one convention.
    """
    px, py, pz = (np.asarray(v, dtype=float) for v in (px, py, pz))
    return np.sqrt((px - sx) ** 2 + (py - sy) ** 2
                   + (pz - float(source_z_repo)) ** 2)


def _cable_tangents(cable, dist_m) -> np.ndarray:
    """Unit tangent of the cable at each along-cable distance, (n, 2).

    The central-difference step is `max(cable.spacing_m, 1.0)`, typically a
    few metres. Measured directly against the real cable geometry: the
    resulting tangent direction is stable to 0.5-0.7 degrees as the step is
    varied from 2 m to 1 km, so this step size is not a source of meaningful
    error in the axial response it feeds.
    """
    d = np.asarray(dist_m, dtype=float)
    step = max(float(cable.spacing_m), 1.0)
    x0, y0, _ = cable.chan_to_xyz(d - step)
    x1, y1, _ = cable.chan_to_xyz(d + step)
    tx, ty = x1 - x0, y1 - y0
    n = np.hypot(tx, ty)
    n = np.where(n > 0, n, 1.0)
    return np.stack([tx / n, ty / n], axis=1)


def _axial_response_column(sub: pd.DataFrame, cables: dict, source_xy) -> np.ndarray:
    """`das_axial_response` for every row of `sub`, each against ITS OWN
    channel's cable tangent -- not a tangent borrowed from a neighbouring
    pick.

    `sub` is required to carry a plain RangeIndex (0..len(sub)-1): the
    per-cable group's pandas index labels are then identical to their
    position in `sub`, so `group.index` can be used directly to place each
    row's result back into `out` by position, with no `index.get_loc` lookup
    (fragile against duplicate index values) anywhere in this function.
    Tangents for a group are computed from that SAME group's `dist` values,
    in that group's own row order, so index label i's tangent is always
    built from index label i's own `dist` -- the ordering is never
    reshuffled between the tangent computation and the assignment.
    """
    if not sub.index.equals(pd.RangeIndex(len(sub))):
        raise ValueError("sub must have a plain RangeIndex (reset_index(drop=True))")

    sx, sy = source_xy
    out = np.empty(len(sub), dtype=float)
    for cable_name, group in sub.groupby("cable", sort=False):
        tangents = _cable_tangents(cables[cable_name], group["dist"].to_numpy(float))
        gx = group["px"].to_numpy(float)
        gy = group["py"].to_numpy(float)
        for row_pos, cx, cy, (tx, ty) in zip(group.index, gx, gy, tangents):
            out[row_pos] = das_axial_response((sx, sy), (cx, cy), (tx, ty))
    return out


def analysis_frame(picks: pd.DataFrame, cables: dict, grid, model, *,
                   source_z: float, min_picks: int = 8,
                   flip_threshold_m: float = 500.0) -> tuple[pd.DataFrame, dict]:
    """Per-pick frame for the data comparison, plus the denominators.

    POSITIONS ARE AN INPUT, NOT A RESULT (spec section 6.1). Range needs a
    source position, and source positions come from `solve()`, which itself
    uses a propagation model -- the very thing this comparison is meant to
    test. That circularity is broken here, not ignored, by an explicit
    argument, not a proof:

      * positions come from the `model` passed in (normally `ConstantC`) and
        are stated as an INPUT to this frame, never re-derived or treated as
        independently confirmed by anything below;
      * the sensitivity of that input is already measured elsewhere -- Plan A
        section 3.5 found a depth-dependent effective c moves basin-stable
        positions a median 9.0-9.5 m, negligible as a covariate at the 5-40 km
        ranges here;
      * calls whose top two basins are confidently, independently resolved
        AND separated by more than `flip_threshold_m` are excluded entirely
        (counted as `calls_flipped`), because for those the position
        difference between models is a median 1,273 m and the "negligible
        sensitivity" argument above does not hold for them.

    This function therefore never uses the solved positions as ground truth
    to grade the propagation model against; it only uses them as the fixed
    geometry needed to compute range, receiver depth, and axial response for
    the (separate) data-vs-model comparison that consumes this frame's
    output.

    Every call in `picks` is accounted for by exactly one of the denominator
    buckets: `calls_eligible + calls_below_min_picks + calls_unsolved +
    calls_flipped + calls_path_failed + calls_geometry_error ==
    calls_total`.

    A per-call geometry degeneracy (e.g. a solved source position that
    coincides horizontally with one of its own channels, which
    `das_axial_response` refuses rather than divide by zero) is caught,
    counted in `calls_geometry_error`, and that one call is skipped -- it
    must never abort the whole run and lose every other call already
    processed (Task 11 runs this over all 628 calls; one degenerate
    geometry cannot be allowed to cost the rest).
    """
    from dclde_localization.physics.bathymetry import depth_along_path
    from dclde_soundspeed.experiments.ablation import solve_all

    calls_total = int(picks["call"].nunique())
    below_calls = {call for call, sub in picks.groupby("call") if len(sub) < min_picks}
    below = len(below_calls)

    solutions = solve_all(picks, model, source_z=source_z, min_picks=min_picks)
    unsolved = calls_total - below - len(solutions)

    flipped = 0
    path_failed = 0
    geometry_error = 0
    rows = []
    for call, sol in solutions.items():
        # A call whose two best basins are confidently, independently
        # resolved AND far apart is a flip risk for this model comparison;
        # per the docstring above it is counted, never merged into the frame.
        if sol.separation_m is not None and sol.separation_m > flip_threshold_m:
            flipped += 1
            continue

        sub = picks[picks["call"] == call].reset_index(drop=True)
        best = sol.best
        rng = _range_to_channels(best.x, best.y, source_z,
                                 sub["px"], sub["py"], sub["pz"])

        ok_all = True
        reason = ""
        depths = np.full(len(sub), np.nan)
        for i, (cx, cy) in enumerate(zip(sub["px"].to_numpy(float),
                                         sub["py"].to_numpy(float))):
            prof = depth_along_path(grid, best.x, best.y, cx, cy, n_steps=16)
            if not prof.ok:
                ok_all = False
                reason = prof.reason
                break
            depths[i] = prof.depth_m[-1]
        if not ok_all:
            path_failed += 1
            continue

        # Only das_axial_response's own degenerate-geometry refusal is
        # caught and counted here -- e.g. a solved source position that
        # coincides horizontally with one of its own channels. Caught by
        # TYPE (`DegenerateGeometryError`), not by matching the exception's
        # message text: a message is not an interface, and this repo has
        # already paid once for coupling two modules through prose (commit
        # babbf50, "Replace profile-source string matching with a
        # structured validation field") -- a reworded message would
        # silently stop being caught here with nothing to flag the drift.
        # `_axial_response_column`'s own defensive check (the
        # sub-must-carry-a-plain-RangeIndex invariant this task's tangent
        # mapping correctness rests on) and a `ValueError` from a corrupted
        # upstream column (e.g. non-numeric `px`/`py`/`pz`/`dist`) are both
        # plain `ValueError`, not `DegenerateGeometryError`, so both still
        # propagate and crash loudly rather than being silently refiled as
        # a geometry miscount.
        try:
            axial = _axial_response_column(sub, cables, (best.x, best.y))
        except DegenerateGeometryError:
            geometry_error += 1
            continue

        rows.append(pd.DataFrame({
            "call": call, "cable": sub["cable"].to_numpy(),
            "dist": sub["dist"].to_numpy(float), "snr": sub["snr"].to_numpy(float),
            "px": sub["px"].to_numpy(float), "py": sub["py"].to_numpy(float),
            "pz": sub["pz"].to_numpy(float),
            "sx": best.x, "sy": best.y,
            "range_m": rng, "receiver_z_m": to_down(sub["pz"].to_numpy(float)),
            "axial_response": axial,
            "water_depth_receiver_m": depths,
            "path_ok": True, "path_reason": reason,
        }))

    frame = (pd.concat(rows, ignore_index=True)[_FRAME_COLUMNS] if rows
             else pd.DataFrame(columns=_FRAME_COLUMNS))
    den = {
        "calls_total": calls_total,
        "calls_below_min_picks": below,
        "calls_unsolved": int(unsolved),
        "calls_flipped": int(flipped),
        "calls_path_failed": int(path_failed),
        "calls_geometry_error": int(geometry_error),
        "calls_eligible": int(frame["call"].nunique()) if len(frame) else 0,
        "picks_eligible": int(len(frame)),
    }
    return frame, den


MAX_RANGE_COLUMNS = ["call", "n_picks", "max_range_m", "min_range_m",
                     "max_range_snr", "receiver_z_at_max_m"]


def max_range_table(frame, denominators) -> pd.DataFrame:
    """Result 2 (spec section 6.3): the furthest range at which each call was
    still detected.

    The most robust data contact available -- it needs no calibration of any
    kind, so the denoised SNR's unknown gain cannot touch it.

    ITS WEAKNESS, STATED WITH IT. Maximum observed range is bounded by cable
    extent and by which channels were examined, not by physics alone. It can
    therefore falsify a hypothesis that predicts detection ranges SHORTER than
    what was observed, and cannot distinguish two hypotheses that both predict
    detection beyond the cable. That asymmetry is a property of the test.
    """
    if len(frame) == 0:
        out = pd.DataFrame(columns=MAX_RANGE_COLUMNS)
        out.attrs["denominators"] = dict(denominators)
        return out

    rows = []
    for call, sub in frame.groupby("call"):
        k = int(sub["range_m"].to_numpy().argmax())
        rows.append({
            "call": call,
            "n_picks": int(len(sub)),
            "max_range_m": float(sub["range_m"].iloc[k]),
            "min_range_m": float(sub["range_m"].min()),
            "max_range_snr": float(sub["snr"].iloc[k]),
            "receiver_z_at_max_m": float(sub["receiver_z_m"].iloc[k]),
        })
    out = pd.DataFrame(rows, columns=MAX_RANGE_COLUMNS)
    out.attrs["denominators"] = dict(denominators)
    return out


def censored_fit(y, X, call_index, *, threshold: float, censored_mask=None,
                 offset=None) -> dict:
    """Left-CENSORED Gaussian regression with a free intercept per call.

    NAMING, PRECISELY. This is a Type-I Tobit / left-censored likelihood:
    every row of `y` is an OBSERVED value, and `censored_mask` rows are known
    only to be at-or-below `threshold` (they contribute
    Phi((threshold - mu)/sigma) instead of a density). This is NOT the same
    problem as TRUNCATION, where sub-threshold rows are simply ABSENT from
    the sample -- see `decay_fit_table`'s docstring for why the real DAS
    picks are the truncated case, not the censored one this function
    actually implements, and why that gap is not closed here.

    WHY A FREE INTERCEPT PER CALL. Source level varies per call and the denoiser
    gain is unknown, so absolute level carries no information. Only shape does.

    `offset` (default: none, i.e. zero) is an OPTIONAL fixed per-observation
    additive term with its coefficient PINNED AT 1 -- e.g. a hypothesis's
    predicted (range, depth) shape in dB, offered as a covariate that is not
    itself fitted. `X` may then carry zero columns (`n_beta == 0`): the model
    reduces to `mu = offset + alpha[call]`, i.e. per-call intercepts around a
    fully fixed shape, and `loglik` becomes exactly the statistic spec
    section 6.4 asks for -- the fitted log-likelihood of that hypothesis's
    shape against the data. This is the mechanism `decay_fit_table`'s
    free-slope descriptive fit does NOT use (there, the range/depth/
    projection coefficients are estimated, not pinned).
    """
    y = np.asarray(y, dtype=float)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    call_index = np.asarray(call_index)
    if censored_mask is None:
        censored_mask = np.zeros(y.shape, dtype=bool)
    censored_mask = np.asarray(censored_mask, dtype=bool)
    if offset is None:
        offset = np.zeros_like(y)
    offset = np.asarray(offset, dtype=float)

    codes, uniq = pd.factorize(call_index)
    n_calls = len(uniq)
    n_beta = X.shape[1]

    def unpack(theta):
        return theta[:n_beta], theta[n_beta:n_beta + n_calls], np.exp(theta[-1])

    def neg_loglik(theta):
        beta, alpha, sigma = unpack(theta)
        mu = X @ beta + offset + alpha[codes]
        ll = np.empty_like(y)
        obs = ~censored_mask
        ll[obs] = stats.norm.logpdf(y[obs], mu[obs], sigma)
        if censored_mask.any():
            ll[censored_mask] = stats.norm.logcdf(
                (threshold - mu[censored_mask]) / sigma)
        return -float(ll.sum())

    def neg_loglik_grad(theta):
        """Analytic gradient of `neg_loglik`, supplied so L-BFGS-B does not
        fall back to a finite-difference gradient.

        WHY THIS EXISTS. With no `jac`, scipy estimates the gradient by
        finite differences: one extra objective evaluation per parameter,
        per iteration. At real scale (~381 call intercepts + 2 betas + 1
        log-sigma) that is ~384 evaluations per gradient, which is
        simultaneously why the unconstrained fit was slow (~4-5 s per call)
        AND why it exhausted L-BFGS-B's default evaluation budget before
        converging ("TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT") at higher
        censoring thresholds on the real data. Supplying the exact gradient
        changes neither the likelihood, the censoring rule, nor the
        bootstrap design -- only how the same objective is minimised.

        DERIVATION. Write z_i = (y_i - mu_i)/sigma for an uncensored pick and
        c_i = (threshold - mu_i)/sigma for a censored one, with
        mu_i = X_i @ beta + alpha[call_i]. Then, writing g_mu_i for
        d(loglik_i)/d(mu_i) and g_ls_i for d(loglik_i)/d(log_sigma):

          uncensored:  g_mu_i = z_i / sigma            g_ls_i = z_i**2 - 1
          censored:    g_mu_i = -r_i / sigma           g_ls_i = -r_i * c_i

        where r_i = phi(c_i)/Phi(c_i) is the inverse Mills ratio (the
        standard-normal hazard at c_i), computed as
        exp(logpdf(c_i) - logcdf(c_i)) rather than a raw ratio so it stays
        finite even where Phi(c_i) itself underflows to 0 in linear space
        (deep censoring, large negative c_i) -- `scipy.stats.norm.logcdf`
        uses a numerically stable tail approximation there that a plain
        `norm.cdf` division would lose.

        d(loglik)/d(beta)     = X.T @ g_mu           (chain rule, d(mu)/d(beta) = X)
        d(loglik)/d(alpha_j)  = sum of g_mu_i over i with call_i == j
                                 (d(mu_i)/d(alpha_j) is a one-hot indicator)
        d(loglik)/d(log_sigma) = sum of g_ls_i

        `neg_loglik` is `-loglik`, so the returned gradient is the negation
        of the above. Verified against `scipy.optimize.check_grad` in
        `paper/tests/test_propagation.py::test_analytic_gradient_matches_finite_differences`
        at both a fully-uncensored and a partially-censored point.
        """
        beta, alpha, sigma = unpack(theta)
        mu = X @ beta + offset + alpha[codes]
        g_mu = np.empty_like(y)
        g_ls = np.empty_like(y)

        obs = ~censored_mask
        z = (y[obs] - mu[obs]) / sigma
        g_mu[obs] = z / sigma
        g_ls[obs] = z ** 2 - 1.0

        if censored_mask.any():
            c = (threshold - mu[censored_mask]) / sigma
            r = np.exp(stats.norm.logpdf(c) - stats.norm.logcdf(c))
            g_mu[censored_mask] = -r / sigma
            g_ls[censored_mask] = -r * c

        d_beta = X.T @ g_mu
        d_alpha = np.bincount(codes, weights=g_mu, minlength=n_calls)
        d_logsigma = g_ls.sum()

        grad = np.concatenate([d_beta, d_alpha, [d_logsigma]])
        return -grad

    # Start from an OLS fit on the uncensored rows, which is cheap and close.
    #
    # WARM START ONLY -- this does not touch the likelihood, the optimizer,
    # or any reported number; it only picks where L-BFGS-B starts. The
    # previous version built a dense (n_obs, n_beta + n_calls) design with a
    # one-hot column per call (`np.eye(n_calls)[codes]`) and solved it with
    # `np.linalg.lstsq`. At real scale (~11,000 picks, ~500 calls) that dense
    # `lstsq` dominates the entire `censored_fit` call (~11 s of ~11 s
    # measured), which makes a 200-draw whole-call bootstrap take tens of
    # minutes.
    #
    # By the Frisch-Waugh-Lovell theorem, regressing `y` on `X` plus a full
    # set of call dummies gives EXACTLY the same beta as: demean `y` and `X`
    # within each call, then regress the demeaned `y` on the demeaned `X`
    # alone (a 2-column OLS here, not an (n_beta + n_calls)-column one). Each
    # call's dummy coefficient (its intercept) is then recovered as the mean
    # residual `y - X @ beta` within that call. This is the identical
    # closed-form OLS answer, not an approximation of it, computed ~1700x
    # faster on this design (measured: 0.01 s vs ~11.3 s) because it never
    # materializes the (n_obs, n_calls) dummy block. See
    # `paper/tests/test_propagation.py::test_warm_start_matches_dense_ols_on_uncensored_data`
    # for the equivalence check, and `docs/.../task-11-report.md` for timing.
    #
    # A call with zero uncensored rows (all its picks censored) gets no
    # within-call mean to demean by; its intercept is left at 0.0, matching
    # what the dense `lstsq` warm start already did for such a column (an
    # all-censored call. i.e. all-zero design column, gets the minimum-norm
    # coefficient of 0 from `lstsq` too) -- it is a warm start the optimizer
    # then refines against the real, correctly-censored likelihood, not a
    # final answer either way.
    # `offset` (if any) is a FIXED shape, not a fitted quantity, so the warm
    # start demeans the RESIDUAL after subtracting it (`r = y - offset`)
    # rather than `y` itself -- the same Frisch-Waugh-Lovell argument applies
    # unchanged once `offset` is treated as already-known data rather than
    # part of the design.
    obs = ~censored_mask
    y_obs, X_obs, codes_obs, offset_obs = y[obs], X[obs], codes[obs], offset[obs]
    r_obs = y_obs - offset_obs

    counts = np.bincount(codes_obs, minlength=n_calls).astype(float)
    counts_safe = np.where(counts > 0, counts, 1.0)
    call_sum_r = np.bincount(codes_obs, weights=r_obs, minlength=n_calls)
    call_mean_r = call_sum_r / counts_safe
    call_sum_X = np.zeros((n_calls, n_beta))
    for j in range(n_beta):
        call_sum_X[:, j] = np.bincount(codes_obs, weights=X_obs[:, j], minlength=n_calls)
    call_mean_X = call_sum_X / counts_safe[:, None]

    r_dm = r_obs - call_mean_r[codes_obs]
    X_dm = X_obs - call_mean_X[codes_obs]
    beta0 = (np.linalg.lstsq(X_dm, r_dm, rcond=None)[0] if n_beta > 0
            else np.zeros(0))

    alpha0 = np.zeros(n_calls)
    has_obs = counts > 0
    alpha0[has_obs] = call_mean_r[has_obs] - call_mean_X[has_obs] @ beta0

    resid = r_obs - (X_obs @ beta0 + alpha0[codes_obs])
    theta0 = np.concatenate([beta0, alpha0, [np.log(max(resid.std(), 1e-3))]])

    res = optimize.minimize(neg_loglik, theta0, jac=neg_loglik_grad,
                            method="L-BFGS-B", options={"maxiter": 5000})
    beta, _, sigma = unpack(res.x)
    return {"beta": beta, "sigma": float(sigma), "n": int(y.size),
            "n_censored": int(censored_mask.sum()), "loglik": -float(res.fun),
            "converged": bool(res.success)}


def decay_fit_table(frame, *, threshold: float, n_bootstrap: int = 200,
                    seed: int = 0, zero_snr_rule: str = "censored",
                    projection: str = "fitted") -> pd.DataFrame:
    """Result 3 (spec section 6.4): pick SNR on range AND receiver depth
    (AND, by default, the fibre's axial projection) jointly, per-call
    intercepts free, censored at `threshold`.

    Range, receiver depth and the projection term are correlated on a
    shelf-to-abyssal cable with axial sensing, so they are fitted JOINTLY and
    no one coefficient may be reported alone.

    The interval comes from resampling WHOLE CALLS, not individual picks: picks
    within a call share a source, a source level and a propagation path, so
    resampling picks would understate the interval.

    DESCRIPTIVE, NOT THE PRE-REGISTERED HYPOTHESIS TEST. This function fits
    FREE slopes for range, depth and (optionally) axial projection, and a
    reader is naturally invited to eyeball those coefficients against a
    physics benchmark (e.g. -10 for spherical spreading). Spec section 6.4
    forbids exactly that substitution for the actual hypothesis test: the
    pre-registered statistic is the DIFFERENCE IN FITTED LOG-LIKELIHOOD
    between H-geometric and H-modal, each offered as a FIXED shape (not a
    free slope) -- see `hypothesis_comparison_table`. This function's output
    is a legitimate, useful DESCRIPTIVE result, but it is not that test and
    must not be reported as if it were.

    CENSORED, NOT TRUNCATED -- see `CAVEAT_CENSORED_VS_TRUNCATED` for the
    full statement. In short: this is a left-censored (Type-I Tobit)
    likelihood, which assumes sub-threshold rows are PRESENT with a known
    bound. The real picks are actually left-TRUNCATED -- sub-threshold rows
    are simply ABSENT -- and censoring is not the same correction for that
    problem; it does not add missing rows, so it cannot fully correct the
    selection bias truncation causes. The numerical effect is nonetheless
    small on this data (the censored fraction is 0.3-9.1% across the
    threshold sweep, and results move by roughly 0.01 between the two
    zero-SNR rules below), and rewriting the estimator is out of scope for
    this task (see `CAVEAT_CENSORED_VS_TRUNCATED` for the truncated-normal
    likelihood that would be the correct fix, left as a follow-up).

    ZERO-SNR RULE (spec section 6.4). 31 of the 11,117 real annotation-
    snapshot picks carry exactly zero SNR -- not "low", exactly 0.00; 16 of
    those fall within this function's eligible fitting sample (the other 15
    belong to calls excluded upstream of this frame). Spec 6.4 requires the
    16 be treated as CENSORED AT THE THRESHOLD rather than dropped (an
    exact-zero reading is still information: the pick's true level was at or
    below whatever the threshold was, the same as any other below-threshold
    pick), but ALSO that the result be reported both ways so the rule's own
    influence on the fit is visible to a reader, not asserted away.
    `zero_snr_rule` selects which:

      "censored" (default) -- the exact-zero picks stay in the sample and
          fall into the ordinary `snr <= threshold` censoring mask exactly
          like any other below-threshold pick; they contribute through the
          Tobit log-CDF term, never as a literal density at snr=0.
      "dropped" -- the exact-zero picks are removed from the sample before
          fitting, as if they were missing rather than informative.

    The returned `zero_snr_rule` column names which rule produced that row,
    so the two can be told apart in a table that carries both (as
    `propagation_decay_fit.csv` does, once per swept threshold).

    AXIAL PROJECTION CONFOUND. `frame["axial_response"]` (the fibre's cos^2
    endfire response) is computed for every pick but was, in an earlier
    version of this function, never entered into the regression. That is a
    confound, not a simplification: see `CAVEAT_AXIAL_PROJECTION_CONFOUND`
    for the measured within-call correlation between projection and range
    that makes omitting it push extra long-range SNR into the range
    coefficient and bias the fitted decay shallow. `projection` selects
    whether the term is fitted:

      "fitted" (default) -- a third design column,
          `10*log10(clip(axial_response, AXIAL_RESPONSE_FLOOR, None))`, is
          added as a FREE coefficient. PHYSICS BENCHMARK: the expected
          coefficient is 2, not 1. `axial_response` = cos^2(theta) is itself
          an AMPLITUDE factor for axial strain, not a power factor: a DAS
          fibre's strain response is built from TWO cosine projections
          multiplied together -- one projecting particle motion onto the
          fibre axis, one from the along-fibre spatial-gradient (strain, not
          displacement) sensing mechanism -- i.e. amplitude ~ cos(theta) *
          cos(theta) = cos^2(theta) = axial_response. Whether the pick SNR is
          10*log10(power) with power ~ amplitude^2, or already
          20*log10(amplitude), the expected contribution to SNR in dB is
          20*log10(axial_response) = 2 * (10*log10(axial_response)) -- so a
          fitted coefficient near 2 on this column, not 1, is what the
          physics predicts either way; the factor of 2 comes from
          `axial_response` already being an amplitude quantity, not from the
          power-vs-amplitude convention on SNR. Fitting rather than assuming
          this coefficient lets the data say so instead.
      "omitted" -- the term is left out, reproducing the earlier, confounded
          two-term regression, so its effect on the range/depth coefficients
          is directly visible by comparing the two `projection` values in
          the same table.

    The returned `projection` column names which of the two produced that
    row.
    """
    if zero_snr_rule not in ("censored", "dropped"):
        raise ValueError(
            f"zero_snr_rule must be 'censored' or 'dropped', got {zero_snr_rule!r}")
    if projection not in ("fitted", "omitted"):
        raise ValueError(
            f"projection must be 'fitted' or 'omitted', got {projection!r}")

    # `axial_response` is always required, regardless of `projection`, so
    # "fitted" and "omitted" are fit on the identical sample and remain
    # comparable -- otherwise a NaN axial_response would silently make the
    # two arms differ in n_picks/n_calls for a reason having nothing to do
    # with whether the term is fitted.
    f = frame.dropna(subset=["range_m", "receiver_z_m", "snr", "axial_response"]).copy()
    if zero_snr_rule == "dropped":
        f = f[f["snr"] != 0.0].copy()
    if f.empty:
        return pd.DataFrame(columns=["term", "estimate", "ci_lo", "ci_hi",
                                     "n_picks", "n_calls", "n_censored",
                                     "zero_snr_rule", "projection",
                                     "converged", "n_draws", "loglik"])

    def design(sub):
        cols = [
            np.log10(sub["range_m"].to_numpy(float)),
            sub["receiver_z_m"].to_numpy(float) / 1000.0,
        ]
        if projection == "fitted":
            cols.append(10.0 * np.log10(np.clip(
                sub["axial_response"].to_numpy(float), AXIAL_RESPONSE_FLOOR, None)))
        X = np.column_stack(cols)
        y = sub["snr"].to_numpy(float)
        return y, X, sub["call"].to_numpy(), y <= threshold

    y, X, ci, mask = design(f)
    point = censored_fit(y, X, ci, threshold=threshold, censored_mask=mask)

    rng = np.random.default_rng(seed)
    calls = np.array(sorted(f["call"].unique()))
    draws = []
    for _ in range(int(n_bootstrap)):
        pick = rng.choice(calls, size=calls.size, replace=True)
        parts = []
        for k, c in enumerate(pick):
            part = f[f["call"] == c].copy()
            part["call"] = f"{c}__{k}"   # a call drawn twice is two calls
            parts.append(part)
        yb, Xb, cib, mb = design(pd.concat(parts, ignore_index=True))
        out = censored_fit(yb, Xb, cib, threshold=threshold, censored_mask=mb)
        if out["converged"]:
            draws.append(out["beta"])

    draws = np.array(draws) if draws else np.empty((0, X.shape[1]))
    terms = ["log10_range", "receiver_depth_km"]
    if projection == "fitted":
        terms.append("axial_projection_db")
    rows = []
    for j, term in enumerate(terms):
        lo, hi = ((np.percentile(draws[:, j], [2.5, 97.5]))
                  if len(draws) else (np.nan, np.nan))
        rows.append({"term": term, "estimate": float(point["beta"][j]),
                     "ci_lo": float(lo), "ci_hi": float(hi),
                     "n_picks": point["n"], "n_calls": int(calls.size),
                     "n_censored": point["n_censored"],
                     "zero_snr_rule": zero_snr_rule,
                     "projection": projection,
                     "converged": point["converged"],
                     "n_draws": int(len(draws)),
                     "loglik": point["loglik"]})
    out = pd.DataFrame(rows)
    # Persist the joint bootstrap draws (not just their marginal 2.5/97.5
    # percentiles) so a reader can form a correctly-intervalled linear
    # combination of terms -- e.g. beta_range/beta_proj -- that the summary
    # table's independent per-term CIs cannot support. `terms` names the
    # columns of `draws` in order (one draw per row).
    out.attrs["draws"] = draws
    out.attrs["terms"] = terms
    return out


def hypothesis_comparison_table(frame, hypotheses: dict, *, threshold: float,
                                n_bootstrap: int = 200, seed: int = 0,
                                zero_snr_rule: str = "censored"):
    """The PRE-REGISTERED hypothesis test, spec section 6.4: "each hypothesis
    supplies a predicted (range, depth) shape; each is offered to the same
    per-call-intercept censored fit as a FIXED regressor, and the hypotheses
    are compared by the DIFFERENCE IN FITTED LOG-LIKELIHOOD, with a
    confidence interval obtained by resampling whole calls... No statistic
    other than this one may be introduced after seeing the data."

    This is NOT `decay_fit_table`: that function fits FREE slopes and is a
    separate, descriptive result (see its own docstring). Here, each
    hypothesis's shape is pinned as a fixed offset (`censored_fit`'s
    `offset` argument, coefficient exactly 1) and only the per-call
    intercepts and sigma are fitted -- the comparison is therefore about
    which shape the data tolerate better, not about re-estimating a slope
    the hypothesis already specifies.

    `hypotheses` maps a display name (e.g. "H-geometric", "H-modal") to the
    name of a column in `frame` holding that hypothesis's predicted per-pick
    SNR shape in dB (same sign convention as `hypothesis_shape_geometric`:
    higher predicted shape means higher predicted SNR). Rows with a NaN in
    ANY named shape column are dropped for ALL hypotheses, so every
    hypothesis is compared on the identical sample.

    Returns `(summary, deltas)`:
      `summary` -- one row per hypothesis: its fitted `loglik`, `converged`,
          `n_picks`, `n_calls`, `n_censored`, `n_draws`. `n_draws` is
          STRICTER than "this hypothesis's own fit converged on that draw":
          the code requires EVERY hypothesis passed in (all of `names`, not
          just this one) to converge on a draw before that draw's loglik is
          recorded for ANY hypothesis, so `n_draws` is identical across
          every row of `summary` and counts draws where the WHOLE set
          converged.
      `deltas` -- one row per unordered pair of hypotheses: `delta_loglik`
          (`loglik_a - loglik_b` at the point fit) with a whole-call
          bootstrap `ci_lo`/`ci_hi`. Each bootstrap draw resamples calls
          ONCE and fits every hypothesis on that SAME resampled sample
          (paired, not independent draws per hypothesis), so the interval
          reflects the hypotheses' actual disagreement on shared resampled
          data, not independent sampling noise from resampling twice. A
          draw contributes to a pair's interval only if EVERY hypothesis
          passed in converged on it (the same whole-set condition as
          `summary`'s `n_draws`, not merely the two in this particular
          pair).

    Same zero-SNR-rule and censored-vs-truncated caveats as `decay_fit_table`
    apply identically here -- see `CAVEAT_ZERO_SNR_RULE` and
    `CAVEAT_CENSORED_VS_TRUNCATED`.
    """
    if zero_snr_rule not in ("censored", "dropped"):
        raise ValueError(
            f"zero_snr_rule must be 'censored' or 'dropped', got {zero_snr_rule!r}")
    if len(hypotheses) < 2:
        raise ValueError("hypothesis_comparison_table needs at least 2 hypotheses to compare")

    names = list(hypotheses)
    shape_cols = list(hypotheses.values())
    f = frame.dropna(subset=["range_m", "receiver_z_m", "snr", *shape_cols]).copy()
    if zero_snr_rule == "dropped":
        f = f[f["snr"] != 0.0].copy()

    summary_cols = ["hypothesis", "loglik", "converged", "n_picks", "n_calls",
                    "n_censored", "n_draws", "threshold", "zero_snr_rule"]
    delta_cols = ["comparison", "delta_loglik", "ci_lo", "ci_hi", "n_draws",
                 "threshold", "zero_snr_rule"]
    if f.empty:
        return pd.DataFrame(columns=summary_cols), pd.DataFrame(columns=delta_cols)

    def fit_one(sub, shape_col):
        y = sub["snr"].to_numpy(float)
        offset = sub[shape_col].to_numpy(float)
        X = np.zeros((len(sub), 0))
        ci = sub["call"].to_numpy()
        mask = y <= threshold
        return censored_fit(y, X, ci, threshold=threshold, censored_mask=mask,
                            offset=offset)

    point = {name: fit_one(f, hypotheses[name]) for name in names}

    rng = np.random.default_rng(seed)
    calls = np.array(sorted(f["call"].unique()))
    loglik_draws = {name: [] for name in names}
    for _ in range(int(n_bootstrap)):
        pick = rng.choice(calls, size=calls.size, replace=True)
        parts = []
        for k, c in enumerate(pick):
            part = f[f["call"] == c].copy()
            part["call"] = f"{c}__{k}"   # a call drawn twice is two calls
            parts.append(part)
        resampled = pd.concat(parts, ignore_index=True)
        draw_result = {name: fit_one(resampled, hypotheses[name]) for name in names}
        if all(r["converged"] for r in draw_result.values()):
            for name in names:
                loglik_draws[name].append(draw_result[name]["loglik"])

    summary_rows = [{
        "hypothesis": name, "loglik": point[name]["loglik"],
        "converged": point[name]["converged"], "n_picks": point[name]["n"],
        "n_calls": int(calls.size), "n_censored": point[name]["n_censored"],
        "n_draws": len(loglik_draws[name]), "threshold": threshold,
        "zero_snr_rule": zero_snr_rule,
    } for name in names]

    delta_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            n_paired = min(len(loglik_draws[a]), len(loglik_draws[b]))
            deltas = (np.array(loglik_draws[a][:n_paired])
                     - np.array(loglik_draws[b][:n_paired]))
            lo, hi = ((np.percentile(deltas, [2.5, 97.5])) if n_paired
                      else (np.nan, np.nan))
            delta_rows.append({
                "comparison": f"{a} - {b}",
                "delta_loglik": point[a]["loglik"] - point[b]["loglik"],
                "ci_lo": float(lo), "ci_hi": float(hi), "n_draws": n_paired,
                "threshold": threshold, "zero_snr_rule": zero_snr_rule,
            })

    return (pd.DataFrame(summary_rows, columns=summary_cols),
           pd.DataFrame(delta_rows, columns=delta_cols))


def compute_pick_paths(frame, grid, *, n_steps: int = 32):
    """Each pick's REAL bathymetry path (via `depth_along_path`), computed
    ONCE, independent of seabed -- mode-solving is the only seabed-dependent
    step, so this is the part `modal_shape_for_seabed` reuses across every
    seabed rather than paying for `len(seabeds)` times over.

    Returns `(paths, path_refused_reason)`, two lists parallel to `frame`'s
    rows: `paths[i]` is a `PathProfile` or `None` (refused), and
    `path_refused_reason[i]` names why when `paths[i]` is `None` (else
    `None`).
    """
    from dclde_localization.physics.bathymetry import depth_along_path

    n = len(frame)
    range_m = frame["range_m"].to_numpy(float)
    receiver_z = frame["receiver_z_m"].to_numpy(float)
    sx = frame["sx"].to_numpy(float)
    sy = frame["sy"].to_numpy(float)
    px = frame["px"].to_numpy(float)
    py = frame["py"].to_numpy(float)

    paths = [None] * n
    path_refused_reason = [None] * n
    for i in range(n):
        if not (np.isfinite(range_m[i]) and np.isfinite(receiver_z[i])
               and range_m[i] > 0):
            path_refused_reason[i] = "nan_or_nonpositive_input"
            continue
        prof = depth_along_path(grid, sx[i], sy[i], px[i], py[i], n_steps=n_steps)
        if not prof.ok:
            path_refused_reason[i] = prof.reason
            continue
        # `prof.depth_m` is sampled along `depth_along_path`'s own
        # HORIZONTAL `range_m` axis, which is discarded here in favour of
        # `linspace(0, range_m[i])` -- `range_m[i]` (this frame's `range_m`
        # column) is the SLANT range (source-to-channel 3-D distance), so
        # depth samples meant for a horizontal abscissa are paired with a
        # slightly longer slant one. Measured: slant/horizontal exceeds 1.05
        # for 12.1% of the 8,660 real picks and reaches 4.78 at the closest
        # range. Not corrected here -- both H-geometric and H-modal share
        # this same `range_m` axis, so the mismatch is largely (not
        # perfectly) protected from biasing their delta.
        paths[i] = PathProfile(range_m=np.linspace(0.0, float(range_m[i]), n_steps),
                               depth_m=prof.depth_m, ok=True)
    return paths, path_refused_reason


def modal_shape_for_seabed(paths, path_refused_reason, receiver_z, profile, seabed,
                           *, source_z: float, freq_hz: float = 20.0,
                           nz: int = 400) -> tuple:
    """H-modal's predicted per-pick SNR shape in dB for ONE seabed, given
    paths already computed by `compute_pick_paths`.

    Split out from `modal_shape_columns` (which still loops over every
    seabed and holds every result at once, fine at small scale) so a
    memory-conscious caller can process seabeds ONE AT A TIME -- computing
    this seabed's shape, persisting it, and releasing every intermediate
    the adiabatic solve allocated before starting the next seabed, rather
    than accumulating `len(seabeds)` seabeds' worth of solver state at once
    (see `paper/scripts/run_experiment.py::_write_modal_shapes_incrementally`).

    `source_z` is POSITIVE-DOWN metres (`adiabatic_tl`'s convention -- the
    OPPOSITE sign to `analysis_frame`'s `source_z` argument).

    Returns `(shape_db, counts)` -- see `modal_shape_columns`'s docstring
    for what NaN means and what `counts` reports.
    """
    from collections import Counter

    n = len(paths)
    receiver_z = np.asarray(receiver_z, dtype=float)
    shape = np.full(n, np.nan)
    n_tl_nonfinite = 0
    reasons = Counter(r for r in path_refused_reason if r is not None)
    for i in range(n):
        path = paths[i]
        if path is None:
            continue
        res = adiabatic_tl(profile, seabed, freq_hz, path, source_z,
                           float(receiver_z[i]), coherent=False, nz=nz)
        if not res.ok or not np.isfinite(res.tl_db):
            n_tl_nonfinite += 1
            reasons[res.reason or "non_finite_tl"] += 1
            continue
        shape[i] = -res.tl_db
    n_path_refused = sum(1 for r in path_refused_reason if r is not None)
    counts = {
        "n_picks": n, "n_path_refused": n_path_refused,
        "n_tl_nonfinite": n_tl_nonfinite,
        "n_predicted": int(np.isfinite(shape).sum()),
        "reasons": dict(reasons),
    }
    return shape, counts


def modal_shape_columns(frame, grid, profile, seabeds: dict, *, source_z: float,
                        freq_hz: float = 20.0, n_steps: int = 32,
                        nz: int = 400) -> dict:
    """H-modal's predicted per-pick SNR shape in dB, for EVERY named seabed,
    computed from THIS frame's REAL per-pick bathymetry path (via
    `depth_along_path`) -- not `predicted_tl_table`'s idealised linear paths.
    Substituting that coarser grid for a real per-pick solve would knowingly
    weaken the one test spec section 6.4 pre-registers, so it is not done.

    `source_z` is POSITIVE-DOWN metres (`adiabatic_tl`'s convention -- the
    OPPOSITE sign to `analysis_frame`'s `source_z` argument, which is
    negative-down like the rest of this frame; callers passing
    `frame`-derived source_z must flip its sign).

    A thin convenience wrapper around `compute_pick_paths` (called once) and
    `modal_shape_for_seabed` (called once per seabed): holds every seabed's
    result in memory at once, which is fine for tests and small frames. A
    caller processing the real ~8,660-pick frame across 5 seabeds where peak
    memory matters should call those two functions directly instead and
    persist/release each seabed's result before computing the next (see
    `paper/scripts/run_experiment.py`).

    Returns `{seabed_name: (shape_db, counts)}`. `shape_db` is NaN -- never
    silently dropped from the returned array -- for any pick whose modal
    prediction is unusable: a refused bathymetry path (`depth_along_path`
    itself failing), or an adiabatic solve that is `ok=False`, or one that is
    `ok=True` but reports a non-finite `tl_db` (+inf: no mode is trapped
    where the path starts, or every mode was stripped before the receiver).
    `counts` names exactly how many of each and why
    (`n_path_refused`, `n_tl_nonfinite`, `n_predicted`, `reasons`), because a
    hypothesis that cannot predict some picks is itself a finding, and
    dropping them silently would change the sample the hypotheses are
    compared on.
    """
    paths, path_refused_reason = compute_pick_paths(frame, grid, n_steps=n_steps)
    receiver_z = frame["receiver_z_m"].to_numpy(float)
    return {
        name: modal_shape_for_seabed(paths, path_refused_reason, receiver_z,
                                     profile, seabed, source_z=source_z,
                                     freq_hz=freq_hz, nz=nz)
        for name, seabed in seabeds.items()
    }


CAVEAT_DB_SCALE_UNCALIBRATED = (
    "The delta log-likelihood compares two SHAPES (H-geometric's bare "
    "spreading law vs. H-modal's per-seabed adiabatic prediction) under an "
    "ASSUMED dB scale for the pick SNR -- and that scale is not itself "
    "established on this data. The axial-projection coefficient fitted in "
    "propagation_decay_fit.csv (~0.15-0.17) sits well below the physics "
    "benchmark of 2 predicted for a properly-scaled dB quantity (see that "
    "table's projection-column docstring for the derivation), which is "
    "itself evidence the assumed scale is imperfect. A reader must NOT take "
    "a decisive delta log-likelihood here as settling the underlying "
    "propagation physics when the observable's own calibration is in "
    "question; it settles only which SHAPE is less inconsistent with the "
    "data under the scale assumed.")
