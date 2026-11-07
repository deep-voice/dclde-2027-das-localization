"""Reproduce the published baseline and compare to it per call.

Goestchel, Wilcock & Abadi (2026) released 176 localizations for batch 1 of the
DAS fin-whale annotations. Their CSV carries a `utc` column (the call's
absolute time) but no call id, so the join goes through the window timestamp
encoded in each annotation call key plus the arrival time within that window.
Rows that do not match are kept with `call = None` and counted: a join that
silently drops rows would turn a reproduction failure into a smaller,
cleaner-looking sample. The same discipline applies to calls excluded by the
pick-count threshold -- they get a row with a reason, not deletion.

Review round 1 found the join was many-to-one and both effects it hid:

- **The join must be provably unambiguous, not just "nearest".** Fin whales
  call every 10-15 s and several are audible at once, so within a ~90 s
  window several *different* real calls can all be the nearest-in-time
  candidate for the same baseline row (worst observed: 5 baseline rows, at
  5 different real-world positions tens of km apart, all nearest to one
  annotated call). A join keyed on nearest-time alone silently attaches
  someone else's whale to this call's estimate. The default mode of
  `join_to_calls` therefore keeps a row only when the match is mutual and
  unique in both directions -- exactly one candidate call within `tol_s` of
  the baseline row, and exactly one candidate baseline row within `tol_s` of
  that call, narrowed by the baseline's own `call_type` label where
  available (never by our own solved position -- that would assume the
  answer being measured). Anything else is `excluded="ambiguous join"`,
  recorded and counted like every other exclusion in this module.
- **A single `delta_m` merges "wrong basin" with "wrong position".**
  `solve.py`'s module docstring states, as a hard rule, that basin flips and
  position error must be reported separately (`compare()` implements this
  for two of our own `Solution`s). The published baseline is a bare point,
  not a `Solution`, so `compare()` itself does not apply, but the same split
  does: `delta_m` (distance from `sol.best`, i.e. what we would actually
  report) and `delta_nearest_basin_m` (distance from whichever of our
  candidate basins is closest to the baseline, i.e. the error if the mirror
  ambiguity had been resolved correctly) are both recorded.

Review round 2 confirmed the default mode's 0/176 result independently
(`utc` is well aligned to real call times -- median signed gap +0.51 s, no
systematic offset -- the problem is purely that ~4 real calls occur every
90 s window, so timing alone cannot single one out) and asked for two more
things:

- A **margin-based** mode (`margin_s=...`): accept a baseline row's nearest
  candidate call only when it beats the second-nearest by at least
  `margin_s`, AND no other accepted row claims the same call. This trades
  "provably unique" for "confidently isolated in time", recovering a small
  but usable subset (21/176 at `margin_s=1.0`) at the cost of a selection
  bias stated explicitly wherever this subset is reported: rows that pass a
  timing-isolation filter are from sparser stretches of the recording by
  construction, not a random sample of the 176. Margin mode intentionally
  does **not** apply the `call_type` filter, both to match the review's own
  computation exactly and because "isolated in time" is a claim about
  timing alone.
- `join_feasibility_sweep` -- a `tol_s` sweep (default mode) and a
  `margin_s` sweep (margin mode), as one committed table, because "only a
  small fraction join unambiguously" is itself a finding about the released
  baseline artifact worth stating on its own, independent of whether this
  codebase can work around it.

Two more things a naive port of `Solution` into a flat table would lose:

- `Solution.separation_m` is `None` whenever it is not resolved between two
  confidently, independently located basins (see solve.py). `None` here means
  "not measured", which is different from "measured and it happened to be
  small" -- so `separation_resolved` is carried alongside it rather than
  collapsing both cases into one column.
- The best basin's `provenance` ("local_min" / "fallback" / "unresolved")
  says whether `our_x`/`our_y` is a genuinely located position or a
  regional/unrefined stand-in. Comparing a fallback position to the published
  baseline as if it were a real localization would misattribute optimizer
  weakness to physics error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dclde_localization.physics.solve import solve
from dclde_localization.physics.sound_speed_models import ConstantC

KEEP = ["utc", "sensor", "call_type", "x_local", "y_local", "z_local", "wrms", "deltax"]


def parse_baseline_csv(path) -> pd.DataFrame:
    """Load the published baseline CSV, keeping the local-frame coordinates
    and discarding the published lat/lon.

    The published lat/lon places calls roughly 270 km west of the cable while
    the local (x_local, y_local, z_local) coordinates are consistent with the
    cable geometry, so carrying both columns forward would invite silently
    using the wrong one downstream.
    """
    df = pd.read_csv(path)
    df["utc"] = pd.to_datetime(df["utc"], utc=True)
    return df[KEEP].copy()


def _window_start(window: str) -> pd.Timestamp:
    date, hh, mm, ss = window.rsplit("_", 3)
    return pd.Timestamp(f"{date} {hh}:{mm}:{ss}", tz="UTC")


def _build_calls_table(picks: pd.DataFrame) -> pd.DataFrame:
    """One row per annotated call: its window-start-plus-earliest-pick
    absolute time, and (when available) its `call_type` -- every call's picks
    share a single `call_type` value (HF or LF; verified on the full picks
    snapshot), so `"first"` is exact, not a lossy summary."""
    has_call_type = "call_type" in picks.columns
    agg = {"window": ("window", "first"), "t": ("time", "min")}
    if has_call_type:
        agg["call_type"] = ("call_type", "first")
    calls = picks.groupby("call").agg(**agg)
    calls["abs_t"] = [_window_start(w) + pd.Timedelta(seconds=float(t))
                      for w, t in zip(calls["window"], calls["t"])]
    return calls


def _gap_matrix(baseline_utc: pd.Series, calls_abs_t: pd.Series) -> np.ndarray:
    """(n_baseline, n_calls) matrix of absolute time gaps, in seconds."""
    b_t = baseline_utc.to_numpy(dtype="datetime64[ns]")
    c_t = calls_abs_t.to_numpy(dtype="datetime64[ns]")
    return np.abs(b_t[:, None] - c_t[None, :]) / np.timedelta64(1, "s")


def join_to_calls(baseline: pd.DataFrame, picks: pd.DataFrame, tol_s: float = 90.0,
                   margin_s: float | None = None) -> pd.DataFrame:
    """Attach a `call` key to each baseline row by absolute time -- but only
    when the match clears an explicit bar for confidence.

    Each call's absolute time is its window's start timestamp (decoded from
    the call key, e.g. "2021-11-04_02_00_02" -> 2021-11-04 02:00:02 UTC) plus
    its earliest pick's arrival time within that window.

    Two modes, selected by `margin_s`:

    **Default (`margin_s=None`) -- provably unique.** A baseline row and a
    call are candidates for each other whenever their absolute times are
    within `tol_s` seconds and (when both sides carry a `call_type` label)
    the baseline's `call_type` is compatible with the call's (an exact match,
    or a baseline row labelled "pair" -- drawing on both bands -- compatible
    with either). A row is assigned a `call` only when that pairing is mutual
    and unique: the baseline row has exactly one candidate call, and that
    call has exactly one candidate baseline row (itself). This is
    intentionally stricter than "nearest": several real calls 10-15 s apart
    can all be the *nearest* candidate for a given baseline row without any
    of them being the *only* one.

    **Margin (`margin_s=<seconds>`) -- confidently isolated in time.** A
    baseline row's nearest candidate call is accepted only if it beats the
    second-nearest candidate's gap by at least `margin_s`, AND no other
    accepted row claims the same call (ties broken in favour of neither --
    both lose the claim). This mode does not apply the `call_type` filter
    (isolation-in-time is a claim about timing alone, and this matches how
    the sensitivity numbers in task-7-report.md were computed). It recovers a
    usable subset at the cost of a selection bias that must be stated
    wherever the subset is used: rows that pass a timing-isolation filter are
    from sparser stretches of the recording by construction, not a random
    sample of the baseline.

    Rows that fail either test keep `call = None` and are never dropped, with
    the reason recorded in `join_status`:
    - `"unmatched"`: no candidate call within `tol_s` at all.
    - `"ambiguous"`: (default mode) more than one candidate on either side;
      (margin mode) the margin over the second-nearest candidate was too small.
    - `"contested"`: (margin mode only) the row passed its own margin test,
      but another row's nearest candidate is the same call.
    - `"matched"`: accepted; `call` and `join_gap_s` (gap to the accepted
      call) are set.

    `n_candidates` (default mode: call_type-aware candidate count within
    `tol_s`, before the mutuality check; margin mode: candidate count within
    `tol_s` regardless of band) is carried through as a diagnostic of how
    crowded each baseline row's neighbourhood is.
    """
    calls = _build_calls_table(picks)
    has_call_type = "call_type" in calls.columns and "call_type" in baseline.columns

    out = baseline.copy()
    n_b, n_c = len(out), len(calls)

    if n_c == 0:
        out["call"] = None
        out["join_gap_s"] = np.nan
        out["join_status"] = "unmatched"
        out["n_candidates"] = 0
        return out

    gap_s = _gap_matrix(out["utc"], calls["abs_t"])
    calls_index = calls.index.to_numpy()

    if margin_s is None:
        within = gap_s <= tol_s
        if has_call_type:
            b_type = out["call_type"].astype(str).str.upper().to_numpy()
            c_type = calls["call_type"].astype(str).str.upper().to_numpy()
            # A stronger, a priori key than time alone -- never our own
            # solved position, which would assume the answer being measured.
            # "pair" draws on both bands, so it is compatible with a call of
            # either band; otherwise the bands must match exactly.
            compat = (b_type[:, None] == "PAIR") | (b_type[:, None] == c_type[None, :])
            candidate = within & compat
        else:
            candidate = within

        row_deg = candidate.sum(axis=1)
        col_deg = candidate.sum(axis=0)

        keys: list = [None] * n_b
        gaps = np.full(n_b, np.nan)
        status = np.empty(n_b, dtype=object)
        for i in range(n_b):
            if row_deg[i] == 0:
                status[i] = "unmatched"
            elif row_deg[i] > 1:
                status[i] = "ambiguous"
            else:
                j = int(np.argmax(candidate[i]))
                if col_deg[j] == 1:
                    keys[i] = calls_index[j]
                    gaps[i] = gap_s[i, j]
                    status[i] = "matched"
                else:
                    status[i] = "ambiguous"

        out["call"] = keys
        out["join_gap_s"] = gaps
        out["join_status"] = status
        out["n_candidates"] = row_deg
        return out

    # Margin mode: time-only, no call_type filter (see docstring).
    masked = np.where(gap_s <= tol_s, gap_s, np.inf)
    n_candidates = np.isfinite(masked).sum(axis=1)

    keys = [None] * n_b
    gaps = np.full(n_b, np.nan)
    status = np.empty(n_b, dtype=object)
    nearest_call_idx = np.full(n_b, -1, dtype=int)

    for i in range(n_b):
        row = masked[i]
        n_finite = int(np.isfinite(row).sum())
        if n_finite == 0:
            status[i] = "unmatched"
            continue
        order = np.argsort(row)
        g1 = row[order[0]]
        g2 = row[order[1]] if n_finite > 1 else np.inf
        margin = g2 - g1
        if margin >= margin_s:
            nearest_call_idx[i] = order[0]
        else:
            status[i] = "ambiguous"

    # Contested check: a call claimed by more than one row that passed its
    # own margin test is not usable by any of them -- there is no principled
    # way to prefer one claimant over another without using position.
    claim_counts: dict[int, int] = {}
    for j in nearest_call_idx:
        if j >= 0:
            claim_counts[j] = claim_counts.get(j, 0) + 1

    for i in range(n_b):
        j = nearest_call_idx[i]
        if j < 0:
            continue  # already "unmatched" or "ambiguous" above
        if claim_counts[j] > 1:
            status[i] = "contested"
        else:
            keys[i] = calls_index[j]
            gaps[i] = gap_s[i, j]
            status[i] = "matched"

    out["call"] = keys
    out["join_gap_s"] = gaps
    out["join_status"] = status
    out["n_candidates"] = n_candidates
    return out


def join_feasibility_sweep(
        baseline: pd.DataFrame, picks: pd.DataFrame, *,
        tol_s_values=(90.0, 30.0, 10.0, 5.0, 2.0, 1.0),
        margin_s_values=(1.0, 2.0, 3.0, 5.0)) -> pd.DataFrame:
    """Tidy (sweep_type, param_value, metric, value) table documenting how
    unusable the timestamp-only join is at various thresholds -- a finding
    about the released baseline artifact (it carries no call id, and the
    call density is too high for timing alone to disambiguate one), not
    something this function tries to work around.

    `sweep_type="tol_s"` rows count `join_status` from the default (unique)
    mode at each `tol_s`. `sweep_type="margin_s"` rows count `join_status`
    from margin mode (at the default `tol_s=90.0`) at each `margin_s`.
    """
    n_total = len(baseline)
    rows = []
    for t in tol_s_values:
        vc = join_to_calls(baseline, picks, tol_s=t)["join_status"].value_counts()
        for metric in ("matched", "ambiguous", "unmatched"):
            rows.append({"sweep_type": "tol_s", "param_value": t, "metric": metric,
                         "value": int(vc.get(metric, 0)), "n_total": n_total})
    for m in margin_s_values:
        vc = join_to_calls(baseline, picks, margin_s=m)["join_status"].value_counts()
        for metric in ("matched", "contested", "ambiguous", "unmatched"):
            rows.append({"sweep_type": "margin_s", "param_value": m, "metric": metric,
                         "value": int(vc.get(metric, 0)), "n_total": n_total})
    return pd.DataFrame(rows)


def run_repro(picks: pd.DataFrame, baseline: pd.DataFrame, cables=None, *,
              source_z: float = -30.0, min_picks: int = 8, tol_s: float = 90.0,
              margin_s: float | None = None) -> pd.DataFrame:
    """Solve each accepted-join call with our estimator and tabulate the
    difference against the published baseline localization for that call.

    `margin_s=None` (default) uses `join_to_calls`'s provably-unique mode,
    which -- at the interface's own `tol_s=90.0` default -- accepts 0/176
    baseline rows (see task-7-report.md). Passing `margin_s` switches to the
    margin-based mode, recovering a small, explicitly biased subset (see
    `join_to_calls`'s docstring): rows selected this way are from sparser,
    less crowded stretches of the recording by construction, not a random
    sample of the 176 -- state that alongside any result computed from them.

    `cables` is accepted (and unused) for interface symmetry with other
    experiment entry points that need cable geometry to build `picks`; by the
    time `picks` reaches this function it already carries `px, py, pz` from
    `attach_xyz`, so `run_repro` itself needs no geometry.

    Every row of the baseline is represented in the output exactly once:
    - no candidate call within `tol_s`: `matched=False`, `excluded="unmatched"`.
    - the match is not accepted by the selected mode: `matched=False`,
      `excluded="ambiguous join"` or `excluded="contested join"` (margin mode
      only -- see `join_to_calls`).
    - matched but excluded by the pick-count threshold: `matched=True`,
      `excluded="too few picks"`, no solved position.
    - matched but the estimator returned no usable basin for this pick
      geometry: `matched=True`, `excluded="no basin found"` (a Task 4
      edge case, not a join or reporting defect -- see task-7-report.md).
    - matched and solved: `matched=True`, `excluded=""`, solved position,
      `delta_m` (distance from `sol.best`, i.e. what we would actually
      report), `delta_nearest_basin_m` (distance from whichever candidate
      basin is closest to the baseline -- separates "wrong position" from
      "right position, wrong side of the cable"), and the best basin's
      `provenance`.
    """
    joined = join_to_calls(baseline, picks, tol_s=tol_s, margin_s=margin_s)
    model = ConstantC()
    rows = []

    def _row(r, *, matched, n_picks, excluded, our_x=np.nan, our_y=np.nan, our_misfit=np.nan,
              delta_m=np.nan, delta_nearest_basin_m=np.nan, n_basins=np.nan,
              separation_m=np.nan, separation_resolved=False, provenance=None):
        return {
            "call": r["call"], "matched": matched, "n_picks": n_picks, "excluded": excluded,
            "their_x": r["x_local"], "their_y": r["y_local"], "their_wrms": r["wrms"],
            "our_x": our_x, "our_y": our_y, "our_misfit": our_misfit,
            "delta_m": delta_m, "delta_nearest_basin_m": delta_nearest_basin_m,
            "n_basins": n_basins,
            "separation_m": separation_m, "separation_resolved": separation_resolved,
            "provenance": provenance, "join_gap_s": r["join_gap_s"]}

    reason_for_status = {"ambiguous": "ambiguous join", "contested": "contested join",
                         "unmatched": "unmatched"}

    for _, r in joined.iterrows():
        # NOT `r["call"] is None`: DataFrame.iterrows() builds each row as a
        # single Series spanning every column, and once that row also holds a
        # real string `call` value (mixed dtypes across rows, as margin mode
        # produces), pandas silently upcasts a None entry to float `nan` --
        # `is None` then never matches, and an unmatched/ambiguous row falls
        # through into the "too few picks" branch below with an empty `sub`.
        # This only surfaces once a DataFrame contains BOTH matched and
        # unmatched rows together, which the default (0/176 matched) run
        # never exercised.
        if pd.isna(r["call"]):
            reason = reason_for_status.get(r["join_status"], "unmatched")
            rows.append(_row(r, matched=False, n_picks=np.nan, excluded=reason))
            continue
        sub = picks[picks["call"] == r["call"]]
        if len(sub) < min_picks:
            rows.append(_row(r, matched=True, n_picks=len(sub), excluded="too few picks"))
            continue

        # The estimator (Task 4) is not guaranteed to find a usable basin for
        # every pick geometry -- a single, near-collinear cable segment can
        # produce a misfit surface whose only candidate minima all land on
        # their own refinement-box boundary and are discarded (solve.py's own
        # "on_boundary" rule), leaving `Solution.basins` empty. That is a
        # genuine estimator failure for this call, not a bug in the join, so
        # it is recorded as an excluded row with a reason rather than either
        # crashing the whole reproduction or silently dropping the call.
        try:
            sol = solve(sub["px"].to_numpy(), sub["py"].to_numpy(), sub["pz"].to_numpy(),
                        sub["time"].to_numpy(), model, sz=source_z)
        except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
            rows.append(_row(r, matched=True, n_picks=len(sub),
                              excluded=f"solver error: {exc}"))
            continue
        if not sol.basins:
            rows.append(_row(r, matched=True, n_picks=len(sub), excluded="no basin found"))
            continue

        nearest_basin_delta = min(
            float(np.hypot(b.x - r["x_local"], b.y - r["y_local"])) for b in sol.basins)
        rows.append(_row(
            r, matched=True, n_picks=len(sub), excluded="",
            our_x=sol.best.x, our_y=sol.best.y, our_misfit=sol.best.misfit,
            delta_m=float(np.hypot(sol.best.x - r["x_local"], sol.best.y - r["y_local"])),
            delta_nearest_basin_m=nearest_basin_delta,
            n_basins=len(sol.basins),
            separation_m=sol.separation_m, separation_resolved=sol.separation_resolved,
            provenance=sol.best.provenance))
    return pd.DataFrame(rows)
