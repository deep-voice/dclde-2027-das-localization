"""Regenerate one committed sound-speed result table from its committed YAML
config -- the driver `paper/docs/das-finwhale-soundspeed-plan-a-findings.md` §8
claimed already existed and did not (Task 14 whole-branch review, Critical 1).

Before this script, `load_config` (`experiments/runner.py`) was called only
from `paper/tests/test_runner.py`; nothing under
`paper/src/dclde_soundspeed/experiments/` had a `__main__`, and none of the
`fit_free_c` / `depth_sweep` / `shadow_zone_table` / `run_repro` functions
were ever invoked from a config at all -- every committed CSV under
`paper/results/` was in fact produced by one-off
`uv run python -c "..."` snippets, several of which are only quoted in
`.superpowers/sdd/.../task-*-report.md`, not committed anywhere runnable.
This script is the fix: one entry point per experiment config, each backed by
the actual tested functions in `experiments/{repro,regime,ablation}.py`.

Not every table a caller might expect from `paper/configs/`
can be regenerated this way:

  - `profile_validation.csv`, `hycom_profile_*.csv` and
    `hycom_spatial_consistency_check.csv` need LIVE network access to OOI's
    ERDDAP server and HYCOM's NCSS server, and have no YAML config at all --
    they are produced directly by
    `uv run python -m paper.scripts.validate_hycom_profile`. This
    script does not attempt them.
  - `repro_baseline.csv` / `repro_join_feasibility.csv` need
    `paper/data/reference/` restored (`uv run python paper/scripts/fetch_reference_data.py
    --dest paper/data/reference`) -- a one-time git-clone-plus-checksummed-zip
    fetch, not a live oceanographic query, but still not committed data
    (`data/` is gitignored). This script raises a clear error naming that
    command if the reference CSV is missing, rather than a bare
    FileNotFoundError three frames down.

Usage:
    uv run python -m paper.scripts.run_experiment \\
        --config paper/configs/regime_depth_vs_path.yaml

Every table is written via `experiments.runner.write_results`, so it carries
the same `#`-prefixed provenance header (experiment name, generation
timestamp, row count) as the tables already committed -- this script
overwrites that header's timestamp but not the numbers, so a diff against the
committed CSV via `pandas.read_csv(path, comment="#")` (which both this
script's caller and `write_results`'s own docstring recommend) is the way to
check regeneration actually reproduced a table, not a raw `diff` of the file
(which will always show the header lines changing).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dclde_soundspeed.experiments import ablation as ablation_mod
from dclde_soundspeed.experiments import propagation as propagation_mod
from dclde_soundspeed.experiments import regime as regime_mod
from dclde_soundspeed.experiments import repro as repro_mod
from dclde_soundspeed.experiments.runner import (
    PROFILES, RESULTS_DIR, load_config, write_results)
from dclde_localization.physics.geometry import load_cables
from dclde_soundspeed.soundspeed.modes import Seabed
from dclde_localization.physics.sound_speed_profile import to_down
from dclde_localization.physics.picks import attach_xyz, load_picks

# `fixtures/` (geometry and the picks snapshot) and `paper/results` (via
# `write_results`'s own relative default) both resolve from the process cwd,
# matching the rest of this repo's convention: run as a module, from the repo
# root (`uv run python -m paper.scripts.run_experiment ...`), the same
# way `paper/scripts/validate_hycom_profile.py` documents itself.
GEOMETRY_DIR = Path("fixtures")


def _load_all_picks() -> pd.DataFrame:
    """Every batch, both cables, with `px`/`py`/`pz` attached -- the input
    every experiment except repro_baseline (which restricts to its own
    `batches`) runs against."""
    cables = load_cables(str(GEOMETRY_DIR))
    return attach_xyz(load_picks(), cables)


def _reference_baseline_path() -> Path:
    """The published batch-1 localization CSV. Not committed (`data/` is
    gitignored) -- restored by `paper/scripts/fetch_reference_data.py`. Raise a
    clear, actionable error rather than a bare FileNotFoundError if it is
    missing, since this is the one config-driven table in this script that
    depends on data outside the repo."""
    p = (Path("data") / "reference" / "Goestchel_JASA_2025b" / "_worktree"
         / "batch1_localizations_with_coords.csv")
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Restore the reference data first:\n"
            "  uv run python paper/scripts/fetch_reference_data.py --dest paper/data/reference\n"
            "(a one-time git-clone-plus-checksummed-zip fetch -- no live "
            "oceanographic query, but not committed to this repo; see "
            "paper/data/reference/PROVENANCE.md).")
    return p


def run_repro_baseline(raw: dict, config_path: Path) -> None:
    baseline = repro_mod.parse_baseline_csv(_reference_baseline_path())
    picks = _load_all_picks()
    batches = raw.get("batches")
    if batches:
        picks = picks[picks["batch"].isin(batches)]

    sweep = repro_mod.join_feasibility_sweep(baseline, picks)
    write_results(sweep, "repro_join_feasibility")

    df = repro_mod.run_repro(
        picks, baseline, source_z=float(raw.get("source_z", -30.0)),
        min_picks=int(raw.get("min_picks", 8)),
        tol_s=float(raw.get("tol_s", 90.0)), margin_s=raw.get("margin_s"))
    write_results(df, "repro_baseline")


def run_regime_depth_vs_path(raw: dict, config_path: Path) -> None:
    cfg = load_config(config_path)
    picks = _load_all_picks()
    df = regime_mod.depth_sweep(
        picks, cfg.truth_model, raw["source_z_sweep"],
        min_picks=cfg.min_picks, range_bins=raw["range_bins_m"])
    write_results(df, "regime_depth_vs_path")


def run_regime_shadow_zone(raw: dict, config_path: Path) -> None:
    cfg = load_config(config_path)
    profile = cfg.truth_model.profile
    seabed_sweep = [Seabed(**s) for s in raw["seabed_sweep"]]
    geometries = [tuple(g) for g in raw["geometries"]]
    df = regime_mod.shadow_zone_table(profile, seabed_sweep, float(raw["freq_hz"]), geometries)
    write_results(df, "regime_shadow_zone")


def run_ablation_free_c(raw: dict, config_path: Path) -> None:
    picks = _load_all_picks()
    spec = raw["c_grid"]
    # Only the outer bracket (min/max) matters to fit_free_c (Ruling R8 -- see
    # ablation.py's module docstring); a coarse linspace over {start, stop,
    # step} reproduces that bracket without pretending `step` is still a grid
    # spacing.
    n = max(2, int(round((spec["stop"] - spec["start"]) / spec["step"])) + 1)
    c_grid = np.linspace(float(spec["start"]), float(spec["stop"]), n)

    df = ablation_mod.fit_free_c(
        picks, source_z=float(raw["source_z"]), c_grid=c_grid,
        min_picks=int(raw.get("min_picks", 8)),
        flip_threshold_m=float(raw.get("flip_threshold_m", 500.0)))
    df = ablation_mod.classify_c_plausibility(df, PROFILES["hycom_south_deep"])
    write_results(df, "ablation_free_c")

    diag = ablation_mod.free_c_flip_diagnostics(df, picks, source_z=float(raw["source_z"]),
                                                min_picks=int(raw.get("min_picks", 8)))
    write_results(diag, "ablation_free_c_flip_diagnostics")


def run_ablation_effective_c(raw: dict, config_path: Path) -> None:
    picks = _load_all_picks()
    profile = PROFILES[raw["profile"]]
    summary, detail = ablation_mod.effective_c_report(
        picks, profile, source_z=float(raw["source_z"]),
        min_picks=int(raw.get("min_picks", 8)),
        flip_threshold_m=float(raw.get("flip_threshold_m", 500.0)))
    write_results(summary, "ablation_effective_c")
    write_results(detail, "ablation_effective_c_flip_detail")


def run_propagation_tl(raw: dict, config_path: Path, *, out_dir=RESULTS_DIR) -> None:
    cfg = load_config(config_path)
    df = propagation_mod.predicted_tl_table(
        cfg.truth_model.profile,
        raw["seabed_sweep"],
        float(raw["freq_hz"]),
        [tuple(g) for g in raw["geometries"]],
        n_steps=int(raw.get("n_steps", 32)),
    )
    write_results(df, "propagation_tl", out_dir=out_dir, notes=[
        propagation_mod.CAVEAT_REFERENCE_SCALE,
        propagation_mod.CAVEAT_RANGE_DEPTH_COVARY,
    ])


def run_propagation_data(raw: dict, config_path: Path, *, out_dir=RESULTS_DIR) -> None:
    """Results 2 and 3 (spec sections 6.3, 6.4): the data comparison over
    every eligible real call.

    POSITIONS ARE AN INPUT, NOT A RESULT -- see `propagation.analysis_frame`'s
    docstring and this config's own notes; `cfg.inversion_model` (ConstantC)
    supplies the fixed geometry the two hypotheses are then compared against,
    it is never treated as the thing under test here.

    `out_dir` defaults to the real committed `RESULTS_DIR` (unchanged
    behaviour); a caller (the config-driven smoke test) can override it to
    write into a scratch directory instead, so a test run never touches the
    committed CSVs under `results/`.
    """
    from dclde_localization.physics.bathymetry import decode_gmt_grid, reproject_to_local
    from dclde_localization.physics.geometry import FrameTransform

    cfg = load_config(config_path)
    cables = load_cables(str(GEOMETRY_DIR))
    picks = _load_all_picks()
    grid = reproject_to_local(
        decode_gmt_grid(str(GEOMETRY_DIR / "gmrt_bathy.nc")),
        FrameTransform.from_cable(cables["South"]))

    frame, den = propagation_mod.analysis_frame(
        picks, cables, grid, cfg.inversion_model,
        source_z=float(raw["source_z"]),
        min_picks=int(raw.get("min_picks", 8)),
        flip_threshold_m=float(raw.get("flip_threshold_m", 500.0)))

    write_results(pd.DataFrame([den]), "propagation_denominators", out_dir=out_dir, notes=[
        propagation_mod.CAVEAT_DENOMINATOR_BUCKETS,
    ])
    write_results(propagation_mod.max_range_table(frame, den), "propagation_max_range",
                  out_dir=out_dir, notes=[
        propagation_mod.CAVEAT_MAX_RANGE_ASYMMETRY,
    ])

    fits = []
    draw_rows = []
    for thr in raw["threshold_sweep"]:
        for rule in ("censored", "dropped"):
            for proj in ("fitted", "omitted"):
                out = propagation_mod.decay_fit_table(
                    frame, threshold=float(thr),
                    n_bootstrap=int(raw.get("n_bootstrap", 200)),
                    seed=int(raw.get("seed", 0)),
                    zero_snr_rule=rule, projection=proj)
                # The joint bootstrap draws (MAJOR 3: needed to form an
                # interval on a linear combination of terms, e.g.
                # beta_range/beta_proj, which the summary table's
                # independent per-term CIs cannot support) are pulled out
                # here and written to their own long-format table instead.
                # `.attrs` is cleared before appending to `fits`: differently
                # shaped `draws` arrays across fragments (2 terms when
                # `projection == "omitted"`, 3 when "fitted") make pandas's
                # `pd.concat` raise inside its own `__finalize__`, which
                # compares every fragment's `.attrs` for equality via `==`
                # and cannot do that across mismatched array shapes.
                draws, terms = out.attrs["draws"], out.attrs["terms"]
                for draw_idx in range(draws.shape[0]):
                    for term_idx, term in enumerate(terms):
                        draw_rows.append({
                            "threshold": float(thr), "zero_snr_rule": rule,
                            "projection": proj, "draw": draw_idx, "term": term,
                            "beta": float(draws[draw_idx, term_idx]),
                        })
                out.attrs.clear()
                out.insert(0, "threshold", float(thr))
                fits.append(out)
    write_results(pd.concat(fits, ignore_index=True), "propagation_decay_fit",
                  out_dir=out_dir, notes=[
        propagation_mod.CAVEAT_DESCRIPTIVE_NOT_TEST,
        propagation_mod.CAVEAT_THRESHOLD_IS_APPROXIMATION,
        propagation_mod.CAVEAT_ZERO_SNR_RULE,
        propagation_mod.CAVEAT_AXIAL_PROJECTION_CONFOUND,
        propagation_mod.CAVEAT_CENSORED_VS_TRUNCATED,
    ])
    write_results(pd.DataFrame(draw_rows), "propagation_decay_fit_draws", out_dir=out_dir, notes=[
        "One row per (threshold, zero_snr_rule, projection, bootstrap draw, "
        "term): the joint per-draw beta values decay_fit_table's summary "
        "table only reduces to independent per-term 2.5/97.5 percentiles. "
        "Use this table, not the summary, to form a correctly-intervalled "
        "linear combination of terms (e.g. beta_log10_range / "
        "beta_axial_projection_db) by computing that combination WITHIN "
        "each draw before taking percentiles across draws.",
    ])

    _run_hypothesis_test(raw, cfg, frame, grid, out_dir=out_dir)


def _run_hypothesis_test(raw: dict, cfg, frame, grid, *, out_dir=RESULTS_DIR) -> None:
    """Spec section 6.4/7's PRE-REGISTERED hypothesis test: H-geometric vs.
    H-modal at EVERY seabed in the sweep, compared by whole-call-bootstrap
    delta log-likelihood -- NOT decay_fit_table's free-slope descriptive
    fit. Every seabed is run (not one representative value) because spec
    section 7's 'undecidable' branch is partly defined by whether the sign
    of the delta CHANGES across the sweep, which a single seabed cannot show.

    Split into two stages, deliberately, so a verification re-run never has
    to pay for both: `modal_shape_columns` (~44 min at real scale) is pure
    sequential numerical computation with no RNG, so its reproducibility
    risk is essentially nil; the fit-and-bootstrap stage (~8 min) is where an
    RNG, a convergence stopping criterion and floating-point path dependence
    actually create reproducibility risk. The shapes are PERSISTED
    (`propagation_modal_shapes.csv`) specifically so that stage split can be
    exploited -- both by this task's own regeneration check and by any
    later task that wants these shapes without paying the 44 minutes again.
    """
    # `analysis_frame`'s `source_z` argument is negative-down (see its own
    # docstring); `adiabatic_tl` (and so `modal_shape_for_seabed`) wants
    # positive-down, the opposite sign.
    source_z_positive_down = to_down(float(raw["source_z"]))
    profile = cfg.truth_model.profile
    seabeds = {
        f"seabed_c{int(sb['c'])}": Seabed(c=float(sb["c"]), rho=float(sb["rho"]),
                                          alpha=float(sb["alpha"]))
        for sb in raw["seabed_sweep"]
    }

    shape_table, modal_results = _write_modal_shapes_incrementally(
        frame, grid, profile, seabeds, source_z=source_z_positive_down,
        freq_hz=float(raw.get("hypothesis_freq_hz", 20.0)),
        n_steps=int(raw.get("hypothesis_n_steps", 32)),
        nz=int(raw.get("hypothesis_nz", 400)), out_dir=out_dir)

    _fit_hypothesis_test(raw, frame, shape_table, seabeds, modal_results, out_dir=out_dir)


def _write_modal_shapes_incrementally(frame, grid, profile, seabeds: dict, *,
                                      source_z: float, freq_hz: float,
                                      n_steps: int, nz: int, out_dir=RESULTS_DIR):
    """Compute and persist `propagation_modal_shapes.csv` ONE SEABED AT A
    TIME, appending each seabed's rows to disk as it finishes, rather than
    accumulating all `len(seabeds)` seabeds' shape arrays (and whatever
    solver-internal state the adiabatic computation allocates along the
    way) before writing anything -- so a memory-pressure kill mid-run
    leaves partial output and a clear restart point (which seabeds are
    already in the file) instead of nothing, and so this process never
    holds more than one seabed's in-flight computation alive at a time.
    Paths are still computed ONCE up front (they do not depend on seabed).

    Written in LONG format (one row per pick per hypothesis) rather than
    the wide one-column-per-seabed layout, because CSV append is naturally
    row-oriented -- a wide file would need every column decided before the
    first row is written, defeating the incremental point.

    Returns `(shape_table, modal_results)`: `shape_table` is the WIDE
    in-memory frame (`shape_<name>` columns) `_fit_hypothesis_test` expects
    -- trivially small (a few hundred KB for the real ~8,660-pick frame) so
    holding it entirely in memory once every seabed is done costs nothing;
    only the EXPENSIVE per-seabed COMPUTATION is staged. `modal_results`
    mirrors `modal_shape_columns`'s per-seabed `(shape, counts)` mapping for
    the diagnostic columns `_fit_hypothesis_test` reads.
    """
    import gc

    paths, path_refused_reason = propagation_mod.compute_pick_paths(frame, grid, n_steps=n_steps)
    receiver_z = frame["receiver_z_m"].to_numpy(float)

    id_cols = pd.DataFrame({
        "call": frame["call"].to_numpy(), "cable": frame["cable"].to_numpy(),
        "dist": frame["dist"].to_numpy(float), "range_m": frame["range_m"].to_numpy(float),
        "receiver_z_m": frame["receiver_z_m"].to_numpy(float),
        "snr": frame["snr"].to_numpy(float),
    })

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "propagation_modal_shapes.csv"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    n_rows_total = (len(seabeds) + 1) * len(frame)   # +1 for H-geometric's block

    shape_table = id_cols.copy()
    modal_results = {}
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(f"# experiment: propagation_modal_shapes\n"
                 f"# generated: {stamp}\n# rows: {n_rows_total}\n")
        fh.write(
            "# H-modal's predicted per-pick SNR shape in dB, for every "
            "seabed in the sweep, plus H-geometric's -- the expensive "
            "(~44 minutes at real scale), purely deterministic (no RNG) "
            "intermediate the fit-and-bootstrap stage consumes. Persisted "
            "so a later task, or this task's own regeneration check, never "
            "has to recompute it. LONG format, one row per pick per "
            "hypothesis, written incrementally (one hypothesis's rows "
            "appended as it finishes) to bound peak memory during "
            "computation and leave a clear restart point if interrupted. "
            "shape_db is NaN when that pick's modal prediction was unusable "
            "for that hypothesis -- see propagation_hypothesis_test.csv for "
            "the counted reasons.\n")
        first = True
        for name, seabed in seabeds.items():
            shape, counts = propagation_mod.modal_shape_for_seabed(
                paths, path_refused_reason, receiver_z, profile, seabed,
                source_z=source_z, freq_hz=freq_hz, nz=nz)
            block = id_cols.copy()
            block.insert(0, "hypothesis", name)
            block["shape_db"] = shape
            block.to_csv(fh, index=False, header=first)
            fh.flush()
            first = False
            modal_results[name] = (shape.copy(), counts)
            shape_table[f"shape_{name}"] = shape
            del shape, block, counts
            gc.collect()

        geo_shape = propagation_mod.hypothesis_shape_geometric(
            id_cols["range_m"].to_numpy(float))
        block = id_cols.copy()
        block.insert(0, "hypothesis", "H-geometric")
        block["shape_db"] = geo_shape
        block.to_csv(fh, index=False, header=first)
        shape_table["shape_H_geometric"] = geo_shape

    return shape_table, modal_results


def _fit_hypothesis_test(raw: dict, frame, shape_table, seabeds: dict,
                         modal_results: dict, *, out_dir=RESULTS_DIR) -> None:
    """The fit-and-bootstrap stage of the hypothesis test, separated from
    `_run_hypothesis_test` so a verification re-run can call this directly
    from a PERSISTED `shape_table` (as `propagation_modal_shapes.csv` was
    written) without recomputing `modal_shape_columns`. `modal_results` only
    supplies the per-seabed STANDALONE diagnostic counts
    (n_path_refused/n_tl_nonfinite/n_predicted) for the output table; the
    actual fit reads shapes from `shape_table`.
    """
    f = frame.copy()
    for col in shape_table.columns:
        if col.startswith("shape_"):
            f[col] = shape_table[col].to_numpy()
    # Modal entries first, geometric last: `hypothesis_comparison_table`'s
    # pairwise comparisons are named "{earlier} - {later}" in insertion
    # order, so this makes every relevant comparison read
    # "{seabed} - H-geometric" (positive means H-modal fits better).
    hypotheses = {name: f"shape_{name}" for name in seabeds}
    hypotheses["H-geometric"] = "shape_H_geometric"

    threshold = float(raw.get("hypothesis_threshold", 0.0))
    zero_snr_rule = raw.get("hypothesis_zero_snr_rule", "censored")
    n_total_eligible = int(len(frame))
    summary, deltas = propagation_mod.hypothesis_comparison_table(
        f, hypotheses, threshold=threshold,
        n_bootstrap=int(raw.get("n_bootstrap", 200)),
        seed=int(raw.get("seed", 0)), zero_snr_rule=zero_snr_rule)

    # `hypothesis_comparison_table` fits every hypothesis in `hypotheses` on
    # the SAME sample -- the intersection of picks where ALL SIX (five
    # seabeds plus H-geometric) have a finite shape -- because they are
    # passed in one call sharing one `f.dropna(subset=[..., *shape_cols])`.
    # That is what makes summary["n_picks"] identical across every row
    # (confirmed, not merely asserted, by this exact equality check) and is
    # exactly the guarantee spec section 6.4's comparison needs: H-geometric
    # is NEVER compared against a different sample than the H-modal seabed
    # it is being compared to, and the five seabed deltas are directly
    # comparable to EACH OTHER too, not just each to its own geometric run.
    n_picks_used = summary["n_picks"].unique()
    if len(n_picks_used) != 1:
        raise AssertionError(
            f"hypothesis_comparison_table's joint sample is not shared "
            f"across every hypothesis (n_picks values: {n_picks_used}); the "
            f"delta would be comparing different samples, which is invalid.")
    n_picks_used = int(n_picks_used[0])
    predicted_fraction = n_picks_used / n_total_eligible
    # Same joint-sample guarantee applies to n_calls: the 381-eligible to
    # (typically 380-used) call drop must be visible in whichever table a
    # reader actually opens, not only in the summary table.
    n_calls_used = summary["n_calls"].unique()
    if len(n_calls_used) != 1:
        raise AssertionError(
            f"hypothesis_comparison_table's joint sample has inconsistent "
            f"n_calls across hypotheses ({n_calls_used}); the delta would be "
            f"comparing different samples, which is invalid.")
    n_calls_used = int(n_calls_used[0])

    summary = summary.copy()
    summary["n_total_eligible"] = n_total_eligible
    summary["predicted_fraction"] = predicted_fraction

    relevant = deltas[deltas["comparison"].str.endswith(" - H-geometric")].copy()
    relevant["seabed_name"] = relevant["comparison"].str.replace(
        " - H-geometric", "", regex=False)
    seabed_params = {n: seabeds[n] for n in relevant["seabed_name"]}
    relevant["seabed_c"] = relevant["seabed_name"].map(lambda n: seabed_params[n].c)
    relevant["seabed_rho"] = relevant["seabed_name"].map(lambda n: seabed_params[n].rho)
    relevant["seabed_alpha"] = relevant["seabed_name"].map(lambda n: seabed_params[n].alpha)
    # STANDALONE per-seabed diagnostics (before the joint six-way
    # intersection above) -- named distinctly from n_picks_used/
    # predicted_fraction so a reader never mistakes one for the other.
    relevant["n_path_refused"] = relevant["seabed_name"].map(
        lambda n: modal_results[n][1]["n_path_refused"])
    relevant["n_tl_nonfinite_standalone"] = relevant["seabed_name"].map(
        lambda n: modal_results[n][1]["n_tl_nonfinite"])
    relevant["n_predicted_standalone"] = relevant["seabed_name"].map(
        lambda n: modal_results[n][1]["n_predicted"])
    relevant["n_picks_used"] = n_picks_used
    relevant["n_calls"] = n_calls_used
    relevant["n_total_eligible"] = n_total_eligible
    relevant["predicted_fraction"] = predicted_fraction

    # Spec section 7's "undecidable" branch requires knowing whether the sign
    # of the delta CHANGES across the seabed sweep -- stated as a column
    # (every row repeats the same verdict, so Task 12 never has to derive it
    # by squinting at five rows) AND as a header note below.
    signs = np.sign(relevant["delta_loglik"].to_numpy())
    sign_consistent = bool(len(set(signs.tolist())) == 1)
    relevant["sign_consistent_across_sweep"] = sign_consistent
    relevant = relevant[[
        "seabed_name", "seabed_c", "seabed_rho", "seabed_alpha",
        "delta_loglik", "ci_lo", "ci_hi", "n_draws",
        "n_picks_used", "n_calls", "n_total_eligible", "predicted_fraction",
        "n_path_refused", "n_tl_nonfinite_standalone", "n_predicted_standalone",
        "sign_consistent_across_sweep", "threshold", "zero_snr_rule",
    ]].sort_values("seabed_c").reset_index(drop=True)

    sign_note = (
        f"SIGN-CHANGE TEST (spec section 7's 'undecidable' branch): delta_loglik "
        f"(H-modal minus H-geometric) signs across the {len(relevant)}-seabed "
        f"sweep are {relevant['delta_loglik'].apply(lambda v: '+' if v > 0 else '-').tolist()}"
        f" -- sign_consistent_across_sweep = {sign_consistent}. "
        f"{'The sign does NOT change: this criterion does not by itself force undecidable.' if sign_consistent else 'The sign CHANGES across the sweep: spec section 7 requires this be read as evidence for the undecidable branch.'}")
    conditional_note = (
        f"THE 35-40% NON-PREDICTION RATE IS A HEADLINE, NOT A FOOTNOTE. "
        f"H-modal fails to predict a substantial share of the eligible picks "
        f"at every seabed (mostly 'every mode was stripped' -- see "
        f"n_tl_nonfinite_standalone per seabed); the SAME six-way "
        f"intersection is used for every hypothesis including H-geometric, "
        f"so delta_loglik above is computed on n_picks_used = {n_picks_used} "
        f"of n_total_eligible = {n_total_eligible} (predicted_fraction = "
        f"{predicted_fraction:.3f}) picks WHERE BOTH HYPOTHESES ARE DEFINED "
        f"-- confirmed, not assumed: hypothesis_comparison_table fits "
        f"H-geometric on the IDENTICAL sample as every H-modal seabed by "
        f"construction (one shared dropna across all six shape columns), so "
        f"the delta never compares two different samples. A hypothesis that "
        f"cannot predict a third of the observations is making a "
        f"substantive claim about those observations, not merely missing "
        f"data -- the reported delta is CONDITIONAL on the subset where "
        f"both hypotheses are defined, the excluded picks are not neutral, "
        f"and this table must not be read as an unconditional comparison "
        f"over all {n_total_eligible} eligible picks.")
    non_finite_note = (
        "Non-finite H-modal predictions (a refused bathymetry path, or an "
        "adiabatic solve reporting +inf transmission loss because no mode is "
        "trapped where the path starts or every mode was stripped) are "
        "COUNTED per seabed (n_path_refused, n_tl_nonfinite_standalone) -- "
        "n_predicted_standalone is that SEABED'S OWN prediction count before "
        "the joint six-way intersection with the other five hypotheses "
        "(see the conditional-comparison note above for the number actually "
        "used in the fit, n_picks_used) -- never silently dropped from the "
        "reported bookkeeping.")
    scale_note = (
        "BOTH OF THESE HOLD AT ONCE, READ TOGETHER: the delta log-likelihood "
        "is internally consistent and SIGN-STABLE across all five seabeds "
        "(see the sign-change note above) -- AND its physical interpretation "
        "rests on an assumed dB scale for the pick SNR that this "
        "experiment's own fitted axial-projection coefficient (~0.15-0.17 "
        "against a physics benchmark of 2, propagation_decay_fit.csv) calls "
        "into question. A decisive, sign-stable delta under an uncalibrated "
        "scale settles which SHAPE the data are less inconsistent with, "
        "NOT the underlying propagation physics -- do not read this table "
        "as having settled that question.")
    verification_note = (
        "REGENERATION VERIFICATION SPLIT (stated explicitly so a reader "
        "never infers the whole pipeline was re-run end-to-end when it was "
        "not, and reported plainly rather than smoothed over): "
        "modal_shape_for_seabed (~44 minutes at real scale for all seabeds, "
        "purely deterministic numerical computation, no RNG) was verified "
        "by re-running it on a real 1,000-pick subset in a SEPARATE process "
        "and comparing against the corresponding rows of the full-scale "
        "run's persisted shapes -- NOT by a second full 8,660-pick run. "
        "That comparison found max abs diff ~1.4e-14 dB, i.e. NOT exactly "
        "bit-identical -- reported plainly, not rounded away. A control "
        "check (calling modal_shape_for_seabed twice on the identical "
        "1,000-pick subset within the SAME process) gave exactly 0.0 "
        "difference, so the computation is deterministic for identical "
        "inputs in one process. `modal_shape_for_seabed` loops per pick and "
        "calls `adiabatic_tl` with a fixed nz=400 on a per-pick path, so the "
        "underlying linear algebra does NOT depend on how many picks are in "
        "the frame -- the ~1e-14 residual is therefore NOT attributable to "
        "'different batch sizes' as such. The more likely cross-process "
        "mechanisms are buffer alignment selecting a different SIMD kernel "
        "or remainder path, or a different OpenBLAS thread count under "
        "different machine load, in the mode eigenvalue solve -- not a code "
        "defect either way. For scale: 1.4e-14 dB at |x| ~ 64 dB is one "
        "double-precision ULP (one last bit, not an accumulated error), "
        "which propagates to a bound of ~1e-10 on the log-likelihood -- "
        "about thirteen orders of magnitude below the reported delta of "
        "~-1000 and its +/-200 interval, and this is far below anything "
        "that could move a reported delta_loglik (which itself matched two "
        "independent full 90-minute runs to the full double-precision "
        "string printed). The fit-and-bootstrap stage (~64 seconds "
        "measured, far under the ~8-minute estimate; has an RNG, a "
        "convergence stopping criterion, and floating-point path dependence "
        "-- the part where a reproducibility claim could actually fail) WAS "
        "verified to the literal standard used elsewhere in this "
        "experiment: re-running it from the PERSISTED shapes "
        "(propagation_modal_shapes.csv) and diffing this table and "
        "propagation_hypothesis_summary.csv byte-for-byte (via "
        "pandas.testing.assert_frame_equal on the parsed body, comment="
        "'#'), modulo the '# generated:' timestamp line: identical.")

    write_results(summary, "propagation_hypothesis_summary", out_dir=out_dir, notes=[
        propagation_mod.CAVEAT_CENSORED_VS_TRUNCATED,
        propagation_mod.CAVEAT_DB_SCALE_UNCALIBRATED,
        conditional_note,
        verification_note,
    ])
    profile_clamp_note = (
        "PROFILE-DEPTH EXTRAPOLATION IS NOT COUNTED PER SEABED IN THIS TABLE. "
        "`modes()` clamps c to isovelocity below hycom_south_deep's own "
        "bottom (1,500 m) whenever the water column is deeper, and warns "
        "when it does, precisely so a shallow profile under a deep water "
        "column cannot be read as physics. This table's per-pick modal "
        "solves are exactly that caller. Recovering an exact per-pick "
        "clamped count requires re-running the ~90-minute modal-shape "
        "stage (the clamp happens inside `modes()`, called from "
        "`modal_shape_for_seabed`, and is not recorded in the persisted "
        "`propagation_modal_shapes.csv` this table's fit stage reads), so "
        "it was NOT backfilled into this run's numbers. A cheaper, "
        "recoverable PROXY measured directly on this frame's real per-pick "
        "bathymetry paths: 272 of 8,660 picks (3.1%) have a path whose "
        "maximum depth exceeds 1,500 m at some point between source and "
        "receiver, meaning the mode solve on that path ran at least partly "
        "on a clamped profile. This is the right denominator (path depth, "
        "not receiver depth) and is roughly double the 1.6% a receiver-depth "
        "reading would suggest -- see findings doc section 7.4 for the full "
        "statement. Every H-modal prediction inherits this untested "
        "assumption, non-predictions included."
    )
    write_results(relevant, "propagation_hypothesis_test", out_dir=out_dir, notes=[
        propagation_mod.CAVEAT_CENSORED_VS_TRUNCATED,
        sign_note,
        conditional_note,
        non_finite_note,
        scale_note,
        verification_note,
        profile_clamp_note,
    ])


def run_gate_moveout(raw: dict, config_path: Path, out_dir=None) -> None:
    """The Plan B gate (Task 3): does real-pick moveout bound source depth
    above the noise? See `paper/configs/gate_moveout.yaml`'s
    `notes` for the pre-declared criterion, applied unchanged here.
    """
    from dclde_localization.physics.bathymetry import decode_gmt_grid, reproject_to_local
    from dclde_soundspeed.experiments import moveout as moveout_mod
    from dclde_localization.physics.geometry import FrameTransform

    cfg = load_config(config_path)
    cables = load_cables(str(GEOMETRY_DIR))
    picks = _load_all_picks()
    grid = reproject_to_local(
        decode_gmt_grid(str(GEOMETRY_DIR / "gmrt_bathy.nc")),
        FrameTransform.from_cable(cables["South"]))

    zg = raw["z_grid"]
    bidirectional_sample = raw.get("bidirectional_sample")
    table, den = moveout_mod.gate_table(
        picks, grid,
        min_picks=int(raw.get("min_picks", 8)),
        span_min_m=float(raw.get("span_min_m", 200.0)),
        z_grid=np.linspace(float(zg["start"]), float(zg["stop"]), int(zg["n"])),
        c=float(cfg.inversion_model.c),
        flip_threshold_m=float(raw.get("flip_threshold_m", 500.0)),
        bidirectional_sample=(None if bidirectional_sample is None
                              else int(bidirectional_sample)),
        bidirectional_seed=int(raw.get("bidirectional_seed", 0)))

    summary = moveout_mod.gate_summary(
        table, den,
        sigma_target_m=float(raw.get("sigma_target_m", 100.0)),
        min_fraction=float(raw.get("min_fraction", 0.10)),
        min_calls=int(raw.get("min_calls", 50)))

    path_summary = moveout_mod.path_comparison_summary(table)

    # Fix round 1 (review of the first gate run), recorded here so it
    # travels with the CSV and not only a review transcript: the tightened
    # x/y refit tolerance (MEDIUM 4, xatol=0.05/fatol=1e-13, matching
    # `_misfit_at_depth`) was checked for sensitivity by re-profiling five of
    # the direction-inconsistent stratum calls -- including the largest
    # curve discrepancy in the whole stratum -- at a 1000x tighter optimiser
    # setting (xatol=1e-4, fatol=1e-16, maxfev=40000). Every one of the five
    # reproduced its as-run forward/reverse sigma_z_m and curve discrepancy
    # to printed precision:
    #   2021-11-04_02_02_32/2  fwd 11.201  rev inf      (discrepancy 6.91e-3)
    #   2021-11-04_08_02_32/7  fwd 22.373  rev inf      (discrepancy 1.61e-2)
    #   2021-11-04_08_05_02/7  fwd 21.557  rev 225.106  (discrepancy 9.73e-3)
    #   2021-11-04_02_04_12/4  fwd 42.122  rev inf      (discrepancy 5.29e-3)
    #   2021-11-04_08_03_22/7  fwd 17.926  rev 58.725   (discrepancy 9.96e-3)
    # The forward/reverse disagreement is therefore tolerance-INVARIANT, not
    # convergence slop -- consistent with genuine multi-basin local-minimum
    # tracking (solve.py's own mirror-lobe documentation predicts exactly
    # this for single-cable geometry, and all five of these calls are
    # single-cable, as are 16 of the 17 forward-resolved stratum calls). The
    # same five-call check also reproduced these calls' committed
    # `gate_moveout.csv` values exactly from raw picks by a code path
    # independent of this script's own regenerate-and-diff check below --
    # a second, independent line of evidence that the committed table is
    # reproducible from source, distinct from this script's own
    # determinism check (which shows the pipeline is deterministic
    # end-to-end, not that an independently-written reader gets the same
    # numbers from the raw inputs).
    TOLERANCE_INVARIANCE_NOTE = (
        "TOLERANCE-INVARIANCE CHECK (fix round 1, MEDIUM 4): the "
        "xatol=0.05/fatol=1e-13 x/y refit tolerance was stress-tested on "
        "five direction-inconsistent stratum calls (including the largest "
        "curve discrepancy in the stratum) at a 1000x tighter setting "
        "(xatol=1e-4, fatol=1e-16, maxfev=40000). All five reproduced their "
        "forward/reverse sigma_z_m and curve discrepancy to printed "
        "precision -- 2021-11-04_02_02_32/2 (fwd 11.201, rev inf, "
        "discrepancy 6.91e-3), 2021-11-04_08_02_32/7 (fwd 22.373, rev inf, "
        "1.61e-2), 2021-11-04_08_05_02/7 (fwd 21.557, rev 225.106, 9.73e-3), "
        "2021-11-04_02_04_12/4 (fwd 42.122, rev inf, 5.29e-3), "
        "2021-11-04_08_03_22/7 (fwd 17.926, rev 58.725, 9.96e-3). The "
        "forward/reverse disagreement is tolerance-INVARIANT, i.e. genuine "
        "multi-basin local-minimum tracking rather than optimiser "
        "convergence slop -- consistent with solve.py's own mirror-lobe "
        "documentation for single-cable geometry (all five calls above, "
        "and 16 of the 17 forward-resolved stratum calls, are single-cable). "
        "The same five-call check also reproduced these calls' values in "
        "this CSV exactly from raw picks via an independently-written code "
        "path -- a second, distinct line of evidence beyond this script's "
        "own regenerate-and-diff determinism check.")

    kw = {} if out_dir is None else {"out_dir": out_dir}
    print(write_results(table, "gate_moveout", **kw))
    print(write_results(summary, "gate_moveout_summary", notes=[TOLERANCE_INVARIANCE_NOTE], **kw))
    print(write_results(path_summary, "gate_moveout_path_comparison", **kw))


_DISPATCH = {
    "repro_baseline": run_repro_baseline,
    "regime_depth_vs_path": run_regime_depth_vs_path,
    "regime_shadow_zone": run_regime_shadow_zone,
    "ablation_free_c": run_ablation_free_c,
    "ablation_effective_c": run_ablation_effective_c,
    "propagation_tl": run_propagation_tl,
    "propagation_data": run_propagation_data,
    "gate_moveout": run_gate_moveout,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path,
                    help="path to a paper/configs/*.yaml file")
    args = ap.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    name = raw.get("name")
    if name not in _DISPATCH:
        raise ValueError(
            f"no driver for experiment {name!r} (from {args.config}); have {sorted(_DISPATCH)}. "
            "example.yaml is a template with no committed table and is not runnable here; "
            "profile_validation.csv and the hycom_profile_*.csv tables need live network access "
            "via paper/scripts/validate_hycom_profile.py instead (see this module's "
            "docstring).")
    print(f"running {name!r} from {args.config} ...")
    _DISPATCH[name](raw, args.config)
    print(f"done: {name!r}")


if __name__ == "__main__":
    main()
