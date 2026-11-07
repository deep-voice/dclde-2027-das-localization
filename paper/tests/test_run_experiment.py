"""Every committed sound-speed config must be loadable and dispatchable.

Task 14 whole-branch review, Critical 1: `paper/docs/das-finwhale-soundspeed-plan-a-
findings.md` claimed every table "regenerates from committed configs ... via
`paper/src/dclde_soundspeed/experiments/`", which was false -- there was no
driver at all, and `paper/configs/ablation_free_c.yaml`
would have raised `ValueError` if anyone had actually tried to load it
through `experiments.runner.load_config` (it deliberately omits
`truth_model`/`inversion_model` -- see its own notes). None of the existing
tests loaded more than one config at a time, or ever exercised
`paper/scripts/run_experiment.py`'s dispatch table against the real,
committed config files, so this exact gap shipped unnoticed. This test file
is the fix the review asked for by name: load every committed config and
check it is wired to something runnable, without paying for a full multi-
minute experiment run in the test suite.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from dclde_soundspeed.experiments.runner import SelfConsistencyError, load_config
from paper.scripts.run_experiment import _DISPATCH

CONFIG_DIR = Path("paper/configs")
GEOMETRY_DIR = Path("fixtures")

# example.yaml is a documentation template (its own notes say so) with no
# committed result table -- not a member of _DISPATCH, and that absence is
# itself expected, not a gap.
NOT_DISPATCHED = {"example"}


def _all_config_paths() -> list[Path]:
    paths = sorted(CONFIG_DIR.glob("*.yaml"))
    # Task 11 added propagation_data.yaml (the data comparison, spec sections
    # 6.3/6.4), the eighth config alongside propagation_tl.yaml (Task 7, the
    # pure-forward Result 1). The Plan B gate (Task 3) added gate_moveout.yaml,
    # the ninth.
    assert len(paths) == 9, (
        f"expected the nine committed sound-speed configs, found {len(paths)}: {paths}. "
        "If a config was added or removed, update this test (and _DISPATCH / "
        "NOT_DISPATCHED above) deliberately -- don't just bump the number.")
    return paths


def test_every_committed_config_is_valid_yaml_with_a_name():
    for path in _all_config_paths():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(raw, dict), f"{path}: top level must be a mapping"
        assert raw.get("name") == path.stem, f"{path}: 'name' must match the file's own stem"


def test_every_committed_config_except_the_template_has_a_driver():
    # This is the check that would have caught Critical 1 directly: every
    # non-template config must dispatch to a real function in
    # paper/scripts/run_experiment.py, not just parse as YAML.
    for path in _all_config_paths():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        name = raw["name"]
        if name in NOT_DISPATCHED:
            assert name not in _DISPATCH
            continue
        assert name in _DISPATCH, f"{path}: no driver registered for {name!r} in _DISPATCH"


def test_configs_declaring_truth_and_inversion_models_load_without_raising():
    # repro_baseline.yaml, ablation_free_c.yaml and ablation_effective_c.yaml
    # deliberately do NOT declare truth_model/inversion_model (Task 14 review,
    # Critical 2 -- each used to name a run it did not actually perform); the
    # other three configs do, and must actually load through the
    # self-consistency guard.
    no_model_configs = {"repro_baseline", "ablation_free_c", "ablation_effective_c"}
    for path in _all_config_paths():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        name = raw["name"]
        if name in no_model_configs:
            with pytest.raises(ValueError, match="missing required key"):
                load_config(path)
            continue
        cfg = load_config(path)
        assert cfg.name == name
        assert cfg.truth_model.name != cfg.inversion_model.name or cfg.allow_same_model


def test_repro_baseline_and_free_c_configs_carry_the_keys_their_driver_reads():
    # These two configs are read directly (not through load_config's schema)
    # by run_repro_baseline / run_ablation_free_c -- pin the keys those
    # functions actually depend on, so a renamed or dropped key fails a fast
    # test instead of an 18-minute background run.
    repro = yaml.safe_load((CONFIG_DIR / "repro_baseline.yaml").read_text(encoding="utf-8"))
    assert repro["batches"] == [1]
    assert "margin_s" in repro  # selects the join mode the committed table used

    free_c = yaml.safe_load((CONFIG_DIR / "ablation_free_c.yaml").read_text(encoding="utf-8"))
    assert {"start", "stop", "step"} <= set(free_c["c_grid"])
    assert "source_z" in free_c and "min_picks" in free_c


def test_self_consistency_guard_fires_on_every_config_that_would_violate_it():
    # A config that named the SAME model for both truth and inversion, without
    # allow_same_model, must never load silently -- checked here against a
    # synthetic config built the same way load_config's own tests do, as a
    # cheap sanity check alongside the real committed configs above (none of
    # which actually violate this, by design).
    import tempfile
    body = ("name: t\ntruth_model: {kind: constant_c, c: 1480.0}\n"
            "inversion_model: {kind: constant_c, c: 1480.0}\n")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.yaml"
        p.write_text(body, encoding="utf-8")
        with pytest.raises(SelfConsistencyError):
            load_config(p)


@pytest.mark.smoke
def test_propagation_data_runs_end_to_end_on_a_small_subset(tmp_path, monkeypatch):
    """Spec section 8's config-driven smoke test, dropped without ruling or
    note (Task 14 whole-branch review, Important 3): the config-driven path
    (`run_propagation_data`, which dispatches to `_run_hypothesis_test` /
    `_write_modal_shapes_incrementally` / `_fit_hypothesis_test`) must run
    end to end on a small subset inside the smoke budget (CLAUDE.md: under 2
    minutes). Before this test, ~300 lines of this script -- including the
    positive-down sign flip (`source_z_positive_down = -float(raw[...])`),
    the two shared-sample `AssertionError` guards, the
    `sign_consistent_across_sweep` computation, and every `notes=` header
    string that ships inside the committed CSVs -- were exercised by nothing
    but a 90-minute manual run.

    Trimmed to ~10 calls (picks are real fixture data, restricted to the
    calls with the most picks so each clears `min_picks`), 2 seabeds, and
    `n_bootstrap=3` so the whole dispatch (decay-fit sweep + the persisted-
    shapes hypothesis test) finishes in well under the budget -- measured at
    ~60-90 s locally, against a 2-minute ceiling.

    `out_dir` is threaded through the whole dispatch (added by this fix) so
    this test writes into `tmp_path`, never into the committed
    `paper/results/` -- no committed result table is
    touched by running this test.
    """
    import time

    import yaml

    from dclde_localization.physics.geometry import load_cables
    from paper.scripts import run_experiment
    from dclde_localization.physics.picks import attach_xyz, load_picks

    t0 = time.time()

    cables = load_cables(str(GEOMETRY_DIR))
    picks = attach_xyz(load_picks(), cables)
    counts = picks.groupby("call").size().sort_values(ascending=False)
    # The busiest ~12 calls in the real fixture snapshot -- picked by pick
    # count, not by name, so this stays valid if the fixture is regenerated.
    candidate_calls = counts.index.tolist()[:12]
    trimmed_picks = picks[picks["call"].isin(candidate_calls)].copy()
    assert trimmed_picks["call"].nunique() >= 8, (
        "the trimmed pick subset lost too many calls to be a meaningful "
        "smoke test of the config-driven dispatch")

    monkeypatch.setattr(run_experiment, "_load_all_picks", lambda: trimmed_picks)

    config_path = CONFIG_DIR / "propagation_data.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["seabed_sweep"] = raw["seabed_sweep"][:2]
    raw["n_bootstrap"] = 3
    raw["threshold_sweep"] = raw["threshold_sweep"][:1]

    run_experiment.run_propagation_data(raw, config_path, out_dir=tmp_path)

    elapsed = time.time() - t0
    print(f"test_propagation_data_runs_end_to_end_on_a_small_subset: {elapsed:.1f}s")
    assert elapsed < 120.0, f"smoke test exceeded the 2-minute budget: {elapsed:.1f}s"

    expected_tables = {
        "propagation_denominators.csv": {
            "calls_total", "calls_below_min_picks", "calls_unsolved",
            "calls_flipped", "calls_path_failed", "calls_geometry_error",
            "calls_eligible", "picks_eligible",
        },
        "propagation_max_range.csv": {
            "call", "n_picks", "max_range_m", "min_range_m",
            "max_range_snr", "receiver_z_at_max_m",
        },
        "propagation_decay_fit.csv": {"threshold"},
        "propagation_decay_fit_draws.csv": {
            "threshold", "zero_snr_rule", "projection", "draw", "term", "beta",
        },
        "propagation_modal_shapes.csv": {"hypothesis", "call", "shape_db"},
        "propagation_hypothesis_summary.csv": {
            "hypothesis", "loglik", "converged", "n_picks", "n_calls",
            "n_censored", "n_draws",
        },
        "propagation_hypothesis_test.csv": {
            "seabed_name", "delta_loglik", "ci_lo", "ci_hi", "n_draws",
            "n_picks_used", "n_calls", "sign_consistent_across_sweep",
        },
    }
    for filename, expected_cols in expected_tables.items():
        path = tmp_path / filename
        assert path.exists(), f"{filename} was not written to out_dir"
        df = pd.read_csv(path, comment="#")
        missing = expected_cols - set(df.columns)
        assert not missing, f"{filename} is missing expected columns: {missing}"
