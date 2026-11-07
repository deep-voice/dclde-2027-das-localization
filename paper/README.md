# paper/ — sound speed and source depth on the DAS fin-whale data

Research toward a publication, not the challenge submission: can ocean sound-speed structure and
whale source depth be recovered from (or improve) DAS arrival-time localization on the DCLDE 2027
Oregon cables? Imported from `deep-voice/dclde_2027`, branch `sound-speed-research` at `0191c54`,
where the full commit history lives.

**Status:** the Plan B gate failed. Source depth is not recoverable from moveout on this dataset.
Read [the gate findings](docs/das-finwhale-plan-b-gate-findings.md) before starting new work.

## Read in this order

| Doc | What it holds |
|---|---|
| [docs/sound-speed-feasibility.md](docs/sound-speed-feasibility.md) | The original idea and the decision gate |
| [docs/das-finwhale-soundspeed-plan-a-findings.md](docs/das-finwhale-soundspeed-plan-a-findings.md) | Plan A: sound-speed profiles, propagation regime, effective/free-c ablations |
| [docs/das-finwhale-transmission-loss-findings.md](docs/das-finwhale-transmission-loss-findings.md) | Transmission loss and modal propagation (the brief is next to it) |
| [docs/das-finwhale-plan-b-gate-findings.md](docs/das-finwhale-plan-b-gate-findings.md) | Plan B depth-observability gate: FAIL |
| [docs/pitch/](docs/pitch/README.md) | Team pitch deck "Can a fibre hear depth?" and its figures |
| `docs/specs/`, `docs/plans/` | The design specs and implementation plans behind each findings doc |

## Layout

| Path | What |
|---|---|
| `src/dclde_soundspeed/soundspeed/` | Ray tracing, normal modes, transmission loss, HYCOM/ERDDAP reanalysis profiles |
| `src/dclde_soundspeed/experiments/` | One module per experiment (repro, ablation, regime, moveout, propagation) plus the runner |
| `scripts/` | `run_experiment.py` (regenerates a result table from its config), figure and recon scripts, `fetch_reference_data.py` |
| `configs/` | One YAML per result table |
| `results/` | Committed result tables (CSV), each named after its config |
| `data/reference/` | Downloaded reference data (gitignored except `PROVENANCE.md`) |
| `tests/` | Unit tests for everything above |

It builds on the shared localization core in [`src/dclde_localization/physics/`](../src/dclde_localization/physics/)
(forward model, solver, geometry, bathymetry, picks loader, sound-speed profile and models).

## Running

Everything runs **from the repo root**; file paths in the code are relative to it.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt -r paper/requirements.txt -e . -e paper
.venv/bin/python -m pytest paper/tests -q            # ~3 min
.venv/bin/python -m paper.scripts.run_experiment --config paper/configs/repro_baseline.yaml
```

Some experiments need reference data first: `python -m paper.scripts.fetch_reference_data --dest paper/data/reference`
(see `data/reference/PROVENANCE.md`).

## Conventions carried over

- Every rate states its denominator and n; every result regenerates from a committed config and a fixed seed.
- Result tables are compared against a named baseline on the same calls, never reported as a bare number.
