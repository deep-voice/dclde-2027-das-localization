# DCLDE 2027 — DAS Fin Whale Localization

Physics-based baseline for localizing fin whale calls from **Distributed Acoustic
Sensing (DAS)** arrival-time picks, for the
[DCLDE 2027](https://www.soest.hawaii.edu/ore/dclde/) fin whale localization
challenge.

Two seafloor fiber-optic cables ("north" and "south") off the Oregon coast act as
dense arrays of acoustic sensors. Each annotated fin whale call gives a set of
arrival-time picks at known distances along the fiber. This repo turns those picks
into an estimated whale position by TDOA-style grid search plus local refinement.

## Approach

For a candidate whale position on a grid, the predicted arrival time at each pick
point is `t0 + distance / c`. Rather than fitting the unknown emission time `t0`
as a free parameter, we take it as the **median residual**, which makes the fit L1
(robust to mis-picks); the misfit is then the **sum of absolute residuals**. The
grid search is fully vectorized over the grid, and the optimum is refined with
Nelder-Mead.

Two configurations are evaluated:

- **L-1** — one cable at a time. Weak geometry: a single near-linear array leaves
  the position poorly constrained along the cable axis.
- **L-2** — both cables combined, for calls picked up on each. Crossing baselines
  give a much better-conditioned solution.

## Layout

| Path | What |
|---|---|
| [src/dclde_localization/geometry.py](src/dclde_localization/geometry.py) | Cable geometry CSVs; interpolates distance-along-fiber → physical (x, y, depth) |
| [src/dclde_localization/picks.py](src/dclde_localization/picks.py) | Loads annotation CSVs, builds dataset-unique call IDs, matches calls across cables |
| [src/dclde_localization/localize.py](src/dclde_localization/localize.py) | The localizer: vectorized L1 grid search + Nelder-Mead refinement |
| [src/dclde_localization/evaluate.py](src/dclde_localization/evaluate.py) | Synthetic-truth trials and cross-cable consistency checks |
| [scripts/download_dclde_2027.py](scripts/download_dclde_2027.py) | Fetches the dataset from the public GCS bucket (resumable, no credentials) |
| [scripts/run_baseline.py](scripts/run_baseline.py) | Runs the baseline over the whole annotation set and prints summary stats |
| [tests/](tests/) | Unit tests, including a recovers-a-known-position end-to-end check |
| `dclde_localization/` | The organizers' original explainer notebook, kept for provenance |

## Setup

Python 3.12+ is required.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
```

Use `requirements.lock.txt` instead for a byte-identical rebuild of the verified
environment (full transitive pin).

## Getting the data

The dataset is **not** in this repo. `dclde_2027_listing.json` is a manifest of the
242 public objects in the `noaa-passive-bioacoustic` GCS bucket; the download
script reads it and pulls each file over the anonymous JSON/media API, so no
`gcloud`/`gsutil` credentials are needed.

```bash
.venv/bin/python scripts/download_dclde_2027.py   # ~resumable; skips complete files
```

This writes `dclde_2027_das-finwhale-localization/` containing `annotations/` and
`geometry/`, both gitignored.

## Running

```bash
.venv/bin/python scripts/run_baseline.py
.venv/bin/python -m pytest tests/ -q
```

## On evaluation

**There is no official metric, baseline, or ground truth for this challenge yet.**
The training annotations contain only arrival *picks*, never a true whale position,
so nothing in the released data supports a real accuracy number. Until the
organizers publish the held-out test set, two proxies stand in:

1. **Synthetic ground truth** — place a whale at a known position on the *real*
   cable geometry, simulate picks with configurable timing noise and sound-speed
   error, localize, and measure true error in meters. This is the only genuine
   accuracy number currently obtainable.
2. **Cross-cable consistency** — for calls seen on both cables, compare each
   single-cable (L-1) estimate against the combined (L-2) estimate. An empirical
   real-data sanity check, not an accuracy measure.

`scripts/run_baseline.py` reports `mean(|residual|) * c` as an "error" in meters.
Treat this as a **goodness-of-fit proxy, not a position error** — a confidently
wrong solution can have a small residual.
