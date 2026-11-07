# Plan B Gate — Depth-Sensitivity and Moveout-Shape Test

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide, before any depth-observability work is built, whether the moveout curve on real picks carries source-depth information above the per-call noise — and answer the direct-versus-bottom-bounce path question from travel time rather than amplitude.

**Architecture:** One new module, `experiments/moveout.py`. It profiles source depth out of the misfit (minimising over `x`, `y`, with `t0` already analytic) to get a per-call marginal depth uncertainty, and it adds a flat-bottom image-source travel-time model so the direct and bounce hypotheses can be compared on curvature. A config, a driver entry, two result tables, and a findings note that applies the pre-declared gate criterion.

**Tech Stack:** Python 3, NumPy, SciPy, pandas, pytest, `uv`. No new dependencies.

**Spec:** [`paper/docs/specs/2026-09-21-plan-b-depth-observability-design.md`](../specs/2026-09-21-plan-b-depth-observability-design.md)

## Global Constraints

- **`src/dclde_localization/physics/forward.py` is FROZEN.** Never modify it. Every task ends with `git diff --exit-code src/dclde_localization/physics/forward.py` clean. The same applies to `paper/src/dclde_soundspeed/soundspeed/modes.py`.
- **The null is `ConstantC(1480)` straight-ray with source depth fixed at −30 m.** Everything is reported against it on the same calls.
- **Depth sign:** positive-down inside `soundspeed/`, negative-down everywhere else (`pz`, `source_z`). Convert only through `to_down` / `to_repo` from `soundspeed.profile`. This module works in the repo's negative-down convention throughout and never calls into `soundspeed/`.
- **No cross-task-package imports.** `dclde.das_finwhale` must not import `dclde.narw` or `dclde.killer_whales`. `dclde_localization.physics.picks` is allowed (the existing driver already uses it).
- **Every rate states its denominator and n** at the point of use.
- **Basin-flipping calls are counted and never merged** into a pooled statistic, in tables or prose.
- **Result tables regenerate from committed configs** via `uv run python -m paper.scripts.run_experiment --config <path>`, and that claim is verified by regenerating and diffing, never asserted.
- Run everything from the repo root. Python is `.venv/Scripts/python.exe` (Windows); `uv run` also works.
- MEASURED baseline: `pytest paper/tests viz/tests -q` = **390 passed**. Must not go down.

## The gate criterion — fixed here, before any code runs

Copied into the config's `notes` verbatim at Task 3, and applied unchanged at Task 4.

- **Stratum:** calls with `n_picks >= 8`, receiver-depth span **>= 200 m**, basin-stable (not flipped), and solved under the null. This is the spec's own middle stratum.
- **Statistic:** per-call **marginal** 1σ on source depth, `sigma_z_m`, from profiling `x` and `y` out of the misfit at each candidate depth (`t0` is already analytic via `best_t0`).
- **Criterion to proceed to §7.4:** at least **10%** of the stratum, **and at least 50 calls in absolute terms**, achieve `sigma_z_m <= 100 m`.
- **Why 100 m:** the null assumes −30 m and fin whales call shallow. A resolution coarser than ±100 m cannot improve on simply assuming the null's depth, so it would be resolution without information.
- **Why 10% and 50 calls:** enough to support the stratified reporting the spec requires, and enough that the outcome is not carried by a handful of favourable geometries. It asks "is there a usable subset", not "is it usable everywhere".

**Recorded so the threshold cannot look reverse-engineered in either direction.** A linearised scoping estimate made before this plan (440 calls, ≥100 m span) put ~33% of calls below 200 m and roughly 15–18% below 100 m. That estimate is **optimistic** — it treats the depth direction as independent of `x`, `y` and `t0`, and treats a structured residual as white — so the joint profile measured here will be worse. The criterion is genuinely in play and may fail. **Failing is a complete, publishable outcome**, not a reason to revisit the threshold.

---

## File Structure

| File | Responsibility |
|---|---|
| `paper/src/dclde_soundspeed/experiments/moveout.py` | **Create.** Depth profiling, marginal σ_z, flat-bottom bounce travel time, direct-vs-bounce comparison, the two table builders. |
| `paper/configs/gate_moveout.yaml` | **Create.** The gate's parameters and the criterion verbatim. |
| `paper/scripts/run_experiment.py` | **Modify.** One new dispatch entry. |
| `paper/tests/test_moveout.py` | **Create.** |
| `paper/results/gate_moveout.csv` | **Create.** Per-call: stratum flags, `sigma_z_m`, direct/bounce misfits. |
| `paper/results/gate_moveout_summary.csv` | **Create.** The criterion evaluated, with its denominators. |
| `paper/docs/das-finwhale-plan-b-gate-findings.md` | **Create.** The gate verdict. |

---

### Task 1: Profiled depth misfit and marginal σ_z

**Files:**
- Create: `paper/src/dclde_soundspeed/experiments/moveout.py`
- Test: `paper/tests/test_moveout.py`

**Interfaces:**
- Consumes: `dclde_localization.physics.forward.misfit_value`, `best_t0` (frozen, read-only); `dclde_localization.physics.solve.solve`; `dclde_localization.physics.sound_speed_models.ConstantC`.
- Produces:
  - `DEFAULT_C = 1480.0`, `NULL_SOURCE_Z = -30.0`
  - `depth_profile(px, py, pz, obs_t, *, z_grid, c=DEFAULT_C, x0=None, y0=None) -> tuple[np.ndarray, np.ndarray]` — returns `(z_grid, profiled_misfit)`, the L2 misfit minimised over `x`, `y` at each candidate depth.
  - `marginal_sigma_z(z_grid, profiled_misfit, n_picks, resid_sigma) -> float` — half-width where Δχ² = 1; `np.inf` when the profile is flat or unbounded within the grid.
  - `robust_sigma(residuals) -> float` — 1.4826 × MAD.

**Why L2 here specifically.** The repo's default objective is `l1`, but a Δχ² interpretation of a misfit width requires a Gaussian likelihood. This module profiles under **L2** and says so; the L1 solve is still what supplies the starting position. Mixing them silently would make `sigma_z_m` uninterpretable.

- [ ] **Step 1: Write the failing tests**

Create `paper/tests/test_moveout.py`:

```python
import numpy as np
import pytest

from dclde_soundspeed.experiments.moveout import (
    DEFAULT_C, NULL_SOURCE_Z, depth_profile, marginal_sigma_z, robust_sigma)


def synthetic_call(n=24, z_src=-120.0, depth_span=400.0, rng_seed=0, noise_s=0.0):
    """A cable arc of receivers spanning `depth_span` of depth, and arrivals
    generated from a known source. Returns (px, py, pz, obs_t, truth)."""
    rng = np.random.default_rng(rng_seed)
    t = np.linspace(-1.0, 1.0, n)
    px = 8000.0 * t
    py = 3000.0 + 500.0 * t**2
    pz = -300.0 - depth_span * (t + 1.0) / 2.0
    sx, sy = 500.0, -2000.0
    r = np.sqrt((px - sx) ** 2 + (py - sy) ** 2 + (pz - z_src) ** 2)
    obs_t = r / DEFAULT_C + 12.3 + rng.normal(0.0, noise_s, n)
    return px, py, pz, obs_t, (sx, sy, z_src)


def test_robust_sigma_is_mad_based_and_ignores_a_gross_outlier():
    r = np.concatenate([np.full(50, 0.0), np.array([1000.0])])
    assert robust_sigma(r) == pytest.approx(0.0, abs=1e-12)
    r2 = np.concatenate([np.array([-1.0, 1.0] * 25), np.array([1e6])])
    assert robust_sigma(r2) == pytest.approx(1.4826, rel=1e-6)


def test_depth_profile_minimum_sits_at_the_true_depth_when_noise_free():
    px, py, pz, obs_t, (_, _, z_true) = synthetic_call(z_src=-120.0, noise_s=0.0)
    z_grid = np.linspace(-400.0, 0.0, 81)
    zs, mis = depth_profile(px, py, pz, obs_t, z_grid=z_grid)
    assert zs.shape == mis.shape == z_grid.shape
    assert zs[int(np.argmin(mis))] == pytest.approx(z_true, abs=10.0)


def test_depth_profile_is_a_real_profile_not_a_fixed_xy_slice():
    """x and y must be re-minimised at every candidate depth. If they were held
    at the null solution the profile would be spuriously sharp, which is the
    exact way this measurement overstates depth resolution."""
    px, py, pz, obs_t, _ = synthetic_call(z_src=-120.0, noise_s=0.0)
    z_grid = np.linspace(-400.0, 0.0, 41)
    _, profiled = depth_profile(px, py, pz, obs_t, z_grid=z_grid)

    from dclde_localization.physics.forward import best_t0, misfit_value, travel_time
    sx, sy = 500.0, -2000.0
    fixed = []
    for z in z_grid:
        tt = travel_time(sx, sy, z, px, py, pz, DEFAULT_C)
        fixed.append(misfit_value(obs_t, tt, "l2"))
    fixed = np.array(fixed)
    # Profiling can only lower the misfit, never raise it.
    assert np.all(profiled <= fixed + 1e-9)
    # And it must actually move somewhere: a pure slice would be identical.
    assert np.any(profiled < fixed - 1e-6)


def test_marginal_sigma_z_widens_as_the_noise_grows():
    z_grid = np.linspace(-400.0, 0.0, 161)
    px, py, pz, obs_t, _ = synthetic_call(noise_s=0.0)
    _, mis = depth_profile(px, py, pz, obs_t, z_grid=z_grid)
    s_small = marginal_sigma_z(z_grid, mis, n_picks=len(px), resid_sigma=0.005)
    s_big = marginal_sigma_z(z_grid, mis, n_picks=len(px), resid_sigma=0.050)
    assert np.isfinite(s_small) and np.isfinite(s_big)
    assert s_big > s_small


def test_marginal_sigma_z_is_infinite_when_the_profile_is_flat():
    z_grid = np.linspace(-400.0, 0.0, 41)
    flat = np.full_like(z_grid, 0.25)
    assert not np.isfinite(
        marginal_sigma_z(z_grid, flat, n_picks=20, resid_sigma=0.01))


def test_marginal_sigma_z_is_infinite_when_the_minimum_is_pinned_to_an_edge():
    """A minimum at the grid edge means the data did not bound the depth; it
    must not be reported as a resolved one."""
    z_grid = np.linspace(-400.0, 0.0, 41)
    ramp = np.linspace(0.10, 0.40, z_grid.size)  # minimum at the first sample
    assert not np.isfinite(
        marginal_sigma_z(z_grid, ramp, n_picks=20, resid_sigma=0.01))
```

- [ ] **Step 2: Run them to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest paper/tests/test_moveout.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'dclde_soundspeed.experiments.moveout'`.

- [ ] **Step 3: Implement**

Create `paper/src/dclde_soundspeed/experiments/moveout.py`:

```python
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
module never calls into `soundspeed/`, so no conversion arises.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize

from dclde_localization.physics.forward import best_t0, misfit_value, travel_time

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


def _misfit_at_depth(z, px, py, pz, obs_t, c, x0, y0) -> float:
    """L2 misfit at source depth `z`, minimised over x and y.

    `t0` needs no minimisation -- `forward.best_t0` solves it in closed form.
    """
    def obj(p):
        tt = travel_time(p[0], p[1], z, px, py, pz, c)
        return misfit_value(obs_t, tt, "l2")

    res = optimize.minimize(obj, np.array([x0, y0], dtype=float),
                            method="Nelder-Mead",
                            options={"xatol": 1.0, "fatol": 1e-9, "maxiter": 2000})
    return float(res.fun)


def depth_profile(px, py, pz, obs_t, *, z_grid, c=DEFAULT_C, x0=None, y0=None):
    """Profiled misfit versus source depth.

    At every candidate depth, x and y are RE-MINIMISED. That is the whole point:
    depth trades off against horizontal range and emission time, so a profile
    taken at a fixed horizontal position is spuriously sharp and would overstate
    what a depth solve can achieve.
    """
    px, py, pz, obs_t = (np.asarray(v, dtype=float) for v in (px, py, pz, obs_t))
    z_grid = np.asarray(z_grid, dtype=float)
    if x0 is None:
        x0 = float(px.mean())
    if y0 is None:
        y0 = float(py.mean())

    out = np.empty(z_grid.size, dtype=float)
    gx, gy = float(x0), float(y0)
    for i, z in enumerate(z_grid):
        out[i] = _misfit_at_depth(z, px, py, pz, obs_t, c, gx, gy)
    return z_grid, out


def marginal_sigma_z(z_grid, profiled_misfit, n_picks, resid_sigma) -> float:
    """Half-width of the profiled depth minimum at Delta-chi-squared = 1.

    Returns `inf` -- meaning "this call does not bound depth" -- when the
    profile is flat, when the minimum sits on a grid edge (the data did not
    bound it, the grid did), or when the threshold is not crossed on both
    sides inside the grid.
    """
    z = np.asarray(z_grid, dtype=float)
    m = np.asarray(profiled_misfit, dtype=float)
    if z.size < 3 or not np.all(np.isfinite(m)) or resid_sigma <= 0:
        return float("inf")

    k = int(np.argmin(m))
    if k == 0 or k == z.size - 1:
        return float("inf")

    # L2 misfit is an RMS residual, so n * (m^2 - m_min^2) / sigma^2 is the
    # chi-squared increase.
    chi2 = n_picks * (m**2 - m[k] ** 2) / (resid_sigma**2)
    if not np.any(chi2[:k] >= _DELTA_CHI2_1SIGMA) or \
       not np.any(chi2[k:] >= _DELTA_CHI2_1SIGMA):
        return float("inf")

    left = np.interp(_DELTA_CHI2_1SIGMA, chi2[: k + 1][::-1], z[: k + 1][::-1])
    right = np.interp(_DELTA_CHI2_1SIGMA, chi2[k:], z[k:])
    return float(abs(right - left) / 2.0)
```

- [ ] **Step 4: Run the tests**

```bash
.venv/Scripts/python.exe -m pytest paper/tests/test_moveout.py -q
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git diff --exit-code src/dclde_localization/physics/forward.py
git add paper/src/dclde_soundspeed/experiments/moveout.py paper/tests/test_moveout.py
git commit -m "Profile source depth out of the misfit, with a marginal width

x and y are re-minimised at every candidate depth. A profile taken at a fixed
horizontal position is spuriously sharp, because depth trades off against range
and emission time -- that is exactly how a depth-resolution claim gets
overstated, so the test pins profiled <= fixed and requires it to differ.

An edge-pinned minimum returns inf rather than a number: the grid bounded it,
not the data."
```

---

### Task 2: Flat-bottom bounce travel time and the curvature comparison

**Files:**
- Modify: `paper/src/dclde_soundspeed/experiments/moveout.py`
- Test: `paper/tests/test_moveout.py`

**Interfaces:**
- Produces:
  - `bounce_travel_time(sx, sy, sz, px, py, pz, c, bottom_z) -> np.ndarray` — image-source travel time for one seafloor reflection.
  - `path_comparison(px, py, pz, obs_t, *, bottom_z, c=DEFAULT_C, source_z=NULL_SOURCE_Z) -> dict` with keys `misfit_direct`, `misfit_bounce`, `delta_misfit` (bounce − direct, so negative favours bounce), `x_direct`, `y_direct`, `x_bounce`, `y_bounce`.

The bounce model mirrors the source in a flat seafloor at `bottom_z` (negative-down). The image sits at `2 * bottom_z - sz`, so

```
t_bounce = sqrt((px - sx)^2 + (py - sy)^2 + (2*bottom_z - sz - pz)^2) / c
```

- [ ] **Step 1: Write the failing tests**

Append to `paper/tests/test_moveout.py`:

```python
from dclde_soundspeed.experiments.moveout import bounce_travel_time, path_comparison
from dclde_localization.physics.forward import travel_time


def test_bounce_is_always_slower_than_the_direct_path():
    px = np.array([1000.0, 5000.0, 20000.0])
    py = np.zeros(3)
    pz = np.array([-400.0, -400.0, -400.0])
    d = travel_time(0.0, 0.0, -30.0, px, py, pz, DEFAULT_C)
    b = bounce_travel_time(0.0, 0.0, -30.0, px, py, pz, DEFAULT_C, bottom_z=-1200.0)
    assert np.all(b > d)


def test_bounce_reduces_to_the_direct_path_as_the_bottom_approaches_the_source():
    """With the seafloor at the source depth the image coincides with the
    source, so the two travel times must agree exactly."""
    px = np.array([3000.0, 9000.0])
    py = np.array([100.0, -250.0])
    pz = np.array([-500.0, -700.0])
    d = travel_time(0.0, 0.0, -30.0, px, py, pz, DEFAULT_C)
    b = bounce_travel_time(0.0, 0.0, -30.0, px, py, pz, DEFAULT_C, bottom_z=-30.0)
    assert np.allclose(b, d)


def test_bounce_and_direct_differ_in_CURVATURE_not_just_offset():
    """The gate rests on the two paths tracing different SHAPES. If they
    differed only by a constant, `best_t0` would absorb the difference and no
    timing test could tell them apart."""
    px = np.linspace(2000.0, 30000.0, 12)
    py = np.zeros(12)
    pz = np.full(12, -600.0)
    d = travel_time(0.0, 0.0, -30.0, px, py, pz, DEFAULT_C)
    b = bounce_travel_time(0.0, 0.0, -30.0, px, py, pz, DEFAULT_C, bottom_z=-1500.0)
    diff = b - d
    assert diff.std() > 1e-3, "difference is nearly constant; t0 would absorb it"


def test_path_comparison_prefers_the_model_that_generated_the_data():
    px = np.linspace(2000.0, 25000.0, 20)
    py = 500.0 * np.ones(20)
    pz = np.linspace(-300.0, -900.0, 20)
    bottom = -1400.0
    t_b = bounce_travel_time(1000.0, -3000.0, -30.0, px, py, pz, DEFAULT_C, bottom) + 7.0
    out = path_comparison(px, py, pz, t_b, bottom_z=bottom)
    assert out["misfit_bounce"] < out["misfit_direct"]
    assert out["delta_misfit"] < 0

    t_d = travel_time(1000.0, -3000.0, -30.0, px, py, pz, DEFAULT_C) + 7.0
    out2 = path_comparison(px, py, pz, t_d, bottom_z=bottom)
    assert out2["misfit_direct"] < out2["misfit_bounce"]
    assert out2["delta_misfit"] > 0
```

- [ ] **Step 2: Run them to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest paper/tests/test_moveout.py -q -k "bounce or path_comparison"
```
Expected: FAIL — `cannot import name 'bounce_travel_time'`.

- [ ] **Step 3: Implement**

Append to `paper/src/dclde_soundspeed/experiments/moveout.py`:

```python
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

    res = optimize.minimize(obj, np.array([x0, y0], dtype=float),
                            method="Nelder-Mead",
                            options={"xatol": 1.0, "fatol": 1e-9, "maxiter": 2000})
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
```

- [ ] **Step 4: Run the tests**

```bash
.venv/Scripts/python.exe -m pytest paper/tests/test_moveout.py -q
```
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git diff --exit-code src/dclde_localization/physics/forward.py
git add paper/src/dclde_soundspeed/experiments/moveout.py paper/tests/test_moveout.py
git commit -m "Add a flat-bottom bounce model and the direct-vs-bounce comparison

The gate rests on the two paths tracing different SHAPES, so a test pins that
their difference is not merely a constant -- if it were, best_t0 would absorb
it and no timing test could discriminate them at all.

Flat-bottom is conservative for a discrimination test: a sloping seafloor makes
the curvatures differ more, not less."
```

---

### Task 3: The gate experiment, config and driver

**Files:**
- Modify: `paper/src/dclde_soundspeed/experiments/moveout.py`, `paper/scripts/run_experiment.py`
- Create: `paper/configs/gate_moveout.yaml`
- Test: `paper/tests/test_moveout.py`

**Interfaces:**
- Consumes: `analysis_frame`-style inputs — `load_picks`/`attach_xyz`/`load_cables`, plus `depth_along_path` from `dclde_localization.physics.bathymetry` for the seafloor depth the bounce model needs.
- Produces:
  - `gate_table(picks, cables, grid, *, min_picks=8, span_min_m=200.0, z_grid=None, c=DEFAULT_C, flip_threshold_m=500.0) -> tuple[pd.DataFrame, dict]`
  - `gate_summary(table, denominators, *, sigma_target_m=100.0, min_fraction=0.10, min_calls=50) -> pd.DataFrame`

Per-call table columns: `call, n_picks, depth_span_m, in_stratum, solved, flipped, sigma_z_m, z_at_min_m, resid_sigma_ms, bottom_z_m, misfit_direct, misfit_bounce, delta_misfit`.

- [ ] **Step 1: Write the failing tests**

Append to `paper/tests/test_moveout.py`:

```python
import pandas as pd

from dclde_soundspeed.experiments.moveout import gate_summary


def test_gate_summary_applies_the_criterion_and_shows_its_denominators():
    tbl = pd.DataFrame({
        "call": [f"c{i}" for i in range(100)],
        "in_stratum": [True] * 80 + [False] * 20,
        "sigma_z_m": [50.0] * 30 + [400.0] * 50 + [np.inf] * 20,
    })
    out = gate_summary(tbl, {"calls_total": 120}, sigma_target_m=100.0,
                       min_fraction=0.10, min_calls=50)
    row = out.iloc[0]
    assert row["n_stratum"] == 80
    assert row["n_resolved"] == 30
    assert row["fraction_resolved"] == pytest.approx(30 / 80)
    # 30 >= 50 is FALSE, so the absolute-count leg fails even though 37.5% > 10%
    assert bool(row["passes_fraction"]) is True
    assert bool(row["passes_min_calls"]) is False
    assert bool(row["gate_passes"]) is False


def test_gate_summary_passes_only_when_both_legs_pass():
    tbl = pd.DataFrame({
        "call": [f"c{i}" for i in range(600)],
        "in_stratum": [True] * 600,
        "sigma_z_m": [50.0] * 60 + [400.0] * 540,
    })
    out = gate_summary(tbl, {"calls_total": 628}, sigma_target_m=100.0,
                       min_fraction=0.10, min_calls=50)
    row = out.iloc[0]
    assert row["n_resolved"] == 60
    assert bool(row["passes_fraction"]) is True   # 10.0% >= 10%
    assert bool(row["passes_min_calls"]) is True  # 60 >= 50
    assert bool(row["gate_passes"]) is True


def test_gate_summary_counts_infinite_sigma_as_unresolved_not_missing():
    tbl = pd.DataFrame({
        "call": ["a", "b", "c"],
        "in_stratum": [True, True, True],
        "sigma_z_m": [10.0, np.inf, np.nan],
    })
    out = gate_summary(tbl, {"calls_total": 3}, sigma_target_m=100.0,
                       min_fraction=0.10, min_calls=1)
    assert out.iloc[0]["n_stratum"] == 3
    assert out.iloc[0]["n_resolved"] == 1
```

- [ ] **Step 2: Run them to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest paper/tests/test_moveout.py -q -k gate_summary
```
Expected: FAIL — `cannot import name 'gate_summary'`.

- [ ] **Step 3: Implement the two table builders**

Append to `paper/src/dclde_soundspeed/experiments/moveout.py`:

```python
import pandas as pd

GATE_COLUMNS = ["call", "n_picks", "depth_span_m", "in_stratum", "solved", "flipped",
                "sigma_z_m", "z_at_min_m", "resid_sigma_ms", "bottom_z_m",
                "misfit_direct", "misfit_bounce", "delta_misfit"]


def gate_table(picks, cables, grid, *, min_picks=8, span_min_m=200.0, z_grid=None,
               c=DEFAULT_C, flip_threshold_m=500.0):
    """Per-call depth sensitivity and path comparison, plus the denominators.

    The stratum is pre-declared (see the config's notes): n_picks >= min_picks,
    receiver-depth span >= span_min_m, basin-stable, solved under the null.
    Calls outside it are still rowed, with `in_stratum` False, so a reader can
    see what was excluded rather than inferring it from a missing row.
    """
    from dclde_localization.physics.bathymetry import depth_along_path
    from dclde_localization.physics.solve import solve
    from dclde_localization.physics.sound_speed_models import ConstantC

    if z_grid is None:
        z_grid = np.linspace(-600.0, 0.0, 121)
    model = ConstantC(c)

    rows, flipped_n, unsolved_n = [], 0, 0
    for call, sub in picks.groupby("call"):
        px, py, pz = (sub[k].to_numpy(float) for k in ("px", "py", "pz"))
        obs_t = sub["time"].to_numpy(float)
        span = float(abs(pz.max() - pz.min()))
        base = {"call": call, "n_picks": int(len(sub)), "depth_span_m": span,
                "in_stratum": False, "solved": False, "flipped": False,
                "sigma_z_m": np.inf, "z_at_min_m": np.nan,
                "resid_sigma_ms": np.nan, "bottom_z_m": np.nan,
                "misfit_direct": np.nan, "misfit_bounce": np.nan,
                "delta_misfit": np.nan}

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
            base["sigma_z_m"] = marginal_sigma_z(zs, mis, len(sub), sigma_r)
            base["z_at_min_m"] = float(zs[int(np.argmin(mis))])

            prof = depth_along_path(grid, b.x, b.y, float(px.mean()), float(py.mean()),
                                    n_steps=16)
            if prof.ok:
                bottom = -float(np.max(prof.depth_m))
                base["bottom_z_m"] = bottom
                base.update({k: v for k, v in
                             path_comparison(px, py, pz, obs_t, bottom_z=bottom,
                                             c=c).items()
                             if k in ("misfit_direct", "misfit_bounce", "delta_misfit")})
        rows.append(base)

    table = pd.DataFrame(rows, columns=GATE_COLUMNS)
    den = {"calls_total": int(picks["call"].nunique()),
           "calls_below_min_picks": int((table["n_picks"] < min_picks).sum()),
           "calls_unsolved": int(unsolved_n),
           "calls_flipped": int(flipped_n),
           "calls_in_stratum": int(table["in_stratum"].sum()),
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
    """
    strat = table[table["in_stratum"].astype(bool)]
    n_stratum = int(len(strat))
    finite = np.isfinite(strat["sigma_z_m"].to_numpy(float))
    n_resolved = int((finite & (strat["sigma_z_m"].to_numpy(float) <= sigma_target_m)).sum())
    frac = float(n_resolved / n_stratum) if n_stratum else 0.0
    passes_fraction = bool(frac >= min_fraction)
    passes_min_calls = bool(n_resolved >= min_calls)

    row = {"sigma_target_m": sigma_target_m, "min_fraction": min_fraction,
           "min_calls": min_calls, "n_stratum": n_stratum, "n_resolved": n_resolved,
           "fraction_resolved": frac, "passes_fraction": passes_fraction,
           "passes_min_calls": passes_min_calls,
           "gate_passes": bool(passes_fraction and passes_min_calls)}
    row.update({k: v for k, v in denominators.items()})
    return pd.DataFrame([row])
```

- [ ] **Step 4: Run the tests**

```bash
.venv/Scripts/python.exe -m pytest paper/tests/test_moveout.py -q
```
Expected: 13 passed.

- [ ] **Step 5: Write the config**

Create `paper/configs/gate_moveout.yaml`:

```yaml
name: gate_moveout
notes: >
  The Plan B gate. Design note:
  paper/docs/specs/2026-09-21-plan-b-depth-observability-design.md

  QUESTION. Does the moveout curve on real picks carry source-depth information
  above the per-call noise? Plan A's transmission-loss study closed the
  amplitude route -- the receiver-depth coefficient of received level has an
  interval spanning zero in all 16 configurations -- so depth, if reachable at
  all, is reachable through travel-time moveout. This measures whether it is,
  before any depth machinery is built.

  THE CRITERION, FIXED BEFORE THIS RAN AND APPLIED UNCHANGED.
  Stratum: n_picks >= 8, receiver-depth span >= 200 m, basin-stable, solved
  under the null. Statistic: per-call MARGINAL 1-sigma on source depth, from
  profiling x and y out of the misfit at each candidate depth (t0 is analytic).
  Proceed to depth observability only if at least 10% of the stratum AND at
  least 50 calls achieve sigma_z <= 100 m.
  Why 100 m: the null assumes -30 m and fin whales call shallow, so a
  resolution coarser than +/-100 m cannot improve on simply assuming the null's
  depth. Why 10% and 50 calls: enough to support stratified reporting, and
  enough that the outcome is not carried by a few favourable geometries. It
  asks whether a usable SUBSET exists, not whether depth works everywhere.

  RECORDED SO THE THRESHOLD CANNOT LOOK REVERSE-ENGINEERED. A linearised
  scoping estimate made before the plan (440 calls, >=100 m span) put ~33% of
  calls below 200 m and roughly 15-18% below 100 m. That estimate is OPTIMISTIC
  -- it treats the depth direction as independent of x, y and t0, and treats a
  structured residual as white -- so the joint profile measured here will be
  worse. The criterion is genuinely in play and may fail. Failing is a
  complete, publishable outcome, not a reason to revisit the threshold.

  OBJECTIVE. The profile is computed under L2, not the repo default L1, because
  a Delta-chi-squared reading of a misfit width requires a Gaussian likelihood.
  The L1 solve still supplies the starting position. Stated rather than assumed.

  inversion_model is the null and is what the profile is computed against.
  truth_model is unused by gate_table (which takes arrays, not an
  ExperimentConfig) and is set to path_average only to satisfy the loader's
  self-consistency guard, matching the convention in regime_shadow_zone.yaml.
truth_model: {kind: path_average, profile: hycom_south_deep}
inversion_model: {kind: constant_c, c: 1480.0}
min_picks: 8
span_min_m: 200.0
z_grid: {start: -600.0, stop: 0.0, n: 121}
sigma_target_m: 100.0
min_fraction: 0.10
min_calls: 50
flip_threshold_m: 500.0
```

- [ ] **Step 6: Wire the driver**

In `paper/scripts/run_experiment.py`, add after the other dispatch functions:

```python
def run_gate_moveout(raw: dict, config_path: Path, out_dir=None) -> None:
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
    table, den = moveout_mod.gate_table(
        picks, cables, grid,
        min_picks=int(raw.get("min_picks", 8)),
        span_min_m=float(raw.get("span_min_m", 200.0)),
        z_grid=np.linspace(float(zg["start"]), float(zg["stop"]), int(zg["n"])),
        c=float(cfg.inversion_model.c),
        flip_threshold_m=float(raw.get("flip_threshold_m", 500.0)))

    summary = moveout_mod.gate_summary(
        table, den,
        sigma_target_m=float(raw.get("sigma_target_m", 100.0)),
        min_fraction=float(raw.get("min_fraction", 0.10)),
        min_calls=int(raw.get("min_calls", 50)))

    kw = {} if out_dir is None else {"out_dir": out_dir}
    print(write_results(table, "gate_moveout", **kw))
    print(write_results(summary, "gate_moveout_summary", **kw))
```

Register it: `"gate_moveout": run_gate_moveout,` in `_DISPATCH`. Add `import numpy as np` at the top if absent.

- [ ] **Step 7: Verify the config loads and the driver dispatches, then run it**

```bash
.venv/Scripts/python.exe -m pytest paper/tests -q
uv run python -m paper.scripts.run_experiment --config paper/configs/gate_moveout.yaml
```
Time the run and report it. `depth_profile` does a 2-D minimisation at each of 121 depths per call, so budget accordingly; if the full run would exceed ~30 minutes, say so with the measurement before launching rather than after.

- [ ] **Step 8: Verify regenerability, then commit**

```bash
cp paper/results/gate_moveout.csv /tmp/gate_a.csv
uv run python -m paper.scripts.run_experiment --config paper/configs/gate_moveout.yaml
.venv/Scripts/python.exe -c "
import pandas as pd
a = pd.read_csv('/tmp/gate_a.csv', comment='#')
b = pd.read_csv('paper/results/gate_moveout.csv', comment='#')
pd.testing.assert_frame_equal(a, b); print('regenerates identically')
"
git diff --exit-code src/dclde_localization/physics/forward.py
git add -A && git commit -m "Run the Plan B gate over the real picks

The criterion was fixed in the config before this ran and is applied unchanged.
Calls outside the stratum are rowed with in_stratum False rather than dropped,
so a reader sees what was excluded instead of inferring it from a missing row.
A non-finite sigma_z is unresolved, not missing: it means the call did not bound
depth, which is an answer."
```

---

### Task 4: The gate verdict

**Files:**
- Create: `paper/docs/das-finwhale-plan-b-gate-findings.md`
- Modify: `paper/docs/specs/2026-09-21-plan-b-depth-observability-design.md` (record the outcome against §3)

- [ ] **Step 1: Re-read both tables before quoting anything**

```bash
.venv/Scripts/python.exe -c "
import pandas as pd
for n in ('gate_moveout','gate_moveout_summary'):
    d = pd.read_csv(f'paper/results/{n}.csv', comment='#')
    print('==', n, d.shape); print(d.head(15).to_string()); print()
"
```

- [ ] **Step 2: Write the findings note**

Create `paper/docs/das-finwhale-plan-b-gate-findings.md`, short, with these sections:

1. **What the gate asked** — quote the criterion from the design note §3, and state that it was fixed before the run.
2. **Denominators** — from `gate_moveout_summary.csv`: total calls, below min picks, unsolved, flipped, in stratum. Every later rate is over the stratum, named.
3. **Depth sensitivity** — the `sigma_z_m` distribution over the stratum: median, quartiles, how many are non-finite (did not bound depth) and what that means. Report the resolved count and fraction against the 100 m target.
4. **The path question, from timing** — the `delta_misfit` distribution: how many calls favour direct, how many favour bounce, and the magnitude. State whether this agrees with the transmission-loss study's amplitude-based ruling. **If it disagrees, that is a finding and must be reported as one, not resolved in favour of the earlier answer.**
5. **Verdict** — `gate_passes` true or false, and the consequence: proceed to §7.4, or stop. If it fails, say plainly what that means — this geometry cannot support depth inference at this pick precision — and do not hedge it into a maybe.
6. **Limitations** — the flat-bottom bounce approximation; the L2-versus-L1 objective choice; that `sigma_z_m` is a local curvature width and not a full posterior; that the structured 76–82 ms range residual Plan A measured is treated here as noise when it is partly model error.

Rules: every rate states its denominator and n at the point of use; flipped calls counted, never merged; no number quoted that was not re-read from its table in Step 1.

- [ ] **Step 3: Record the outcome in the design note**

Append to the design note's §3 a short "Outcome" subsection: the date, `gate_passes`, the resolved count and fraction, and a link to the findings note. This closes the loop so the design note does not read as still-open.

- [ ] **Step 4: Verify and commit**

```bash
.venv/Scripts/python.exe -m pytest paper/tests viz/tests -q
git diff --exit-code src/dclde_localization/physics/forward.py
git add docs/ && git commit -m "Report the Plan B gate verdict"
```

---

## Self-Review

**Spec coverage.**

| Design note section | Task |
|---|---|
| §3 gate — depth sensitivity vs pick noise | 1 (profile + σ_z), 3 (over real picks) |
| §3 gate — direct vs bottom-bounce, from timing | 2 (model + comparison), 3 |
| §3 gate criterion fixed before running | stated in this plan's header, copied verbatim into the config at Task 3 Step 5, applied at Task 3 Step 3 (`gate_summary`) |
| §3 "stops at the gate" is publishable | Task 4 Step 2 §5 requires it stated plainly, unhedged |
| §2 null is `ConstantC(1480)` at −30 m | Task 3 (`NULL_SOURCE_Z`, `inversion_model`) |
| §4(a) identifiability, stratified | Task 3's `in_stratum` + `sigma_z_m`; full §4 is §7.4, out of scope here |
| §6 reporting rules | Global Constraints; Task 4 Step 2 |
| §7 scope — no depth maps, no §7.5 | nothing in this plan produces them |

**Deliberately out of scope:** §4(b) synthetic recovery, §5 the `c(z)` comparison, and all of §7.4. Those are gated on this plan's outcome and will be planned only if it passes.

**Placeholder scan.** Clean — every code step carries its code, every test its assertions.

**Type consistency.** `depth_profile` returns `(z_grid, misfit)` and is consumed that way in `gate_table`. `marginal_sigma_z(z_grid, profiled_misfit, n_picks, resid_sigma)` matches both its test and its call site. `path_comparison`'s returned keys (`misfit_direct`, `misfit_bounce`, `delta_misfit`, `x_direct`, `y_direct`, `x_bounce`, `y_bounce`) are filtered to the three the table carries. `bounce_travel_time(sx, sy, sz, px, py, pz, c, bottom_z)` mirrors `forward.travel_time`'s argument order with `bottom_z` appended. `gate_summary(table, denominators, *, sigma_target_m, min_fraction, min_calls)` matches its tests and the driver. `solve(...)`, `Solution.best`, `.separation_m`, `depth_along_path(...).ok/.depth_m`, `write_results(df, name, out_dir=...)` all match existing signatures.

**Known risk, flagged not hidden.** `depth_profile` runs a 2-D Nelder–Mead at each of 121 depths per call — roughly 121 × a few hundred function evaluations × ~400 calls. Task 3 Step 7 requires timing it and reporting before launching if it would exceed ~30 minutes. If it does, the cheapest honest reduction is a coarser `z_grid` (the config exposes it) rather than trimming the stratum, because the stratum is pre-declared and the grid is not.
