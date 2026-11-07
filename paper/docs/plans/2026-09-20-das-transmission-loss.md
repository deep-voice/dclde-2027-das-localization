# DAS Transmission Loss — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide whether the first arrival on the DAS fin-whale cables at 20 Hz over 5–40 km is consistent with a direct water path or a trapped / bottom-interacting one — or whether this dataset cannot distinguish them.

**Architecture:** Three new units. `bathymetry.py` moves out of `viz/` into the task package and gains path sampling. `soundspeed/tl.py` turns the existing lossless mode solver into a transmission-loss calculation: per-mode sediment attenuation, source excitation, coherent and incoherent mode sums, adiabatic range stepping with mode stripping, and the fiber's axial projection. `experiments/propagation.py` produces three result tables — predicted TL, observed maximum detection range, and a censored joint fit of pick SNR on range and receiver depth — driven by YAML through the existing config driver.

**Tech Stack:** Python 3, NumPy, SciPy, pandas, pytest, `uv`. No new dependencies beyond SciPy, which is already a dependency.

**Spec:** [`paper/docs/specs/2026-09-20-das-transmission-loss-design.md`](../specs/2026-09-20-das-transmission-loss-design.md)

## Global Constraints

- **`src/dclde_localization/physics/forward.py` is frozen.** Do not modify it. Every task ends with `git diff --exit-code src/dclde_localization/physics/forward.py` clean.
- **Depth sign:** positive-down inside `soundspeed/`, negative-down everywhere else. Convert only through `to_down` / `to_repo` from `soundspeed.profile`.
- **No cross-package imports between task packages.** `dclde.das_finwhale` must not import `dclde.narw` or `dclde.killer_whales`. It may import `viz` only where the repo already does (`paper/scripts/run_experiment.py` imports `dclde_localization.physics.picks`); new physics code must not.
- **Absolute SNR level is never used.** Only shape versus range and versus depth. Any fit must carry a free per-call intercept.
- **Basin-flipping calls are never merged** into a pooled statistic, in tables or in prose.
- **Every rate and count states its denominator and n** at the point of use.
- **Every result table regenerates** from a committed config via `uv run python -m paper.scripts.run_experiment --config <path>`, and that claim is verified by regenerating and diffing, never asserted.
- Run everything from the repo root under `uv run`.
- Tests: `uv run pytest paper/tests -q`. The suite is currently **252 passing**; it must never go down.
- `Seabed.alpha` is in **dB per wavelength**.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/dclde_localization/physics/bathymetry.py` | **Create** (moved from `viz/bathymetry.py`). GMRT fetch/decode/reproject/validate, plus water depth along a path. |
| `viz/bathymetry.py` | **Modify** → thin re-export from the new home, so `viz`'s importers and tests are untouched. |
| `paper/src/dclde_soundspeed/soundspeed/tl.py` | **Create.** Modal attenuation, excitation, mode sums, adiabatic stepping, fiber projection. |
| `paper/src/dclde_soundspeed/soundspeed/__init__.py` | **Modify.** Export the new names. |
| `paper/src/dclde_soundspeed/experiments/propagation.py` | **Create.** The three result tables. |
| `paper/configs/propagation_tl.yaml` | **Create.** Result 1. |
| `paper/configs/propagation_data.yaml` | **Create.** Results 2 and 3. |
| `paper/scripts/run_experiment.py` | **Modify.** Two new dispatch entries. |
| `tests/physics/test_bathymetry_path.py` | **Create.** |
| `paper/tests/test_tl.py` | **Create.** |
| `paper/tests/test_propagation.py` | **Create.** |
| `paper/docs/das-finwhale-transmission-loss-findings.md` | **Create.** The verdict. |

---

### Task 1: Move `bathymetry` into the task package

The spec's §5.1. `das_finwhale` must not reach into the visualizer package, and there must be one definition. This task is **behaviour-preserving**: `viz`'s existing tests are the check.

**Files:**
- Create: `src/dclde_localization/physics/bathymetry.py`
- Modify: `viz/bathymetry.py` (becomes a re-export)
- Test: `viz/tests/test_bathymetry.py` (unchanged — that is the point)

**Interfaces:**
- Consumes: nothing.
- Produces: `GeoGrid`, `LocalGrid`, `ValidationReport`, `GMRT_BOUNDS`, `GMRT_URL`, `NO_DATA`, `fetch_gmrt`, `decode_gmt_grid`, `reproject_to_local`, `validate_against_cable`, `_sample` — all importable from `dclde_localization.physics.bathymetry` with identical behaviour.

- [ ] **Step 1: Confirm the current state is green, so a later failure is attributable**

```bash
uv run pytest viz/tests/test_bathymetry.py -q
```
Expected: PASS.

- [ ] **Step 2: Move the file with git, preserving history**

```bash
git mv viz/bathymetry.py src/dclde_localization/physics/bathymetry.py
```

- [ ] **Step 3: Replace `viz/bathymetry.py` with a re-export**

Create `viz/bathymetry.py`:

```python
"""Re-export of the bathymetry module, which now lives in the task package.

It moved to `dclde_localization.physics.bathymetry` because the sound-speed work needs
water depth along a propagation path, and `dclde.das_finwhale` must not import
the visualizer package to get it. One definition, imported both ways.
"""

from dclde_localization.physics.bathymetry import (  # noqa: F401
    GMRT_BOUNDS,
    GMRT_URL,
    NO_DATA,
    GeoGrid,
    LocalGrid,
    ValidationReport,
    _sample,
    decode_gmt_grid,
    fetch_gmrt,
    reproject_to_local,
    validate_against_cable,
)

__all__ = ["GMRT_BOUNDS", "GMRT_URL", "NO_DATA", "GeoGrid", "LocalGrid",
           "ValidationReport", "decode_gmt_grid", "fetch_gmrt",
           "reproject_to_local", "validate_against_cable"]
```

`_sample` is re-exported although it is private because `viz/tests/test_bathymetry.py` imports it directly; dropping it would break that test, and this task must not change tests.

- [ ] **Step 4: Fix the import in the new module's own docstring reference**

`src/dclde_localization/physics/bathymetry.py` needs no code change — it has no `viz` imports. Confirm:

```bash
grep -n "viz" src/dclde_localization/physics/bathymetry.py
```
Expected: no output.

- [ ] **Step 5: Verify both import paths work and agree**

```bash
uv run python -c "
import viz.bathymetry as a, dclde_localization.physics.bathymetry as b
assert a.decode_gmt_grid is b.decode_gmt_grid
assert a.NO_DATA == b.NO_DATA == -32768
print('same object, same constants')
"
```
Expected: `same object, same constants`.

- [ ] **Step 6: Run the untouched viz test and the full suites**

```bash
uv run pytest viz/tests/test_bathymetry.py -q
uv run pytest paper/tests -q
cd viz && npm test; cd ..
```
Expected: viz bathymetry tests PASS unchanged, `252 passed`, 49 JS tests pass.

- [ ] **Step 7: Verify the frozen module and commit**

```bash
git diff --exit-code src/dclde_localization/physics/forward.py
git add -A
git commit -m "Move bathymetry into the task package so the physics can use it

dclde.das_finwhale must not import the visualizer package to get water depth
along a propagation path. viz/bathymetry.py becomes a re-export, so viz's
importers and its tests are untouched -- which is the check that the move was
behaviour-preserving."
```

---

### Task 2: Water depth along a source→receiver path

Spec §5.1. The input the adiabatic calculation steps over.

**Files:**
- Modify: `src/dclde_localization/physics/bathymetry.py`
- Test: `tests/physics/test_bathymetry_path.py`

**Interfaces:**
- Consumes: `LocalGrid`, `NO_DATA` from Task 1.
- Produces:
  - `class PathProfile` — frozen dataclass with `range_m: np.ndarray` (0 → total range, `n_steps` points), `depth_m: np.ndarray` (positive-down water depth at each), `ok: bool`, `reason: str` (empty when `ok`).
  - `depth_along_path(grid: LocalGrid, x0: float, y0: float, x1: float, y1: float, *, n_steps: int = 64) -> PathProfile`

- [ ] **Step 1: Write the failing tests**

Create `tests/physics/test_bathymetry_path.py`:

```python
import numpy as np
import pytest

from dclde_localization.physics.bathymetry import NO_DATA, LocalGrid, depth_along_path


def _grid(z):
    """A 1 km-spaced local grid holding `z` (int16 metres, negative = below sea level)."""
    z = np.asarray(z, dtype=np.int16)
    ny, nx = z.shape
    return LocalGrid(x0=0.0, y0=0.0, dx=1000.0, dy=1000.0, nx=nx, ny=ny, z=z)


def test_constant_depth_path_returns_that_depth_everywhere():
    grid = _grid(np.full((5, 5), -800))
    p = depth_along_path(grid, 0.0, 0.0, 4000.0, 0.0, n_steps=9)
    assert p.ok
    assert p.depth_m.shape == (9,)
    # Positive-DOWN water depth, so the -800 m elevation becomes +800 m of water.
    assert np.allclose(p.depth_m, 800.0)


def test_range_axis_spans_zero_to_the_horizontal_separation():
    grid = _grid(np.full((5, 5), -800))
    p = depth_along_path(grid, 0.0, 0.0, 3000.0, 4000.0, n_steps=5)
    assert p.ok
    assert p.range_m[0] == pytest.approx(0.0)
    assert p.range_m[-1] == pytest.approx(5000.0)  # 3-4-5 triangle
    assert np.all(np.diff(p.range_m) > 0)


def test_a_sloping_path_reports_the_slope():
    # Depth deepens with x: -100, -200, -300, -400, -500 m elevation.
    z = np.tile(np.array([-100, -200, -300, -400, -500], dtype=np.int16), (3, 1))
    p = depth_along_path(_grid(z), 0.0, 1000.0, 4000.0, 1000.0, n_steps=5)
    assert p.ok
    assert np.allclose(p.depth_m, [100.0, 200.0, 300.0, 400.0, 500.0])


def test_a_path_leaving_the_grid_fails_explicitly_and_does_not_extrapolate():
    grid = _grid(np.full((5, 5), -800))
    p = depth_along_path(grid, 0.0, 0.0, 99_000.0, 0.0, n_steps=9)
    assert not p.ok
    assert "outside" in p.reason
    # It must NOT quietly hand back a usable-looking profile.
    assert p.depth_m.size == 0


def test_a_path_crossing_a_no_data_hole_fails_rather_than_interpolating_across_it():
    z = np.full((5, 5), -800, dtype=np.int16)
    z[:, 2] = NO_DATA
    p = depth_along_path(_grid(z), 0.0, 1000.0, 4000.0, 1000.0, n_steps=5)
    assert not p.ok
    assert "no-data" in p.reason
    assert p.depth_m.size == 0


def test_land_is_rejected_because_there_is_no_water_column_to_propagate_in():
    z = np.full((5, 5), -800, dtype=np.int16)
    z[:, 2] = 150  # above sea level
    p = depth_along_path(_grid(z), 0.0, 1000.0, 4000.0, 1000.0, n_steps=5)
    assert not p.ok
    assert "land" in p.reason


def test_zero_length_path_is_rejected_rather_than_dividing_by_zero():
    grid = _grid(np.full((5, 5), -800))
    p = depth_along_path(grid, 1000.0, 1000.0, 1000.0, 1000.0, n_steps=5)
    assert not p.ok
    assert "zero-length" in p.reason
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest tests/physics/test_bathymetry_path.py -q
```
Expected: FAIL — `ImportError: cannot import name 'depth_along_path'`.

- [ ] **Step 3: Implement**

Append to `src/dclde_localization/physics/bathymetry.py`:

```python
@dataclass(frozen=True)
class PathProfile:
    """Water depth sampled along a straight horizontal path.

    range_m: horizontal distance from the start, 0 .. total, `n_steps` points.
    depth_m: water depth, POSITIVE DOWN, at each of those ranges.
    ok:      whether the profile is usable. When False, `range_m` and `depth_m`
             are EMPTY -- a failed path must not hand back something that looks
             like an answer.
    reason:  why it failed; empty when `ok`.

    A path that leaves the grid, crosses a no-data hole, or crosses land fails
    rather than interpolating over the gap. Interpolating across a no-data hole
    in bathymetry invents a waveguide, and the whole point of this module is
    that the waveguide's shape along the path is the physics.
    """

    range_m: np.ndarray
    depth_m: np.ndarray
    ok: bool
    reason: str = ""


def _failed(reason: str) -> PathProfile:
    empty = np.empty(0, dtype=float)
    return PathProfile(range_m=empty, depth_m=empty, ok=False, reason=reason)


def depth_along_path(grid: LocalGrid, x0: float, y0: float, x1: float, y1: float,
                     *, n_steps: int = 64) -> PathProfile:
    """Sample water depth along the straight path from (x0, y0) to (x1, y1).

    Nearest-neighbour on the regular local grid, matching `_sample`'s convention
    (see its docstring for why nearest-neighbour rather than a floor lookup).
    """
    if n_steps < 2:
        raise ValueError(f"n_steps must be at least 2, got {n_steps}")

    total = float(np.hypot(x1 - x0, y1 - y0))
    if total <= 0.0:
        return _failed("zero-length path: source and receiver are at the same "
                       "horizontal position")

    s = np.linspace(0.0, 1.0, int(n_steps))
    xs = x0 + s * (x1 - x0)
    ys = y0 + s * (y1 - y0)

    i = np.round((xs - grid.x0) / grid.dx).astype(int)
    j = np.round((ys - grid.y0) / grid.dy).astype(int)
    inside = (i >= 0) & (i < grid.nx) & (j >= 0) & (j < grid.ny)
    if not inside.all():
        return _failed(f"path leaves the bathymetry grid: {int((~inside).sum())} "
                       f"of {int(n_steps)} samples outside")

    elev = grid.z[j, i].astype(float)
    if np.any(elev == NO_DATA):
        return _failed(f"path crosses a no-data hole in the bathymetry: "
                       f"{int((elev == NO_DATA).sum())} of {int(n_steps)} samples")
    if np.any(elev >= 0.0):
        return _failed(f"path crosses land: {int((elev >= 0.0).sum())} of "
                       f"{int(n_steps)} samples are at or above sea level")

    return PathProfile(range_m=s * total, depth_m=-elev, ok=True)
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/physics/test_bathymetry_path.py -q
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git diff --exit-code src/dclde_localization/physics/forward.py
git add src/dclde_localization/physics/bathymetry.py tests/physics/test_bathymetry_path.py
git commit -m "Sample water depth along a propagation path

Fails explicitly on a path that leaves the grid, crosses a no-data hole or
crosses land, and returns an EMPTY profile when it fails -- interpolating
across a bathymetry gap would invent the waveguide whose shape is the physics
this experiment is about."
```

---

### Task 3: Modal attenuation from sediment `alpha`

Spec §5.2. `Seabed.alpha` exists and is deliberately unconsumed by `modes()`; this is where it starts doing work.

**The derivation, so the constant is auditable.** In the density-reduced form the modes satisfy `phi'' + (omega^2/c^2 - kr^2) phi = 0` with `∫ phi^2 dz = 1`. Attenuation enters as a complex sound speed `c -> c(1 + i*eta)` with `eta = alpha_n * c / omega`, where `alpha_n` is in nepers/m. To first order `omega^2/c^2 -> (omega^2/c^2)(1 - 2*i*eta)`, so the perturbation to the eigenvalue is

```
d(kr^2) = -2i * ∫ eta(z) * omega^2/c(z)^2 * phi(z)^2 dz
        = -2i * omega * ∫ (alpha_n(z)/c(z)) * phi(z)^2 dz
d(kr)   = d(kr^2) / (2*kr)
```

giving the magnitude used below:

```
Im(kr_m) = (omega / kr_m) * ∫ (alpha_n(z)/c(z)) * phi_m(z)^2 dz
```

with the integral non-zero only in the sediment. Step 1's test checks this against an independently evaluated closed form for the Pekeris exponential tail.

**Files:**
- Create: `paper/src/dclde_soundspeed/soundspeed/tl.py`
- Test: `paper/tests/test_tl.py`

**Interfaces:**
- Consumes: `ModeSet`, `Seabed`, `modes` from `soundspeed.modes`; `SoundSpeedProfile` from `soundspeed.profile`.
- Produces:
  - `alpha_nepers_per_m(alpha_db_per_wavelength: float, freq_hz: float, c_m_s: float) -> float`
  - `modal_attenuation(modeset: ModeSet, seabed: Seabed) -> np.ndarray` — shape `(n_modes,)`, nepers/m, non-negative.

- [ ] **Step 1: Write the failing tests**

Create `paper/tests/test_tl.py`:

```python
import numpy as np
import pytest

from dclde_soundspeed.soundspeed.modes import Seabed, modes
from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile
from dclde_soundspeed.soundspeed.tl import alpha_nepers_per_m, modal_attenuation

C_WATER = 1480.0


def isovelocity(depth_m=1000.0, c=C_WATER, n=64):
    z = np.linspace(0.0, depth_m, n)
    return SoundSpeedProfile(z_down=z, c=np.full(n, c), source="test isovelocity")


def test_db_per_wavelength_converts_to_nepers_per_metre():
    # 1 dB/wavelength at 20 Hz in a 1700 m/s sediment: wavelength 85 m,
    # so 1/85 dB/m, and a neper is 20/ln(10) = 8.685889... dB.
    got = alpha_nepers_per_m(1.0, 20.0, 1700.0)
    expected = (1.0 * 20.0 / 1700.0) / (20.0 / np.log(10.0))
    assert got == pytest.approx(expected, rel=1e-12)
    assert got == pytest.approx(0.0013540, abs=1e-7)


def test_conversion_scales_linearly_in_alpha_and_frequency():
    a = alpha_nepers_per_m(0.2, 20.0, 1700.0)
    assert alpha_nepers_per_m(0.4, 20.0, 1700.0) == pytest.approx(2 * a)
    assert alpha_nepers_per_m(0.2, 40.0, 1700.0) == pytest.approx(2 * a)


def test_a_lossless_seabed_gives_exactly_zero_attenuation_for_every_mode():
    ms = modes(isovelocity(), Seabed(c=1700.0, rho=1.8, alpha=0.0), 20.0, 1000.0)
    assert ms.n_modes > 0, "need trapped modes for this test to mean anything"
    att = modal_attenuation(ms, Seabed(c=1700.0, rho=1.8, alpha=0.0))
    assert att.shape == (ms.n_modes,)
    assert np.all(att == 0.0), "a non-attenuating seabed must not acquire loss"


def test_attenuation_is_positive_and_linear_in_alpha():
    ms = modes(isovelocity(), Seabed(c=1700.0, rho=1.8, alpha=0.15), 20.0, 1000.0)
    a1 = modal_attenuation(ms, Seabed(c=1700.0, rho=1.8, alpha=0.15))
    a2 = modal_attenuation(ms, Seabed(c=1700.0, rho=1.8, alpha=0.30))
    assert np.all(a1 > 0)
    # The perturbation is first-order, so doubling alpha doubles the loss.
    assert np.allclose(a2, 2 * a1, rtol=1e-10)


def test_higher_modes_attenuate_more_because_they_penetrate_the_sediment_further():
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    ms = modes(isovelocity(), seabed, 20.0, 1000.0)
    if ms.n_modes < 2:
        pytest.skip("needs at least two trapped modes")
    att = modal_attenuation(ms, seabed)
    # Mode 1 is the shallowest-grazing (largest kr) and the best trapped.
    assert att[0] < att[-1]


def test_matches_an_independently_evaluated_closed_form_on_the_sediment_tail():
    """Independent oracle.

    In the sediment a trapped mode is phi(z) = A*exp(-gamma*(z-D)), with
    gamma = sqrt(kr^2 - (omega/c_sed)^2). The perturbation integral over the
    tail then has the closed form

        (omega / kr) * (alpha_n / c_sed) * A^2 / (2*gamma)

    which is evaluated here from the mode's own kr and its amplitude at the
    seafloor, independently of how `modal_attenuation` does its quadrature.
    """
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    depth = 1000.0
    ms = modes(isovelocity(depth_m=depth), seabed, 20.0, depth)
    assert ms.n_modes > 0
    got = modal_attenuation(ms, seabed)

    omega = 2 * np.pi * ms.freq_hz
    a_n = alpha_nepers_per_m(seabed.alpha, ms.freq_hz, seabed.c)
    floor = np.argmin(np.abs(ms.z_down - depth))
    expected = []
    for m in range(ms.n_modes):
        kr = ms.kr[m]
        gamma = np.sqrt(kr**2 - (omega / seabed.c) ** 2)
        amp = ms.phi[floor, m]
        expected.append((omega / kr) * (a_n / seabed.c) * amp**2 / (2 * gamma))

    assert np.allclose(got, expected, rtol=0.05), (
        f"numeric quadrature {got} disagrees with the closed form {expected}")
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest paper/tests/test_tl.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'dclde_soundspeed.soundspeed.tl'`.

- [ ] **Step 3: Implement**

Create `paper/src/dclde_soundspeed/soundspeed/tl.py`:

```python
"""Transmission loss from the trapped modes `modes.py` computes.

`modes.py` answers EXISTENCE: are there trapped modes, and how many. That is
not arrival. A mode can exist and still be irrelevant if a source at -30 m does
not excite it, or if it does not survive 5-40 km of a shelf-to-abyssal margin.
This module supplies the missing three: per-mode sediment attenuation (the
`Seabed.alpha` that `modes()` deliberately carries without consuming), source
excitation from the mode shape at source depth, and the mode sum that turns
those into a received level.

Depths are POSITIVE DOWN throughout, as everywhere in this package.

What this module does NOT do, and what a reader must not assume from it:
  * Adiabatic range stepping bounds mode coupling, it does not resolve it. The
    Oregon margin is steep and the adiabatic approximation assumes gentle
    slopes.
  * There is no gauge-length response. At 20 Hz the wavelength is ~74 m against
    a gauge length of order 10 m, so it is second order -- but it is absent,
    not included.
  * Absolute level is not meaningful against this dataset's denoised SNR
    product. Use shapes.
"""

from __future__ import annotations

import numpy as np

from dclde_soundspeed.soundspeed.modes import ModeSet, Seabed

# A neper is 20/ln(10) dB. Named because getting this factor wrong is the
# classic silent error in attenuation code, and a bare 8.686 is unreadable.
DB_PER_NEPER = 20.0 / np.log(10.0)


def alpha_nepers_per_m(alpha_db_per_wavelength: float, freq_hz: float,
                       c_m_s: float) -> float:
    """Convert `Seabed.alpha` (dB per WAVELENGTH) to nepers per metre.

    dB/m = alpha_dB_per_wavelength * freq / c, since one wavelength is c/freq.
    """
    alpha_db_per_wavelength = float(alpha_db_per_wavelength)
    freq_hz = float(freq_hz)
    c_m_s = float(c_m_s)
    if alpha_db_per_wavelength < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha_db_per_wavelength}")
    if freq_hz <= 0 or c_m_s <= 0:
        raise ValueError(f"freq_hz and c must be positive, got {freq_hz}, {c_m_s}")
    return (alpha_db_per_wavelength * freq_hz / c_m_s) / DB_PER_NEPER


def modal_attenuation(modeset: ModeSet, seabed: Seabed) -> np.ndarray:
    """Per-mode attenuation Im(kr), nepers/m, from sediment absorption.

    First-order perturbation of the eigenvalue by a complex sediment sound
    speed (see the plan's Task 3 for the derivation):

        Im(kr_m) = (omega / kr_m) * INTEGRAL (alpha_n(z) / c(z)) phi_m(z)^2 dz

    with `phi` the orthonormal, density-reduced shapes `ModeSet` already holds,
    and the integrand non-zero only in the sediment. A seabed with `alpha = 0`
    returns exactly zero for every mode.
    """
    if modeset.n_modes == 0:
        return np.empty(0, dtype=float)
    if seabed.alpha == 0.0:
        return np.zeros(modeset.n_modes, dtype=float)

    omega = 2.0 * np.pi * modeset.freq_hz
    a_n = alpha_nepers_per_m(seabed.alpha, modeset.freq_hz, seabed.c)

    z = modeset.z_down
    dz = float(z[1] - z[0])
    in_sediment = ~modeset.in_water

    # phi is orthonormal under sum(phi^2)*dz, so the quadrature matches the
    # normalisation the shapes were built with -- do not swap in a fancier rule
    # without renormalising, or the two disagree at the few-percent level.
    weight = np.where(in_sediment, a_n / seabed.c, 0.0)
    integral = (weight[:, None] * modeset.phi**2).sum(axis=0) * dz
    return omega * integral / modeset.kr
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest paper/tests/test_tl.py -q
```
Expected: 6 passed. If the closed-form test fails by more than 5%, the quadrature and the normalisation disagree — fix that rather than loosening the tolerance, because every later number rests on it.

- [ ] **Step 5: Commit**

```bash
git diff --exit-code src/dclde_localization/physics/forward.py
git add paper/src/dclde_soundspeed/soundspeed/tl.py paper/tests/test_tl.py
git commit -m "Give each mode its attenuation from the sediment alpha

Seabed.alpha has been carried unconsumed since Task 10 of Plan A, documented as
'for downstream use'. This is that use: first-order perturbation of the
eigenvalue by a complex sediment sound speed, checked against an independently
evaluated closed form for the exponential tail rather than against itself."
```

---

### Task 4: Excitation and the range-independent mode sum

Spec §5.3.

**Files:**
- Modify: `paper/src/dclde_soundspeed/soundspeed/tl.py`
- Test: `paper/tests/test_tl.py`

**Interfaces:**
- Consumes: `modal_attenuation`, `ModeSet`.
- Produces:
  - `excitation(modeset: ModeSet, source_z_down: float) -> np.ndarray` — `phi_m(z_s)`, shape `(n_modes,)`.
  - `mode_sum_tl(modeset, seabed, source_z_down, receiver_z_down, range_m, *, coherent=True) -> np.ndarray` — TL in dB, broadcast over `range_m`. `+inf` where no mode is trapped.

- [ ] **Step 1: Write the failing tests**

Append to `paper/tests/test_tl.py`:

```python
from dclde_soundspeed.soundspeed.tl import excitation, mode_sum_tl


def test_a_source_at_the_surface_excites_nothing_because_phi_is_pinned_to_zero():
    ms = modes(isovelocity(), Seabed(c=1700.0, rho=1.8, alpha=0.15), 20.0, 1000.0)
    assert np.allclose(excitation(ms, 0.0), 0.0)


def test_excitation_is_the_mode_shape_at_source_depth():
    ms = modes(isovelocity(), Seabed(c=1700.0, rho=1.8, alpha=0.15), 20.0, 1000.0)
    z_s = 300.0
    got = excitation(ms, z_s)
    k = np.argmin(np.abs(ms.z_down - z_s))
    assert np.allclose(got, ms.phi[k, :], rtol=1e-6)


def test_a_source_at_a_node_is_not_coupled_and_at_an_antinode_is_maximally_coupled():
    """Mode 1 in an isovelocity guide is a half-sine: node at the surface,
    antinode near mid-depth. A source at a node drives that mode far less than
    one at its antinode -- this is the term that decides whether a whale at
    -30 m can drive a mode trapped in a deep guide at all."""
    depth = 1000.0
    ms = modes(isovelocity(depth_m=depth), Seabed(c=1700.0, rho=1.8, alpha=0.15),
               20.0, depth)
    phi1 = ms.phi[ms.in_water, 0]
    z_in = ms.z_down[ms.in_water]
    antinode_z = float(z_in[np.argmax(np.abs(phi1))])
    near_node_z = 5.0
    assert abs(excitation(ms, near_node_z)[0]) < 0.2 * abs(excitation(ms, antinode_z)[0])


def test_tl_grows_with_range():
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    ms = modes(isovelocity(), seabed, 20.0, 1000.0)
    r = np.array([1e3, 5e3, 20e3, 40e3])
    tl = mode_sum_tl(ms, seabed, 30.0, 500.0, r, coherent=False)
    assert np.all(np.diff(tl) > 0), f"TL must increase with range, got {tl}"


def test_incoherent_sum_has_no_interference_nulls_and_coherent_one_may():
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    ms = modes(isovelocity(), seabed, 20.0, 1000.0)
    if ms.n_modes < 2:
        pytest.skip("interference needs at least two modes")
    r = np.linspace(2e3, 20e3, 400)
    inc = mode_sum_tl(ms, seabed, 30.0, 500.0, r, coherent=False)
    coh = mode_sum_tl(ms, seabed, 30.0, 500.0, r, coherent=True)
    # The incoherent sum is smooth; the coherent one is not.
    assert np.abs(np.diff(inc, 2)).max() < np.abs(np.diff(coh, 2)).max()


def test_attenuation_strictly_increases_tl():
    lossless = Seabed(c=1700.0, rho=1.8, alpha=0.0)
    lossy = Seabed(c=1700.0, rho=1.8, alpha=0.30)
    ms = modes(isovelocity(), lossless, 20.0, 1000.0)
    r = np.array([20e3])
    assert (mode_sum_tl(ms, lossy, 30.0, 500.0, r, coherent=False)[0]
            > mode_sum_tl(ms, lossless, 30.0, 500.0, r, coherent=False)[0])


def test_no_trapped_modes_gives_infinite_tl_rather_than_a_misleading_number():
    """A seabed slower than the water traps nothing. That is a physical answer,
    and it must not come back as a finite level."""
    slow = Seabed(c=1400.0, rho=1.8, alpha=0.15)
    ms = modes(isovelocity(), slow, 20.0, 1000.0)
    assert ms.n_modes == 0
    tl = mode_sum_tl(ms, slow, 30.0, 500.0, np.array([10e3]))
    assert np.all(np.isinf(tl))
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest paper/tests/test_tl.py -q -k "excitation or node or tl or sum or trapped"
```
Expected: FAIL — `cannot import name 'excitation'`.

- [ ] **Step 3: Implement**

Append to `paper/src/dclde_soundspeed/soundspeed/tl.py`:

```python
def excitation(modeset: ModeSet, source_z_down: float) -> np.ndarray:
    """Mode shape at source depth -- how strongly a source at `source_z_down`
    drives each mode.

    This is the term that separates "the modes are not there" from "the modes
    are there but the whale does not excite them". They are different findings
    with different consequences, so this is reported per mode rather than
    folded into a total.
    """
    if modeset.n_modes == 0:
        return np.empty(0, dtype=float)
    z = modeset.z_down
    k = int(np.argmin(np.abs(z - float(source_z_down))))
    return modeset.phi[k, :].astype(float)


def mode_sum_tl(modeset: ModeSet, seabed: Seabed, source_z_down: float,
                receiver_z_down: float, range_m, *, coherent: bool = True) -> np.ndarray:
    """Transmission loss, dB re 1 m, at each range in `range_m`.

    Far-field normal-mode sum:

        p(r,z) ~ (i e^{-i pi/4} / (rho(z_s) sqrt(8 pi r)))
                 * SUM_m phi_m(z_s) phi_m(z) e^{i kr_m r} / sqrt(kr_m)

    with per-mode attenuation applied as e^{-Im(kr_m) r}. The density prefactor
    is a constant here (it does not vary with range or mode) and TL is used
    only as a SHAPE against this dataset's uncalibrated denoised SNR, so it is
    omitted rather than invented from an unknown source-depth density.

    `coherent=False` sums mode powers instead of amplitudes. Both are produced
    and carried together deliberately: the coherent sum's interference nulls
    are far sharper than this geometry's uncertainty justifies, so a verdict
    must never rest on where a null falls.

    Returns +inf where no mode is trapped -- a physical answer, not a failure,
    and one that must not come back looking like a finite level.
    """
    r = np.atleast_1d(np.asarray(range_m, dtype=float))
    if np.any(r <= 0):
        raise ValueError("range_m must be positive; the far-field mode sum "
                         "diverges at r = 0")
    if modeset.n_modes == 0:
        return np.full(r.shape, np.inf)

    a_m = excitation(modeset, source_z_down)
    z = modeset.z_down
    k_r = int(np.argmin(np.abs(z - float(receiver_z_down))))
    b_m = modeset.phi[k_r, :].astype(float)

    kr = modeset.kr.astype(float)
    att = modal_attenuation(modeset, seabed)

    # (n_ranges, n_modes)
    decay = np.exp(-att[None, :] * r[:, None])
    amp = (a_m * b_m / np.sqrt(kr))[None, :] * decay

    if coherent:
        phase = np.exp(1j * kr[None, :] * r[:, None])
        field = np.abs((amp * phase).sum(axis=1))
    else:
        field = np.sqrt((amp**2).sum(axis=1))

    p = field / np.sqrt(8.0 * np.pi * r)
    with np.errstate(divide="ignore"):
        return -20.0 * np.log10(np.where(p > 0, p, 0.0))
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest paper/tests/test_tl.py -q
```
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git diff --exit-code src/dclde_localization/physics/forward.py
git add paper/src/dclde_soundspeed/soundspeed/tl.py paper/tests/test_tl.py
git commit -m "Add source excitation and the range-independent mode sum

Excitation is reported per mode, not folded into a total: 'the modes are not
there' and 'the modes are there but a source at -30 m does not drive them' are
different findings. No trapped modes returns +inf rather than a finite-looking
level. Coherent and incoherent sums are both produced, because the coherent
nulls are sharper than this geometry's uncertainty justifies."
```

---

### Task 5: Adiabatic range stepping and mode stripping

Spec §5.4. The mechanism this whole approach exists to capture.

**Files:**
- Modify: `paper/src/dclde_soundspeed/soundspeed/tl.py`
- Test: `paper/tests/test_tl.py`

**Interfaces:**
- Consumes: `mode_sum_tl`, `modal_attenuation`, `excitation`, `PathProfile` from `bathymetry`, `modes`, `SoundSpeedProfile`.
- Produces:
  - `class AdiabaticResult` — frozen dataclass: `tl_db: float`, `n_modes_start: int`, `n_modes_end: int`, `n_stripped: int`, `n_unresolved_steps: int`, `ok: bool`, `reason: str`.
  - `adiabatic_tl(profile, seabed, freq_hz, path, source_z_down, receiver_z_down, *, coherent=True, nz=2000) -> AdiabaticResult`

- [ ] **Step 1: Write the failing tests**

Append to `paper/tests/test_tl.py`:

```python
from dclde_localization.physics.bathymetry import PathProfile
from dclde_soundspeed.soundspeed.tl import adiabatic_tl


def flat_path(depth_m, total_m, n=24):
    return PathProfile(range_m=np.linspace(0.0, total_m, n),
                       depth_m=np.full(n, depth_m), ok=True)


def test_a_constant_depth_path_reduces_exactly_to_the_range_independent_sum():
    """The strongest check on the range stepping: with nothing varying, the
    adiabatic machinery must reproduce the plain mode sum."""
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    depth, rng = 1000.0, 20_000.0
    prof = isovelocity(depth_m=depth)
    ms = modes(prof, seabed, 20.0, depth)

    direct = mode_sum_tl(ms, seabed, 30.0, 500.0, np.array([rng]), coherent=False)[0]
    got = adiabatic_tl(prof, seabed, 20.0, flat_path(depth, rng), 30.0, 500.0,
                       coherent=False)

    assert got.ok
    assert got.n_stripped == 0
    assert got.tl_db == pytest.approx(direct, rel=1e-3)


def test_a_shallowing_path_strips_modes_and_never_regains_one():
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    n = 20
    path = PathProfile(range_m=np.linspace(0.0, 30_000.0, n),
                       depth_m=np.linspace(1400.0, 120.0, n), ok=True)
    got = adiabatic_tl(isovelocity(depth_m=1400.0), seabed, 20.0, path, 30.0, 100.0,
                       coherent=False)
    assert got.ok
    assert got.n_stripped > 0
    assert got.n_modes_end <= got.n_modes_start
    assert got.n_modes_end == got.n_modes_start - got.n_stripped


def test_stripping_a_mode_can_only_increase_transmission_loss():
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    prof = isovelocity(depth_m=1400.0)
    n = 20
    flat = PathProfile(range_m=np.linspace(0.0, 30_000.0, n),
                       depth_m=np.full(n, 1400.0), ok=True)
    shoaling = PathProfile(range_m=np.linspace(0.0, 30_000.0, n),
                           depth_m=np.linspace(1400.0, 200.0, n), ok=True)
    a = adiabatic_tl(prof, seabed, 20.0, flat, 30.0, 200.0, coherent=False)
    b = adiabatic_tl(prof, seabed, 20.0, shoaling, 30.0, 200.0, coherent=False)
    assert b.n_stripped > 0
    assert b.tl_db > a.tl_db


def test_a_failed_path_is_refused_rather_than_silently_treated_as_flat():
    seabed = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    bad = PathProfile(range_m=np.empty(0), depth_m=np.empty(0), ok=False,
                      reason="path crosses land: 3 of 9 samples")
    got = adiabatic_tl(isovelocity(), seabed, 20.0, bad, 30.0, 500.0)
    assert not got.ok
    assert "land" in got.reason
    assert not np.isfinite(got.tl_db)


def test_losing_every_mode_gives_infinite_loss():
    slow_guide = Seabed(c=1700.0, rho=1.8, alpha=0.15)
    n = 12
    # Shoal to a few metres: nothing stays trapped at 20 Hz.
    path = PathProfile(range_m=np.linspace(0.0, 30_000.0, n),
                       depth_m=np.linspace(1400.0, 3.0, n), ok=True)
    got = adiabatic_tl(isovelocity(depth_m=1400.0), slow_guide, 20.0, path, 30.0, 3.0,
                       coherent=False)
    assert got.ok
    assert got.n_modes_end == 0
    assert np.isinf(got.tl_db)
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest paper/tests/test_tl.py -q -k "adiabatic or strip or path or losing or constant_depth"
```
Expected: FAIL — `cannot import name 'adiabatic_tl'`.

- [ ] **Step 3: Implement**

Append to `paper/src/dclde_soundspeed/soundspeed/tl.py` (and add `from dataclasses import dataclass` plus the `modes`/`SoundSpeedProfile`/`PathProfile` imports at the top):

```python
@dataclass(frozen=True)
class AdiabaticResult:
    """One adiabatic mode-sum transmission loss along one path.

    tl_db:              transmission loss, dB. +inf if every mode was stripped.
    n_modes_start:      modes trapped in the guide at the source end.
    n_modes_end:        modes still trapped at the receiver end.
    n_stripped:         how many were lost along the way. Reported because mode
                        stripping, not sediment absorption, is the dominant
                        loss on a shelf-to-abyssal path -- it is the number that
                        explains the answer.
    n_unresolved_steps: steps where the local mode solve returned
                        `n_modes < n_modes_expected`, i.e. a shape was missing
                        entirely. Counted rather than absorbed. Steps that were
                        merely `converged=False` with an exact count are NOT
                        counted here -- per `ModeSet`'s own documentation that
                        is the common, expected case near the trapping
                        threshold, and discarding it would bias away exactly
                        the marginal regime this experiment is about.
    ok / reason:        false when the path itself was unusable.
    """

    tl_db: float
    n_modes_start: int
    n_modes_end: int
    n_stripped: int
    n_unresolved_steps: int
    ok: bool
    reason: str = ""


def adiabatic_tl(profile, seabed: Seabed, freq_hz: float, path,
                 source_z_down: float, receiver_z_down: float, *,
                 coherent: bool = True, nz: int = 2000) -> AdiabaticResult:
    """Adiabatic mode-sum TL over a range-varying waveguide.

    Each mode's phase accumulates as INTEGRAL kr_m(r') dr' along the path, with
    the amplitude factor sqrt(kr_m(0)/kr_m(r)). The local mode set is re-solved
    against the local water depth at each step.

    A mode is counted as STRIPPED when it stops being trapped in the local
    guide. `modes()` answers that exactly through `n_modes_expected`, the
    truncation-free cutoff count its docstring certifies as a physical
    statement rather than a solver outcome. Once stripped a mode contributes
    nothing further and does not return if the guide deepens again.

    Modes are tracked by index (mode 1 is the best trapped everywhere), so
    stripping removes from the high-index end. That is exact for a guide that
    only shoals and is the standard adiabatic bookkeeping otherwise.

    ADIABATIC BOUNDS MODE COUPLING, IT DOES NOT RESOLVE IT. This approximation
    assumes gentle slopes and the Oregon margin is steep. Read the result as a
    bound.
    """
    if not path.ok:
        return AdiabaticResult(np.inf, 0, 0, 0, 0, ok=False, reason=path.reason)

    r = np.asarray(path.range_m, dtype=float)
    d = np.asarray(path.depth_m, dtype=float)
    if r.size < 2:
        return AdiabaticResult(np.inf, 0, 0, 0, 0, ok=False,
                               reason="path has fewer than two samples")

    n_unresolved = 0
    n_start = None
    n_live = None
    phase = None         # accumulated INTEGRAL kr dr, per live mode
    atten = None         # accumulated INTEGRAL Im(kr) dr, per live mode
    kr_first = None      # kr at the source end, for the amplitude factor
    kr_last = None
    a_m = None           # excitation at the source end
    b_m = None           # receiver shape at the receiver end

    prev_kr = prev_att = None
    for i, (ri, di) in enumerate(zip(r, d)):
        ms = modes(profile, seabed, freq_hz, float(di), nz=nz)
        if ms.n_modes < ms.n_modes_expected:
            n_unresolved += 1

        n_here = ms.n_modes
        if i == 0:
            n_start = n_here
            n_live = n_here
            if n_live == 0:
                return AdiabaticResult(np.inf, 0, 0, 0, n_unresolved, ok=True)
            phase = np.zeros(n_live)
            atten = np.zeros(n_live)
            kr_first = ms.kr[:n_live].copy()
            a_m = excitation(ms, source_z_down)[:n_live]
            prev_kr = ms.kr[:n_live].copy()
            prev_att = modal_attenuation(ms, seabed)[:n_live]
            continue

        n_live = min(n_live, n_here)
        if n_live == 0:
            return AdiabaticResult(np.inf, int(n_start), 0, int(n_start),
                                   n_unresolved, ok=True)

        kr_here = ms.kr[:n_live]
        att_here = modal_attenuation(ms, seabed)[:n_live]
        dr = float(ri - r[i - 1])
        # Trapezoid along the path, truncated to the modes still alive.
        phase = phase[:n_live] + 0.5 * (prev_kr[:n_live] + kr_here) * dr
        atten = atten[:n_live] + 0.5 * (prev_att[:n_live] + att_here) * dr

        prev_kr, prev_att = kr_here.copy(), att_here.copy()
        kr_last = kr_here.copy()
        b_m = ms.phi[int(np.argmin(np.abs(ms.z_down - receiver_z_down))), :n_live]

    n_stripped = int(n_start - n_live)
    total_r = float(r[-1] - r[0])

    amp = (a_m[:n_live] * b_m * np.sqrt(kr_first[:n_live]) / kr_last
           * np.exp(-atten))
    if coherent:
        field = np.abs((amp * np.exp(1j * phase)).sum())
    else:
        field = float(np.sqrt((amp**2).sum()))

    p = field / np.sqrt(8.0 * np.pi * total_r)
    tl = np.inf if p <= 0 else float(-20.0 * np.log10(p))
    return AdiabaticResult(tl, int(n_start), int(n_live), n_stripped,
                           n_unresolved, ok=True)
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest paper/tests/test_tl.py -q
```
Expected: 18 passed. The reduction test (constant depth → plain mode sum, `rel=1e-3`) is the one that matters; if it fails, the amplitude factor or the phase quadrature is wrong.

- [ ] **Step 5: Commit**

```bash
git diff --exit-code src/dclde_localization/physics/forward.py
git add paper/src/dclde_soundspeed/soundspeed/tl.py paper/tests/test_tl.py
git commit -m "Step the mode sum along real bathymetry and strip modes that stop being trapped

Mode stripping, not sediment absorption, is the dominant loss on a
shelf-to-abyssal path, and a range-independent calculation structurally cannot
express it. Stripping is decided by n_modes_expected, the truncation-free
cutoff count modes.py certifies as physics rather than a solver outcome. A
constant-depth path reduces to the plain mode sum to 1e-3, which is the real
check on the stepping."
```

---

### Task 6: The fiber's axial projection

Spec §5.5. Without this the prediction is a pressure field and the measurement is axial strain, and the mismatch is range-correlated — exactly the shape the experiment reads.

**Files:**
- Modify: `paper/src/dclde_soundspeed/soundspeed/tl.py`
- Test: `paper/tests/test_tl.py`

**Interfaces:**
- Produces: `das_axial_response(source_xy, channel_xy, tangent_xy) -> float` — the `cos^2(theta)` factor in [0, 1].

- [ ] **Step 1: Write the failing tests**

Append to `paper/tests/test_tl.py`:

```python
from dclde_soundspeed.soundspeed.tl import das_axial_response


def test_endfire_arrival_along_the_fibre_is_fully_sensed():
    # Source due west, cable running east-west: propagation is along the fibre.
    assert das_axial_response((-10_000.0, 0.0), (0.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)


def test_broadside_arrival_is_rejected():
    # Source due south, cable east-west: propagation is perpendicular.
    assert das_axial_response((0.0, -10_000.0), (0.0, 0.0), (1.0, 0.0)) == pytest.approx(0.0, abs=1e-12)


def test_forty_five_degrees_gives_one_half():
    assert das_axial_response((-1000.0, -1000.0), (0.0, 0.0), (1.0, 0.0)) == pytest.approx(0.5)


def test_response_does_not_depend_on_the_sign_of_the_tangent():
    a = das_axial_response((-1000.0, -300.0), (0.0, 0.0), (1.0, 0.0))
    b = das_axial_response((-1000.0, -300.0), (0.0, 0.0), (-1.0, 0.0))
    assert a == pytest.approx(b)


def test_response_is_bounded_in_zero_one():
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = rng.normal(size=2) * 1e4
        t = rng.normal(size=2)
        v = das_axial_response(tuple(s), (0.0, 0.0), tuple(t))
        assert 0.0 <= v <= 1.0


def test_a_source_on_top_of_the_channel_is_refused_rather_than_returning_nan():
    with pytest.raises(ValueError, match="zero-length"):
        das_axial_response((0.0, 0.0), (0.0, 0.0), (1.0, 0.0))
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest paper/tests/test_tl.py -q -k axial
```
Expected: FAIL — `cannot import name 'das_axial_response'`.

- [ ] **Step 3: Implement**

Append to `paper/src/dclde_soundspeed/soundspeed/tl.py`:

```python
def das_axial_response(source_xy, channel_xy, tangent_xy) -> float:
    """Fraction of the arrival a DAS channel senses: cos^2(theta), with theta
    the angle between the horizontal propagation direction and the local cable
    tangent.

    DAS measures axial strain, so a broadside arrival is rejected and an endfire
    one is fully sensed. This is not a refinement: it varies strongly and
    systematically with geometry along these cables, and leaving it out would
    put a large range-correlated error into the prediction and invite it to be
    read as propagation physics.

    The gauge-length response is NOT included. At 20 Hz the wavelength is ~74 m
    against a gauge length of order 10 m, so it is second order -- but it is
    absent, not accounted for.
    """
    sx, sy = (float(v) for v in source_xy)
    cx, cy = (float(v) for v in channel_xy)
    tx, ty = (float(v) for v in tangent_xy)

    dx, dy = cx - sx, cy - sy
    dn = np.hypot(dx, dy)
    tn = np.hypot(tx, ty)
    if dn == 0.0:
        raise ValueError("zero-length source-to-channel vector: the source is "
                         "at the channel, so there is no propagation direction")
    if tn == 0.0:
        raise ValueError("zero-length cable tangent")

    cos_theta = (dx * tx + dy * ty) / (dn * tn)
    return float(min(1.0, cos_theta**2))
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest paper/tests/test_tl.py -q
```
Expected: 24 passed.

- [ ] **Step 5: Export the new names and commit**

Edit `paper/src/dclde_soundspeed/soundspeed/__init__.py` — add to the imports and `__all__`:

```python
from dclde_soundspeed.soundspeed.tl import (
    AdiabaticResult, adiabatic_tl, alpha_nepers_per_m, das_axial_response,
    excitation, modal_attenuation, mode_sum_tl)
```

and extend `__all__` with `"AdiabaticResult", "adiabatic_tl", "alpha_nepers_per_m", "das_axial_response", "excitation", "modal_attenuation", "mode_sum_tl"`.

```bash
uv run pytest paper/tests -q
git diff --exit-code src/dclde_localization/physics/forward.py
git add paper/src/dclde_soundspeed/soundspeed/
git commit -m "Project the predicted arrival onto the fibre

The prediction is a pressure field; the measurement is axial strain. The
cos^2 projection varies systematically with geometry along these cables, so
omitting it would leave a range-correlated error free to be read as propagation
physics. Gauge-length response is out of scope and said to be, not assumed in."
```

---

### Task 7: Result 1 — the predicted transmission-loss table

Spec §6.2. Pure forward calculation, no data. This alone answers the literal form of the findings' item 1.

**Files:**
- Create: `paper/src/dclde_soundspeed/experiments/propagation.py`
- Create: `paper/configs/propagation_tl.yaml`
- Modify: `paper/scripts/run_experiment.py`
- Test: `paper/tests/test_propagation.py`

**Interfaces:**
- Consumes: `adiabatic_tl`, `mode_sum_tl`, `excitation`, `modes`, `Seabed`, `PathProfile`, `write_results`.
- Produces: `predicted_tl_table(profile, seabed_sweep, freq_hz, geometries, *, n_steps=32) -> pd.DataFrame` with columns `source_z_m, receiver_z_m, range_m, water_depth_m, seabed_c, seabed_rho, seabed_alpha, n_modes_start, n_modes_end, n_stripped, n_unresolved_steps, excitation_rms, tl_modal_coherent_db, tl_modal_incoherent_db, tl_geometric_db, path_ok`.

`geometries` is a list of `(source_z_m, receiver_z_m, range_m, water_depth_source_m, water_depth_receiver_m)`. The path is a straight linear interpolation between the two depths — this table is a *forward* calculation over idealized geometries, so it does not need real bathymetry; Task 8 brings that in.

- [ ] **Step 1: Write the failing tests**

Create `paper/tests/test_propagation.py`:

```python
import numpy as np
import pandas as pd
import pytest

from dclde_soundspeed.experiments.propagation import (
    geometric_tl, predicted_tl_table)
from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile

SWEEP = [
    {"c": 1550.0, "rho": 1.5, "alpha": 0.05},
    {"c": 1900.0, "rho": 2.1, "alpha": 0.40},
]
GEOMS = [
    (30.0, 500.0, 10_000.0, 1000.0, 1000.0),
    (30.0, 1000.0, 30_000.0, 1400.0, 1200.0),
]


def isovelocity(depth_m=1500.0, c=1480.0, n=64):
    z = np.linspace(0.0, depth_m, n)
    return SoundSpeedProfile(z_down=z, c=np.full(n, c), source="test isovelocity")


def test_geometric_tl_is_cylindrical_spreading_plus_absorption():
    # 10 log10(r) + alpha*r/1000, with alpha in dB/km.
    assert geometric_tl(1000.0, alpha_db_per_km=0.0) == pytest.approx(30.0)
    assert geometric_tl(10_000.0, alpha_db_per_km=0.0) == pytest.approx(40.0)
    assert geometric_tl(10_000.0, alpha_db_per_km=1.0) == pytest.approx(50.0)


def test_geometric_tl_grows_monotonically():
    r = np.array([1e3, 5e3, 20e3, 40e3])
    tl = geometric_tl(r, alpha_db_per_km=0.1)
    assert np.all(np.diff(tl) > 0)


def test_table_has_one_row_per_geometry_and_seabed():
    df = predicted_tl_table(isovelocity(), SWEEP, 20.0, GEOMS, n_steps=6)
    assert len(df) == len(GEOMS) * len(SWEEP)


def test_table_carries_the_columns_the_findings_quote():
    df = predicted_tl_table(isovelocity(), SWEEP, 20.0, GEOMS, n_steps=6)
    for col in ("source_z_m", "receiver_z_m", "range_m", "seabed_c",
                "seabed_alpha", "n_modes_start", "n_modes_end", "n_stripped",
                "n_unresolved_steps", "excitation_rms",
                "tl_modal_coherent_db", "tl_modal_incoherent_db",
                "tl_geometric_db", "path_ok"):
        assert col in df.columns, col


def test_a_faster_seabed_traps_more_modes():
    df = predicted_tl_table(isovelocity(), SWEEP, 20.0, GEOMS[:1], n_steps=6)
    slow = df[df.seabed_c == 1550.0].n_modes_start.iloc[0]
    fast = df[df.seabed_c == 1900.0].n_modes_start.iloc[0]
    assert fast >= slow


def test_stripped_modes_are_reported_not_hidden():
    """The second geometry shoals from 1400 m to 1200 m. Whatever the count
    does, the column must exist and be internally consistent."""
    df = predicted_tl_table(isovelocity(), SWEEP, 20.0, GEOMS, n_steps=8)
    assert (df.n_modes_end == df.n_modes_start - df.n_stripped).all()


def test_excitation_rms_falls_as_the_source_moves_toward_the_surface():
    """phi is pinned to zero at the surface, so a shallower source drives the
    modes less. This is the term that decides whether a whale at -30 m matters."""
    deep = predicted_tl_table(isovelocity(), SWEEP[:1], 20.0,
                              [(300.0, 500.0, 10_000.0, 1000.0, 1000.0)], n_steps=6)
    shallow = predicted_tl_table(isovelocity(), SWEEP[:1], 20.0,
                                 [(3.0, 500.0, 10_000.0, 1000.0, 1000.0)], n_steps=6)
    assert shallow.excitation_rms.iloc[0] < deep.excitation_rms.iloc[0]
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest paper/tests/test_propagation.py -q
```
Expected: FAIL — `No module named 'dclde_soundspeed.experiments.propagation'`.

- [ ] **Step 3: Implement**

Create `paper/src/dclde_soundspeed/experiments/propagation.py`:

```python
"""Does the first arrival travel a direct water path, or a trapped one?

Plan A established that trapped modes EXIST wherever ray theory called the
geometry shadowed (findings section 3.3). Existence is not arrival. This module
asks the next question: does such a mode survive 5-40 km across a
shelf-to-abyssal margin, and does a source at -30 m excite it -- and can this
dataset tell the difference.

Two hypotheses, compared by the SHAPE of received level against range and
against receiver depth. Never by absolute level: the pick SNR comes from a
denoised strain product with unknown gain.

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

from dclde_localization.physics.bathymetry import PathProfile
from dclde_soundspeed.soundspeed.modes import Seabed
from dclde_soundspeed.soundspeed.tl import adiabatic_tl, excitation

# Thorp-scale water absorption at 20 Hz is ~0.001 dB/km -- utterly negligible
# over 40 km (0.04 dB). It is carried as an explicit parameter rather than
# dropped so that the null is a stated model, not an unexamined 10*log10(r).
DEFAULT_WATER_ALPHA_DB_PER_KM = 0.001


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


def _linear_path(range_m: float, d_source: float, d_receiver: float,
                 n_steps: int) -> PathProfile:
    return PathProfile(range_m=np.linspace(0.0, float(range_m), int(n_steps)),
                       depth_m=np.linspace(float(d_source), float(d_receiver),
                                           int(n_steps)),
                       ok=True)


def predicted_tl_table(profile, seabed_sweep, freq_hz: float, geometries,
                       *, n_steps: int = 32,
                       water_alpha_db_per_km: float = DEFAULT_WATER_ALPHA_DB_PER_KM
                       ) -> pd.DataFrame:
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
    """
    rows = []
    for (sz, rz, rng, d_src, d_rcv) in geometries:
        path = _linear_path(rng, d_src, d_rcv, n_steps)
        for sb in seabed_sweep:
            seabed = Seabed(c=float(sb["c"]), rho=float(sb["rho"]),
                            alpha=float(sb["alpha"]))
            coh = adiabatic_tl(profile, seabed, freq_hz, path, sz, rz,
                               coherent=True, nz=400)
            inc = adiabatic_tl(profile, seabed, freq_hz, path, sz, rz,
                               coherent=False, nz=400)

            from dclde_soundspeed.soundspeed.modes import modes as _modes
            ms0 = _modes(profile, seabed, freq_hz, float(d_src), nz=400)
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
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest paper/tests/test_propagation.py -q
```
Expected: 7 passed.

- [ ] **Step 5: Add the config**

Create `paper/configs/propagation_tl.yaml`:

```yaml
name: propagation_tl
notes: >
  Result 1 of the transmission-loss design (spec section 6.2). Pure forward
  calculation, no data: does a mode trapped at these geometries survive 5-40 km
  and does a source at -30 m excite it.

  truth_model names hycom_south_deep for the same reason
  regime_shadow_zone.yaml does: oregon_nov is hand-typed indicative numbers,
  never measured or reanalysis, and no conclusion may rest on it.
  hycom_south_deep covers 0-1500 m (98.4% of picks) and was itself never
  validated in-situ -- its credibility is inferred from hycom_oregon_nov's
  0.47 m/s median agreement with in-situ CTD 87 km away at the shallower OOI
  site. Every profile-dependent number here inherits that caveat; the spec's
  two-independent-product bar (WOA23) is still unmet.

  inversion_model is unused by predicted_tl_table (which takes a profile
  directly, not an ExperimentConfig) and is set to constant_c only to satisfy
  the loader's self-consistency guard, matching regime_shadow_zone.yaml.

  SEABED SWEEP PROVENANCE. Seabed geoacoustics are unknown at this site, so
  they are swept and the spread across the sweep IS the uncertainty. The middle
  three entries are the ones regime_shadow_zone.yaml already used; this config
  widens them at both ends to span soft mud through sand/gravel, following the
  standard geoacoustic compilations for continental-margin sediments (Hamilton,
  "Geoacoustic modeling of the sea floor", JASA 68:1313, 1980, and the
  sediment-property tables in Jensen et al., Computational Ocean Acoustics).
  NO OREGON-MARGIN-SPECIFIC MEASUREMENT WAS FOUND for these two cables. The
  sweep width therefore stands in for site uncertainty and must not be
  presented as site knowledge. If a site-specific source turns up, narrow the
  sweep and say so here.
truth_model: {kind: raytrace, profile: hycom_south_deep}
inversion_model: {kind: constant_c, c: 1480.0}
freq_hz: 20.0
n_steps: 48
seabed_sweep:
  - {c: 1500.0, rho: 1.4, alpha: 0.02}   # soft mud, barely faster than water
  - {c: 1550.0, rho: 1.5, alpha: 0.05}
  - {c: 1700.0, rho: 1.8, alpha: 0.15}
  - {c: 1900.0, rho: 2.1, alpha: 0.40}
  - {c: 2100.0, rho: 2.3, alpha: 0.80}   # sand/gravel
# (source_z, receiver_z, range, water_depth_at_source, water_depth_at_receiver)
geometries:
  - [30.0, 100.0, 5000.0, 200.0, 120.0]
  - [30.0, 150.0, 10000.0, 600.0, 200.0]
  - [30.0, 500.0, 20000.0, 1200.0, 600.0]
  - [30.0, 1000.0, 30000.0, 1400.0, 1100.0]
  - [30.0, 1575.0, 40000.0, 1600.0, 1600.0]
```

- [ ] **Step 6: Wire the driver**

In `paper/scripts/run_experiment.py`, add the dispatch function before the `DISPATCH` dict:

```python
def run_propagation_tl(raw: dict, config_path: Path) -> None:
    cfg = load_config(config_path)
    df = propagation.predicted_tl_table(
        cfg.truth_model.profile,
        raw["seabed_sweep"],
        float(raw["freq_hz"]),
        [tuple(g) for g in raw["geometries"]],
        n_steps=int(raw.get("n_steps", 32)),
    )
    print(write_results(df, "propagation_tl"))
```

and register it:

```python
    "propagation_tl": run_propagation_tl,
```

Add `from dclde_soundspeed.experiments import propagation` to the imports.

- [ ] **Step 7: Verify the config loads and the driver dispatches**

```bash
uv run python -m paper.scripts.run_experiment --config paper/configs/propagation_tl.yaml
uv run pytest paper/tests -q
```
Expected: writes `paper/results/propagation_tl.csv`; suite green.

- [ ] **Step 8: Commit**

```bash
git diff --exit-code src/dclde_localization/physics/forward.py
git add paper/src/dclde_soundspeed/experiments/propagation.py paper/configs/propagation_tl.yaml paper/scripts/run_experiment.py paper/tests/test_propagation.py paper/results/propagation_tl.csv
git commit -m "Predict transmission loss for both hypotheses over the seabed sweep

Result 1: the forward answer to 'does a trapped mode survive 5-40 km and does a
source at -30 m excite it'. Every row separates 'no modes', 'modes stripped'
and 'modes present but not excited', which are three different answers. The
seabed sweep's provenance and the absence of any Oregon-margin-specific
measurement are stated in the config rather than left implied."
```

---

### Task 8: The analysis frame — ranges, eligibility, and the fiber projection per pick

Spec §6.1 and §3.1. Everything data-side needs this one table, so it is built and tested once.

**Files:**
- Modify: `paper/src/dclde_soundspeed/experiments/propagation.py`
- Test: `paper/tests/test_propagation.py`

**Interfaces:**
- Consumes: `solve_all` from `experiments.ablation`, `depth_along_path`, `das_axial_response`, `CableGeometry`.
- Produces: `analysis_frame(picks, cables, grid, model, *, source_z, min_picks=8, flip_threshold_m=500.0) -> tuple[pd.DataFrame, dict]` — the per-pick frame plus a denominators dict.

Frame columns: `call, cable, dist, snr, px, py, pz, sx, sy, range_m, receiver_z_m, axial_response, water_depth_receiver_m, path_ok, path_reason`. Denominators dict keys: `calls_total, calls_below_min_picks, calls_unsolved, calls_flipped, calls_eligible, picks_eligible, calls_path_failed`.

- [ ] **Step 1: Write the failing tests**

Append to `paper/tests/test_propagation.py`:

```python
from dclde_soundspeed.experiments.propagation import analysis_frame


def test_analysis_frame_reports_every_denominator_it_drops_on(monkeypatch):
    """The denominators dict must account for every call: eligible + below-min
    + unsolved + flipped + path-failed == total. A silently dropped call is how
    a rate acquires the wrong denominator."""
    from dclde_localization.physics.geometry import load_cables
    from dclde_localization.physics.bathymetry import decode_gmt_grid, reproject_to_local
    from dclde_localization.physics.geometry import FrameTransform
    from dclde_localization.physics.sound_speed_models import ConstantC
    from dclde_localization.physics.picks import attach_xyz, load_picks

    cables = load_cables("fixtures")
    picks = attach_xyz(load_picks(), cables)
    # A small, deterministic subset -- this test guards the bookkeeping, not
    # the science, and the full solve is minutes long.
    keep = sorted(picks["call"].unique())[:6]
    picks = picks[picks["call"].isin(keep)].copy()

    grid = reproject_to_local(decode_gmt_grid("fixtures/gmrt_bathy.nc"),
                              FrameTransform.from_cable(cables["South"]))
    frame, den = analysis_frame(picks, cables, grid, ConstantC(1480.0), source_z=-30.0)

    assert den["calls_total"] == len(keep)
    accounted = (den["calls_eligible"] + den["calls_below_min_picks"]
                 + den["calls_unsolved"] + den["calls_flipped"]
                 + den["calls_path_failed"])
    assert accounted == den["calls_total"], (den, accounted)
    assert den["picks_eligible"] == len(frame)


def test_range_is_source_to_channel_not_along_cable_distance():
    """`dist` is distance along the fibre. Using it as range would be wrong by
    tens of kilometres, and it is the single easiest mistake to make here."""
    from dclde_soundspeed.experiments.propagation import _range_to_channels

    sx, sy = 0.0, 0.0
    px = np.array([3000.0, 0.0])
    py = np.array([0.0, 4000.0])
    pz = np.array([-500.0, -500.0])
    got = _range_to_channels(sx, sy, -30.0, px, py, pz)
    assert got[0] == pytest.approx(np.hypot(3000.0, 470.0))
    assert got[1] == pytest.approx(np.hypot(4000.0, 470.0))


def test_axial_response_column_is_bounded_and_varies():
    from dclde_localization.physics.geometry import load_cables
    from dclde_localization.physics.bathymetry import decode_gmt_grid, reproject_to_local
    from dclde_localization.physics.geometry import FrameTransform
    from dclde_localization.physics.sound_speed_models import ConstantC
    from dclde_localization.physics.picks import attach_xyz, load_picks

    cables = load_cables("fixtures")
    picks = attach_xyz(load_picks(), cables)
    keep = sorted(picks["call"].unique())[:6]
    picks = picks[picks["call"].isin(keep)].copy()
    grid = reproject_to_local(decode_gmt_grid("fixtures/gmrt_bathy.nc"),
                              FrameTransform.from_cable(cables["South"]))
    frame, _ = analysis_frame(picks, cables, grid, ConstantC(1480.0), source_z=-30.0)
    if frame.empty:
        pytest.skip("no eligible calls in this subset")
    assert frame.axial_response.between(0.0, 1.0).all()
    assert frame.axial_response.std() > 0
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest paper/tests/test_propagation.py -q -k "analysis_frame or range_is or axial_response_column"
```
Expected: FAIL — `cannot import name 'analysis_frame'`.

- [ ] **Step 3: Implement**

Append to `paper/src/dclde_soundspeed/experiments/propagation.py`:

```python
def _range_to_channels(sx: float, sy: float, source_z_repo: float,
                       px, py, pz) -> np.ndarray:
    """Slant source-to-channel range, metres.

    `picks.dist` is distance ALONG THE FIBRE, not range. Confusing the two is
    wrong by tens of kilometres and is the easiest mistake in this module, so
    range is computed here, once, from positions.
    """
    px, py, pz = (np.asarray(v, dtype=float) for v in (px, py, pz))
    return np.sqrt((px - sx) ** 2 + (py - sy) ** 2
                   + (pz - float(source_z_repo)) ** 2)


def _cable_tangents(cable, dist_m) -> np.ndarray:
    """Unit tangent of the cable at each along-cable distance, (n, 2)."""
    d = np.asarray(dist_m, dtype=float)
    step = max(float(cable.spacing_m), 1.0)
    x0, y0, _ = cable.chan_to_xyz(d - step)
    x1, y1, _ = cable.chan_to_xyz(d + step)
    tx, ty = x1 - x0, y1 - y0
    n = np.hypot(tx, ty)
    n = np.where(n > 0, n, 1.0)
    return np.stack([tx / n, ty / n], axis=1)


def analysis_frame(picks, cables, grid, model, *, source_z: float,
                   min_picks: int = 8, flip_threshold_m: float = 500.0):
    """Per-pick frame for the data comparison, plus the denominators.

    POSITIONS ARE AN INPUT, NOT A RESULT (spec section 6.1). Range needs a
    source position and source positions come from `solve()`, which uses a
    propagation model -- which is what is under test. The circularity is broken,
    not ignored:

      * positions come from the `model` passed in, normally ConstantC, and are
        stated as an input;
      * the sensitivity is already measured -- Plan A section 3.5 found a
        depth-dependent effective c moves basin-stable positions a median
        9.0-9.5 m, negligible as a covariate at 5-40 km;
      * basin-FLIPPING calls are excluded, because for those the position
        difference is a median 1,273 m and that argument does not hold.

    Returns `(frame, denominators)`. Every call is accounted for in the
    denominators: eligible + below_min_picks + unsolved + flipped + path_failed
    equals total.
    """
    from dclde_localization.physics.bathymetry import depth_along_path
    from dclde_soundspeed.experiments.ablation import solve_all
    from dclde_soundspeed.soundspeed.tl import das_axial_response

    calls_total = int(picks["call"].nunique())
    below = int(sum(1 for _, s in picks.groupby("call") if len(s) < min_picks))

    solutions = solve_all(picks, model, source_z=source_z, min_picks=min_picks)
    unsolved = calls_total - below - len(solutions)

    flipped = 0
    path_failed = 0
    rows = []
    for call, sol in solutions.items():
        # A call whose two best basins are confidently separated is a flip
        # risk; per the reporting contract it is counted, never merged.
        if sol.separation_m is not None and sol.separation_m > flip_threshold_m:
            flipped += 1
            continue

        sub = picks[picks["call"] == call]
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

        axial = np.empty(len(sub))
        for name, idx in sub.groupby("cable").groups.items():
            part = sub.loc[idx]
            tangents = _cable_tangents(cables[name], part["dist"].to_numpy(float))
            for k, (pos, t) in enumerate(zip(part.index, tangents)):
                axial[sub.index.get_loc(pos)] = das_axial_response(
                    (best.x, best.y),
                    (float(sub.at[pos, "px"]), float(sub.at[pos, "py"])),
                    (float(t[0]), float(t[1])))

        rows.append(pd.DataFrame({
            "call": call, "cable": sub["cable"].to_numpy(),
            "dist": sub["dist"].to_numpy(float), "snr": sub["snr"].to_numpy(float),
            "px": sub["px"].to_numpy(float), "py": sub["py"].to_numpy(float),
            "pz": sub["pz"].to_numpy(float),
            "sx": best.x, "sy": best.y,
            "range_m": rng, "receiver_z_m": -sub["pz"].to_numpy(float),
            "axial_response": axial,
            "water_depth_receiver_m": depths,
            "path_ok": True, "path_reason": reason,
        }))

    frame = (pd.concat(rows, ignore_index=True) if rows
             else pd.DataFrame(columns=["call", "cable", "dist", "snr", "px", "py",
                                        "pz", "sx", "sy", "range_m", "receiver_z_m",
                                        "axial_response", "water_depth_receiver_m",
                                        "path_ok", "path_reason"]))
    den = {
        "calls_total": calls_total,
        "calls_below_min_picks": below,
        "calls_unsolved": int(unsolved),
        "calls_flipped": int(flipped),
        "calls_path_failed": int(path_failed),
        "calls_eligible": int(frame["call"].nunique()) if len(frame) else 0,
        "picks_eligible": int(len(frame)),
    }
    return frame, den
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest paper/tests/test_propagation.py -q
```
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git diff --exit-code src/dclde_localization/physics/forward.py
git add paper/src/dclde_soundspeed/experiments/propagation.py paper/tests/test_propagation.py
git commit -m "Build the analysis frame: real ranges, eligibility, fibre projection

picks.dist is distance along the fibre, not source-receiver range; a test pins
that distinction because confusing them is wrong by tens of kilometres.
Positions are declared an INPUT, with Plan A's measured 9.0-9.5 m sensitivity
as the argument and basin-flipping calls excluded, since for those the figure
is 1,273 m. Every call is accounted for in the denominators -- eligible plus
each drop reason equals the total, asserted."
```

---

### Task 9: Result 2 — observed maximum detection range

Spec §6.3. The most robust data contact available: no calibration at all.

**Files:**
- Modify: `paper/src/dclde_soundspeed/experiments/propagation.py`
- Test: `paper/tests/test_propagation.py`

**Interfaces:**
- Produces: `max_range_table(frame, denominators) -> pd.DataFrame` — one row per eligible call: `call, n_picks, max_range_m, min_range_m, max_range_snr, receiver_z_at_max_m`; carries the denominators in `df.attrs["denominators"]`.

- [ ] **Step 1: Write the failing tests**

Append to `paper/tests/test_propagation.py`:

```python
from dclde_soundspeed.experiments.propagation import max_range_table


def _frame(rows):
    return pd.DataFrame(rows)


def test_max_range_is_per_call_and_picks_the_furthest_pick():
    frame = _frame([
        {"call": "a", "range_m": 5000.0, "snr": 12.0, "receiver_z_m": 300.0},
        {"call": "a", "range_m": 22_000.0, "snr": 7.0, "receiver_z_m": 800.0},
        {"call": "b", "range_m": 9000.0, "snr": 15.0, "receiver_z_m": 400.0},
    ])
    out = max_range_table(frame, {"calls_total": 3})
    assert len(out) == 2
    a = out[out.call == "a"].iloc[0]
    assert a.max_range_m == pytest.approx(22_000.0)
    assert a.min_range_m == pytest.approx(5000.0)
    assert a.max_range_snr == pytest.approx(7.0)
    assert a.receiver_z_at_max_m == pytest.approx(800.0)
    assert a.n_picks == 2


def test_denominators_travel_with_the_table_so_a_rate_cannot_lose_them():
    frame = _frame([{"call": "a", "range_m": 5000.0, "snr": 12.0,
                     "receiver_z_m": 300.0}])
    den = {"calls_total": 628, "calls_eligible": 1}
    out = max_range_table(frame, den)
    assert out.attrs["denominators"] == den


def test_an_empty_frame_gives_an_empty_table_not_a_crash():
    out = max_range_table(_frame([]), {"calls_total": 0})
    assert out.empty
    assert list(out.columns)[:3] == ["call", "n_picks", "max_range_m"]
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest paper/tests/test_propagation.py -q -k max_range
```
Expected: FAIL — `cannot import name 'max_range_table'`.

- [ ] **Step 3: Implement**

Append to `paper/src/dclde_soundspeed/experiments/propagation.py`:

```python
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
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest paper/tests/test_propagation.py -q
```
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git diff --exit-code src/dclde_localization/physics/forward.py
git add paper/src/dclde_soundspeed/experiments/propagation.py paper/tests/test_propagation.py
git commit -m "Report the furthest range each call was still detected at

Result 2, the one data contact needing no calibration. Its asymmetry is
documented where it is computed: it can falsify a hypothesis predicting
detection ranges shorter than observed, and cannot separate two that both
predict detection beyond the cable."
```

---

### Task 10: Result 3 — the censored joint fit

Spec §6.4. The test with the most power and the most ways to go wrong.

**Files:**
- Modify: `paper/src/dclde_soundspeed/experiments/propagation.py`
- Test: `paper/tests/test_propagation.py`

**Interfaces:**
- Produces:
  - `censored_fit(y, X, call_index, *, threshold, censored_mask=None) -> dict` with keys `beta` (np.ndarray, one per column of `X`), `sigma`, `n`, `n_censored`, `loglik`, `converged`.
  - `decay_fit_table(frame, *, threshold, n_bootstrap=200, seed=0) -> pd.DataFrame`

A **Tobit** (left-censored Gaussian) likelihood with a free intercept per call. Uncensored picks contribute the Gaussian density; censored ones contribute `Phi((threshold - mu)/sigma)`.

- [ ] **Step 1: Write the failing tests**

Append to `paper/tests/test_propagation.py`:

```python
from dclde_soundspeed.experiments.propagation import censored_fit, decay_fit_table


def _synthetic(n_calls=40, per_call=30, slope=-8.0, sigma=2.0, seed=0):
    """Known slope, known threshold. log10(range) is the regressor."""
    rng = np.random.default_rng(seed)
    call_index, x, y = [], [], []
    for c in range(n_calls):
        intercept = rng.normal(60.0, 5.0)          # per-call source level
        lr = rng.uniform(np.log10(3e3), np.log10(6e4), per_call)
        obs = intercept + slope * lr + rng.normal(0.0, sigma, per_call)
        call_index += [c] * per_call
        x.append(lr)
        y.append(obs)
    return np.array(call_index), np.concatenate(x), np.concatenate(y)


def test_censored_fit_recovers_a_known_slope_when_nothing_is_censored():
    ci, x, y = _synthetic()
    X = x[:, None]
    out = censored_fit(y, X, ci, threshold=-1e9)   # threshold below all data
    assert out["converged"]
    assert out["n_censored"] == 0
    assert out["beta"][0] == pytest.approx(-8.0, abs=0.4)


def test_the_naive_uncensored_fit_is_biased_and_the_censored_one_is_not():
    """The censoring correction must have teeth.

    Picks exist only where SNR crossed a detection threshold, so the sample is
    truncated and the truncation bites hardest at long range -- exactly where
    the hypotheses differ. A test that only showed the censored fit working
    would pass trivially on a correction that did nothing, so this asserts the
    naive fit is WRONG by a margin and the censored fit is right.
    """
    ci, x, y = _synthetic()
    thresh = float(np.quantile(y, 0.35))           # censor the lowest 35%
    seen = y >= thresh

    naive = censored_fit(y[seen], x[seen, None], ci[seen], threshold=-1e9)
    corrected = censored_fit(np.where(seen, y, thresh), x[:, None], ci,
                             threshold=thresh, censored_mask=~seen)

    assert corrected["converged"]
    # Truncation flattens the apparent decay: the naive slope is too shallow.
    assert naive["beta"][0] > -8.0 + 1.0, (
        f"naive slope {naive['beta'][0]:.2f} was not detectably biased; "
        "this test has no teeth unless it is")
    assert corrected["beta"][0] == pytest.approx(-8.0, abs=0.7)
    assert abs(corrected["beta"][0] + 8.0) < abs(naive["beta"][0] + 8.0)


def test_per_call_intercepts_absorb_arbitrary_per_call_level_offsets():
    """Absolute level is unusable (denoised product, unknown gain), so the fit
    must be invariant to shifting each call by an arbitrary constant."""
    ci, x, y = _synthetic()
    rng = np.random.default_rng(7)
    offsets = rng.normal(0.0, 30.0, ci.max() + 1)
    a = censored_fit(y, x[:, None], ci, threshold=-1e9)
    b = censored_fit(y + offsets[ci], x[:, None], ci, threshold=-1e9)
    assert b["beta"][0] == pytest.approx(a["beta"][0], abs=1e-4)


def test_decay_fit_reports_range_and_depth_separately():
    frame = pd.DataFrame({
        "call": np.repeat([f"c{i}" for i in range(12)], 20),
        "range_m": np.tile(np.linspace(4e3, 5e4, 20), 12),
        "receiver_z_m": np.tile(np.linspace(100.0, 1400.0, 20), 12),
        "snr": np.tile(np.linspace(20.0, 8.0, 20), 12),
        "axial_response": 1.0,
    })
    out = decay_fit_table(frame, threshold=0.0, n_bootstrap=20, seed=0)
    assert {"term", "estimate", "ci_lo", "ci_hi"} <= set(out.columns)
    assert {"log10_range", "receiver_depth_km"} <= set(out.term)


def test_bootstrap_resamples_whole_calls_not_individual_picks():
    """Picks within a call are correlated, so resampling picks would understate
    the interval. Two runs with the same seed must agree exactly, and the
    interval must widen when there are fewer calls to draw from."""
    def frame_with(n_calls):
        return pd.DataFrame({
            "call": np.repeat([f"c{i}" for i in range(n_calls)], 20),
            "range_m": np.tile(np.linspace(4e3, 5e4, 20), n_calls),
            "receiver_z_m": np.tile(np.linspace(100.0, 1400.0, 20), n_calls),
            "snr": np.tile(np.linspace(20.0, 8.0, 20), n_calls)
                   + np.repeat(np.linspace(0, 5, n_calls), 20),
            "axial_response": 1.0,
        })
    a = decay_fit_table(frame_with(30), threshold=0.0, n_bootstrap=40, seed=1)
    b = decay_fit_table(frame_with(30), threshold=0.0, n_bootstrap=40, seed=1)
    pd.testing.assert_frame_equal(a, b)

    few = decay_fit_table(frame_with(6), threshold=0.0, n_bootstrap=40, seed=1)
    wide = (few.ci_hi - few.ci_lo).mean()
    narrow = (a.ci_hi - a.ci_lo).mean()
    assert wide > narrow
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest paper/tests/test_propagation.py -q -k "censored or decay or bootstrap or intercepts"
```
Expected: FAIL — `cannot import name 'censored_fit'`.

- [ ] **Step 3: Implement**

Append to `paper/src/dclde_soundspeed/experiments/propagation.py` (add `from scipy import optimize, stats` at the top):

```python
def censored_fit(y, X, call_index, *, threshold: float, censored_mask=None) -> dict:
    """Left-censored (Tobit) Gaussian regression with a free intercept per call.

    WHY CENSORING RATHER THAN A PLAIN FIT. Picks exist only where the received
    SNR crossed a detection threshold, so the observed sample is truncated, and
    the truncation bites hardest at long range -- exactly where the two
    hypotheses differ. An uncensored fit flattens the apparent decay and would
    bias the comparison toward whichever hypothesis decays more slowly.

    WHY A FREE INTERCEPT PER CALL. Source level varies per call and the denoiser
    gain is unknown, so absolute level carries no information. Only shape does.

    `censored_mask` marks observations known only to be BELOW `threshold`; they
    contribute Phi((threshold - mu)/sigma) instead of a density.
    """
    y = np.asarray(y, dtype=float)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    call_index = np.asarray(call_index)
    if censored_mask is None:
        censored_mask = np.zeros(y.shape, dtype=bool)
    censored_mask = np.asarray(censored_mask, dtype=bool)

    codes, uniq = pd.factorize(call_index)
    n_calls = len(uniq)
    n_beta = X.shape[1]

    def unpack(theta):
        return theta[:n_beta], theta[n_beta:n_beta + n_calls], np.exp(theta[-1])

    def neg_loglik(theta):
        beta, alpha, sigma = unpack(theta)
        mu = X @ beta + alpha[codes]
        ll = np.empty_like(y)
        obs = ~censored_mask
        ll[obs] = stats.norm.logpdf(y[obs], mu[obs], sigma)
        if censored_mask.any():
            ll[censored_mask] = stats.norm.logcdf(
                (threshold - mu[censored_mask]) / sigma)
        return -float(ll.sum())

    # Start from an OLS fit on the uncensored rows, which is cheap and close.
    obs = ~censored_mask
    design = np.column_stack([X, np.eye(n_calls)[codes]])
    start_coef, *_ = np.linalg.lstsq(design[obs], y[obs], rcond=None)
    resid = y[obs] - design[obs] @ start_coef
    theta0 = np.concatenate([start_coef, [np.log(max(resid.std(), 1e-3))]])

    res = optimize.minimize(neg_loglik, theta0, method="L-BFGS-B",
                            options={"maxiter": 5000})
    beta, _, sigma = unpack(res.x)
    return {"beta": beta, "sigma": float(sigma), "n": int(y.size),
            "n_censored": int(censored_mask.sum()), "loglik": -float(res.fun),
            "converged": bool(res.success)}


def decay_fit_table(frame, *, threshold: float, n_bootstrap: int = 200,
                    seed: int = 0) -> pd.DataFrame:
    """Result 3 (spec section 6.4): pick SNR on range AND receiver depth
    jointly, per-call intercepts free, censored at `threshold`.

    Range and receiver depth are correlated on a shelf-to-abyssal cable, so
    they are fitted JOINTLY and neither coefficient may be reported alone.

    The interval comes from resampling WHOLE CALLS, not individual picks: picks
    within a call share a source, a source level and a propagation path, so
    resampling picks would understate the interval.
    """
    f = frame.dropna(subset=["range_m", "receiver_z_m", "snr"]).copy()
    if f.empty:
        return pd.DataFrame(columns=["term", "estimate", "ci_lo", "ci_hi",
                                     "n_picks", "n_calls", "n_censored"])

    def design(sub):
        X = np.column_stack([
            np.log10(sub["range_m"].to_numpy(float)),
            sub["receiver_z_m"].to_numpy(float) / 1000.0,
        ])
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
    rows = []
    for j, term in enumerate(terms):
        lo, hi = ((np.percentile(draws[:, j], [2.5, 97.5]))
                  if len(draws) else (np.nan, np.nan))
        rows.append({"term": term, "estimate": float(point["beta"][j]),
                     "ci_lo": float(lo), "ci_hi": float(hi),
                     "n_picks": point["n"], "n_calls": int(calls.size),
                     "n_censored": point["n_censored"]})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest paper/tests/test_propagation.py -q
```
Expected: 18 passed. The teeth test is the one that matters: if the naive fit is *not* detectably biased, the synthetic setup is not censoring hard enough and the test is worthless — fix the setup, do not relax the assertion.

- [ ] **Step 5: Commit**

```bash
git diff --exit-code src/dclde_localization/physics/forward.py
git add paper/src/dclde_soundspeed/experiments/propagation.py paper/tests/test_propagation.py
git commit -m "Fit SNR on range and depth jointly, with detection censoring

Picks exist only above a detection threshold, so the sample is truncated where
the hypotheses differ most; an uncensored fit would flatten the decay and
favour whichever hypothesis decays more slowly. The censoring test has teeth --
it asserts the naive fit IS biased, not merely that the corrected one works.
Range and depth are correlated on a shelf-to-abyssal cable, so they are fitted
jointly and neither is reported alone. The interval resamples whole calls."
```

---

### Task 11: Wire the data experiment and produce the tables

**Files:**
- Modify: `paper/src/dclde_soundspeed/experiments/propagation.py`, `paper/scripts/run_experiment.py`
- Create: `paper/configs/propagation_data.yaml`
- Test: `paper/tests/test_propagation.py`

**Interfaces:**
- Produces: `run_propagation_data(raw, config_path)` writing `propagation_max_range.csv` and `propagation_decay_fit.csv`, plus `propagation_denominators.csv`.

- [ ] **Step 1: Write the config**

Create `paper/configs/propagation_data.yaml`:

```yaml
name: propagation_data
notes: >
  Results 2 and 3 of the transmission-loss design (spec sections 6.3, 6.4).

  POSITIONS ARE AN INPUT, NOT A RESULT. Range needs a source position, and
  source positions come from solve(), which uses a propagation model -- the
  thing under test. inversion_model below is the model that SUPPLIES those
  positions (ConstantC, the existing baseline). Plan A section 3.5 measured the
  sensitivity: a depth-dependent effective c moves basin-stable positions a
  median 9.0-9.5 m, negligible as a covariate at 5-40 km. Basin-flipping calls
  are excluded entirely, because for those the figure is a median 1,273 m and
  that argument does not hold.

  THRESHOLD SWEEP. The detection threshold is not recorded in the annotation
  snapshot, so it is swept rather than invented. The observed pick SNR runs
  0.00-31.50 with p25 10.46 and median 13.54 (n = 11,117); 31 picks are exactly
  zero and are treated as censored AT the threshold rather than dropped, with
  the table reporting the result both ways so the rule's influence is visible.
truth_model: {kind: raytrace, profile: hycom_south_deep}
inversion_model: {kind: constant_c, c: 1480.0}
source_z: -30.0
min_picks: 8
flip_threshold_m: 500.0
n_bootstrap: 200
seed: 0
threshold_sweep: [0.0, 4.0, 6.0, 8.0]
```

- [ ] **Step 2: Write the failing test**

Append to `paper/tests/test_propagation.py`:

```python
def test_every_soundspeed_config_loads_or_documents_why_it_raises():
    """Guard against a config that names a run it does not perform -- the
    Critical 2 defect Plan A's whole-branch review found."""
    import glob
    from dclde_soundspeed.experiments.runner import load_config
    from dclde_soundspeed.experiments.runner import SelfConsistencyError

    known_raisers = {"ablation_free_c.yaml"}
    for path in sorted(glob.glob("paper/configs/*.yaml")):
        name = path.replace("\\", "/").split("/")[-1]
        if name in known_raisers:
            with pytest.raises((ValueError, SelfConsistencyError)):
                load_config(path)
        else:
            cfg = load_config(path)
            assert cfg.name


def test_the_new_configs_are_registered_with_the_driver():
    from paper.scripts.run_experiment import DISPATCH
    assert "propagation_tl" in DISPATCH
    assert "propagation_data" in DISPATCH
```

- [ ] **Step 3: Run it to verify it fails**

```bash
uv run pytest paper/tests/test_propagation.py -q -k "registered or every_soundspeed"
```
Expected: FAIL — `propagation_data` not in `DISPATCH`.

- [ ] **Step 4: Implement the dispatch**

Add to `paper/scripts/run_experiment.py`:

```python
def run_propagation_data(raw: dict, config_path: Path) -> None:
    from dclde_localization.physics.bathymetry import decode_gmt_grid, reproject_to_local
    from dclde_localization.physics.geometry import FrameTransform

    cfg = load_config(config_path)
    cables = load_cables(str(GEOMETRY_DIR))
    picks = _load_all_picks()
    grid = reproject_to_local(
        decode_gmt_grid(str(GEOMETRY_DIR / "gmrt_bathy.nc")),
        FrameTransform.from_cable(cables["South"]))

    frame, den = propagation.analysis_frame(
        picks, cables, grid, cfg.inversion_model,
        source_z=float(raw["source_z"]),
        min_picks=int(raw.get("min_picks", 8)),
        flip_threshold_m=float(raw.get("flip_threshold_m", 500.0)))

    print(write_results(pd.DataFrame([den]), "propagation_denominators"))
    print(write_results(propagation.max_range_table(frame, den),
                        "propagation_max_range"))

    fits = []
    for thr in raw["threshold_sweep"]:
        out = propagation.decay_fit_table(
            frame, threshold=float(thr),
            n_bootstrap=int(raw.get("n_bootstrap", 200)),
            seed=int(raw.get("seed", 0)))
        out.insert(0, "threshold", float(thr))
        fits.append(out)
    print(write_results(pd.concat(fits, ignore_index=True),
                        "propagation_decay_fit"))
```

Register it: `"propagation_data": run_propagation_data,`. The GMRT fixture is at `fixtures/gmrt_bathy.nc`, which `GEOMETRY_DIR` already points at.

- [ ] **Step 5: Run the tests, then the experiment**

```bash
uv run pytest paper/tests -q
uv run python -m paper.scripts.run_experiment --config paper/configs/propagation_data.yaml
```
Expected: suite green; three tables written. This run solves every eligible call and bootstraps, so budget tens of minutes.

- [ ] **Step 6: Verify regenerability, as a claim rather than an assertion**

```bash
cp paper/results/propagation_max_range.csv /tmp/a.csv
uv run python -m paper.scripts.run_experiment --config paper/configs/propagation_data.yaml
uv run python -c "
import pandas as pd
a = pd.read_csv('/tmp/a.csv', comment='#')
b = pd.read_csv('paper/results/propagation_max_range.csv', comment='#')
pd.testing.assert_frame_equal(a, b)
print('regenerates identically')
"
```
Expected: `regenerates identically`.

- [ ] **Step 7: Commit**

```bash
git diff --exit-code src/dclde_localization/physics/forward.py
git add paper/configs/propagation_data.yaml paper/scripts/run_experiment.py paper/tests/test_propagation.py paper/results/propagation_*.csv
git commit -m "Run the data comparison and commit its tables

Denominators get their own table so no rate can be quoted without them. The
detection threshold is swept rather than invented, and the 31 exactly-zero-SNR
picks are censored at the threshold with the result reported both ways.
Regenerability is verified by regenerating and diffing, not asserted."
```

---

### Task 12: The verdict

Spec §7 and §9. The verdict criteria were fixed in the spec before any result existed; this task applies them.

**Files:**
- Create: `paper/docs/das-finwhale-transmission-loss-findings.md`
- Modify: `paper/docs/das-finwhale-soundspeed-plan-a-findings.md` (mark item 1 of §6 as discharged, with a link)

- [ ] **Step 1: Re-read every table before quoting it**

```bash
uv run python -c "
import pandas as pd, glob
for p in sorted(glob.glob('paper/results/propagation_*.csv')):
    d = pd.read_csv(p, comment='#')
    print('==', p, d.shape); print(d.head(12).to_string()); print()
"
```

Every number in the write-up comes from this output or from a committed source named at the point of use. Anything measured but not table-backed is marked **[not table-backed]**, as Plan A's findings do.

- [ ] **Step 2: Write the findings document**

Create `paper/docs/das-finwhale-transmission-loss-findings.md` with these sections:

1. **What was asked** — quote the spec's §1 question and note the verdict criteria were fixed in §7 in advance.
2. **What was measured** — the denominators from `propagation_denominators.csv` (calls total, below min picks, unsolved, flipped, path-failed, eligible; picks eligible), then one subsection per result table.
3. **Result 1 — does a trapped mode survive and get excited** — from `propagation_tl.csv`. Report `n_modes_start`, `n_stripped`, `excitation_rms` and both TL columns across the seabed sweep, and say plainly whether the answer changes sign across the sweep.
4. **Result 2 — observed maximum detection range** — with its stated asymmetry (§6.3) repeated, not just cross-referenced.
5. **Result 3 — the censored joint fit** — both coefficients together, never one alone, with the whole-call bootstrap interval and the threshold sweep. State how much the censoring correction moved the estimate; if it moved it little, say so.
6. **Verdict** — exactly one of the spec §7 branches: direct path, bottom-interacting/trapped, or undecidable. State which, why, and what it means for Plan A's spec §7.4 — whether depth observability is now interpretable and under what null.
7. **Limitations** — all six of spec §9, each in its own words, not as a list of cross-references.
8. **Reproducing this** — the per-table regenerability table, in the form Plan A's §8 uses, with the verification actually run.

Rules, all of which Plan A's review caught being violated at least once: every rate states its denominator at the point of use; basin-flipping calls never merged, in tables *or* prose; no absolute SNR level quoted anywhere; the seabed sweep is site uncertainty, not site knowledge.

- [ ] **Step 3: Verify every quoted number against its table**

For each number in the document, open the CSV it came from and confirm it. Then check that no percentage appearing in both a table and the prose disagrees between them:

```bash
uv run python -c "
import re, io
t = io.open('paper/docs/das-finwhale-transmission-loss-findings.md', encoding='utf-8').read()
nums = re.findall(r'\d+\.?\d*%', t)
print(sorted(set(nums)))
"
```
Confirm each against its table by hand.

- [ ] **Step 4: Mark the Plan A item discharged**

In `paper/docs/das-finwhale-soundspeed-plan-a-findings.md` §6, item 1, append:

```markdown
   **Done (2026-09-20).** Answered in
   [`paper/docs/das-finwhale-transmission-loss-findings.md`](../das-finwhale-transmission-loss-findings.md).
```

- [ ] **Step 5: Final verification**

```bash
uv run pytest paper/tests -q
uv run pytest viz/tests -q
cd viz && npm test; cd ..
git diff --exit-code src/dclde_localization/physics/forward.py
```
Expected: all green, `forward.py` clean across the whole branch.

- [ ] **Step 6: Commit**

```bash
git add paper/docs/das-finwhale-transmission-loss-findings.md paper/docs/das-finwhale-soundspeed-plan-a-findings.md
git commit -m "Report the transmission-loss verdict on Q-path

Applies the verdict criteria the spec fixed in section 7 before any result
existed, so the branch taken was not chosen after seeing the numbers. States
what it means for Plan A's section 7.4 -- whether depth observability is now
interpretable, and under what null."
```

---

## Self-Review

**Spec coverage.**

| Spec section | Task |
|---|---|
| §3 data and denominators | 8 (denominators dict, asserted complete) |
| §3.1 eligibility, basin-stable | 8 |
| §4 the two hypotheses | 7 (`geometric_tl` vs `adiabatic_tl`) |
| §5.1 bathymetry move + path sampling | 1, 2 |
| §5.2 modal attenuation, dB/λ conversion | 3 |
| §5.3 excitation and mode sum | 4 |
| §5.4 adiabatic stepping, mode stripping, `converged` handling | 5 |
| §5.5 fiber projection, gauge length out of scope | 6 |
| §5.6 seabed sweep with provenance | 7 (config notes) |
| §6.1 range, circularity | 8 |
| §6.2 Result 1 | 7 |
| §6.3 Result 2 | 9 |
| §6.4 Result 3, censoring, whole-call bootstrap, zero-SNR rule | 10, 11 |
| §7 verdict criteria | 12 |
| §8 testing (all 7 named cases) | 2, 3, 4, 5, 6, 10 (teeth), 1 (viz unchanged) |
| §9 limitations | 12 §7, plus module docstrings in 5 and 6 |
| §10 reporting rules | 11 step 6, 12 steps 2–3 |
| §11 scope | nothing outside it |

**Placeholder scan.** Clean — every code step carries its code, every test its assertions. No "similar to Task N".

**Type consistency.** `PathProfile` (Task 2) is constructed in tests in Task 5 and consumed by `adiabatic_tl` (5) and `predicted_tl_table` (7). `AdiabaticResult` fields (`tl_db`, `n_modes_start`, `n_modes_end`, `n_stripped`, `n_unresolved_steps`, `ok`, `reason`) are read in Task 7's table builder with those exact names. `analysis_frame`'s columns (Task 8) are exactly what `max_range_table` (9) and `decay_fit_table` (10) read: `call`, `range_m`, `receiver_z_m`, `snr`, `axial_response`. `censored_fit`'s returned keys (`beta`, `sigma`, `n`, `n_censored`, `loglik`, `converged`) are read in `decay_fit_table` and in Task 10's tests. `Seabed(c, rho, alpha)` matches the existing frozen dataclass. `solve_all(picks, model, *, source_z, min_picks)` matches `ablation.py:97`. `write_results(df, name)` matches `runner.py:176`.

**Known risk, flagged rather than hidden.** Task 11's full run re-solves the local mode set at every path step for every pick, which is the expensive part. If it proves too slow, cache mode sets by rounded water depth — the guide only depends on local depth, so a 10 m-resolution cache is exact to well within the seabed sweep's spread. Do that as a measured optimization inside Task 11, not as a speculative one earlier.
