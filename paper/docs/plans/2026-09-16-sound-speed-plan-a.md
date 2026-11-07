# Sound-Speed Research, Plan A — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the published DAS fin-whale localization baseline, settle whether the assumed straight-line propagation path is the real one, and run the L-1 sound-speed ablations — producing committed result tables that say what the evidence supports.

**Architecture:** A `SoundSpeedModel` layer sits above a frozen `forward.py`, so the visualizer's Python↔JavaScript parity contract is untouched while new physics (path-averaged `c`, refracted rays, modal group delay) plugs in behind one interface. The estimator returns *every* misfit basin rather than a point, because the objective is bimodal and a median over basin flips hides the only interesting failure. Experiments are YAML-configured, take `truth_model` and `inversion_model` as separate required arguments so a self-consistent simulation cannot happen by accident, and write versioned result tables to disk.

**Tech Stack:** Python 3.13, numpy, scipy, pandas, xarray, pyproj, h5py, gsw (new), `uv` for environments, pytest. No W&B. No new JavaScript.

**Spec:** [`paper/docs/specs/2026-09-15-sound-speed-research-design.md`](../specs/2026-09-15-sound-speed-research-design.md) — read it before starting; this plan argues from it and does not restate its reasoning.

## Global Constraints

- **`src/dclde_localization/physics/forward.py` is FROZEN.** It is bit-for-bit parity-tested against `viz/template/lib/forward.mjs`, which the published visualizer runs in the browser. Do not change its signatures, its arithmetic, or its module-level constants. New physics layers above it.
- **Never cross-import between task packages.** `dclde.das_finwhale` must not import `dclde.narw` or `dclde.killer_whales`. Shared helpers go in `dclde.common`.
- **Depth sign convention is a trap.** The repo's geometry frame is **negative-down** (`cable.depth` runs 11 down to −1575; `pz` is negative). Oceanography is **positive-down**. `SoundSpeedProfile` stores **positive-down** internally and every conversion is explicit and tested. A silent sign flip here corrupts every number downstream and will not raise.
- **Every reported result states its denominator**: how many calls, which batches and cables, how many excluded and why. Non-negotiable per CLAUDE.md.
- **Every experiment is config-driven YAML** in `paper/configs/`. No hard-coded hyperparameters. **No W&B** — result tables are committed files.
- Run everything with `uv run`. Tests: `uv run pytest paper/tests -q`.
- `DEFAULT_C = 1480.0` and `OBJECTIVES = ("l1", "l2")` come from `dclde_localization.physics.forward`. Never redefine them.
- Cable geometry lives at `fixtures/{north,south}_DAS_multicoord.csv`, loaded with `dclde_localization.physics.geometry.load_cables("fixtures")`.
- Picks load with `dclde_localization.physics.picks.load_picks()` then `dclde_localization.physics.picks.attach_xyz(picks, cables)`, giving columns `call, window, call_id, cable, dist, time, snr, call_type, batch, px, py, pz`.
- Commit after every task. Message body explains *why*, not *what*. End with:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## File Structure

| File | Responsibility |
|---|---|
| `paper/src/dclde_soundspeed/soundspeed/__init__.py` | Package marker; re-exports the public names |
| `src/dclde_localization/physics/sound_speed_profile.py` | `SoundSpeedProfile`: depth→sound-speed, path averaging, T/S conversion |
| `src/dclde_localization/physics/sound_speed_models.py` | `SoundSpeedModel` protocol, `ConstantC`, `PathAverage` |
| `paper/src/dclde_soundspeed/soundspeed/raytrace.py` | Layer-arc ray tracer, `RayTrace` model, `max_direct_range` |
| `paper/src/dclde_soundspeed/soundspeed/modes.py` | Range-independent normal-mode solver, modal group delay |
| `paper/src/dclde_soundspeed/soundspeed/erddap.py` | OOI ERDDAP client, QARTOD filtering |
| `paper/src/dclde_soundspeed/soundspeed/reanalysis.py` | HYCOM / WOA23 profiles, in-situ validation |
| `src/dclde_localization/physics/solve.py` | Multi-basin estimator and the basin-stable/flip comparison contract |
| `paper/src/dclde_soundspeed/experiments/runner.py` | YAML config → experiment, self-consistency guard, result tables |
| `paper/src/dclde_soundspeed/experiments/repro.py` | Baseline reproduction and per-call join |
| `paper/scripts/fetch_reference_data.py` | Re-downloadable provenance for the 3.6 GB of reference data |
| `paper/configs/*.yaml` | One YAML per experiment |
| `paper/tests/test_*.py` | Unit tests |
| `paper/results/*.csv` | Committed result tables |

---

### Task 1: Sound-speed profile with an explicit depth sign convention

**Files:**
- Create: `paper/src/dclde_soundspeed/soundspeed/__init__.py`
- Create: `src/dclde_localization/physics/sound_speed_profile.py`
- Modify: `requirements.txt` (add `gsw`)
- Test: `tests/physics/test_sound_speed_profile.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `SoundSpeedProfile(z_down, c, source)` — frozen dataclass; `z_down` ascending, metres **positive down**; `c` m/s; `source` a provenance string.
  - `SoundSpeedProfile.c_at(z_down) -> np.ndarray`
  - `SoundSpeedProfile.path_average(z1_down, z2_down) -> float` — harmonic mean (mean slowness)
  - `SoundSpeedProfile.constant(c, z_max=3000.0) -> SoundSpeedProfile` (classmethod)
  - `SoundSpeedProfile.from_ts(z_down, temperature_c, salinity_psu, lat, lon, source) -> SoundSpeedProfile` (classmethod, TEOS-10 via `gsw`)
  - `to_down(pz) -> np.ndarray` and `to_repo(z_down) -> np.ndarray` — the only sanctioned sign conversions.

- [ ] **Step 1: Write the failing tests**

```python
# tests/physics/test_sound_speed_profile.py
import numpy as np
import pytest

from dclde_localization.physics.forward import DEFAULT_C
from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile, to_down, to_repo

OREGON_Z = np.array([0.0, 100.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0])
OREGON_C = np.array([1491.0, 1480.0, 1478.0, 1477.0, 1479.0, 1484.0, 1490.0])


def test_sign_conversion_round_trips_and_flips():
    # The repo frame is negative-down; the profile is positive-down. A silent
    # flip here would corrupt every downstream number without raising.
    pz = np.array([11.0, -30.0, -589.0, -1575.0])
    assert np.allclose(to_down(pz), [-11.0, 30.0, 589.0, 1575.0])
    assert np.allclose(to_repo(to_down(pz)), pz)


def test_constant_profile_path_average_is_that_constant():
    p = SoundSpeedProfile.constant(DEFAULT_C)
    assert p.path_average(30.0, 1575.0) == pytest.approx(DEFAULT_C, rel=1e-12)
    assert p.path_average(500.0, 500.0) == pytest.approx(DEFAULT_C, rel=1e-12)


def test_path_average_is_harmonic_not_arithmetic():
    # Travel time is distance / c, so the average that preserves travel time is
    # the mean of SLOWNESS. With a strong gradient the two differ measurably and
    # using the arithmetic mean would bias every range.
    p = SoundSpeedProfile(np.array([0.0, 1000.0]), np.array([1400.0, 1600.0]), "synthetic")
    harmonic = p.path_average(0.0, 1000.0)
    arithmetic = 1500.0
    assert harmonic < arithmetic
    assert harmonic == pytest.approx(1000.0 / np.trapezoid(
        1.0 / np.linspace(1400.0, 1600.0, 20001), np.linspace(0.0, 1000.0, 20001)), rel=1e-6)


def test_path_average_is_symmetric_in_its_endpoints():
    p = SoundSpeedProfile(OREGON_Z, OREGON_C, "oregon-nov")
    assert p.path_average(30.0, 1200.0) == pytest.approx(p.path_average(1200.0, 30.0), rel=1e-15)


def test_c_at_clamps_outside_the_profile_rather_than_extrapolating():
    # Extrapolating a sound-speed profile past its deepest sample invents data.
    # Clamping is wrong too, but it is wrong in a bounded, visible way.
    p = SoundSpeedProfile(OREGON_Z, OREGON_C, "oregon-nov")
    assert p.c_at(5000.0) == pytest.approx(OREGON_C[-1])
    assert p.c_at(-10.0) == pytest.approx(OREGON_C[0])


def test_from_ts_reproduces_a_known_sound_speed():
    # TEOS-10 at S=35 psu, T=10 C, ~1000 dbar is close to 1500 m/s. This is a
    # coarse anchor: its job is to catch swapped T/S arguments, which would give
    # a wildly wrong answer, not to validate gsw.
    p = SoundSpeedProfile.from_ts(
        z_down=np.array([1000.0]), temperature_c=np.array([10.0]),
        salinity_psu=np.array([35.0]), lat=45.0, lon=-125.0, source="anchor")
    assert 1490.0 < float(p.c[0]) < 1520.0


def test_rejects_unsorted_or_mismatched_input():
    with pytest.raises(ValueError, match="ascending"):
        SoundSpeedProfile(np.array([100.0, 0.0]), np.array([1480.0, 1490.0]), "bad")
    with pytest.raises(ValueError, match="same length"):
        SoundSpeedProfile(np.array([0.0, 100.0]), np.array([1480.0]), "bad")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/physics/test_sound_speed_profile.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'dclde_soundspeed.soundspeed'`

- [ ] **Step 3: Add the `gsw` dependency**

Append `gsw` to `requirements.txt` under the `# data / audio / DAS` block, then:

```bash
uv pip install -e .
```

- [ ] **Step 4: Write the implementation**

```python
# paper/src/dclde_soundspeed/soundspeed/__init__.py
"""Sound-speed structure for DAS fin-whale localization.

Layers above the frozen forward.py so the visualizer's Python/JavaScript
parity contract is untouched.
"""

from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile, to_down, to_repo

__all__ = ["SoundSpeedProfile", "to_down", "to_repo"]
```

```python
# src/dclde_localization/physics/sound_speed_profile.py
"""Depth-dependent sound speed.

Depth convention: this module is POSITIVE DOWN, the oceanographic convention and
the one gsw expects. The rest of the repo's geometry frame is NEGATIVE DOWN
(cable depths run +11 to -1575 m). `to_down` and `to_repo` are the only
sanctioned conversions; a sign error here is silent and corrupts everything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_QUAD_POINTS = 4001


def to_down(pz):
    """Repo frame (negative down) -> profile frame (positive down)."""
    return -np.asarray(pz, dtype=float)


def to_repo(z_down):
    """Profile frame (positive down) -> repo frame (negative down)."""
    return -np.asarray(z_down, dtype=float)


@dataclass(frozen=True)
class SoundSpeedProfile:
    z_down: np.ndarray
    c: np.ndarray
    source: str

    def __post_init__(self):
        z = np.asarray(self.z_down, dtype=float)
        c = np.asarray(self.c, dtype=float)
        if z.shape != c.shape:
            raise ValueError(f"z_down and c must be the same length: {z.shape} vs {c.shape}")
        if z.size < 2:
            raise ValueError("a profile needs at least two samples")
        if np.any(np.diff(z) <= 0):
            raise ValueError("z_down must be strictly ascending (positive down)")
        object.__setattr__(self, "z_down", z)
        object.__setattr__(self, "c", c)

    def c_at(self, z_down):
        """Sound speed at depth. Clamped outside the profile, never extrapolated."""
        return np.interp(np.asarray(z_down, dtype=float), self.z_down, self.c)

    def path_average(self, z1_down, z2_down) -> float:
        """Harmonic mean of c between two depths: the mean of slowness.

        Travel time is distance / c, so the average preserving travel time
        averages 1/c. The arithmetic mean would bias every range.
        """
        lo, hi = sorted((float(z1_down), float(z2_down)))
        if hi - lo < 1e-9:
            return float(self.c_at(lo))
        zz = np.linspace(lo, hi, _QUAD_POINTS)
        return float((hi - lo) / np.trapezoid(1.0 / self.c_at(zz), zz))

    @classmethod
    def constant(cls, c: float, z_max: float = 3000.0) -> "SoundSpeedProfile":
        return cls(np.array([0.0, float(z_max)]), np.array([float(c), float(c)]),
                   f"constant:{c:g}")

    @classmethod
    def from_ts(cls, z_down, temperature_c, salinity_psu, lat, lon, source):
        """Build c(z) from in-situ temperature and practical salinity (TEOS-10)."""
        import gsw

        z = np.asarray(z_down, dtype=float)
        t = np.asarray(temperature_c, dtype=float)
        sp = np.asarray(salinity_psu, dtype=float)
        p = gsw.p_from_z(-z, lat)                    # gsw wants height, negative down
        sa = gsw.SA_from_SP(sp, p, lon, lat)
        ct = gsw.CT_from_t(sa, t, p)
        return cls(z, np.asarray(gsw.sound_speed(sa, ct, p), dtype=float), source)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/physics/test_sound_speed_profile.py -q`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add requirements.txt paper/src/dclde_soundspeed/soundspeed/ tests/physics/test_sound_speed_profile.py
git commit -m "Add SoundSpeedProfile with an explicit depth sign convention

The repo geometry frame is negative-down and oceanography is positive-down;
mixing them is silent and corrupts every downstream number, so the conversion
is a named, tested pair rather than an inline minus sign. Path averaging is
harmonic because travel time averages slowness, not speed."
```

---

### Task 2: The `SoundSpeedModel` interface, `ConstantC` and `PathAverage`

**Files:**
- Create: `src/dclde_localization/physics/sound_speed_models.py`
- Modify: `paper/src/dclde_soundspeed/soundspeed/__init__.py`
- Test: `tests/physics/test_sound_speed_models.py`

**Interfaces:**
- Consumes: `SoundSpeedProfile`, `to_down` from Task 1; `forward.travel_time`, `forward.DEFAULT_C`.
- Produces:
  - `SoundSpeedModel` — Protocol with `name: str` and `travel_time(sx, sy, sz, px, py, pz) -> np.ndarray`. Positional signature matches `forward.travel_time` minus its trailing `c`, so the two are directly comparable. All coordinates are in the **repo frame** (negative down).
  - `ConstantC(c: float = DEFAULT_C)` — `name` is `f"constant_c:{c:g}"`
  - `PathAverage(profile: SoundSpeedProfile)` — `name` is `f"path_average:{profile.source}"`

- [ ] **Step 1: Write the failing tests**

```python
# tests/physics/test_sound_speed_models.py
import numpy as np
import pytest

from dclde_localization.physics import forward
from dclde_localization.physics.forward import DEFAULT_C
from dclde_localization.physics.sound_speed_models import ConstantC, PathAverage
from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile

PX = np.array([12000.0, 20000.0, 35000.0, 60000.0])
PY = np.array([18000.0, 19000.0, 21000.0, 24000.0])
PZ = np.array([-80.0, -220.0, -640.0, -1500.0])
SRC = (37000.0, 22000.0, -30.0)


def test_constant_c_is_identical_to_the_frozen_forward_model():
    # This is the contract that keeps one definition of the physics. If it ever
    # fails, either forward.py changed (it is frozen) or ConstantC drifted.
    got = ConstantC(DEFAULT_C).travel_time(*SRC, PX, PY, PZ)
    want = forward.travel_time(*SRC, PX, PY, PZ, DEFAULT_C)
    assert np.array_equal(got, want)


def test_constant_c_matches_forward_at_a_non_default_speed():
    got = ConstantC(1440.0).travel_time(*SRC, PX, PY, PZ)
    want = forward.travel_time(*SRC, PX, PY, PZ, 1440.0)
    assert np.array_equal(got, want)


def test_path_average_with_a_constant_profile_reduces_to_constant_c():
    # PathAverage adds a depth-dependent speed; with no depth dependence it must
    # collapse exactly onto the null model, or comparisons between them are
    # measuring the implementation rather than the physics.
    pa = PathAverage(SoundSpeedProfile.constant(DEFAULT_C))
    got = pa.travel_time(*SRC, PX, PY, PZ)
    want = ConstantC(DEFAULT_C).travel_time(*SRC, PX, PY, PZ)
    assert np.allclose(got, want, rtol=1e-12, atol=0.0)


def test_path_average_differs_per_channel_under_a_real_profile():
    # The whole point: c_eff depends on BOTH endpoint depths, so two channels at
    # different depths get different effective speeds from the same source.
    prof = SoundSpeedProfile(
        np.array([0.0, 100.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0]),
        np.array([1491.0, 1480.0, 1478.0, 1477.0, 1479.0, 1484.0, 1490.0]),
        "oregon-nov")
    tt = PathAverage(prof).travel_time(*SRC, PX, PY, PZ)
    ref = ConstantC(DEFAULT_C).travel_time(*SRC, PX, PY, PZ)
    delta = tt - ref
    assert np.ptp(delta) > 1e-3, "effective speed should vary across channel depths"


def test_path_average_depends_on_source_depth():
    # c_eff(z_s, z_c) cannot be precomputed per channel. This test is why the
    # interface takes positions rather than an array of speeds.
    prof = SoundSpeedProfile(np.array([0.0, 1000.0]), np.array([1400.0, 1600.0]), "steep")
    pa = PathAverage(prof)
    shallow = pa.travel_time(37000.0, 22000.0, -10.0, PX, PY, PZ)
    deep = pa.travel_time(37000.0, 22000.0, -400.0, PX, PY, PZ)
    geom_only = (forward.travel_time(37000.0, 22000.0, -400.0, PX, PY, PZ, DEFAULT_C)
                 - forward.travel_time(37000.0, 22000.0, -10.0, PX, PY, PZ, DEFAULT_C))
    assert not np.allclose(deep - shallow, geom_only, rtol=1e-6)


def test_names_identify_the_model_and_its_profile():
    assert ConstantC(1480.0).name == "constant_c:1480"
    assert PathAverage(SoundSpeedProfile.constant(1480.0)).name == "path_average:constant:1480"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/physics/test_sound_speed_models.py -q`
Expected: `ModuleNotFoundError: No module named 'dclde_localization.physics.sound_speed_models'`

- [ ] **Step 3: Write the implementation**

```python
# src/dclde_localization/physics/sound_speed_models.py
"""Propagation models, all sharing one interface.

`c` cannot be precomputed per channel and handed to forward.travel_time: the
path average between source depth and channel depth depends on the source depth,
which is the unknown being solved for. So this is a model object, and its
interface takes positions rather than speeds.

All coordinates are in the REPO frame (negative down). Conversion to the
profile's positive-down frame happens here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from dclde_localization.physics.forward import DEFAULT_C, travel_time as _straight_ray
from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile, to_down


@runtime_checkable
class SoundSpeedModel(Protocol):
    name: str

    def travel_time(self, sx, sy, sz, px, py, pz) -> np.ndarray:
        """Travel time from source (sx, sy, sz) to each pick point."""
        ...


@dataclass(frozen=True)
class ConstantC:
    """The published baseline: one speed everywhere, straight rays.

    Delegates to the frozen forward.py so there is exactly one definition of
    this arithmetic in the codebase, enforced by test_models.py.
    """

    c: float = DEFAULT_C

    @property
    def name(self) -> str:
        return f"constant_c:{self.c:g}"

    def travel_time(self, sx, sy, sz, px, py, pz) -> np.ndarray:
        return _straight_ray(sx, sy, sz, px, py, pz, self.c)


@dataclass(frozen=True)
class PathAverage:
    """Straight geometric path, but traversed at the depth-averaged speed.

    Fermat's principle makes travel time stationary with respect to path
    perturbation, so to first order the refraction correction IS the change in
    path-averaged speed; the bending itself is second order. RayTrace (Task 3)
    measures what is left over.
    """

    profile: SoundSpeedProfile

    @property
    def name(self) -> str:
        return f"path_average:{self.profile.source}"

    def travel_time(self, sx, sy, sz, px, py, pz) -> np.ndarray:
        px = np.asarray(px, dtype=float)
        py = np.asarray(py, dtype=float)
        pz = np.asarray(pz, dtype=float)
        dist = np.sqrt((px - sx) ** 2 + (py - sy) ** 2 + (pz - sz) ** 2)
        zs_down = float(to_down(sz))
        zc_down = to_down(pz)
        c_eff = np.array([self.profile.path_average(zs_down, z) for z in np.atleast_1d(zc_down)])
        return dist / c_eff.reshape(np.shape(dist)) if np.shape(dist) else dist / c_eff[0]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/physics/test_sound_speed_models.py -q`
Expected: 6 passed

- [ ] **Step 5: Re-export from the package**

Add to `paper/src/dclde_soundspeed/soundspeed/__init__.py`:

```python
from dclde_localization.physics.sound_speed_models import ConstantC, PathAverage, SoundSpeedModel

__all__ = ["SoundSpeedProfile", "to_down", "to_repo",
           "SoundSpeedModel", "ConstantC", "PathAverage"]
```

- [ ] **Step 6: Confirm the frozen module is untouched**

Run: `git diff --exit-code src/dclde_localization/physics/forward.py && cd viz && npm test`
Expected: no diff, and the JavaScript parity suite passes (49 tests)

- [ ] **Step 7: Commit**

```bash
git add paper/src/dclde_soundspeed/soundspeed/ tests/physics/test_sound_speed_models.py
git commit -m "Add the SoundSpeedModel interface with ConstantC and PathAverage

ConstantC delegates to the frozen forward.py and a test asserts bit equality,
so there stays exactly one definition of the physics rather than two that drift.
The interface takes positions rather than an array of speeds because the path
average depends on source depth, which is the unknown being solved for."
```

---

### Task 3: Ray tracer and the `RayTrace` model

**Files:**
- Create: `paper/src/dclde_soundspeed/soundspeed/raytrace.py`
- Test: `paper/tests/test_raytrace.py`
- Reference: `paper/scripts/recon_propagation_regime.py` holds a working, self-tested implementation of this arithmetic — promote it rather than rewrite it.

**Interfaces:**
- Consumes: `SoundSpeedProfile`, `to_down` (Task 1); `ConstantC` (Task 2) for tests.
- Produces:
  - `trace(profile, z1_down, z2_down, p) -> tuple[float, float] | None` — (range, time) for ray parameter `p`; `None` if the ray turns before spanning the depths.
  - `max_direct_range(profile, z1_down, z2_down) -> float`
  - `direct_time(profile, z1_down, z2_down, target_range) -> float | None`
  - `RayTrace(profile, fallback="path_average")` — a `SoundSpeedModel`. Where no direct ray exists it falls back to `PathAverage` **and records which picks fell back** in `RayTrace.last_fallback_fraction`.

- [ ] **Step 1: Write the failing tests**

```python
# paper/tests/test_raytrace.py
import numpy as np
import pytest

from dclde_localization.physics.forward import DEFAULT_C
from dclde_localization.physics.sound_speed_models import ConstantC
from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile
from dclde_soundspeed.soundspeed.raytrace import (
    RayTrace, direct_time, max_direct_range, trace)

OREGON = SoundSpeedProfile(
    np.array([0.0, 100.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0]),
    np.array([1491.0, 1480.0, 1478.0, 1477.0, 1479.0, 1484.0, 1490.0]),
    "oregon-nov")
FLAT = SoundSpeedProfile.constant(DEFAULT_C)


@pytest.mark.parametrize("z1,z2,rng", [(30.0, 100.0, 5000.0), (30.0, 1575.0, 30000.0),
                                       (100.0, 500.0, 10000.0)])
def test_constant_profile_reproduces_the_straight_line_time(z1, z2, rng):
    # The tracer must collapse onto simple geometry when there is no gradient.
    # Without this, any difference it reports could be its own quadrature error.
    got = direct_time(FLAT, z1, z2, rng)
    want = float(np.hypot(rng, z2 - z1) / DEFAULT_C)
    assert got is not None
    assert got == pytest.approx(want, rel=1e-9)


def test_downward_refraction_limits_how_far_a_direct_ray_reaches():
    # The finding this whole module exists to express: with c falling from the
    # surface, a shallow source cannot reach a shallow receiver at range.
    assert max_direct_range(OREGON, 30.0, 100.0) == pytest.approx(1400.0, abs=300.0)
    assert max_direct_range(OREGON, 30.0, 1575.0) == pytest.approx(15900.0, abs=1500.0)


def test_no_direct_ray_beyond_the_maximum_range():
    mx = max_direct_range(OREGON, 30.0, 500.0)
    assert direct_time(OREGON, 30.0, 500.0, mx * 0.8) is not None
    assert direct_time(OREGON, 30.0, 500.0, mx * 1.5) is None


def test_constant_profile_has_no_range_limit_at_these_scales():
    # An iso-speed ocean does not refract, so nothing is shadowed. This guards
    # against the range limit being an artefact of the tracer rather than physics.
    assert max_direct_range(FLAT, 30.0, 500.0) > 1e6


def test_trace_returns_none_when_the_ray_turns():
    assert trace(OREGON, 30.0, 1575.0, 1.0 / 1000.0) is None


def test_raytrace_model_reduces_to_constant_c_on_a_flat_profile():
    px = np.array([12000.0, 35000.0])
    py = np.array([18000.0, 21000.0])
    pz = np.array([-220.0, -640.0])
    got = RayTrace(FLAT).travel_time(20000.0, 19000.0, -30.0, px, py, pz)
    want = ConstantC(DEFAULT_C).travel_time(20000.0, 19000.0, -30.0, px, py, pz)
    assert np.allclose(got, want, rtol=1e-8)


def test_raytrace_records_what_fraction_of_picks_had_no_direct_ray():
    # A model that silently substitutes a different physics for some picks would
    # make its own results uninterpretable. The fraction must be observable.
    px = np.array([12000.0, 60000.0])
    py = np.array([18000.0, 24000.0])
    pz = np.array([-100.0, -1500.0])
    m = RayTrace(OREGON)
    m.travel_time(37000.0, 22000.0, -30.0, px, py, pz)
    assert 0.0 <= m.last_fallback_fraction <= 1.0
    assert m.last_fallback_fraction > 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest paper/tests/test_raytrace.py -q`
Expected: `ModuleNotFoundError: ... soundspeed.raytrace`

- [ ] **Step 3: Write the implementation**

Promote the tracer from `paper/scripts/recon_propagation_regime.py`, changing only its profile source (module constants → a `SoundSpeedProfile` argument) and adding the model wrapper:

```python
# paper/src/dclde_soundspeed/soundspeed/raytrace.py
"""Refracted rays through a depth-dependent profile.

Within a layer of constant gradient the ray is a circular arc, so range and
travel time are closed-form and the grazing limit stays finite. A quadrature in
depth diverges there and silently reports "no path" for rays that do exist --
this cost an hour during reconnaissance, so the closed form is not an
optimisation, it is the correctness fix.

Depths here are POSITIVE DOWN (profile frame). The model wrapper at the bottom
converts from the repo frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from dclde_localization.physics.sound_speed_models import PathAverage
from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile, to_down


def _layers(profile: SoundSpeedProfile, z1: float, z2: float):
    edges = sorted({z1, z2} | {float(e) for e in profile.z_down if z1 < e < z2})
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        ca, cb = float(profile.c_at(a)), float(profile.c_at(b))
        out.append((a, b, ca, cb, (cb - ca) / (b - a)))
    return out


def trace(profile, z1_down, z2_down, p):
    """Range and travel time for ray parameter p = cos(theta)/c, or None if it turns."""
    lo, hi = sorted((float(z1_down), float(z2_down)))
    rng = t = 0.0
    for za, zb, ca, cb, g in _layers(profile, lo, hi):
        sa2, sb2 = 1.0 - (p * ca) ** 2, 1.0 - (p * cb) ** 2
        if sa2 < 0.0 or sb2 < 0.0:
            return None
        sa, sb = np.sqrt(sa2), np.sqrt(sb2)
        if abs(g) < 1e-12:
            if sa < 1e-15:
                return None
            rng += p * ca * (zb - za) / sa
            t += (zb - za) / (ca * sa)
        else:
            rng += abs((sa - sb) / (p * g))
            t += abs(np.log((cb * (1.0 + sa)) / (ca * (1.0 + sb))) / g)
    return rng, t


def _p_grazing(profile, z1_down, z2_down) -> float:
    lo, hi = sorted((float(z1_down), float(z2_down)))
    return 1.0 / float(np.max(profile.c_at(np.linspace(lo, hi, 2001))))


def max_direct_range(profile, z1_down, z2_down) -> float:
    """Longest range a direct, water-borne ray can span between two depths."""
    got = trace(profile, z1_down, z2_down, _p_grazing(profile, z1_down, z2_down) * (1 - 1e-15))
    return float(got[0]) if got else 0.0


def direct_time(profile, z1_down, z2_down, target_range):
    """Travel time of the direct eigenray, or None if none reaches target_range."""
    target_range = float(target_range)
    if max_direct_range(profile, z1_down, z2_down) < target_range:
        return None
    pmax = _p_grazing(profile, z1_down, z2_down)

    def miss(p):
        got = trace(profile, z1_down, z2_down, p)
        return np.inf if got is None else got[0] - target_range

    lo = pmax * 1e-9
    if miss(lo) > 0:
        return None
    # Range is violently sensitive to p near grazing; interpolating across the
    # bracket costs ~1 ms, the size of the effect being measured. Bisect.
    p = brentq(miss, lo, pmax * (1 - 1e-15), xtol=1e-20, rtol=8.9e-16, maxiter=300)
    return float(trace(profile, z1_down, z2_down, p)[1])


@dataclass
class RayTrace:
    """Refracted direct ray, falling back to path-average where none exists.

    The fallback is not a detail to hide: where it fires, this model is not
    describing the propagation that actually happened, and the fraction is
    reported so results can say so.
    """

    profile: SoundSpeedProfile
    last_fallback_fraction: float = field(default=0.0, init=False)

    @property
    def name(self) -> str:
        return f"raytrace:{self.profile.source}"

    def travel_time(self, sx, sy, sz, px, py, pz) -> np.ndarray:
        px = np.asarray(px, dtype=float)
        py = np.asarray(py, dtype=float)
        pz = np.asarray(pz, dtype=float)
        horiz = np.sqrt((px - sx) ** 2 + (py - sy) ** 2)
        zs, zc = float(to_down(sz)), to_down(pz)
        fallback = PathAverage(self.profile)
        out = np.empty(horiz.shape, dtype=float)
        n_fell_back = 0
        flat_h, flat_z = horiz.ravel(), np.broadcast_to(zc, horiz.shape).ravel()
        for i, (h, z) in enumerate(zip(flat_h, flat_z)):
            t = direct_time(self.profile, zs, float(z), float(h))
            if t is None:
                n_fell_back += 1
                t = float(fallback.travel_time(sx, sy, sz,
                                               np.array([px.ravel()[i]]),
                                               np.array([py.ravel()[i]]),
                                               np.array([pz.ravel()[i]]))[0])
            out.ravel()[i] = t
        self.last_fallback_fraction = n_fell_back / max(1, flat_h.size)
        return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest paper/tests/test_raytrace.py -q`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add paper/src/dclde_soundspeed/soundspeed/raytrace.py paper/tests/test_raytrace.py
git commit -m "Add refracted ray tracing and the RayTrace model

Closed-form circular arcs per constant-gradient layer, because a quadrature in
depth diverges at the grazing limit and reports 'no path' for rays that exist.
RayTrace publishes the fraction of picks with no direct ray rather than
silently substituting path-average physics for them."
```

---

### Task 4: Multi-basin estimator and the comparison contract

**Files:**
- Create: `src/dclde_localization/physics/solve.py`
- Test: `tests/physics/test_solve.py`

**Interfaces:**
- Consumes: `SoundSpeedModel` (Task 2); `forward.best_t0`, `forward.misfit_value`.
- Produces:
  - `Basin(x, y, z, c, t0, misfit)` — frozen dataclass.
  - `Solution(basins, n_picks, model_name)` — `basins` ranked best-first; `.best -> Basin`; `.separation_m -> float | None` (distance between the two best basins, `None` if only one).
  - `solve(px, py, pz, obs_t, model, *, sz=-30.0, objective="l1", bounds=None, free_z=False, n_coarse=161) -> Solution`
  - `Comparison(n, n_stable, n_flipped, flip_fraction, stable_delta_m)` — frozen dataclass; `stable_delta_m` is an `np.ndarray`.
  - `compare(a: dict[str, Solution], b: dict[str, Solution], *, flip_threshold_m=500.0) -> Comparison`

- [ ] **Step 1: Write the failing tests**

```python
# tests/physics/test_solve.py
import numpy as np
import pytest

from dclde_localization.physics.forward import DEFAULT_C
from dclde_localization.physics.solve import Basin, Solution, compare, solve
from dclde_localization.physics.sound_speed_models import ConstantC

MODEL = ConstantC(DEFAULT_C)


def _straight_cable(n=40, x0=10000.0, x1=60000.0, y=20000.0, depth=-300.0):
    return (np.linspace(x0, x1, n), np.full(n, y), np.full(n, depth))


def _synthesise(src, cable, model=MODEL, t0=7.5):
    px, py, pz = cable
    return model.travel_time(*src, px, py, pz) + t0


def test_recovers_a_known_source_from_noiseless_arrivals():
    cable = _straight_cable()
    truth = (35000.0, 26000.0, -30.0)
    sol = solve(*cable, _synthesise(truth, cable), MODEL, sz=-30.0)
    assert np.hypot(sol.best.x - truth[0], sol.best.y - truth[1]) < 60.0


def test_returns_both_lobes_for_a_single_straight_cable():
    # A straight cable cannot tell which side the source is on. An estimator
    # that returns one point has thrown that away, and every ablation built on
    # it would report basin flips as small position changes.
    cable = _straight_cable()
    truth = (35000.0, 26000.0, -30.0)
    sol = solve(*cable, _synthesise(truth, cable), MODEL, sz=-30.0)
    assert len(sol.basins) >= 2
    sep = sol.separation_m
    assert sep is not None and sep > 8000.0
    mirrored_y = 2 * 20000.0 - truth[1]
    ys = sorted(b.y for b in sol.basins[:2])
    assert min(abs(ys[0] - mirrored_y), abs(ys[1] - mirrored_y)) < 600.0


def test_the_two_lobes_fit_almost_equally_well():
    cable = _straight_cable()
    sol = solve(*cable, _synthesise((35000.0, 26000.0, -30.0), cable), MODEL, sz=-30.0)
    a, b = sol.basins[0].misfit, sol.basins[1].misfit
    assert b / max(a, 1e-12) < 1.15


def test_two_cables_resolve_the_ambiguity():
    # With a second cable off-axis the mirror lobe stops fitting, so the gap
    # between the best two basins should widen sharply.
    px1, py1, pz1 = _straight_cable()
    px2 = np.linspace(20000.0, 50000.0, 30)
    py2 = np.linspace(38000.0, 41000.0, 30)
    pz2 = np.full(30, -700.0)
    px, py, pz = (np.r_[px1, px2], np.r_[py1, py2], np.r_[pz1, pz2])
    obs = MODEL.travel_time(35000.0, 26000.0, -30.0, px, py, pz) + 7.5
    sol = solve(px, py, pz, obs, MODEL, sz=-30.0)
    assert sol.basins[1].misfit / max(sol.basins[0].misfit, 1e-12) > 3.0


def test_t0_is_recovered_and_is_shared_across_all_picks():
    cable = _straight_cable()
    sol = solve(*cable, _synthesise((35000.0, 26000.0, -30.0), cable, t0=12.25), MODEL, sz=-30.0)
    assert sol.best.t0 == pytest.approx(12.25, abs=0.05)


def test_compare_separates_basin_flips_from_position_deltas():
    # The reporting contract: a call that jumped to the mirror lobe must never
    # be averaged into a median position change.
    def sol_at(x, y):
        return Solution(basins=(Basin(x=x, y=y, z=-30.0, c=DEFAULT_C, t0=0.0, misfit=1.0),),
                        n_picks=20, model_name="test")

    a = {"c1": sol_at(0.0, 0.0), "c2": sol_at(0.0, 0.0), "c3": sol_at(0.0, 0.0)}
    b = {"c1": sol_at(80.0, 0.0), "c2": sol_at(120.0, 0.0), "c3": sol_at(9000.0, 0.0)}
    cmp = compare(a, b, flip_threshold_m=500.0)
    assert cmp.n == 3
    assert cmp.n_flipped == 1
    assert cmp.n_stable == 2
    assert np.median(cmp.stable_delta_m) == pytest.approx(100.0)
    assert 9000.0 not in cmp.stable_delta_m


def test_compare_ignores_calls_missing_from_either_side():
    def sol_at(x):
        return Solution(basins=(Basin(x=x, y=0.0, z=-30.0, c=DEFAULT_C, t0=0.0, misfit=1.0),),
                        n_picks=20, model_name="test")

    cmp = compare({"a": sol_at(0.0), "b": sol_at(0.0)}, {"a": sol_at(10.0)})
    assert cmp.n == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/physics/test_solve.py -q`
Expected: `ModuleNotFoundError: No module named 'dclde_localization.physics.solve'`

- [ ] **Step 3: Write the implementation**

```python
# src/dclde_localization/physics/solve.py
"""Localization estimator that returns every misfit basin, not a point.

The objective is bimodal for a single cable: the source could be on either side.
A local optimiser seeded from a coarse grid lands in whichever lobe the grid
happened to favour, and a small parameter change can flip it. If an ablation
reports "positions moved a median of 80 m" while some calls jumped kilometres to
the mirror lobe, the median has hidden the only interesting thing that happened.

So solve() returns ranked basins, and compare() reports basin-stable deltas and
flip counts as two separate numbers. Merging them is prohibited by this
contract, not by convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from dclde_localization.physics.forward import DEFAULT_C, best_t0, misfit_value

DEFAULT_SOURCE_Z = -30.0
MIN_BASIN_SEPARATION_M = 1500.0
MAX_BASINS = 4


@dataclass(frozen=True)
class Basin:
    x: float
    y: float
    z: float
    c: float
    t0: float
    misfit: float


@dataclass(frozen=True)
class Solution:
    basins: tuple[Basin, ...]
    n_picks: int
    model_name: str

    @property
    def best(self) -> Basin:
        return self.basins[0]

    @property
    def separation_m(self) -> float | None:
        if len(self.basins) < 2:
            return None
        a, b = self.basins[0], self.basins[1]
        return float(np.hypot(a.x - b.x, a.y - b.y))


def _misfit_at(x, y, sz, px, py, pz, obs_t, model, objective):
    tt = model.travel_time(x, y, sz, px, py, pz)
    return misfit_value(obs_t, tt, objective), float(best_t0(obs_t, tt, objective))


def _local_minima(field, xs, ys):
    """Grid cells lower than all eight neighbours, as (misfit, x, y), best first."""
    f = field
    out = []
    for iy in range(1, f.shape[0] - 1):
        for ix in range(1, f.shape[1] - 1):
            w = f[iy - 1:iy + 2, ix - 1:ix + 2]
            if f[iy, ix] <= w.min():
                out.append((float(f[iy, ix]), float(xs[ix]), float(ys[iy])))
    return sorted(out)


def solve(px, py, pz, obs_t, model, *, sz=DEFAULT_SOURCE_Z, objective="l1",
          bounds=None, free_z=False, n_coarse=161) -> Solution:
    px, py, pz, obs_t = (np.asarray(v, dtype=float) for v in (px, py, pz, obs_t))
    if bounds is None:
        pad = 45_000.0
        bounds = (px.min() - pad, px.max() + pad, py.min() - pad, py.max() + pad)
    x0, x1, y0, y1 = bounds

    xs = np.linspace(x0, x1, n_coarse)
    ys = np.linspace(y0, y1, n_coarse)
    gx, gy = np.meshgrid(xs, ys)
    field = np.empty(gx.shape, dtype=float)
    for iy in range(gx.shape[0]):
        for ix in range(gx.shape[1]):
            field[iy, ix] = _misfit_at(gx[iy, ix], gy[iy, ix], sz,
                                       px, py, pz, obs_t, model, objective)[0]

    seeds, basins = _local_minima(field, xs, ys), []
    for _, sx, sy in seeds:
        if any(np.hypot(sx - b[0], sy - b[1]) < MIN_BASIN_SEPARATION_M for b in
               [(bb.x, bb.y) for bb in basins]):
            continue
        p0 = [sx, sy, sz] if free_z else [sx, sy]

        def obj(p):
            z = p[2] if free_z else sz
            return _misfit_at(p[0], p[1], z, px, py, pz, obs_t, model, objective)[0]

        res = minimize(obj, p0, method="Nelder-Mead",
                       options={"xatol": 0.5, "fatol": 1e-9, "maxiter": 4000})
        zx = float(res.x[2]) if free_z else sz
        mis, t0 = _misfit_at(res.x[0], res.x[1], zx, px, py, pz, obs_t, model, objective)
        basins.append(Basin(x=float(res.x[0]), y=float(res.x[1]), z=zx,
                            c=getattr(model, "c", DEFAULT_C), t0=t0, misfit=float(mis)))
        if len(basins) >= MAX_BASINS:
            break

    basins.sort(key=lambda b: b.misfit)
    return Solution(basins=tuple(basins), n_picks=int(px.size), model_name=model.name)


@dataclass(frozen=True)
class Comparison:
    n: int
    n_stable: int
    n_flipped: int
    flip_fraction: float
    stable_delta_m: np.ndarray


def compare(a: dict, b: dict, *, flip_threshold_m: float = 500.0) -> Comparison:
    """Position change between two solution sets, flips reported separately."""
    shared = sorted(set(a) & set(b))
    deltas, flips = [], 0
    for k in shared:
        pa, pb = a[k].best, b[k].best
        d = float(np.hypot(pa.x - pb.x, pa.y - pb.y))
        if d > flip_threshold_m:
            flips += 1
        else:
            deltas.append(d)
    n = len(shared)
    return Comparison(n=n, n_stable=n - flips, n_flipped=flips,
                      flip_fraction=(flips / n if n else 0.0),
                      stable_delta_m=np.asarray(deltas, dtype=float))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/physics/test_solve.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/dclde_localization/physics/solve.py tests/physics/test_solve.py
git commit -m "Add a multi-basin estimator and the flip-vs-delta reporting contract

The single-cable objective is bimodal, so returning a point discards the
ambiguity and lets a mirror-lobe flip masquerade as a small position change in
every downstream ablation. solve() returns ranked basins; compare() reports
basin-stable deltas and flip counts as two numbers that cannot be merged."
```

---

### Task 5: Experiment runner with the self-consistency guard

**Files:**
- Create: `paper/src/dclde_soundspeed/experiments/__init__.py`
- Create: `paper/src/dclde_soundspeed/experiments/runner.py`
- Create: `paper/configs/example.yaml`
- Test: `paper/tests/test_runner.py`

**Interfaces:**
- Consumes: `ConstantC`, `PathAverage` (Task 2); `RayTrace` (Task 3); `SoundSpeedProfile` (Task 1).
- Produces:
  - `ExperimentConfig` — frozen dataclass with `name, truth_model, inversion_model, objective, source_z, batches, cables, min_picks, free_z, allow_same_model, notes`.
  - `load_config(path) -> ExperimentConfig` — raises `SelfConsistencyError` when truth and inversion models are the same and `allow_same_model` is not explicitly true.
  - `SelfConsistencyError(ValueError)`
  - `build_model(spec: dict) -> SoundSpeedModel` — from `{kind: constant_c|path_average|raytrace, c?: float, profile?: str}`.
  - `write_results(df, name, out_dir="paper/results") -> Path` — writes CSV with a provenance header comment.

- [ ] **Step 1: Write the failing tests**

```python
# paper/tests/test_runner.py
import textwrap

import pytest

from dclde_soundspeed.experiments.runner import (
    SelfConsistencyError, build_model, load_config)


def _write(tmp_path, body):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_guard_rejects_identical_models(tmp_path):
    # The failure mode this whole guard exists for: simulate and invert with the
    # same physics and truth comes back by construction, measuring nothing.
    # It must be impossible by accident, not merely discouraged.
    cfg = _write(tmp_path, """
        name: bad
        truth_model: {kind: constant_c, c: 1480.0}
        inversion_model: {kind: constant_c, c: 1480.0}
        """)
    with pytest.raises(SelfConsistencyError, match="same model"):
        load_config(cfg)


def test_guard_can_be_overridden_only_explicitly(tmp_path):
    cfg = _write(tmp_path, """
        name: null-check
        truth_model: {kind: constant_c, c: 1480.0}
        inversion_model: {kind: constant_c, c: 1480.0}
        allow_same_model: true
        notes: deliberate null experiment
        """)
    got = load_config(cfg)
    assert got.allow_same_model is True


def test_accepts_differing_models(tmp_path):
    cfg = _write(tmp_path, """
        name: free-c
        truth_model: {kind: constant_c, c: 1480.0}
        inversion_model: {kind: constant_c, c: 1440.0}
        """)
    got = load_config(cfg)
    assert got.truth_model.name != got.inversion_model.name


def test_defaults_match_the_published_baseline(tmp_path):
    cfg = _write(tmp_path, """
        name: d
        truth_model: {kind: constant_c, c: 1480.0}
        inversion_model: {kind: path_average, profile: oregon_nov}
        """)
    got = load_config(cfg)
    assert got.source_z == -30.0
    assert got.objective == "l1"
    assert got.min_picks == 8


def test_build_model_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown model kind"):
        build_model({"kind": "telepathy"})


def test_build_model_rejects_a_missing_profile():
    with pytest.raises(ValueError, match="profile"):
        build_model({"kind": "path_average"})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest paper/tests/test_runner.py -q`
Expected: `ModuleNotFoundError: ... experiments.runner`

- [ ] **Step 3: Write the implementation**

```python
# paper/src/dclde_soundspeed/experiments/__init__.py
"""Config-driven numerical experiments for the sound-speed research."""
```

```python
# paper/src/dclde_soundspeed/experiments/runner.py
"""YAML config -> experiment, with the self-consistency guard.

The way this research fails silently is by simulating and inverting with the
same physics: truth comes back by construction and the experiment measures its
own conditioning rather than anything about the ocean. truth_model and
inversion_model are therefore separate REQUIRED keys, neither defaulting to the
other, and identical models are refused unless a config says allow_same_model
in so many words.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dclde_localization.physics.forward import DEFAULT_C, OBJECTIVES
from dclde_localization.physics.sound_speed_models import ConstantC, PathAverage
from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile
from dclde_soundspeed.soundspeed.raytrace import RayTrace

RESULTS_DIR = Path("paper/results")

# Indicative Oregon margin, November. Superseded by measured/reanalysis
# profiles in Tasks 8-9; kept as a named fallback so early experiments run.
_OREGON_NOV = SoundSpeedProfile(
    np.array([0.0, 100.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0]),
    np.array([1491.0, 1480.0, 1478.0, 1477.0, 1479.0, 1484.0, 1490.0]),
    "oregon_nov_indicative")

PROFILES = {"oregon_nov": _OREGON_NOV}


class SelfConsistencyError(ValueError):
    """Raised when an experiment would invert with the physics it simulated."""


def build_model(spec: dict):
    kind = spec.get("kind")
    if kind == "constant_c":
        return ConstantC(float(spec.get("c", DEFAULT_C)))
    if kind in ("path_average", "raytrace"):
        key = spec.get("profile")
        if key not in PROFILES:
            raise ValueError(f"unknown or missing profile {key!r}; have {sorted(PROFILES)}")
        prof = PROFILES[key]
        return PathAverage(prof) if kind == "path_average" else RayTrace(prof)
    raise ValueError(f"unknown model kind {kind!r}")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    truth_model: object
    inversion_model: object
    objective: str = "l1"
    source_z: float = -30.0
    batches: tuple = ()
    cables: tuple = ("North", "South")
    min_picks: int = 8
    free_z: bool = False
    allow_same_model: bool = False
    notes: str = ""
    raw: dict = field(default_factory=dict)


def load_config(path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("name", "truth_model", "inversion_model"):
        if key not in raw:
            raise ValueError(f"config is missing required key {key!r}")

    truth, inversion = build_model(raw["truth_model"]), build_model(raw["inversion_model"])
    allow_same = bool(raw.get("allow_same_model", False))
    if truth.name == inversion.name and not allow_same:
        raise SelfConsistencyError(
            f"{raw['name']}: truth and inversion use the same model ({truth.name}). "
            "Inverting with the physics you simulated recovers truth by construction. "
            "Set allow_same_model: true only for a deliberate null experiment.")

    objective = raw.get("objective", "l1")
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")

    return ExperimentConfig(
        name=raw["name"], truth_model=truth, inversion_model=inversion,
        objective=objective, source_z=float(raw.get("source_z", -30.0)),
        batches=tuple(raw.get("batches", ())),
        cables=tuple(raw.get("cables", ("North", "South"))),
        min_picks=int(raw.get("min_picks", 8)),
        free_z=bool(raw.get("free_z", False)),
        allow_same_model=allow_same, notes=raw.get("notes", ""), raw=raw)


def write_results(df: pd.DataFrame, name: str, out_dir=RESULTS_DIR) -> Path:
    """Write a result table with a provenance header, so a CSV explains itself."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.csv"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(f"# experiment: {name}\n# generated: {stamp}\n# rows: {len(df)}\n")
        df.to_csv(fh, index=False)
    return path
```

```yaml
# paper/configs/example.yaml
name: example
notes: >
  Template. truth_model and inversion_model are both required and must differ
  unless allow_same_model is set, so a self-consistent experiment cannot happen
  by accident.
truth_model: {kind: path_average, profile: oregon_nov}
inversion_model: {kind: constant_c, c: 1480.0}
objective: l1
source_z: -30.0
min_picks: 8
cables: [North, South]
free_z: false
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest paper/tests/test_runner.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add paper/src/dclde_soundspeed/experiments/ paper/configs/ paper/tests/test_runner.py
git commit -m "Add the experiment runner and the self-consistency guard

truth_model and inversion_model are separate required keys and identical
models are refused unless a config says allow_same_model explicitly. Inverting
with the physics you simulated recovers truth by construction, which is the
mistake most likely to invalidate everything downstream, so it is refused by
the loader rather than left to discipline."
```

---

### Task 6: Re-downloadable provenance for the reference data

**Files:**
- Create: `paper/scripts/fetch_reference_data.py`
- Create: `paper/data/reference/PROVENANCE.md` (tracked; `data/` is gitignored, so force-add this one file)
- Test: `paper/tests/test_fetch_reference.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `colon_safe(path: str) -> str` — the `:`→`_` rename
  - `verify_checksum(path, expected_sha256) -> bool`
  - `write_manifest(rows, path)` / `read_manifest(path)`

**Why now:** the 3.6 GB of reference data is temporary on this machine by agreement, and deletion must cost a teammate one command, not a reconstruction. Build this before the experiments depend on the data, not after it is deleted.

- [ ] **Step 1: Write the failing tests**

```python
# paper/tests/test_fetch_reference.py
import hashlib

import pytest

from paper.scripts.fetch_reference_data import colon_safe, read_manifest, verify_checksum, write_manifest


def test_colon_rename_matches_the_dataset_timestamp_normalisation():
    # The dataset writes the same timestamp three ways; underscores are the
    # canonical one (src/dclde_localization/physics/picks.py already normalises to it), and NTFS forbids
    # colons outright, so this is both a convention and a platform requirement.
    assert colon_safe("data/detections/peaks_tp_North_2021-11-04_02:00:02_ipi3.nc") == \
        "data/detections/peaks_tp_North_2021-11-04_02_00_02_ipi3.nc"


def test_colon_rename_leaves_clean_paths_alone():
    assert colon_safe("data/north_DAS_multicoord.csv") == "data/north_DAS_multicoord.csv"


def test_checksum_detects_a_corrupt_file(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    good = hashlib.sha256(b"hello").hexdigest()
    assert verify_checksum(p, good) is True
    assert verify_checksum(p, hashlib.sha256(b"goodbye").hexdigest()) is False


def test_manifest_round_trips(tmp_path):
    rows = [{"upstream_path": "a/b:c.nc", "local_path": "a/b_c.nc",
             "renamed": True, "bytes": 12},
            {"upstream_path": "x.csv", "local_path": "x.csv", "renamed": False, "bytes": 3}]
    p = tmp_path / "m.csv"
    write_manifest(rows, p)
    got = read_manifest(p)
    assert got == rows
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest paper/tests/test_fetch_reference.py -q`
Expected: `ModuleNotFoundError: No module named 'paper.scripts.fetch_reference_data'`
(If `scripts/` lacks `__init__.py`, add an empty one so the test can import it.)

- [ ] **Step 3: Write the implementation**

```python
# paper/scripts/fetch_reference_data.py
"""Re-download the reference data this research compares against.

Two artifacts, both public, ~3.6 GB together. They are deliberately NOT
committed (data/ is gitignored) and are deleted from local machines once the
experiments consuming them have produced committed result tables. This script
plus the recorded checksums is the reproducibility guarantee.

Windows cannot check either out directly: 32 of the repo's 45 files and all 180
archive members carry colons in their names, which NTFS forbids. Git aborts the
checkout wholesale. Everything is materialised through a ':' -> '_' rename with
a manifest recording every mapping.

    uv run python paper/scripts/fetch_reference_data.py --dest paper/data/reference
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import zipfile
from pathlib import Path

REPO_URL = "https://github.com/Ocean-Data-Lab/Goestchel_JASA_2025b.git"
ZENODO_URL = "https://zenodo.org/records/21220346/files/associations.zip?download=1"
MANIFEST_FIELDS = ["upstream_path", "local_path", "renamed", "bytes"]


def colon_safe(path: str) -> str:
    """':' -> '_', matching the underscore normalisation src/dclde_localization/physics/picks.py uses."""
    return path.replace(":", "_")


def verify_checksum(path, expected_sha256: str) -> bool:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == expected_sha256


def write_manifest(rows, path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_manifest(path):
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return [{"upstream_path": r["upstream_path"], "local_path": r["local_path"],
                 "renamed": r["renamed"] == "True", "bytes": int(r["bytes"])}
                for r in csv.DictReader(fh)]


def materialise_repo(dest: Path) -> list[dict]:
    """Clone bare, then write each blob under a colon-safe name."""
    repo = dest / "Goestchel_JASA_2025b"
    if not (repo / ".git").exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo)], check=False)
    out, rows = repo / "_worktree", []

    def git(*a):
        return subprocess.run(["git", "-C", str(repo), *a],
                              check=True, stdout=subprocess.PIPE).stdout

    for line in git("ls-tree", "-r", "HEAD",
                    "--format=%(objectname) %(path)").decode().splitlines():
        sha, path = line.split(" ", 1)
        safe = colon_safe(path)
        target = out / safe
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git("cat-file", "blob", sha))
        rows.append({"upstream_path": path, "local_path": safe,
                     "renamed": path != safe, "bytes": target.stat().st_size})
    write_manifest(rows, repo / "_worktree_manifest.csv")
    return rows


def materialise_zip(dest: Path) -> list[dict]:
    archive = dest / "associations.zip"
    if not archive.exists():
        subprocess.run(["curl", "-sL", ZENODO_URL, "-o", str(archive)], check=True)
    out, rows = dest / "associations", []
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            safe = colon_safe(info.filename)
            target = out / safe
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src:
                target.write_bytes(src.read())
            rows.append({"upstream_path": info.filename, "local_path": safe,
                         "renamed": info.filename != safe, "bytes": info.file_size})
    write_manifest(rows, dest / "associations_manifest.csv")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default="paper/data/reference", type=Path)
    ap.add_argument("--skip-zenodo", action="store_true",
                    help="skip the 3.4 GB association archive")
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    rows = materialise_repo(args.dest)
    print(f"reference repo: {len(rows)} files, "
          f"{sum(r['renamed'] for r in rows)} renamed, "
          f"{sum(r['bytes'] for r in rows) / 1e6:.0f} MB")
    if not args.skip_zenodo:
        rows = materialise_zip(args.dest)
        print(f"associations: {len(rows)} files, "
              f"{sum(r['bytes'] for r in rows) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
```

```markdown
<!-- paper/data/reference/PROVENANCE.md -->
# Reference data provenance

Not committed (`data/` is gitignored) and deleted from local machines once the
experiments consuming it have produced committed result tables. Restore with:

    uv run python paper/scripts/fetch_reference_data.py --dest paper/data/reference

| Artifact | Source | Size | Licence |
|---|---|---|---|
| Reference implementation | `github.com/Ocean-Data-Lab/Goestchel_JASA_2025b` | 204 MB | GPL-3.0 |
| Association output, 5 batches | Zenodo record `21220346` | 3.67 GB | see record |

Paper: Goestchel, Q., Wilcock, W. S. D. & Abadi, S. (2026). *JASA* 160(1), 135–146.
doi:10.1121/10.0044257

**Colons in filenames.** 32 of the repo's 45 files and all 180 archive members
carry colons (`..._2021-11-04_02:00:02_...`). NTFS forbids them, so `git clone`
fetches the objects and then aborts the checkout wholesale. The fetch script
renames `:` to `_` and writes a manifest of every mapping.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest paper/tests/test_fetch_reference.py -q`
Expected: 4 passed

- [ ] **Step 5: Verify the script reproduces what is already on disk**

Run: `uv run python paper/scripts/fetch_reference_data.py --dest paper/data/reference --skip-zenodo`
Expected: `reference repo: 45 files, 32 renamed, 204 MB`

- [ ] **Step 6: Commit**

```bash
git add paper/scripts/fetch_reference_data.py paper/tests/test_fetch_reference.py
git add -f paper/data/reference/PROVENANCE.md
git commit -m "Add a re-downloadable provenance script for the reference data

The 3.6 GB is temporary on local machines by agreement, so deleting it has to
cost a teammate one command rather than a reconstruction. Also encodes the
colon-rename that Windows forces: git aborts the checkout wholesale on the 32
colon-named files rather than skipping them."
```

---

### Task 7: Baseline reproduction

**Files:**
- Create: `paper/src/dclde_soundspeed/experiments/repro.py`
- Create: `paper/configs/repro_baseline.yaml`
- Test: `paper/tests/test_repro.py`
- Output: `paper/results/repro_baseline.csv`

**Interfaces:**
- Consumes: `solve`, `Solution` (Task 4); `ConstantC` (Task 2); `write_results` (Task 5); `dclde_localization.physics.picks`.
- Produces:
  - `parse_baseline_csv(path) -> pd.DataFrame` — columns `utc, sensor, call_type, x_local, y_local, z_local, wrms, deltax`
  - `join_to_calls(baseline: pd.DataFrame, picks: pd.DataFrame, tol_s=90.0) -> pd.DataFrame` — adds a `call` column by matching `utc` to window start plus fitted `t0`; unmatched rows keep `call = None` and are **counted, never dropped silently**
  - `run_repro(picks, baseline, cables, *, source_z=-30.0) -> pd.DataFrame`

- [ ] **Step 1: Write the failing tests**

```python
# paper/tests/test_repro.py
import numpy as np
import pandas as pd
import pytest

from dclde_soundspeed.experiments.repro import join_to_calls, parse_baseline_csv

BASELINE_HEAD = (
    "utc,sensor,call_type,x_local,y_local,z_local,x_utm,y_utm,z_utm,lat,lon,wrms,deltax\n"
    "2021-11-04 02:00:08,hf,pair,37000.55,22344.41,-30.0,68607.0,5007455.9,-30.0,"
    "45.0888,-128.4819,0.1259,8.796\n"
    "2021-11-04 02:00:51,hf,pair,37011.21,22382.38,-30.0,68596.3,5007493.9,-30.0,"
    "45.0891,-128.4821,0.1271,10.809\n")


def test_parse_keeps_local_coordinates_and_discards_published_latlon(tmp_path):
    # The published lat/lon place calls ~270 km west of the cable while the local
    # coordinates are sound. Carrying both invites using the wrong one.
    p = tmp_path / "b.csv"
    p.write_text(BASELINE_HEAD, encoding="utf-8")
    df = parse_baseline_csv(p)
    assert list(df.columns) == ["utc", "sensor", "call_type", "x_local", "y_local",
                                "z_local", "wrms", "deltax"]
    assert df["x_local"].iloc[0] == pytest.approx(37000.55)
    assert pd.api.types.is_datetime64_any_dtype(df["utc"])


def test_join_matches_a_call_within_its_window(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text(BASELINE_HEAD, encoding="utf-8")
    baseline = parse_baseline_csv(p)
    picks = pd.DataFrame({
        "call": ["2021-11-04_02_00_02/1"] * 3 + ["2021-11-04_02_00_02/2"] * 3,
        "window": ["2021-11-04_02_00_02"] * 6,
        "time": [6.0, 6.4, 6.8, 49.0, 49.4, 49.8]})
    got = join_to_calls(baseline, picks, tol_s=90.0)
    assert got["call"].notna().sum() == 2
    assert got["call"].iloc[0] == "2021-11-04_02_00_02/1"


def test_join_reports_unmatched_rows_rather_than_dropping_them(tmp_path):
    # A join that silently drops rows turns a reproduction failure into a
    # smaller, cleaner-looking sample.
    p = tmp_path / "b.csv"
    p.write_text(BASELINE_HEAD, encoding="utf-8")
    baseline = parse_baseline_csv(p)
    picks = pd.DataFrame({"call": ["2021-11-09_09_09_09/1"], "window": ["2021-11-09_09_09_09"],
                          "time": [1.0]})
    got = join_to_calls(baseline, picks, tol_s=90.0)
    assert len(got) == len(baseline)
    assert got["call"].isna().all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest paper/tests/test_repro.py -q`
Expected: `ModuleNotFoundError: ... experiments.repro`

- [ ] **Step 3: Write the implementation**

```python
# paper/src/dclde_soundspeed/experiments/repro.py
"""Reproduce the published baseline and compare to it per call.

Their released CSV carries a `utc` column (the call's absolute time) but no call
id, so the join goes through the window timestamp encoded in the annotation call
key plus the arrival time within that window. Rows that do not match are kept
with call = None and counted: a join that silently drops rows would turn a
reproduction failure into a smaller, cleaner-looking sample.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dclde_localization.physics.solve import solve
from dclde_localization.physics.sound_speed_models import ConstantC

KEEP = ["utc", "sensor", "call_type", "x_local", "y_local", "z_local", "wrms", "deltax"]


def parse_baseline_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["utc"] = pd.to_datetime(df["utc"], utc=True)
    return df[KEEP].copy()


def _window_start(window: str) -> pd.Timestamp:
    date, hh, mm, ss = window.rsplit("_", 3)
    return pd.Timestamp(f"{date} {hh}:{mm}:{ss}", tz="UTC")


def join_to_calls(baseline: pd.DataFrame, picks: pd.DataFrame, tol_s: float = 90.0):
    """Attach a `call` key to each baseline row by absolute time."""
    first = picks.groupby("call").agg(window=("window", "first"), t=("time", "min"))
    first["abs_t"] = [_window_start(w) + pd.Timedelta(seconds=float(t))
                      for w, t in zip(first["window"], first["t"])]

    out = baseline.copy()
    keys, deltas = [], []
    for stamp in out["utc"]:
        gap = (first["abs_t"] - stamp).abs()
        i = gap.to_numpy().argmin()
        if gap.iloc[i].total_seconds() <= tol_s:
            keys.append(first.index[i])
            deltas.append(gap.iloc[i].total_seconds())
        else:
            keys.append(None)
            deltas.append(np.nan)
    out["call"] = keys
    out["join_gap_s"] = deltas
    return out


def run_repro(picks: pd.DataFrame, baseline: pd.DataFrame, *, source_z=-30.0,
              min_picks=8) -> pd.DataFrame:
    """Solve each joined call with our estimator and tabulate the difference."""
    joined = join_to_calls(baseline, picks)
    model = ConstantC()
    rows = []
    for _, r in joined.iterrows():
        if r["call"] is None:
            rows.append({"call": None, "matched": False})
            continue
        sub = picks[picks["call"] == r["call"]]
        if len(sub) < min_picks:
            rows.append({"call": r["call"], "matched": True, "n_picks": len(sub),
                         "excluded": "too few picks"})
            continue
        sol = solve(sub["px"].to_numpy(), sub["py"].to_numpy(), sub["pz"].to_numpy(),
                    sub["time"].to_numpy(), model, sz=source_z)
        rows.append({
            "call": r["call"], "matched": True, "n_picks": len(sub), "excluded": "",
            "their_x": r["x_local"], "their_y": r["y_local"], "their_wrms": r["wrms"],
            "our_x": sol.best.x, "our_y": sol.best.y, "our_misfit": sol.best.misfit,
            "delta_m": float(np.hypot(sol.best.x - r["x_local"],
                                      sol.best.y - r["y_local"])),
            "n_basins": len(sol.basins), "separation_m": sol.separation_m,
            "join_gap_s": r["join_gap_s"]})
    return pd.DataFrame(rows)
```

```yaml
# paper/configs/repro_baseline.yaml
name: repro_baseline
notes: >
  Reproduce Goestchel et al. batch 1. Our estimator against their published
  localizations, joined per call through the utc timestamp. No pass/fail
  threshold is set in advance: a fabricated threshold is only a way of not
  looking. Report the distribution and investigate the tail individually.
truth_model: {kind: constant_c, c: 1480.0}
inversion_model: {kind: path_average, profile: oregon_nov}
allow_same_model: false
objective: l1
source_z: -30.0
min_picks: 8
batches: [1]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest paper/tests/test_repro.py -q`
Expected: 3 passed

- [ ] **Step 5: Run the reproduction and commit the result table**

```bash
uv run python -c "
import pandas as pd
from dclde_localization.physics.geometry import load_cables
from dclde_soundspeed.experiments.repro import parse_baseline_csv, run_repro
from dclde_soundspeed.experiments.runner import write_results
from dclde_localization.physics.picks import attach_xyz, load_picks
cables = load_cables('fixtures')
picks = attach_xyz(load_picks(), cables)
base = parse_baseline_csv('paper/data/reference/Goestchel_JASA_2025b/_worktree/batch1_localizations_with_coords.csv')
df = run_repro(picks, base)
print(f'baseline rows {len(base)}, matched {int(df.matched.sum())}, solved {df.delta_m.notna().sum()}')
print(df.delta_m.describe(percentiles=[.5,.9]).round(1).to_string())
print(write_results(df, 'repro_baseline'))
"
```

Record the matched fraction and the delta distribution in the commit message. **If fewer than half the rows match, stop and report** — the join is wrong and every later comparison inherits it.

- [ ] **Step 6: Commit**

```bash
git add paper/src/dclde_soundspeed/experiments/repro.py paper/tests/test_repro.py \
        paper/configs/repro_baseline.yaml \
        paper/results/repro_baseline.csv
git commit -m "Reproduce the published baseline per call

Their CSV has a utc timestamp but no call id, so the join goes through the
window timestamp plus arrival time. Unmatched rows are kept and counted rather
than dropped, because a silent drop turns a reproduction failure into a
smaller, cleaner-looking sample."
```

---

### Task 8: OOI ERDDAP client with QARTOD filtering

**Files:**
- Create: `paper/src/dclde_soundspeed/soundspeed/erddap.py`
- Test: `paper/tests/test_erddap.py`

**Interfaces:**
- Consumes: `SoundSpeedProfile.from_ts` (Task 1).
- Produces:
  - `DATASETS: dict[str, str]` — the verified dataset ids (see below)
  - `build_url(dataset_id, start, end) -> str`
  - `filter_qartod(df, max_flag=2) -> pd.DataFrame`
  - `fetch_profile(dataset_id, start, end, lat, lon, *, session=None) -> SoundSpeedProfile`

**Verified 2026-09-15 for Nov 2021** — do not substitute other site codes without re-checking:

| Key | Dataset ID | Status |
|---|---|---|
| `ce04ospd_deep` | `ooi-ce04ospd-dp01b-01-ctdpfl105` | **Profiling**, 97–575 m, 129,177 samples |
| `rs01sbpd_deep` | `ooi-rs01sbpd-dp01a-01-ctdpfl104` | No data (404) |
| `rs01sbps_shallow` | `ooi-rs01sbps-sf01a-2a-ctdpfa102` | Parked at ~193 m |

Variables carry the suffix `_profiler_depth_enabled`, e.g. `sea_water_temperature_profiler_depth_enabled`, each with a `_qc_agg` companion. QARTOD flags: 1 pass, 2 not-evaluated, 3 suspect, 4 fail, 9 missing.

- [ ] **Step 1: Write the failing tests**

```python
# paper/tests/test_erddap.py
import numpy as np
import pandas as pd
import pytest

from dclde_soundspeed.soundspeed.erddap import DATASETS, build_url, filter_qartod

T = "sea_water_temperature_profiler_depth_enabled"
S = "sea_water_practical_salinity_profiler_depth_enabled"


def test_url_requests_the_qc_columns_alongside_the_data():
    # Fetching values without their QC flags is how you build a garbage profile:
    # the raw ranges include salinity 33929 PSU and temperature 61 C.
    url = build_url(DATASETS["ce04ospd_deep"], "2021-11-01", "2021-11-06")
    assert "ooi-ce04ospd-dp01b-01-ctdpfl105.csv" in url
    assert f"{T}_qc_agg" in url and f"{S}_qc_agg" in url
    assert "time%3E=2021-11-01" in url or "time>=2021-11-01" in url


def test_qartod_filter_rejects_suspect_and_fail_flags():
    df = pd.DataFrame({
        "z": [-100.0, -200.0, -300.0, -400.0],
        T: [8.0, 9.0, 61.0, 7.5], S: [33.8, 33.9, 33.9, 33929.0],
        f"{T}_qc_agg": [1, 2, 4, 1], f"{S}_qc_agg": [1, 2, 1, 3]})
    got = filter_qartod(df, max_flag=2)
    assert len(got) == 2
    assert 61.0 not in got[T].to_numpy()
    assert 33929.0 not in got[S].to_numpy()


def test_qartod_filter_keeps_everything_when_all_flags_pass():
    df = pd.DataFrame({"z": [-100.0], T: [8.0], S: [33.8],
                       f"{T}_qc_agg": [1], f"{S}_qc_agg": [1]})
    assert len(filter_qartod(df)) == 1


def test_known_gaps_are_recorded_in_the_module():
    # These were measured, not assumed. Anyone reaching for the Slope Base deep
    # profiler should find the note before spending an afternoon on a 404.
    from dclde_soundspeed.soundspeed import erddap
    assert "rs01sbpd_deep" in erddap.DATASETS
    assert "no data" in erddap.NOV_2021_STATUS["rs01sbpd_deep"].lower()
    assert "park" in erddap.NOV_2021_STATUS["rs01sbps_shallow"].lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest paper/tests/test_erddap.py -q`
Expected: `ModuleNotFoundError: ... soundspeed.erddap`

- [ ] **Step 3: Write the implementation**

```python
# paper/src/dclde_soundspeed/soundspeed/erddap.py
"""OOI ERDDAP client for in-situ temperature and salinity.

Coverage for November 2021 is much thinner than the feasibility note assumed;
NOV_2021_STATUS records what was actually measured on 2026-09-15 so nobody
repeats the search. Exactly one instrument near the cables was profiling.

The raw variable ranges are polluted (salinity to 33929 PSU, temperature to
61 C). Always filter on the QARTOD aggregate flags.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile

BASE = "https://erddap.dataexplorer.oceanobservatories.org/erddap/tabledap"

DATASETS = {
    "ce04ospd_deep": "ooi-ce04ospd-dp01b-01-ctdpfl105",
    "rs01sbpd_deep": "ooi-rs01sbpd-dp01a-01-ctdpfl104",
    "rs01sbps_shallow": "ooi-rs01sbps-sf01a-2a-ctdpfa102",
    "ce04osps_shallow": "ooi-ce04osps-sf01b-2a-ctdpfa107",
}

NOV_2021_STATUS = {
    "ce04ospd_deep": "Profiling: 129,177 samples, z -575..-97 m, 479 distinct depths",
    "rs01sbpd_deep": "No data: HTTP 404, zero rows for 25 Oct - 15 Nov 2021",
    "rs01sbps_shallow": "Parked at z -195..-189 m over five days: a point, not a profile",
    "ce04osps_shallow": "Parked at z -197..-192 m",
}

TEMP = "sea_water_temperature_profiler_depth_enabled"
SALT = "sea_water_practical_salinity_profiler_depth_enabled"
QC_PASS_MAX = 2  # 1 pass, 2 not evaluated; 3 suspect and 4 fail are rejected


def build_url(dataset_id: str, start: str, end: str) -> str:
    cols = ",".join(["time", "z", TEMP, SALT, f"{TEMP}_qc_agg", f"{SALT}_qc_agg"])
    return (f"{BASE}/{dataset_id}.csv?{cols}"
            f"&time%3E={start}T00:00:00Z&time%3C={end}T00:00:00Z")


def filter_qartod(df: pd.DataFrame, max_flag: int = QC_PASS_MAX) -> pd.DataFrame:
    keep = pd.Series(True, index=df.index)
    for col in (c for c in df.columns if c.endswith("_qc_agg")):
        keep &= df[col].astype(float) <= max_flag
    return df.loc[keep].copy()


def fetch_profile(dataset_key: str, start: str, end: str, lat: float, lon: float,
                  *, session=None, n_bins: int = 60) -> SoundSpeedProfile:
    """Fetch, QC-filter and depth-bin one instrument into a sound-speed profile."""
    dataset_id = DATASETS[dataset_key]
    get = (session or requests).get
    resp = get(build_url(dataset_id, start, end), timeout=180)
    if resp.status_code == 404:
        raise ValueError(f"{dataset_key} ({dataset_id}) returned no rows for {start}..{end}. "
                         f"Known status: {NOV_2021_STATUS.get(dataset_key, 'unknown')}")
    resp.raise_for_status()
    # Row 0 is the header, row 1 is the units row ERDDAP inserts.
    df = pd.read_csv(io.StringIO(resp.text), skiprows=[1])
    df = filter_qartod(df).dropna(subset=["z", TEMP, SALT])
    if df.empty:
        raise ValueError(f"{dataset_key}: every row failed QARTOD filtering")

    z_down = -df["z"].to_numpy(dtype=float)
    bins = np.linspace(z_down.min(), z_down.max(), n_bins + 1)
    idx = np.clip(np.digitize(z_down, bins) - 1, 0, n_bins - 1)
    grouped = pd.DataFrame({"bin": idx, "z": z_down,
                            "t": df[TEMP].to_numpy(float),
                            "s": df[SALT].to_numpy(float)}).groupby("bin").mean()
    return SoundSpeedProfile.from_ts(
        grouped["z"].to_numpy(), grouped["t"].to_numpy(), grouped["s"].to_numpy(),
        lat, lon, source=f"ooi:{dataset_key}:{start}..{end}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest paper/tests/test_erddap.py -q`
Expected: 4 passed

- [ ] **Step 5: Fetch the real profile and confirm its coverage**

```bash
uv run python -c "
from dclde_soundspeed.soundspeed.erddap import fetch_profile
p = fetch_profile('ce04ospd_deep', '2021-11-01', '2021-11-06', 44.37, -124.95)
print(p.source)
print(f'{len(p.z_down)} bins, {p.z_down.min():.0f}..{p.z_down.max():.0f} m, '
      f'c {p.c.min():.1f}..{p.c.max():.1f} m/s')
"
```

Expected: roughly 97–575 m, sound speeds in the 1475–1490 m/s band. **If `c` falls outside 1450–1550 m/s, stop** — that indicates swapped temperature and salinity arguments, which `from_ts` cannot detect.

- [ ] **Step 6: Commit**

```bash
git add paper/src/dclde_soundspeed/soundspeed/erddap.py paper/tests/test_erddap.py
git commit -m "Add the OOI ERDDAP client with QARTOD filtering

Records what November 2021 coverage actually is, because the feasibility note
claimed full-depth paired profilers and the measurement says one instrument
over 97-575 m, with the Slope Base deep profiler returning nothing at all.
Raw ranges include salinity 33929 PSU, so the QC filter is not optional."
```

---

### Task 9: Reanalysis profile and in-situ validation

**Files:**
- Create: `paper/src/dclde_soundspeed/soundspeed/reanalysis.py`
- Test: `paper/tests/test_reanalysis.py`
- Output: `paper/results/profile_validation.csv`

**Interfaces:**
- Consumes: `SoundSpeedProfile` (Task 1), `fetch_profile` (Task 8).
- Produces:
  - `fetch_hycom_profile(lat, lon, date, *, session=None) -> SoundSpeedProfile`
  - `compare_profiles(a, b, z_lo, z_hi) -> pd.DataFrame` — per-depth difference over the overlapping band
  - `validation_summary(df) -> dict` — `median_abs_diff_ms`, `p90_abs_diff_ms`, `n`

**Why this shape:** OOI covers 97–575 m of a 0–1575 m problem, so reanalysis is the base profile and in-situ is the validator, not the other way round. The disagreement over the overlapping band is the uncertainty on every downstream result and must be reported, not absorbed.

- [ ] **Step 1: Write the failing tests**

```python
# paper/tests/test_reanalysis.py
import numpy as np
import pytest

from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile
from dclde_soundspeed.soundspeed.reanalysis import compare_profiles, validation_summary


def test_comparison_is_restricted_to_the_overlapping_depth_band():
    # Comparing outside the overlap would silently compare a measurement with a
    # clamped extrapolation and call the agreement good.
    a = SoundSpeedProfile(np.array([0.0, 500.0, 1500.0]),
                          np.array([1490.0, 1477.0, 1484.0]), "reanalysis")
    b = SoundSpeedProfile(np.array([100.0, 300.0, 575.0]),
                          np.array([1481.0, 1478.5, 1477.2]), "in-situ")
    df = compare_profiles(a, b, 100.0, 575.0)
    assert df["z_down"].min() >= 100.0
    assert df["z_down"].max() <= 575.0


def test_identical_profiles_show_zero_disagreement():
    p = SoundSpeedProfile(np.array([100.0, 300.0, 575.0]),
                          np.array([1481.0, 1478.5, 1477.2]), "x")
    s = validation_summary(compare_profiles(p, p, 100.0, 575.0))
    assert s["median_abs_diff_ms"] == pytest.approx(0.0, abs=1e-9)


def test_summary_reports_its_denominator():
    a = SoundSpeedProfile(np.array([100.0, 575.0]), np.array([1481.0, 1477.0]), "a")
    b = SoundSpeedProfile(np.array([100.0, 575.0]), np.array([1483.0, 1479.0]), "b")
    s = validation_summary(compare_profiles(a, b, 100.0, 575.0))
    assert s["n"] > 0
    assert s["median_abs_diff_ms"] == pytest.approx(2.0, abs=1e-6)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest paper/tests/test_reanalysis.py -q`
Expected: `ModuleNotFoundError: ... soundspeed.reanalysis`

- [ ] **Step 3: Write the implementation**

```python
# paper/src/dclde_soundspeed/soundspeed/reanalysis.py
"""Full-depth sound-speed profiles from reanalysis, validated against in-situ.

OOI covers 97-575 m of a 0-1575 m problem (see erddap.NOV_2021_STATUS), so this
is the base profile and the in-situ data is the validator. The disagreement over
the overlapping band is the uncertainty carried by every downstream result.

HYCOM GOFS 3.1 is the primary source: open NCSS subsetting, no account. WOA23
monthly climatology is the independent cross-check -- it is a climatological
November, not this November, and results must say so. GLORYS12V1 is better but
needs a Copernicus account and is deliberately off the critical path.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile

HYCOM_NCSS = ("https://ncss.hycom.org/thredds/ncss/GLBy0.08/expt_93.0/ts3z")
_QUAD = 400


def fetch_hycom_profile(lat: float, lon: float, date: str, *, session=None,
                        timeout: int = 300) -> SoundSpeedProfile:
    """One full-depth T/S profile from HYCOM GOFS 3.1, converted to c(z)."""
    params = {
        "var": "water_temp,salinity", "latitude": lat, "longitude": lon,
        "time": f"{date}T12:00:00Z", "vertCoordinate": "all",
        "accept": "csv", "addLatLon": "true",
    }
    get = (session or requests).get
    resp = get(HYCOM_NCSS, params=params, timeout=timeout)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    depth_col = next(c for c in df.columns if c.lower().startswith("depth"))
    t_col = next(c for c in df.columns if "water_temp" in c)
    s_col = next(c for c in df.columns if "salinity" in c)
    df = df.dropna(subset=[depth_col, t_col, s_col]).sort_values(depth_col)
    return SoundSpeedProfile.from_ts(
        df[depth_col].to_numpy(float), df[t_col].to_numpy(float),
        df[s_col].to_numpy(float), lat, lon, source=f"hycom:{date}:{lat:.2f},{lon:.2f}")


def compare_profiles(a: SoundSpeedProfile, b: SoundSpeedProfile,
                     z_lo: float, z_hi: float) -> pd.DataFrame:
    """Per-depth sound-speed difference over the overlapping band only."""
    lo = max(float(z_lo), float(a.z_down.min()), float(b.z_down.min()))
    hi = min(float(z_hi), float(a.z_down.max()), float(b.z_down.max()))
    if hi <= lo:
        raise ValueError(f"profiles do not overlap: {a.source} vs {b.source}")
    zz = np.linspace(lo, hi, _QUAD)
    return pd.DataFrame({"z_down": zz, "c_a": a.c_at(zz), "c_b": b.c_at(zz),
                         "diff_ms": a.c_at(zz) - b.c_at(zz)})


def validation_summary(df: pd.DataFrame) -> dict:
    d = np.abs(df["diff_ms"].to_numpy(float))
    return {"n": int(d.size), "median_abs_diff_ms": float(np.median(d)),
            "p90_abs_diff_ms": float(np.percentile(d, 90)),
            "max_abs_diff_ms": float(d.max())}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest paper/tests/test_reanalysis.py -q`
Expected: 3 passed

- [ ] **Step 5: Run the real validation and commit the table**

```bash
uv run python -c "
import pandas as pd
from dclde_soundspeed.soundspeed.erddap import fetch_profile
from dclde_soundspeed.soundspeed.reanalysis import (
    compare_profiles, fetch_hycom_profile, validation_summary)
from dclde_soundspeed.experiments.runner import write_results
insitu = fetch_profile('ce04ospd_deep', '2021-11-01', '2021-11-06', 44.37, -124.95)
hycom = fetch_hycom_profile(44.37, -124.95, '2021-11-04')
df = compare_profiles(hycom, insitu, 97.0, 575.0)
print(validation_summary(df))
print(write_results(df, 'profile_validation'))
"
```

Put the summary in the commit message. **A median disagreement above ~5 m/s means the reanalysis is not usable as a base profile** — report that rather than proceeding, since it would exceed the effect being chased.

- [ ] **Step 6: Commit**

```bash
git add paper/src/dclde_soundspeed/soundspeed/reanalysis.py paper/tests/test_reanalysis.py \
        paper/results/profile_validation.csv
git commit -m "Add reanalysis profiles validated against the in-situ band

OOI covers 97-575 m of a 0-1575 m problem, so reanalysis is the base and
in-situ is the validator. The disagreement over the overlap is the uncertainty
on every later result, so it is measured and committed rather than assumed
small."
```

---

### Task 10: Normal-mode solver

**Files:**
- Create: `paper/src/dclde_soundspeed/soundspeed/modes.py`
- Test: `paper/tests/test_modes.py`

**Interfaces:**
- Consumes: `SoundSpeedProfile` (Task 1).
- Produces:
  - `Seabed(c, rho, alpha)` — frozen dataclass: sediment sound speed m/s, density ratio, attenuation dB/λ
  - `modes(profile, seabed, freq_hz, water_depth_m, *, nz=2000) -> ModeSet` with `.kr` (horizontal wavenumbers), `.phi` (mode shapes), `.z_down`, `.n_modes`
  - `group_slowness(profile, seabed, freq_hz, water_depth_m, *, dfreq=0.25) -> np.ndarray` — dU/dω per mode, by finite difference
  - `modal_arrival_time(profile, seabed, freq_hz, water_depth_m, range_m, z_src, z_rcv) -> float` — energy-weighted first arrival

**Why it is here and not in phase 2:** a ray-theoretic shadow zone in a waveguide 1.3–7 wavelengths deep is exactly the claim normal modes adjudicate, and Task 12 depends on the answer. `arlpy` and `pyat` both shell out to Acoustics Toolbox Fortran binaries that are not pip-installable and are hostile on Windows, so we solve the Sturm–Liouville eigenproblem directly and validate it against the analytic Pekeris waveguide.

- [ ] **Step 1: Write the failing tests**

```python
# paper/tests/test_modes.py
import numpy as np
import pytest

from dclde_soundspeed.soundspeed.modes import Seabed, modes
from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile

PEKERIS_D = 200.0
PEKERIS_CW = 1500.0
PEKERIS_CB = 1800.0
FREQ = 20.0


def _pekeris_profile():
    return SoundSpeedProfile(np.array([0.0, PEKERIS_D]), np.array([PEKERIS_CW, PEKERIS_CW]),
                             "pekeris")


def test_pekeris_wavenumbers_lie_between_the_two_sound_speeds():
    # Trapped modes must have phase speed between water and seabed speed; this is
    # the defining property of a trapped mode and catches an eigensolver that has
    # picked up leaky or spurious roots.
    w = 2 * np.pi * FREQ
    ms = modes(_pekeris_profile(), Seabed(c=PEKERIS_CB, rho=1.8, alpha=0.0),
               FREQ, PEKERIS_D)
    assert ms.n_modes >= 1
    assert np.all(ms.kr < w / PEKERIS_CW)
    assert np.all(ms.kr > w / PEKERIS_CB)


def test_pekeris_mode_count_matches_the_analytic_estimate():
    # For a Pekeris waveguide the number of trapped modes is
    # floor( 2 f D sqrt(1/cw^2 - 1/cb^2) + 1/2 ) -- a closed form, so this is a
    # real check rather than a restatement of the implementation.
    expected = int(2 * FREQ * PEKERIS_D * np.sqrt(1 / PEKERIS_CW**2 - 1 / PEKERIS_CB**2) + 0.5)
    ms = modes(_pekeris_profile(), Seabed(c=PEKERIS_CB, rho=1.8, alpha=0.0), FREQ, PEKERIS_D)
    assert abs(ms.n_modes - expected) <= 1


def test_mode_shapes_vanish_at_the_pressure_release_surface():
    ms = modes(_pekeris_profile(), Seabed(c=PEKERIS_CB, rho=1.8, alpha=0.0), FREQ, PEKERIS_D)
    assert np.allclose(ms.phi[0, :], 0.0, atol=1e-9)


def test_mode_shapes_are_orthonormal():
    ms = modes(_pekeris_profile(), Seabed(c=PEKERIS_CB, rho=1.8, alpha=0.0), FREQ, PEKERIS_D)
    dz = ms.z_down[1] - ms.z_down[0]
    gram = ms.phi.T @ ms.phi * dz
    assert np.allclose(gram, np.eye(ms.n_modes), atol=5e-2)


def test_deeper_water_traps_more_modes():
    sb = Seabed(c=PEKERIS_CB, rho=1.8, alpha=0.0)
    shallow = modes(_pekeris_profile(), sb, FREQ, PEKERIS_D)
    deep_prof = SoundSpeedProfile(np.array([0.0, 800.0]), np.array([PEKERIS_CW, PEKERIS_CW]),
                                  "pekeris-deep")
    deep = modes(deep_prof, sb, FREQ, 800.0)
    assert deep.n_modes > shallow.n_modes


def test_twenty_hertz_in_shallow_water_traps_very_few_modes():
    # The regime claim in the spec: at 20 Hz in 100-500 m of water the guide is
    # 1.3-7 wavelengths deep, so only a handful of modes exist and ray theory,
    # which needs many, is marginal.
    sb = Seabed(c=1700.0, rho=1.8, alpha=0.0)
    prof = SoundSpeedProfile(np.array([0.0, 150.0]), np.array([1480.0, 1480.0]), "shelf")
    assert modes(prof, sb, 20.0, 150.0).n_modes <= 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest paper/tests/test_modes.py -q`
Expected: `ModuleNotFoundError: ... soundspeed.modes`

- [ ] **Step 3: Write the implementation**

```python
# paper/src/dclde_soundspeed/soundspeed/modes.py
"""Range-independent normal modes for a fluid waveguide over a fluid halfspace.

At 20 Hz, lambda ~ 74 m, and in 100-500 m of water the guide is a few
wavelengths deep. Ray theory -- including the sharp shadow zone it predicts --
is marginal there, and modes are what adjudicate it.

arlpy and pyat both shell out to Acoustics Toolbox Fortran binaries that are not
pip-installable and are hostile on Windows, so the Sturm-Liouville eigenproblem
is solved directly here and validated against the analytic Pekeris waveguide.

Depth-separated equation, pressure-release surface:

    d2 phi/dz2 + (omega^2 / c(z)^2 - kr^2) phi = 0,   phi(0) = 0

Discretised with second-order finite differences. Trapped modes are the
eigenvalues with kr between omega/c_seabed and omega/c_water_min; the halfspace
is imposed as a radiation-like cutoff on that interval rather than modelled
explicitly, which is adequate for counting modes and computing group delay and
is why the Pekeris test tolerates +-1 mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal

from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile


@dataclass(frozen=True)
class Seabed:
    c: float = 1700.0
    rho: float = 1.8
    alpha: float = 0.1


@dataclass(frozen=True)
class ModeSet:
    kr: np.ndarray
    phi: np.ndarray
    z_down: np.ndarray
    freq_hz: float

    @property
    def n_modes(self) -> int:
        return int(self.kr.size)


def modes(profile: SoundSpeedProfile, seabed: Seabed, freq_hz: float,
          water_depth_m: float, *, nz: int = 2000) -> ModeSet:
    omega = 2.0 * np.pi * float(freq_hz)
    z = np.linspace(0.0, float(water_depth_m), nz)
    dz = z[1] - z[0]
    c = profile.c_at(z)

    # Interior nodes only: phi(0) = 0 is the pressure-release surface.
    diag = (omega / c[1:]) ** 2 - 2.0 / dz**2
    off = np.full(diag.size - 1, 1.0 / dz**2)
    vals, vecs = eigh_tridiagonal(diag, off)

    kr2_lo, kr2_hi = (omega / seabed.c) ** 2, (omega / c.min()) ** 2
    keep = (vals > kr2_lo) & (vals < kr2_hi)
    vals, vecs = vals[keep], vecs[:, keep]
    order = np.argsort(vals)[::-1]          # mode 1 first (largest kr)
    vals, vecs = vals[order], vecs[:, order]

    phi = np.vstack([np.zeros((1, vals.size)), vecs])
    norm = np.sqrt(np.sum(phi**2, axis=0) * dz)
    norm[norm == 0] = 1.0
    return ModeSet(kr=np.sqrt(vals), phi=phi / norm, z_down=z, freq_hz=float(freq_hz))


def group_slowness(profile, seabed, freq_hz, water_depth_m, *, dfreq=0.25):
    """dk/domega per mode by central difference: the reciprocal of group speed."""
    lo = modes(profile, seabed, freq_hz - dfreq, water_depth_m)
    hi = modes(profile, seabed, freq_hz + dfreq, water_depth_m)
    n = min(lo.n_modes, hi.n_modes)
    if n == 0:
        return np.array([])
    return (hi.kr[:n] - lo.kr[:n]) / (2.0 * np.pi * 2.0 * dfreq)


def modal_arrival_time(profile, seabed, freq_hz, water_depth_m, range_m,
                       z_src_down, z_rcv_down) -> float:
    """Earliest energy-carrying modal arrival: range times the fastest group slowness.

    An envelope-peak pick is closer to the energy-weighted arrival than to the
    fastest mode, but the fastest mode bounds it, which is what the shadow-zone
    question needs.
    """
    ms = modes(profile, seabed, freq_hz, water_depth_m)
    if ms.n_modes == 0:
        return float("nan")
    sg = group_slowness(profile, seabed, freq_hz, water_depth_m)
    if sg.size == 0:
        return float("nan")
    w_src = np.interp(z_src_down, ms.z_down, np.abs(ms.phi[:, :sg.size]).T.max(axis=0)
                      if sg.size > 1 else np.abs(ms.phi[:, 0]))
    excited = sg[np.isfinite(sg) & (sg > 0)]
    if excited.size == 0 or not np.isfinite(w_src):
        return float("nan")
    return float(range_m * excited.min())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest paper/tests/test_modes.py -q`
Expected: 6 passed

If `test_pekeris_mode_count_matches_the_analytic_estimate` fails by more than one mode, the eigenvalue window is wrong — check the `kr2_lo`/`kr2_hi` bounds before loosening the tolerance. Loosening a test to make it pass is not a fix.

- [ ] **Step 5: Commit**

```bash
git add paper/src/dclde_soundspeed/soundspeed/modes.py paper/tests/test_modes.py
git commit -m "Add a normal-mode solver validated against the Pekeris waveguide

Pulled out of phase 2 because section 7.0 depends on it: a ray-theoretic shadow
zone in a guide a few wavelengths deep is exactly what modes adjudicate. Solved
directly rather than through arlpy or pyat, which shell out to Fortran binaries
that are not pip-installable and are hostile on Windows."
```

---

### Task 11: Experiment §7.0.1 — separate path error from source-depth error

**Files:**
- Create: `paper/src/dclde_soundspeed/experiments/regime.py`
- Create: `paper/configs/regime_depth_vs_path.yaml`
- Test: `paper/tests/test_regime.py`
- Output: `paper/results/regime_depth_vs_path.csv`

**Interfaces:**
- Consumes: `solve` (Task 4); `ConstantC`, `PathAverage` (Task 2); `write_results` (Task 5).
- Produces:
  - `residual_table(picks, model, *, source_z, min_picks=8) -> pd.DataFrame` — one row per pick with `call, residual_s, range_m, channel_z, snr`
  - `stratify(df, range_bins, snr_quartiles=True) -> pd.DataFrame`
  - `depth_sweep(picks, model, z_values, *, min_picks=8) -> pd.DataFrame` — the range trend at each assumed source depth

**The question:** §2.5C found the residual range-trend survives every SNR quartile, but it is confounded with the fixed −30 m source depth, which produces range structure of its own. If sweeping source depth flattens the trend, the signal is depth. If it does not, the signal is the propagation path. §7.4 is uninterpretable until this is answered.

- [ ] **Step 1: Write the failing tests**

```python
# paper/tests/test_regime.py
import numpy as np
import pandas as pd
import pytest

from dclde_soundspeed.experiments.regime import residual_table, stratify
from dclde_localization.physics.sound_speed_models import ConstantC


def _picks_for_a_known_source(n=30, src=(35000.0, 26000.0, -30.0), t0=5.0):
    px = np.linspace(12000.0, 60000.0, n)
    py = np.full(n, 20000.0)
    pz = np.linspace(-80.0, -1500.0, n)
    t = ConstantC().travel_time(*src, px, py, pz) + t0
    return pd.DataFrame({"call": "c1", "px": px, "py": py, "pz": pz,
                         "time": t, "snr": np.linspace(8.0, 25.0, n)})


def test_residuals_are_numerically_zero_for_a_perfectly_matched_model():
    # If the model that generated the arrivals leaves residuals, the estimator
    # or the residual definition is broken, and every stratification built on
    # it is measuring that bug.
    df = residual_table(_picks_for_a_known_source(), ConstantC(), source_z=-30.0, min_picks=8)
    assert np.abs(df["residual_s"]).max() < 1e-3


def test_residual_table_carries_the_axes_the_stratification_needs():
    df = residual_table(_picks_for_a_known_source(), ConstantC(), source_z=-30.0, min_picks=8)
    assert {"call", "residual_s", "range_m", "channel_z", "snr"} <= set(df.columns)
    assert (df["range_m"] > 0).all()


def test_stratify_reports_its_denominator_per_cell():
    df = residual_table(_picks_for_a_known_source(), ConstantC(), source_z=-30.0, min_picks=8)
    out = stratify(df, range_bins=[0, 10_000, 25_000, 100_000])
    assert "n" in out.columns
    assert out["n"].sum() == len(df)


def test_calls_below_the_pick_threshold_are_excluded_not_silently_solved():
    small = _picks_for_a_known_source(n=4)
    df = residual_table(small, ConstantC(), source_z=-30.0, min_picks=8)
    assert df.empty
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest paper/tests/test_regime.py -q`
Expected: `ModuleNotFoundError: ... experiments.regime`

- [ ] **Step 3: Write the implementation**

```python
# paper/src/dclde_soundspeed/experiments/regime.py
"""Section 7.0.1 -- is the range trend in residuals path error or depth error?

Section 2.5C measured a residual trend from -34 ms at short range to +54 ms
beyond 20 km that survives inside every SNR quartile and is strongest at the
HIGHEST SNR, which rules out low-SNR late picks. But it is confounded with the
fixed -30 m source depth, which produces range structure of its own.

Sweep the assumed source depth. If the trend flattens, the signal is depth and
section 7.4 can read it. If it does not, the signal is the propagation path and
the sound-speed programme is aimed at the wrong term.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dclde_localization.physics.forward import best_t0
from dclde_localization.physics.solve import solve


def residual_table(picks: pd.DataFrame, model, *, source_z: float,
                   min_picks: int = 8) -> pd.DataFrame:
    rows = []
    for call, sub in picks.groupby("call"):
        if len(sub) < min_picks:
            continue
        px, py, pz, t = (sub[k].to_numpy(float) for k in ("px", "py", "pz", "time"))
        sol = solve(px, py, pz, t, model, sz=source_z)
        b = sol.best
        tt = model.travel_time(b.x, b.y, source_z, px, py, pz)
        res = t - tt - best_t0(t, tt)
        rows.append(pd.DataFrame({
            "call": call, "residual_s": res,
            "range_m": np.hypot(px - b.x, py - b.y),
            "channel_z": pz, "snr": sub["snr"].to_numpy(float)}))
    return (pd.concat(rows, ignore_index=True) if rows
            else pd.DataFrame(columns=["call", "residual_s", "range_m", "channel_z", "snr"]))


def stratify(df: pd.DataFrame, range_bins, *, snr_quartiles: bool = False) -> pd.DataFrame:
    d = df.copy()
    d["range_bin"] = pd.cut(d["range_m"], range_bins)
    keys = ["range_bin"]
    if snr_quartiles:
        d["snr_q"] = pd.qcut(d["snr"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
        keys.insert(0, "snr_q")
    g = d.groupby(keys, observed=True)["residual_s"]
    return (g.agg(n="size", mean_ms=lambda s: s.mean() * 1e3,
                  std_ms=lambda s: s.std() * 1e3).reset_index())


def depth_sweep(picks: pd.DataFrame, model, z_values, *, min_picks: int = 8,
                range_bins=(0, 5_000, 10_000, 20_000, 30_000, 200_000)) -> pd.DataFrame:
    """Range trend at each assumed source depth. Flat trend => depth explains it."""
    out = []
    for z in z_values:
        df = residual_table(picks, model, source_z=float(z), min_picks=min_picks)
        if df.empty:
            continue
        s = stratify(df, list(range_bins))
        near, far = s.iloc[0]["mean_ms"], s.iloc[-1]["mean_ms"]
        out.append({"source_z": float(z), "n_picks": int(s["n"].sum()),
                    "near_mean_ms": near, "far_mean_ms": far,
                    "range_trend_ms": far - near})
    return pd.DataFrame(out)
```

```yaml
# paper/configs/regime_depth_vs_path.yaml
name: regime_depth_vs_path
notes: >
  Section 7.0.1. Sweep assumed source depth and watch the residual range trend.
  A trend that flattens means the signal is source depth; one that persists
  means it is the propagation path, and the sound-speed programme is aimed at
  the wrong term. This experiment is allowed to conclude the latter.
truth_model: {kind: constant_c, c: 1480.0}
inversion_model: {kind: path_average, profile: oregon_nov}
objective: l1
source_z: -30.0
min_picks: 8
source_z_sweep: [-5.0, -15.0, -30.0, -60.0, -100.0, -150.0, -200.0]
range_bins_m: [0, 5000, 10000, 20000, 30000, 200000]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest paper/tests/test_regime.py -q`
Expected: 4 passed

- [ ] **Step 5: Run the experiment on the real picks and commit the table**

```bash
uv run python -c "
import numpy as np, yaml
from dclde_localization.physics.geometry import load_cables
from dclde_soundspeed.experiments.regime import depth_sweep, residual_table, stratify
from dclde_soundspeed.experiments.runner import write_results
from dclde_localization.physics.sound_speed_models import ConstantC
from dclde_localization.physics.picks import attach_xyz, load_picks
cfg = yaml.safe_load(open('paper/configs/regime_depth_vs_path.yaml'))
picks = attach_xyz(load_picks(), load_cables('fixtures'))
sweep = depth_sweep(picks, ConstantC(), cfg['source_z_sweep'],
                    range_bins=cfg['range_bins_m'])
print(sweep.round(1).to_string(index=False))
print(write_results(sweep, 'regime_depth_vs_path'))
"
```

**Read the output before committing.** If `range_trend_ms` falls close to zero at some source depth, source depth explains the trend and §7.4 is on solid ground. If it stays large at every depth, the propagation path is implicated and that becomes the headline finding — say so in the commit message and in the report.

- [ ] **Step 6: Commit**

```bash
git add paper/src/dclde_soundspeed/experiments/regime.py paper/tests/test_regime.py \
        paper/configs/regime_depth_vs_path.yaml \
        paper/results/regime_depth_vs_path.csv
git commit -m "Separate source-depth error from propagation-path error

The residual range trend survives every SNR quartile but is confounded with the
fixed -30 m source assumption, which produces range structure of its own.
Sweeping assumed depth separates them, and section 7.4 cannot be read until
this says which signal it is looking at."
```

---

### Task 12: Experiment §7.0.3 — is the shadow zone real at 20 Hz?

**Files:**
- Modify: `paper/src/dclde_soundspeed/experiments/regime.py` (add `shadow_zone_table`)
- Create: `paper/configs/regime_shadow_zone.yaml`
- Modify: `paper/tests/test_regime.py`
- Output: `paper/results/regime_shadow_zone.csv`

**Interfaces:**
- Consumes: `max_direct_range` (Task 3); `modes`, `Seabed`, `modal_arrival_time` (Task 10).
- Produces:
  - `shadow_zone_table(profile, seabed_sweep, freq_hz, geometries) -> pd.DataFrame` — per geometry: ray-predicted max direct range, whether modes exist, and the spread across the seabed sweep

**The question:** ray theory says a −30 m source cannot reach a 100 m channel beyond 1.4 km. If trapped modes exist at 20 Hz for that geometry, the shadow zone is a ray artefact and energy arrives anyway — which would mean the ray-theoretic framing of §2.5B overstates the problem. Seabed geoacoustics are unknown, so they are **swept**, and the spread is the uncertainty.

- [ ] **Step 1: Write the failing tests**

```python
# append to paper/tests/test_regime.py
import numpy as np

from dclde_soundspeed.experiments.regime import shadow_zone_table
from dclde_soundspeed.soundspeed.modes import Seabed
from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile

OREGON = SoundSpeedProfile(
    np.array([0.0, 100.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0]),
    np.array([1491.0, 1480.0, 1478.0, 1477.0, 1479.0, 1484.0, 1490.0]),
    "oregon-nov")
SWEEP = [Seabed(c=1600.0, rho=1.6, alpha=0.1), Seabed(c=1800.0, rho=2.0, alpha=0.3)]


def test_shadow_zone_table_reports_both_the_ray_limit_and_the_mode_count():
    got = shadow_zone_table(OREGON, SWEEP, 20.0,
                            [(30.0, 150.0, 5000.0), (30.0, 500.0, 20000.0)])
    assert {"z_src", "z_rcv", "range_m", "ray_max_direct_m", "n_modes_min",
            "n_modes_max", "ray_says_shadowed"} <= set(got.columns)
    assert got["ray_says_shadowed"].all()


def test_seabed_sweep_is_reported_as_a_spread_not_a_single_value():
    # Seabed geoacoustics are unknown here. Fixing them at invented values and
    # reporting one number would present a guess as a measurement.
    got = shadow_zone_table(OREGON, SWEEP, 20.0, [(30.0, 150.0, 5000.0)])
    assert (got["n_modes_max"] >= got["n_modes_min"]).all()


def test_no_geometries_gives_an_empty_table_rather_than_raising():
    assert shadow_zone_table(OREGON, SWEEP, 20.0, []).empty
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest paper/tests/test_regime.py -q -k shadow`
Expected: `ImportError: cannot import name 'shadow_zone_table'`

- [ ] **Step 3: Write the implementation**

```python
# append to paper/src/dclde_soundspeed/experiments/regime.py
from dclde_soundspeed.soundspeed.modes import Seabed, modes
from dclde_soundspeed.soundspeed.raytrace import max_direct_range


def shadow_zone_table(profile, seabed_sweep, freq_hz, geometries) -> pd.DataFrame:
    """Does ray theory's shadow zone survive a modal treatment at 20 Hz?

    Ray theory says a -30 m source cannot reach a 100 m channel past ~1.4 km.
    If trapped modes exist there, the shadow zone is a ray artefact and energy
    arrives regardless. Seabed geoacoustics are unknown, so they are swept and
    the mode count is reported as a range, never a single invented value.
    """
    rows = []
    for z_src, z_rcv, range_m in geometries:
        ray_max = max_direct_range(profile, z_src, z_rcv)
        counts = [modes(profile, sb, freq_hz, max(z_rcv, 1.0)).n_modes
                  for sb in seabed_sweep]
        rows.append({
            "z_src": float(z_src), "z_rcv": float(z_rcv), "range_m": float(range_m),
            "ray_max_direct_m": float(ray_max),
            "ray_says_shadowed": bool(ray_max < range_m),
            "n_modes_min": int(min(counts)), "n_modes_max": int(max(counts)),
            "seabed_c_min": float(min(sb.c for sb in seabed_sweep)),
            "seabed_c_max": float(max(sb.c for sb in seabed_sweep))})
    return pd.DataFrame(rows)
```

```yaml
# paper/configs/regime_shadow_zone.yaml
name: regime_shadow_zone
notes: >
  Section 7.0.3. Ray theory predicts a shadow zone; at 20 Hz in a guide a few
  wavelengths deep, modes are what adjudicate. Seabed geoacoustics are unknown
  and therefore swept, with the spread reported as the uncertainty.
truth_model: {kind: raytrace, profile: oregon_nov}
inversion_model: {kind: constant_c, c: 1480.0}
freq_hz: 20.0
seabed_sweep:
  - {c: 1550.0, rho: 1.5, alpha: 0.05}
  - {c: 1700.0, rho: 1.8, alpha: 0.15}
  - {c: 1900.0, rho: 2.1, alpha: 0.40}
geometries:
  - [30.0, 100.0, 5000.0]
  - [30.0, 150.0, 10000.0]
  - [30.0, 500.0, 20000.0]
  - [30.0, 1000.0, 30000.0]
  - [30.0, 1575.0, 40000.0]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest paper/tests/test_regime.py -q`
Expected: 7 passed

- [ ] **Step 5: Run the experiment and commit the table**

```bash
uv run python -c "
import yaml, pandas as pd
from dclde_soundspeed.experiments.regime import shadow_zone_table
from dclde_soundspeed.experiments.runner import PROFILES, write_results
from dclde_soundspeed.soundspeed.modes import Seabed
cfg = yaml.safe_load(open('paper/configs/regime_shadow_zone.yaml'))
sweep = [Seabed(**s) for s in cfg['seabed_sweep']]
df = shadow_zone_table(PROFILES['oregon_nov'], sweep, cfg['freq_hz'],
                       [tuple(g) for g in cfg['geometries']])
print(df.to_string(index=False))
print(write_results(df, 'regime_shadow_zone'))
"
```

**Interpretation, to be written into the commit message:** if `n_modes_min >= 1` wherever `ray_says_shadowed` is true, the shadow zone is a ray artefact and §2.5B overstates the problem — energy does arrive, and the remaining question is only whether its *timing* differs from the straight-line model. If mode counts collapse to zero, the ray picture is corroborated and propagation-path modelling becomes the headline.

- [ ] **Step 6: Commit**

```bash
git add paper/src/dclde_soundspeed/experiments/regime.py paper/tests/test_regime.py \
        paper/configs/regime_shadow_zone.yaml \
        paper/results/regime_shadow_zone.csv
git commit -m "Test whether the ray-theoretic shadow zone survives a modal treatment

At 20 Hz in a guide a few wavelengths deep a shadow zone is exactly the kind of
ray prediction modes overturn, so the section 2.5B finding cannot stand on rays
alone. Seabed geoacoustics are unknown and are swept rather than invented."
```

---

### Task 13: Experiments §7.2 and §7.3 — free-`c` and effective-`c` ablations

**Files:**
- Create: `paper/src/dclde_soundspeed/experiments/ablation.py`
- Create: `paper/configs/ablation_free_c.yaml`
- Create: `paper/configs/ablation_effective_c.yaml`
- Test: `paper/tests/test_ablation.py`
- Output: `paper/results/ablation_free_c.csv`, `ablation_effective_c.csv`

**Interfaces:**
- Consumes: `solve`, `compare`, `Comparison` (Task 4); `ConstantC`, `PathAverage` (Task 2); `RayTrace` (Task 3); `write_results` (Task 5).
- Produces:
  - `solve_all(picks, model, *, source_z, min_picks=8) -> dict[str, Solution]`
  - `fit_free_c(picks, *, source_z, c_grid, min_picks=8) -> pd.DataFrame` — per call: best `c`, its misfit, and the fixed-1480 misfit
  - `ablation_report(base: dict, variant: dict, label: str) -> dict` — wraps `compare` and **always** emits basin-stable and flip figures separately

- [ ] **Step 1: Write the failing tests**

```python
# paper/tests/test_ablation.py
import numpy as np
import pandas as pd
import pytest

from dclde_soundspeed.experiments.ablation import ablation_report, fit_free_c, solve_all
from dclde_localization.physics.sound_speed_models import ConstantC


def _picks(n=30, src=(35000.0, 26000.0, -30.0), c=1480.0, t0=5.0, call="c1"):
    px = np.linspace(12000.0, 60000.0, n)
    py = np.full(n, 20000.0)
    pz = np.linspace(-80.0, -1500.0, n)
    t = ConstantC(c).travel_time(*src, px, py, pz) + t0
    return pd.DataFrame({"call": call, "px": px, "py": py, "pz": pz,
                         "time": t, "snr": 15.0})


def test_free_c_recovers_the_speed_that_generated_the_arrivals():
    # The estimator must be able to find c before any ablation using it means
    # anything.
    df = fit_free_c(_picks(c=1455.0), source_z=-30.0,
                    c_grid=np.arange(1430.0, 1510.0, 2.5))
    assert abs(float(df["best_c"].iloc[0]) - 1455.0) <= 5.0


def test_free_c_reduces_or_matches_the_fixed_c_misfit():
    # A free parameter can never fit worse than a fixed one at its own optimum.
    # If it does, the search grid or the objective is wrong.
    df = fit_free_c(_picks(c=1455.0), source_z=-30.0,
                    c_grid=np.arange(1430.0, 1510.0, 2.5))
    assert float(df["best_misfit"].iloc[0]) <= float(df["fixed_misfit"].iloc[0]) + 1e-9


def test_ablation_report_never_merges_flips_into_the_delta_statistic():
    # The reporting contract from spec section 5. A merged median would hide the
    # only interesting failure mode.
    base = solve_all(_picks(call="a"), ConstantC(1480.0), source_z=-30.0)
    variant = solve_all(_picks(call="a"), ConstantC(1479.0), source_z=-30.0)
    rep = ablation_report(base, variant, "tiny-c-change")
    assert {"label", "n", "n_stable", "n_flipped", "flip_fraction",
            "stable_delta_median_m", "stable_delta_p90_m"} <= set(rep)
    assert rep["n"] == rep["n_stable"] + rep["n_flipped"]


def test_solve_all_skips_calls_below_the_pick_threshold():
    small = _picks(n=4, call="tiny")
    assert solve_all(small, ConstantC(), source_z=-30.0, min_picks=8) == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest paper/tests/test_ablation.py -q`
Expected: `ModuleNotFoundError: ... experiments.ablation`

- [ ] **Step 3: Write the implementation**

```python
# paper/src/dclde_soundspeed/experiments/ablation.py
"""Sections 7.2 and 7.3 -- free-c and effective-c ablations.

Expected outcome for 7.2: a free scalar c absorbs most of the effect for L-1,
closing that branch. That does not stop the programme; this fit is the null
model the depth experiment has to beat.

Every report emits basin-stable deltas and basin-flip counts separately. A call
that jumped to the mirror lobe must never be averaged into a median position
change -- see spec section 5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dclde_localization.physics.forward import DEFAULT_C, best_t0, misfit_value
from dclde_localization.physics.solve import compare, solve
from dclde_localization.physics.sound_speed_models import ConstantC


def solve_all(picks: pd.DataFrame, model, *, source_z: float,
              min_picks: int = 8) -> dict:
    out = {}
    for call, sub in picks.groupby("call"):
        if len(sub) < min_picks:
            continue
        out[call] = solve(sub["px"].to_numpy(float), sub["py"].to_numpy(float),
                          sub["pz"].to_numpy(float), sub["time"].to_numpy(float),
                          model, sz=source_z)
    return out


def fit_free_c(picks: pd.DataFrame, *, source_z: float, c_grid,
               min_picks: int = 8) -> pd.DataFrame:
    """Add c as a fifth unknown, by profiling it on a grid."""
    rows = []
    for call, sub in picks.groupby("call"):
        if len(sub) < min_picks:
            continue
        px, py, pz, t = (sub[k].to_numpy(float) for k in ("px", "py", "pz", "time"))
        best = (np.inf, np.nan, np.nan, np.nan)
        for c in np.asarray(c_grid, dtype=float):
            sol = solve(px, py, pz, t, ConstantC(float(c)), sz=source_z)
            if sol.best.misfit < best[0]:
                best = (sol.best.misfit, float(c), sol.best.x, sol.best.y)
        fixed = solve(px, py, pz, t, ConstantC(DEFAULT_C), sz=source_z)
        rows.append({"call": call, "n_picks": int(len(sub)),
                     "best_c": best[1], "best_misfit": best[0],
                     "best_x": best[2], "best_y": best[3],
                     "fixed_misfit": fixed.best.misfit,
                     "fixed_x": fixed.best.x, "fixed_y": fixed.best.y,
                     "shift_m": float(np.hypot(best[2] - fixed.best.x,
                                               best[3] - fixed.best.y))})
    return pd.DataFrame(rows)


def ablation_report(base: dict, variant: dict, label: str,
                    *, flip_threshold_m: float = 500.0) -> dict:
    c = compare(base, variant, flip_threshold_m=flip_threshold_m)
    d = c.stable_delta_m
    return {"label": label, "n": c.n, "n_stable": c.n_stable, "n_flipped": c.n_flipped,
            "flip_fraction": c.flip_fraction,
            "stable_delta_median_m": float(np.median(d)) if d.size else float("nan"),
            "stable_delta_p90_m": float(np.percentile(d, 90)) if d.size else float("nan"),
            "stable_delta_max_m": float(d.max()) if d.size else float("nan")}
```

```yaml
# paper/configs/ablation_free_c.yaml
name: ablation_free_c
notes: >
  Section 7.2. Add c as a free scalar and measure how far positions move against
  fixed 1480. Expected to absorb most of the L-1 effect, which closes that
  branch; this fit is the null model the depth experiment must beat.
truth_model: {kind: constant_c, c: 1480.0}
inversion_model: {kind: path_average, profile: oregon_nov}
objective: l1
source_z: -30.0
min_picks: 8
c_grid: {start: 1430.0, stop: 1530.0, step: 2.5}
flip_threshold_m: 500.0
```

```yaml
# paper/configs/ablation_effective_c.yaml
name: ablation_effective_c
notes: >
  Section 7.3. PathAverage and RayTrace against ConstantC and against the
  free-c fit. RayTrace reports the fraction of picks with no direct ray; where
  that fraction is large the comparison is between two models that both
  describe propagation that did not happen, and the report must say so.
truth_model: {kind: raytrace, profile: oregon_nov}
inversion_model: {kind: constant_c, c: 1480.0}
objective: l1
source_z: -30.0
min_picks: 8
flip_threshold_m: 500.0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest paper/tests/test_ablation.py -q`
Expected: 5 passed

- [ ] **Step 5: Run both ablations on the real picks and commit the tables**

```bash
uv run python -c "
import numpy as np, pandas as pd, yaml
from dclde_localization.physics.geometry import load_cables
from dclde_soundspeed.experiments.ablation import ablation_report, fit_free_c, solve_all
from dclde_soundspeed.experiments.runner import PROFILES, write_results
from dclde_localization.physics.sound_speed_models import ConstantC, PathAverage
from dclde_soundspeed.soundspeed.raytrace import RayTrace
from dclde_localization.physics.picks import attach_xyz, load_picks

picks = attach_xyz(load_picks(), load_cables('fixtures'))
g = yaml.safe_load(open('paper/configs/ablation_free_c.yaml'))['c_grid']
free = fit_free_c(picks, source_z=-30.0,
                  c_grid=np.arange(g['start'], g['stop'], g['step']))
print('free-c: n=%d, median best_c=%.1f, median shift=%.0f m'
      % (len(free), free.best_c.median(), free.shift_m.median()))
print(write_results(free, 'ablation_free_c'))

prof = PROFILES['oregon_nov']
base = solve_all(picks, ConstantC(), source_z=-30.0)
reports = [ablation_report(base, solve_all(picks, PathAverage(prof), source_z=-30.0),
                           'path_average vs constant'),
           ablation_report(base, solve_all(picks, RayTrace(prof), source_z=-30.0),
                           'raytrace vs constant')]
df = pd.DataFrame(reports)
print(df.to_string(index=False))
print(write_results(df, 'ablation_effective_c'))
"
```

**Report both halves in the commit message** — the basin-stable median delta *and* the flip fraction. Reporting only the median is the specific failure the estimator was built to prevent.

- [ ] **Step 6: Run the full suite and confirm the frozen module is intact**

```bash
uv run pytest paper/tests viz/tests -q
git diff --exit-code src/dclde_localization/physics/forward.py
cd viz && npm test
```

Expected: all Python tests pass; no diff on `forward.py`; 49 JavaScript tests pass.

- [ ] **Step 7: Commit**

```bash
git add paper/src/dclde_soundspeed/experiments/ablation.py paper/tests/test_ablation.py \
        paper/configs/ablation_*.yaml \
        paper/results/ablation_*.csv
git commit -m "Run the free-c and effective-c ablations

Both report basin-stable position deltas and mirror-lobe flip counts as two
separate numbers, because a median over a set that includes basin flips hides
the only interesting failure. RayTrace also reports how often no direct ray
existed, since where it fell back it is not describing the real propagation."
```

---

### Task 14: Write up Plan A and delete the temporary reference data

**Files:**
- Create: `paper/docs/das-finwhale-soundspeed-plan-a-findings.md`
- Modify: `paper/docs/sound-speed-feasibility.md` (apply the §3 corrections)

**Interfaces:**
- Consumes: every result table in `paper/results/`.
- Produces: the document Plan B is planned from.

- [ ] **Step 1: Write the findings document**

Structure it as: what was asked, what was measured, what the evidence supports, what it does not, and what Plan B should therefore do. Required content:

- Every number carries its denominator: calls, batches, cables, exclusions and why.
- The baseline reproduction: matched fraction and the per-call delta distribution.
- Free-`c`: fitted-`c` distribution, basin-stable delta, **flip fraction stated separately**.
- Effective-`c`: same, plus `RayTrace.last_fallback_fraction`.
- §7.0.1: whether the residual range trend flattens under a source-depth sweep.
- §7.0.3: whether trapped modes exist where ray theory says shadowed, with the seabed sweep as a spread.
- Profile validation: reanalysis-vs-in-situ disagreement in m/s over 97–575 m, and an explicit note that 0–97 m and 575–1575 m are model, not measurement.
- A verdict on **Q-path**, which the spec permits to be "the sound-speed programme is aimed at the wrong term."

- [ ] **Step 2: Apply the §3 corrections to the feasibility note**

Fix, in `paper/docs/sound-speed-feasibility.md`:
1. The OOI availability claim — the Slope Base deep profiler has no Nov 2021 data, both shallow profilers were parked, the site code is `RS01SBPD` not `RS01SBPS`, and the usable instrument is CE04OSPD over 97–575 m.
2. `z = −30.0` is the pair CSV; their single-cable output uses −40.0.
3. "Roughly a day of ingestion work" — replace with what it actually took.
4. The constant-vs-gradient category error, cross-referencing the spec's §2.5.

- [ ] **Step 3: Verify every quoted number against its result table**

For each number in the findings document, open the CSV it came from and confirm it. A findings document that misquotes its own tables is worse than none, because it is trusted.

- [ ] **Step 4: Delete the temporary reference data**

Only after the result tables are committed:

```bash
du -sh paper/data/reference
rm -rf paper/data/reference/associations paper/data/reference/associations.zip paper/data/reference/_probe
uv run python paper/scripts/fetch_reference_data.py --dest paper/data/reference --skip-zenodo
```

Keep the 204 MB repo worktree (the baseline CSV is small and used by the reproduction test); drop the 3.6 GB of association archives. `PROVENANCE.md` and the fetch script restore them.

- [ ] **Step 5: Commit**

```bash
git add paper/docs/das-finwhale-soundspeed-plan-a-findings.md paper/docs/sound-speed-feasibility.md
git commit -m "Report Plan A findings and correct the feasibility note

Applies the four corrections the spec recorded, including the constant-versus-
gradient category error that motivated section 7.0. The 3.6 GB of association
archives is deleted; paper/scripts/fetch_reference_data.py restores it."
```

---

## Self-Review

**Spec coverage.**

| Spec section | Task |
|---|---|
| §4 architecture, frozen `forward.py` | 1–3 (Task 2 Step 6 asserts no diff) |
| §4.1 `SoundSpeedModel`, `ConstantC` equivalence | 2 |
| §4.2 self-consistency guard | 5 |
| §5 multi-basin estimator, flip/delta split | 4, enforced again in 13 |
| §6 reanalysis base, in-situ validation | 8, 9 |
| §7.0 propagation regime | 11 (depth-vs-path), 12 (shadow zone) |
| §7.1 baseline reproduction | 7 |
| §7.2 free-`c` | 13 |
| §7.3 effective-`c` | 13 |
| §7.6 modal calculation, un-gated | 10, 12 |
| §9 provenance and deletion | 6, 14 |
| §10 testing | every task; the Pekeris and mirror-lobe cases carry the real content |

**Deliberately out of scope** (Plan B, per spec §12): §7.4 depth observability, §7.5 pick density, §8 noise model and simulator. Plan B's design depends on Task 11's and Task 13's outputs, so planning it now would mean rewriting it.

**Known gap:** §7.0.2 (the moveout-shape test distinguishing direct from bottom-bounce arrivals) has no task. It needs a bounce-path model that Task 10's range-independent solver does not provide, and Tasks 11 and 12 answer the same question more directly and more cheaply. If both come back ambiguous, §7.0.2 becomes the first task of Plan B rather than an afterthought here.

**Type consistency.** `SoundSpeedModel.travel_time(sx, sy, sz, px, py, pz)` is used identically in Tasks 2, 3, 4, 11 and 13. `Solution.best`/`.separation_m` (Task 4) are consumed in 7, 11 and 13. `write_results(df, name)` (Task 5) is called in 7, 9, 11, 12 and 13. `Seabed(c, rho, alpha)` (Task 10) is constructed in 12. Depths: positive-down inside `soundspeed/`, negative-down everywhere else, converted only through `to_down`/`to_repo`.

**Placeholder scan.** Clean — no TBDs, no "similar to Task N", no steps that describe without showing. Every code step carries the code.
