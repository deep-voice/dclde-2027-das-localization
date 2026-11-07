# DAS Fin-Whale Localization Visualizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained, shareable web page that renders the misfit field for any annotated fin-whale call on the DCLDE 2027 DAS dataset, so left–right ambiguity, range conditioning and depth unobservability are directly visible rather than argued.

**Architecture:** The travel-time physics lives in `src/dclde/das_finwhale/` (our inversion pipeline needs it too) and is mirrored in a small JavaScript module guarded by a cross-language parity test. A Python build step reads the committed pick snapshot, the cable geometry, and a GMRT bathymetry grid, and emits one HTML file with all data inlined as base64 typed arrays. The page computes misfit fields in the browser on demand, so sound speed and assumed source depth are live parameters and the payload stays around 1.3 MB.

**Tech Stack:** Python 3.12, numpy, pandas, scipy, xarray/netcdf4, pyproj, pytest. Vanilla ES modules + Canvas 2D for the page, with no third-party JavaScript at all — the 3D inset is a hand-rolled orthographic projection. Node's built-in test runner for JS.

**Spec:** `viz/docs/specs/2026-09-10-das-localization-visualizer-design.md`

## Global Constraints

- **Frame identity (exact, verified zero-variance over all 77,958 channels):** `utm_x = 435226.0809 − x`, `utm_y = 4985111.5099 + y`. UTM zone 10N is EPSG:32610; WGS84 is EPSG:4326.
- **Channel spacing is exactly `2.041904688` m**, `chan_idx` starts at `942` on both cables, `chan_m0 = 1923.474216`. Max deviation from `chan_m0 + (chan_idx − 942) × spacing` is 7e-12 m.
- **Cable channel counts:** North 31,518; South 46,440. North length 66.3 km (elevation −589 m…+11 m); South 96.7 km (−1575 m…+11 m). Both share a shore landing at (45.20230, −123.96715).
- **Picks:** 11,117 rows total, 628 distinct `(window, call_id)` keys over 60 windows, 166 keys on both cables.
- **Call key format:** `<window>/<call_id>` with the window timestamp underscore-normalized, e.g. `2021-11-04_02_00_02/1`. Never colons or dashes in the time part — the dataset uses both inconsistently.
- **Default sound speed `c = 1480.0` m/s.** Never hardcode it at a call site; always an explicit argument.
- **Default objective:** L1 with `t0` eliminated as the median residual.
- **Contour levels:** 1.05, 1.10, 1.25, 1.50 × best misfit.
- **Frame guard threshold:** 25,000 m median distance from solutions to nearest cable channel.
- **Grid resolution:** long-axis cell count, aspect preserved. Default 260; selectable 130 / 260 / 520.
- **Parity tolerance:** max relative difference per cell < 1e-6 between the Python and JS misfit fields.
- **Page must not fetch anything at runtime, and must reference no external origin at all.** The Artifact CSP blocks all fetch/XHR, and the page carries no third-party library: the 3D inset is a hand-rolled orthographic projection on a canvas, not three.js. A built page must contain zero `https://` references.
- **Theme:** the page must render correctly in light and dark. Define the light palette on bare `:root`; redefine tokens under `@media (prefers-color-scheme: dark)` guarded as `:root:not([data-theme="light"])` and again under `:root[data-theme="dark"]`.
- `viz/` is exempt from the repo's seaborn-only plotting rule. Static figures under `docs/` are not.
- **Do not depend on DAS4Whales** — it is CC BY-NC-SA 4.0 (viral, non-commercial). Reimplement the ten lines of travel-time arithmetic.
- **Node 18 or newer** is required: the JS tests use the built-in `node --test` runner, and `decodeF32`/`decodeI16` use the global `atob`.
- All commits on branch `visualizer`.

---

### Task 1: Cable geometry and frame transforms

**Files:**
- Create: `src/dclde_localization/physics/geometry.py`
- Create: `paper/tests/__init__.py` (empty)
- Create: `tests/physics/test_geometry.py`
- Modify: `requirements.txt` (add `pyproj`)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `CHAN_IDX0 = 942`, `CHAN_SPACING_M = 2.041904688`, `CABLES = ("North", "South")`
  - `CableGeometry` frozen dataclass with fields `name: str`, `chan_m0: float`, `spacing_m: float`, `x/y/depth/lat/lon: np.ndarray`; property `chan_m -> np.ndarray`; method `chan_to_xyz(dist_m: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
  - `load_cable(name: str, csv_path: str | Path) -> CableGeometry`
  - `decimate_indices(n: int, stride: int) -> np.ndarray` — decimated indices that always include the final channel
  - `FrameTransform` frozen dataclass with `x_offset: float`, `y_offset: float`; classmethod `from_cable(cable: CableGeometry) -> FrameTransform`; methods `local_to_utm(x, y)`, `utm_to_local(ux, uy)`, `local_to_wgs84(x, y) -> (lat, lon)`, `wgs84_to_local(lat, lon) -> (x, y)`, `to_local(frame: str, a, b) -> (x, y)`
  - `VALID_FRAMES = ("local_xy", "utm10n", "wgs84")`

- [ ] **Step 1: Add the dependency**

Add `pyproj` to `requirements.txt` under the `# data / audio / DAS` section, then:

```bash
uv pip install -e .
```

- [ ] **Step 2: Write the failing tests**

Create `paper/tests/__init__.py` as an empty file, then `tests/physics/test_geometry.py`:

```python
import numpy as np
import pandas as pd
import pytest

from dclde_localization.physics.geometry import (
    CHAN_IDX0,
    CHAN_SPACING_M,
    CableGeometry,
    FrameTransform,
    decimate_indices,
    load_cable,
)

NORTH_CSV = "fixtures/north_DAS_multicoord.csv"


@pytest.fixture(scope="module")
def north() -> CableGeometry:
    return load_cable("North", NORTH_CSV)


def test_load_cable_shape_and_constants(north):
    assert north.name == "North"
    assert len(north.x) == 31518
    assert north.spacing_m == pytest.approx(CHAN_SPACING_M, abs=1e-9)
    assert north.chan_m0 == pytest.approx(1923.474216, abs=1e-6)


@pytest.mark.parametrize("csv", [
    "fixtures/north_DAS_multicoord.csv",
    "fixtures/south_DAS_multicoord.csv",
])
def test_raw_chan_m_column_is_linear_in_channel_index(csv):
    # The bundle omits chan_m and reconstructs it from start + step, so a
    # non-uniform cable would silently corrupt every pick position.
    #
    # This reads the CSV directly and compares the DATA against the documented
    # constant. Asserting on CableGeometry.chan_m instead would be circular:
    # that property IS the reconstruction, so it would be compared to its own
    # formula and could never fail.
    df = pd.read_csv(csv).sort_values("chan_idx")
    raw = df["chan_m"].to_numpy()
    expected = raw[0] + np.arange(len(raw)) * CHAN_SPACING_M
    assert np.abs(raw - expected).max() < 1e-3


def _corrupt_fixture(tmp_path, column, row, delta):
    df = pd.read_csv("fixtures/north_DAS_multicoord.csv", nrows=200)
    df.loc[row, column] = df.loc[row, column] + delta
    path = tmp_path / "corrupt.csv"
    df.to_csv(path, index=False)
    return path


def test_load_cable_rejects_non_uniform_spacing(tmp_path):
    bad = _corrupt_fixture(tmp_path, "chan_m", 100, 5.0)
    with pytest.raises(ValueError, match="not uniform"):
        load_cable("North", bad)


def test_load_cable_rejects_non_contiguous_chan_idx(tmp_path):
    bad = _corrupt_fixture(tmp_path, "chan_idx", 100, 7)
    with pytest.raises(ValueError, match="contiguous"):
        load_cable("North", bad)


def test_chan_to_xyz_matches_table_at_channel_centres(north):
    idx = np.array([0, 100, 5000, 31517])
    x, y, z = north.chan_to_xyz(north.chan_m[idx])
    assert np.allclose(x, north.x[idx], atol=1e-6)
    assert np.allclose(y, north.y[idx], atol=1e-6)
    assert np.allclose(z, north.depth[idx], atol=1e-6)


def test_chan_to_xyz_clamps_outside_range(north):
    x, y, _ = north.chan_to_xyz([-1e6, 1e9])
    assert x[0] == pytest.approx(north.x[0])
    assert x[1] == pytest.approx(north.x[-1])


def test_frame_transform_offsets_are_exact(north):
    t = FrameTransform.from_cable(north)
    assert t.x_offset == pytest.approx(435226.0809, abs=1e-3)
    assert t.y_offset == pytest.approx(4985111.5099, abs=1e-3)


def test_frame_transform_reproduces_table_utm(north):
    t = FrameTransform.from_cable(north)
    ux, uy = t.local_to_utm(north.x, north.y)
    assert np.abs(ux - north.utm_x).max() < 1e-6
    assert np.abs(uy - north.utm_y).max() < 1e-6


def test_frame_transform_reproduces_table_latlon(north):
    # pyproj UTM10N -> WGS84 must reproduce the dataset's own lat/lon.
    t = FrameTransform.from_cable(north)
    lat, lon = t.local_to_wgs84(north.x, north.y)
    assert np.abs(lat - north.lat).max() < 1e-6
    assert np.abs(lon - north.lon).max() < 1e-6


def test_frame_roundtrip_wgs84(north):
    t = FrameTransform.from_cable(north)
    lat, lon = t.local_to_wgs84(north.x, north.y)
    x2, y2 = t.wgs84_to_local(lat, lon)
    assert np.abs(x2 - north.x).max() < 0.01
    assert np.abs(y2 - north.y).max() < 0.01


def test_to_local_dispatches_on_frame(north):
    t = FrameTransform.from_cable(north)
    x, y = t.to_local("local_xy", north.x[:5], north.y[:5])
    assert np.allclose(x, north.x[:5])
    ux, uy = t.local_to_utm(north.x[:5], north.y[:5])
    x2, y2 = t.to_local("utm10n", ux, uy)
    assert np.allclose(x2, north.x[:5], atol=1e-6)
    with pytest.raises(ValueError, match="unknown frame"):
        t.to_local("mercator", 0, 0)


def test_decimate_indices_always_includes_both_ends():
    idx = decimate_indices(10, 4)
    assert idx[0] == 0
    assert idx[-1] == 9
    assert np.array_equal(idx, np.array([0, 4, 8, 9]))
    # Already-aligned lengths must not duplicate the final index.
    assert np.array_equal(decimate_indices(9, 4), np.array([0, 4, 8]))


@pytest.mark.parametrize("name,csv", [
    ("North", "fixtures/north_DAS_multicoord.csv"),
    ("South", "fixtures/south_DAS_multicoord.csv"),
])
def test_decimation_error_under_one_metre(name, csv):
    # The bundle ships a decimated cable; interpolating back must stay sub-metre.
    # Plain [::4] leaves 1-3 channels past the last kept sample, where np.interp
    # clamps rather than interpolating: that tail costs 1.8 m on North and 6.3 m
    # on South. decimate_indices always keeps the final channel, which is why
    # both the bundle and this test must go through it.
    cable = load_cable(name, csv)
    idx = decimate_indices(len(cable.x), 4)
    xi = np.interp(cable.chan_m, cable.chan_m[idx], cable.x[idx])
    yi = np.interp(cable.chan_m, cable.chan_m[idx], cable.y[idx])
    err = np.hypot(xi - cable.x, yi - cable.y)
    assert err.max() < 1.0, f"{name}: max decimation error {err.max():.3f} m"
```


- [ ] **Step 3: Run the tests to verify they fail**

```bash
uv run pytest tests/physics/test_geometry.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'dclde_localization.physics.geometry'` — the file currently holds only a docstring, so the imported names do not exist.

- [ ] **Step 4: Vendor the geometry fixtures**

The tests read real cable geometry. These two files are inputs, not generated data, and are small enough to commit (3.9 MB + 5.8 MB).

```bash
mkdir -p fixtures
aws s3 cp s3://deepvoice-datasets/dclde_2027_das_finwhale_localization/geometry/north_DAS_multicoord.csv fixtures/
aws s3 cp s3://deepvoice-datasets/dclde_2027_das_finwhale_localization/geometry/south_DAS_multicoord.csv fixtures/
```

If S3 credentials are unavailable, the same objects are public:
`https://storage.googleapis.com/noaa-passive-bioacoustic/dclde/2027/dclde_2027_das-finwhale-localization/geometry/north_DAS_multicoord.csv`

Add an exception to `.gitignore` so the `*.csv` under `fixtures/` is tracked while generated bundles are not:

```
# viz build output (fixtures under fixtures/ ARE tracked)
viz/out/
```

- [ ] **Step 5: Write the implementation**

Replace the contents of `src/dclde_localization/physics/geometry.py`:

```python
"""DAS cable geometry and coordinate frames.

The dataset's "local" XY frame is UTM zone 10N with the x axis flipped:
    utm_x = X_OFFSET - x        utm_y = Y_OFFSET + y
Both offsets have zero variance across all 77,958 channels of both cables, so
this is an identity, not a fit. The offsets are still derived from the geometry
table and asserted constant, so a revised dataset cannot slip past unnoticed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

CABLES = ("North", "South")
CHAN_IDX0 = 942
CHAN_SPACING_M = 2.041904688
VALID_FRAMES = ("local_xy", "utm10n", "wgs84")

UTM10N_EPSG = 32610
WGS84_EPSG = 4326

_TO_WGS84 = Transformer.from_crs(UTM10N_EPSG, WGS84_EPSG, always_xy=True)
_FROM_WGS84 = Transformer.from_crs(WGS84_EPSG, UTM10N_EPSG, always_xy=True)


@dataclass(frozen=True)
class CableGeometry:
    name: str
    chan_m0: float
    spacing_m: float
    x: np.ndarray
    y: np.ndarray
    depth: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    utm_x: np.ndarray
    utm_y: np.ndarray

    @property
    def chan_m(self) -> np.ndarray:
        return self.chan_m0 + np.arange(len(self.x)) * self.spacing_m

    def chan_to_xyz(self, dist_m) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map along-cable distance to local (x, y, depth), clamped at the ends."""
        d = np.asarray(dist_m, dtype=float)
        m = self.chan_m
        return (
            np.interp(d, m, self.x),
            np.interp(d, m, self.y),
            np.interp(d, m, self.depth),
        )


def load_cable(name: str, csv_path: str | Path) -> CableGeometry:
    df = pd.read_csv(csv_path).sort_values("chan_idx").reset_index(drop=True)

    idx = df["chan_idx"].to_numpy()
    if idx[0] != CHAN_IDX0 or not np.array_equal(np.diff(idx), np.ones(len(idx) - 1)):
        raise ValueError(f"{name}: chan_idx must start at {CHAN_IDX0} and be contiguous")

    chan_m = df["chan_m"].to_numpy()
    steps = np.diff(chan_m)
    spacing = float(steps.mean())
    if steps.std() > 1e-6:
        raise ValueError(f"{name}: channel spacing is not uniform (std={steps.std():.3e})")

    return CableGeometry(
        name=name,
        chan_m0=float(chan_m[0]),
        spacing_m=spacing,
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        depth=df["depth"].to_numpy(),
        lat=df["lat"].to_numpy(),
        lon=df["lon"].to_numpy(),
        utm_x=df["utm_x"].to_numpy(),
        utm_y=df["utm_y"].to_numpy(),
    )


def decimate_indices(n: int, stride: int) -> np.ndarray:
    """Indices for a decimated cable, always including the final channel.

    A plain ``[::stride]`` leaves the last ``(n-1) % stride`` channels beyond the
    final kept sample, where interpolation clamps instead of interpolating — 1.8 m
    of position error on North and 6.3 m on South at stride 4. Both the bundle and
    its tests must decimate through this function so they cannot diverge.
    """
    return np.unique(np.r_[np.arange(n)[::stride], n - 1])


def load_cables(geometry_dir: str | Path) -> dict[str, CableGeometry]:
    d = Path(geometry_dir)
    return {n: load_cable(n, d / f"{n.lower()}_DAS_multicoord.csv") for n in CABLES}


@dataclass(frozen=True)
class FrameTransform:
    x_offset: float
    y_offset: float

    @classmethod
    def from_cable(cls, cable: CableGeometry, tol_m: float = 1e-3) -> FrameTransform:
        xs = cable.utm_x + cable.x
        ys = cable.utm_y - cable.y
        if xs.std() > tol_m or ys.std() > tol_m:
            raise ValueError(
                "local<->UTM relation is not a pure flip+translate "
                f"(x std={xs.std():.3e}, y std={ys.std():.3e})"
            )
        return cls(x_offset=float(xs.mean()), y_offset=float(ys.mean()))

    def local_to_utm(self, x, y):
        return self.x_offset - np.asarray(x, float), self.y_offset + np.asarray(y, float)

    def utm_to_local(self, ux, uy):
        return self.x_offset - np.asarray(ux, float), np.asarray(uy, float) - self.y_offset

    def local_to_wgs84(self, x, y):
        ux, uy = self.local_to_utm(x, y)
        lon, lat = _TO_WGS84.transform(ux, uy)
        return np.asarray(lat), np.asarray(lon)

    def wgs84_to_local(self, lat, lon):
        ux, uy = _FROM_WGS84.transform(np.asarray(lon, float), np.asarray(lat, float))
        return self.utm_to_local(ux, uy)

    def to_local(self, frame: str, a, b):
        """Convert a coordinate pair in `frame` to local XY.

        For 'wgs84' the pair is (lat, lon); otherwise it is (x, y) / (easting, northing).
        """
        if frame == "local_xy":
            return np.asarray(a, float), np.asarray(b, float)
        if frame == "utm10n":
            return self.utm_to_local(a, b)
        if frame == "wgs84":
            return self.wgs84_to_local(a, b)
        raise ValueError(f"unknown frame: {frame!r}; expected one of {VALID_FRAMES}")
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
uv run pytest tests/physics/test_geometry.py -v
```

Expected: 10 passed. If `test_frame_transform_reproduces_table_latlon` fails by more than 1e-6 degrees, the dataset's lat/lon was produced by a different geodesy than EPSG:32610 — stop and report rather than loosening the tolerance, because everything downstream inherits this.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .gitignore src/dclde_localization/physics/geometry.py paper/tests/ fixtures/
git commit -m "Add DAS cable geometry loader and exact local/UTM/WGS84 frame transforms"
```

---

### Task 2: Forward model and misfit field (Python)

**Files:**
- Create: `src/dclde_localization/physics/forward.py`
- Create: `tests/physics/test_forward.py`

**Interfaces:**
- Consumes: `CableGeometry` from Task 1.
- Produces:
  - `DEFAULT_C = 1480.0`, `OBJECTIVES = ("l1", "l2")`
  - `travel_time(sx, sy, sz, px, py, pz, c) -> np.ndarray`
  - `best_t0(obs_t, tt, objective="l1") -> float`
  - `misfit_value(obs_t, tt, objective="l1") -> float`
  - `misfit_field(xs, ys, sz, px, py, pz, obs_t, c, objective="l1") -> np.ndarray` shaped `(len(ys), len(xs))`
  - `signed_side(cable, x, y) -> np.ndarray` returning ±1
  - `MirrorPenalty` frozen dataclass `(best_misfit, mirror_misfit, ratio, best_xy, mirror_xy)`
  - `mirror_penalty(cable, xs, ys, sz, px, py, pz, obs_t, c, objective="l1") -> MirrorPenalty`

- [ ] **Step 1: Write the failing tests**

Create `tests/physics/test_forward.py`:

```python
import numpy as np
import pytest

from dclde_localization.physics.forward import (
    DEFAULT_C,
    best_t0,
    misfit_field,
    misfit_value,
    mirror_penalty,
    signed_side,
    travel_time,
)
from dclde_localization.physics.geometry import load_cable

NORTH_CSV = "fixtures/north_DAS_multicoord.csv"


def test_travel_time_is_euclidean_over_c():
    t = travel_time(0.0, 0.0, 0.0, np.array([3000.0]), np.array([4000.0]), np.array([0.0]), 1000.0)
    assert t[0] == pytest.approx(5.0)


def test_travel_time_includes_depth():
    t = travel_time(0.0, 0.0, -30.0, np.array([0.0]), np.array([0.0]), np.array([-1030.0]), 1000.0)
    assert t[0] == pytest.approx(1.0)


def test_best_t0_l1_is_median_and_l2_is_mean():
    tt = np.array([1.0, 2.0, 3.0])
    obs = tt + np.array([10.0, 11.0, 30.0])
    assert best_t0(obs, tt, "l1") == pytest.approx(11.0)
    assert best_t0(obs, tt, "l2") == pytest.approx(17.0)


def test_misfit_value_is_zero_for_a_perfect_fit():
    tt = np.array([1.0, 2.0, 3.5])
    obs = tt + 7.0
    assert misfit_value(obs, tt, "l1") == pytest.approx(0.0)
    assert misfit_value(obs, tt, "l2") == pytest.approx(0.0)


def test_misfit_field_recovers_a_synthetic_source():
    # Straight cable along x at depth 0; source 6 km off-axis.
    px = np.linspace(0.0, 50_000.0, 40)
    py = np.zeros_like(px)
    pz = np.zeros_like(px)
    true_x, true_y, sz = 25_000.0, 6_000.0, 0.0
    obs = 12.5 + travel_time(true_x, true_y, sz, px, py, pz, DEFAULT_C)

    xs = np.linspace(20_000.0, 30_000.0, 101)
    ys = np.linspace(0.0, 12_000.0, 121)
    field = misfit_field(xs, ys, sz, px, py, pz, obs, DEFAULT_C)

    assert field.shape == (len(ys), len(xs))
    j, i = np.unravel_index(field.argmin(), field.shape)
    assert xs[i] == pytest.approx(true_x, abs=150.0)
    assert abs(ys[j]) == pytest.approx(true_y, abs=150.0)
    assert field.min() < 1e-6


def test_misfit_field_is_bimodal_for_a_straight_cable():
    # A perfectly straight cable cannot distinguish the two sides: the mirror
    # solution must fit exactly as well.
    px = np.linspace(0.0, 50_000.0, 40)
    py = np.zeros_like(px)
    pz = np.zeros_like(px)
    obs = travel_time(25_000.0, 6_000.0, 0.0, px, py, pz, DEFAULT_C)

    xs = np.linspace(24_000.0, 26_000.0, 21)
    ys = np.array([-6_000.0, 6_000.0])
    field = misfit_field(xs, ys, 0.0, px, py, pz, obs, DEFAULT_C)
    assert field[0].min() == pytest.approx(field[1].min(), abs=1e-9)


def test_signed_side_splits_around_the_cable():
    cable = load_cable("North", NORTH_CSV)
    # Offsets perpendicular to a locally east-west cable are north/south of it.
    i = 15_000
    x0, y0 = cable.x[i], cable.y[i]
    s = signed_side(cable, np.array([x0, x0]), np.array([y0 + 5_000.0, y0 - 5_000.0]))
    assert s[0] == -s[1]
    assert set(np.unique(s)) <= {-1.0, 1.0}


def test_mirror_penalty_ratio_is_one_for_a_symmetric_synthetic_case():
    class _Straight:
        x = np.linspace(0.0, 50_000.0, 200)
        y = np.zeros(200)

    px = np.linspace(0.0, 50_000.0, 40)
    py = np.zeros_like(px)
    pz = np.zeros_like(px)
    obs = travel_time(25_000.0, 6_000.0, 0.0, px, py, pz, DEFAULT_C)

    xs = np.linspace(23_000.0, 27_000.0, 41)
    ys = np.linspace(-9_000.0, 9_000.0, 91)
    mp = mirror_penalty(_Straight(), xs, ys, 0.0, px, py, pz, obs, DEFAULT_C)
    assert mp.ratio == pytest.approx(1.0, abs=1e-6)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/physics/test_forward.py -v
```

Expected: collection error, `ImportError: cannot import name 'travel_time'`.

- [ ] **Step 3: Write the implementation**

Replace the contents of `src/dclde_localization/physics/forward.py`:

```python
"""Straight-ray travel-time forward model and misfit fields.

Deliberately explicit rather than clever: this arithmetic is mirrored in
viz/template/lib/forward.mjs and the two are held together by a parity test.
Every published DAS localization method fixes the sound speed (1440/1480/1500
m/s); here `c` is always an explicit argument so it can be varied and fitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_C = 1480.0
OBJECTIVES = ("l1", "l2")


def travel_time(sx, sy, sz, px, py, pz, c):
    """Straight-ray travel time from source (sx, sy, sz) to each pick point."""
    dx = np.asarray(px, float) - sx
    dy = np.asarray(py, float) - sy
    dz = np.asarray(pz, float) - sz
    return np.sqrt(dx * dx + dy * dy + dz * dz) / c


def best_t0(obs_t, tt, objective="l1"):
    """Emission time that minimises the objective, eliminated analytically."""
    r = np.asarray(obs_t, float) - np.asarray(tt, float)
    if objective == "l1":
        return float(np.median(r))
    if objective == "l2":
        return float(np.mean(r))
    raise ValueError(f"unknown objective: {objective!r}; expected one of {OBJECTIVES}")


def misfit_value(obs_t, tt, objective="l1"):
    r = np.asarray(obs_t, float) - np.asarray(tt, float) - best_t0(obs_t, tt, objective)
    if objective == "l1":
        return float(np.abs(r).sum())
    return float(np.sqrt(np.mean(r * r)))


def misfit_field(xs, ys, sz, px, py, pz, obs_t, c, objective="l1"):
    """Misfit over a grid. Returns shape (len(ys), len(xs))."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    px = np.asarray(px, float)
    py = np.asarray(py, float)
    pz = np.asarray(pz, float)
    obs = np.asarray(obs_t, float)

    out = np.empty((len(ys), len(xs)), dtype=float)
    dz2 = (pz - sz) ** 2
    for j, wy in enumerate(ys):
        # (npicks, nxs) travel times for this row of the grid. `dy` stays 1-D so
        # that (dy*dy)[:, None] is (npicks, 1); making dy 2-D here would produce a
        # 3-D array that still broadcasts but computes the wrong thing.
        dx = px[:, None] - xs[None, :]
        dy = py - wy
        tt = np.sqrt(dx * dx + (dy * dy)[:, None] + dz2[:, None]) / c
        r = obs[:, None] - tt
        if objective == "l1":
            out[j] = np.abs(r - np.median(r, axis=0)).sum(axis=0)
        elif objective == "l2":
            rc = r - r.mean(axis=0)
            out[j] = np.sqrt((rc * rc).mean(axis=0))
        else:
            raise ValueError(f"unknown objective: {objective!r}")
    return out


def signed_side(cable, x, y):
    """+1 / -1 for which side of the cable each point lies on.

    Uses the local tangent at the nearest cable node, so it follows the cable's
    curvature rather than assuming a straight line.
    """
    cx = np.asarray(cable.x, float)
    cy = np.asarray(cable.y, float)
    x = np.atleast_1d(np.asarray(x, float))
    y = np.atleast_1d(np.asarray(y, float))

    k = np.argmin((cx[None, :] - x[:, None]) ** 2 + (cy[None, :] - y[:, None]) ** 2, axis=1)
    k0 = np.clip(k - 1, 0, len(cx) - 1)
    k1 = np.clip(k + 1, 0, len(cx) - 1)
    tx, ty = cx[k1] - cx[k0], cy[k1] - cy[k0]
    nx, ny = x - cx[k], y - cy[k]
    s = np.sign(tx * ny - ty * nx)
    return np.where(s == 0, 1.0, s)


@dataclass(frozen=True)
class MirrorPenalty:
    best_misfit: float
    mirror_misfit: float
    ratio: float
    best_xy: tuple[float, float]
    mirror_xy: tuple[float, float]


def mirror_penalty(cable, xs, ys, sz, px, py, pz, obs_t, c, objective="l1"):
    """Best misfit on each side of the cable, and the wrong-side penalty ratio.

    Defined per reference cable. With picks from two cables there is no single
    mirror, so callers should report one of these per cable instead.
    """
    field = misfit_field(xs, ys, sz, px, py, pz, obs_t, c, objective)
    gx, gy = np.meshgrid(np.asarray(xs, float), np.asarray(ys, float))
    side = signed_side(cable, gx.ravel(), gy.ravel()).reshape(field.shape)

    out = []
    for sgn in (1.0, -1.0):
        masked = np.where(side == sgn, field, np.inf)
        j, i = np.unravel_index(masked.argmin(), masked.shape)
        out.append((float(masked[j, i]), (float(gx[j, i]), float(gy[j, i]))))
    out.sort(key=lambda t: t[0])
    (m1, p1), (m2, p2) = out
    # A zero best misfit needs care. Both sides zero means both fit perfectly,
    # which is maximum ambiguity, so the ratio is 1.0 — not infinity, which would
    # say the opposite. Only a zero best against a non-zero mirror is infinite.
    if m1 > 0:
        ratio = float(m2 / m1)
    elif m1 == m2:
        ratio = 1.0
    else:
        ratio = float("inf")
    return MirrorPenalty(m1, m2, ratio, p1, p2)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/physics/test_forward.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/dclde_localization/physics/forward.py tests/physics/test_forward.py
git commit -m "Add straight-ray travel-time model, misfit field and mirror penalty"
```

---

### Task 3: JavaScript forward model and cross-language parity test

**Files:**
- Create: `viz/template/lib/forward.mjs`
- Create: `viz/make_parity_fixture.py`
- Create: `viz/fixtures/parity_fixture.json` (generated, committed)
- Create: `viz/tests/test_parity.mjs`
- Create: `viz/package.json`

**Interfaces:**
- Consumes: `misfit_field`, `travel_time`, `best_t0` from Task 2.
- Produces (ES module named exports, mirroring Task 2 exactly):
  - `DEFAULT_C = 1480.0`
  - `travelTime(sx, sy, sz, px, py, pz, c) -> Float64Array`
  - `bestT0(obsT, tt, objective) -> number`
  - `misfitValue(obsT, tt, objective) -> number`
  - `misfitField({xs, ys, sz, px, py, pz, obsT, c, objective}) -> {field: Float64Array, nx, ny}` with `field[j*nx + i]` matching Python's `field[j, i]`

- [ ] **Step 1: Write the fixture generator**

Create `viz/make_parity_fixture.py`:

```python
"""Generate the committed fixture that pins the JS misfit to the Python misfit."""

import json
from pathlib import Path

import numpy as np

from dclde_localization.physics.forward import DEFAULT_C, misfit_field, misfit_value, travel_time

OUT = Path("viz/fixtures/parity_fixture.json")


def main() -> None:
    rng = np.random.default_rng(20260910)
    n = 12
    px = np.linspace(14_000.0, 61_000.0, n)
    py = 20_000.0 + 300.0 * np.sin(px / 9_000.0)
    pz = -400.0 - 300.0 * np.cos(px / 15_000.0)
    obs = 6.5 + np.sqrt((px - 37_000.0) ** 2 + (py - 22_300.0) ** 2 + (pz + 30.0) ** 2) / DEFAULT_C
    obs = obs + rng.normal(0.0, 0.02, n)

    xs = np.linspace(30_000.0, 44_000.0, 9)
    ys = np.linspace(16_000.0, 28_000.0, 7)
    sz = -30.0

    payload = {
        "c": DEFAULT_C,
        "sz": sz,
        "objective": "l1",
        "px": px.tolist(),
        "py": py.tolist(),
        "pz": pz.tolist(),
        "obsT": obs.tolist(),
        "xs": xs.tolist(),
        "ys": ys.tolist(),
    }
    for objective in ("l1", "l2"):
        f = misfit_field(xs, ys, sz, px, py, pz, obs, DEFAULT_C, objective)
        payload[f"expected_{objective}"] = f.ravel().tolist()

    # Scalar-path parity too. Without this, misfitValue would be exported and
    # never exercised in either language against non-degenerate residuals.
    tt_ref = travel_time(37_000.0, 22_300.0, sz, px, py, pz, DEFAULT_C)
    payload["ttRef"] = tt_ref.tolist()
    for objective in ("l1", "l2"):
        payload[f"expected_value_{objective}"] = misfit_value(obs, tt_ref, objective)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the fixture**

```bash
uv run python viz/make_parity_fixture.py
```

Expected: writes `viz/fixtures/parity_fixture.json`, roughly 4–6 KB.

- [ ] **Step 3: Write the failing parity test**

Create `viz/package.json`:

```json
{
  "name": "dclde-viz",
  "private": true,
  "type": "module",
  "scripts": {
    "test": "node --test tests/"
  }
}
```

Create `viz/tests/test_parity.mjs`:

```javascript
import { readFileSync } from "node:fs";
import test from "node:test";
import assert from "node:assert/strict";

import { misfitField, misfitValue, travelTime, bestT0 } from "../template/lib/forward.mjs";

const fx = JSON.parse(readFileSync(new URL("../fixtures/parity_fixture.json", import.meta.url)));

test("travelTime is euclidean over c", () => {
  const t = travelTime(0, 0, 0, [3000], [4000], [0], 1000);
  assert.ok(Math.abs(t[0] - 5) < 1e-12);
});

test("bestT0 matches the l1 median and l2 mean", () => {
  const tt = [1, 2, 3];
  const obs = [11, 13, 33];
  assert.ok(Math.abs(bestT0(obs, tt, "l1") - 11) < 1e-12);
  assert.ok(Math.abs(bestT0(obs, tt, "l2") - 17) < 1e-12);
});

test("bestT0 averages the two middle residuals for an even count", () => {
  // This is the ONLY test that can pin numpy's even-length median convention.
  // The l1 misfit is mathematically invariant to any t0 between the two central
  // order statistics, so a median() that returned a[mid] instead of averaging
  // would produce an identical l1 field and slip past every parity assertion.
  const tt = [1, 2, 3, 4];
  const obs = [11, 13, 15, 34]; // residuals [10, 11, 12, 30]
  assert.ok(Math.abs(bestT0(obs, tt, "l1") - 11.5) < 1e-12);
  assert.ok(Math.abs(bestT0(obs, tt, "l2") - 15.75) < 1e-12);
});

test("an unknown objective throws rather than returning NaN", () => {
  assert.throws(() => bestT0([1], [1], "l3"), /unknown objective/);
  assert.throws(() => misfitValue([1], [1], "l3"), /unknown objective/);
  assert.throws(
    () => misfitField({
      xs: [0], ys: [0], sz: 0, px: [0], py: [0], pz: [0],
      obsT: [0], c: 1480, objective: "l3",
    }),
    /unknown objective/,
  );
});

for (const objective of ["l1", "l2"]) {
  test(`misfitValue matches the Python scalar misfit (${objective})`, () => {
    const got = misfitValue(fx.obsT, fx.ttRef, objective);
    const want = fx[`expected_value_${objective}`];
    const rel = Math.abs(got - want) / Math.max(Math.abs(want), 1e-12);
    assert.ok(rel < 1e-6, `${objective}: got ${got}, want ${want} (rel ${rel})`);
  });
}

for (const objective of ["l1", "l2"]) {
  test(`misfitField matches the Python field (${objective})`, () => {
    const { field, nx, ny } = misfitField({
      xs: fx.xs, ys: fx.ys, sz: fx.sz,
      px: fx.px, py: fx.py, pz: fx.pz,
      obsT: fx.obsT, c: fx.c, objective,
    });
    const expected = fx[`expected_${objective}`];
    assert.equal(nx, fx.xs.length);
    assert.equal(ny, fx.ys.length);
    assert.equal(field.length, expected.length);

    let worst = 0;
    for (let k = 0; k < expected.length; k++) {
      const denom = Math.max(Math.abs(expected[k]), 1e-12);
      worst = Math.max(worst, Math.abs(field[k] - expected[k]) / denom);
    }
    assert.ok(worst < 1e-6, `max relative difference ${worst} exceeds 1e-6`);
  });
}
```

- [ ] **Step 4: Run the test to verify it fails**

```bash
cd viz && npm test
```

Expected: FAIL — `Cannot find module '.../template/lib/forward.mjs'`.

- [ ] **Step 5: Write the JS implementation**

Create `viz/template/lib/forward.mjs`. Note the `export` keywords: the build step in Task 7 strips them when inlining, and a build test asserts none survive.

```javascript
// Mirror of src/dclde_localization/physics/forward.py. Held together by viz/tests/test_parity.mjs.
// Keep this boringly explicit; a clever rewrite here silently corrupts every picture.

export const DEFAULT_C = 1480.0;

export function travelTime(sx, sy, sz, px, py, pz, c) {
  const n = px.length;
  const out = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const dx = px[k] - sx;
    const dy = py[k] - sy;
    const dz = pz[k] - sz;
    out[k] = Math.sqrt(dx * dx + dy * dy + dz * dz) / c;
  }
  return out;
}

function median(values) {
  const a = Float64Array.from(values).sort();
  const n = a.length;
  if (n === 0) return NaN;
  const mid = n >> 1;
  return n % 2 ? a[mid] : 0.5 * (a[mid - 1] + a[mid]);
}

function mean(values) {
  let s = 0;
  for (let k = 0; k < values.length; k++) s += values[k];
  return s / values.length;
}

export function bestT0(obsT, tt, objective = "l1") {
  const n = obsT.length;
  const r = new Float64Array(n);
  for (let k = 0; k < n; k++) r[k] = obsT[k] - tt[k];
  if (objective === "l1") return median(r);
  if (objective === "l2") return mean(r);
  throw new Error(`unknown objective: ${objective}`);
}

export function misfitValue(obsT, tt, objective = "l1") {
  const n = obsT.length;
  const t0 = bestT0(obsT, tt, objective);
  if (objective === "l1") {
    let s = 0;
    for (let k = 0; k < n; k++) s += Math.abs(obsT[k] - tt[k] - t0);
    return s;
  }
  let s = 0;
  for (let k = 0; k < n; k++) {
    const d = obsT[k] - tt[k] - t0;
    s += d * d;
  }
  return Math.sqrt(s / n);
}

export function misfitField({ xs, ys, sz, px, py, pz, obsT, c, objective = "l1" }) {
  const nx = xs.length;
  const ny = ys.length;
  const np_ = px.length;
  const field = new Float64Array(nx * ny);

  const dz2 = new Float64Array(np_);
  for (let k = 0; k < np_; k++) {
    const dz = pz[k] - sz;
    dz2[k] = dz * dz;
  }

  const r = new Float64Array(np_);
  for (let j = 0; j < ny; j++) {
    const wy = ys[j];
    for (let i = 0; i < nx; i++) {
      const wx = xs[i];
      for (let k = 0; k < np_; k++) {
        const dx = px[k] - wx;
        const dy = py[k] - wy;
        r[k] = obsT[k] - Math.sqrt(dx * dx + dy * dy + dz2[k]) / c;
      }
      let acc = 0;
      if (objective === "l1") {
        const t0 = median(r);
        for (let k = 0; k < np_; k++) acc += Math.abs(r[k] - t0);
      } else if (objective === "l2") {
        const t0 = mean(r);
        for (let k = 0; k < np_; k++) {
          const d = r[k] - t0;
          acc += d * d;
        }
        acc = Math.sqrt(acc / np_);
      } else {
        throw new Error(`unknown objective: ${objective}`);
      }
      field[j * nx + i] = acc;
    }
  }
  return { field, nx, ny };
}
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
cd viz && npm test
```

Expected: 4 passed. A failure here means the two implementations disagree — fix the code, never the tolerance.

- [ ] **Step 7: Commit**

```bash
git add viz/package.json viz/template/lib/forward.mjs viz/make_parity_fixture.py viz/fixtures/parity_fixture.json viz/tests/test_parity.mjs
git commit -m "Mirror the forward model in JS with a Python parity test"
```

---

### Task 4: Picks loader and call keys

**Files:**
- Create: `src/dclde_localization/physics/picks.py`
- Create: `tests/physics/test_picks.py`

**Interfaces:**
- Consumes: `CableGeometry` from Task 1.
- Produces:
  - `PICKS_SNAPSHOT = "fixtures/das_calls.csv.gz"`
  - `call_key(filename: str, call_id: int) -> str`
  - `load_picks(path=PICKS_SNAPSHOT) -> pd.DataFrame` with columns `call, cable, dist, time, snr, call_type, batch, window, call_id`
  - `attach_xyz(picks: pd.DataFrame, cables: dict[str, CableGeometry]) -> pd.DataFrame` adding `px, py, pz`

The committed snapshot is the source of truth here — it holds all 11,117 picks and needs no network access.

- [ ] **Step 1: Write the failing tests**

Create `tests/physics/test_picks.py`:

```python
import numpy as np
import pytest

from dclde_localization.physics.geometry import load_cable
from dclde_localization.physics.picks import attach_xyz, call_key, load_picks


def test_call_key_normalises_colons_and_dashes():
    assert call_key("Denoised_SNR_North_2021-11-04_02:00:02.nc", 1) == "2021-11-04_02_00_02/1"
    assert call_key("Denoised_SNR_South_2021-11-04_02-00-02.nc", 7) == "2021-11-04_02_00_02/7"
    assert call_key("annotated_calls_north_2021-11-04_02_00_02.csv", 3) == "2021-11-04_02_00_02/3"


def test_call_key_rejects_a_filename_with_no_timestamp():
    with pytest.raises(ValueError, match="no timestamp"):
        call_key("Denoised_SNR_North.nc", 1)


@pytest.fixture(scope="module")
def picks():
    return load_picks()


def test_load_picks_matches_known_totals(picks):
    assert len(picks) == 11117
    assert picks["call"].nunique() == 628
    assert picks["window"].nunique() == 60
    assert set(picks["cable"].unique()) == {"North", "South"}


def test_keys_on_both_cables_match_the_verified_count(picks):
    per_key = picks.groupby("call")["cable"].nunique()
    assert int((per_key == 2).sum()) == 166


@pytest.fixture(scope="module")
def cables():
    return {n: load_cable(n, f"fixtures/{n.lower()}_DAS_multicoord.csv")
            for n in ("North", "South")}


def test_attach_xyz_fills_every_row(picks, cables):
    out = attach_xyz(picks, cables)
    assert {"px", "py", "pz"} <= set(out.columns)
    assert out[["px", "py", "pz"]].notna().all().all()
    assert len(out) == len(picks)


@pytest.mark.parametrize("name,other", [("North", "South"), ("South", "North")])
def test_attach_xyz_uses_each_cable_its_own_geometry(picks, cables, name, other):
    # The real failure mode is cross-wiring: giving South's picks positions from
    # the North cable. Comparing against the SAME interpolation attach_xyz used
    # would be circular and would report exactly zero error, so instead require
    # that each cable's picks are clearly NOT explained by the other cable.
    out = attach_xyz(picks, cables)
    sub = out[out.cable == name]
    wx, wy, _ = cables[other].chan_to_xyz(sub["dist"].to_numpy())
    d = np.hypot(sub["px"].to_numpy() - wx, sub["py"].to_numpy() - wy)
    assert d.max() > 1000.0, f"{name} picks are indistinguishable from {other} geometry"


def test_attach_xyz_raises_on_an_unknown_cable_label(picks, cables):
    bad = picks.head(50).copy()
    bad.loc[bad.index[0], "cable"] = "Souh"
    with pytest.raises(ValueError, match="unknown cable"):
        attach_xyz(bad, cables)


@pytest.mark.parametrize("name,csv", [
    ("North", "fixtures/north_DAS_multicoord.csv"),
    ("South", "fixtures/south_DAS_multicoord.csv"),
])
def test_chan_m_matches_the_cables_own_arc_length(name, csv):
    # An independent check on what `dist` MEANS, which attach_xyz depends on
    # entirely: the dataset's along-cable distance should equal the arc length of
    # its own coordinate polyline. This compares the chan_m column against the
    # x/y/depth columns, so unlike an interpolation round-trip it would catch a
    # wrong unit or offset in `dist`. Measured agreement is 0.19% (North) and
    # 0.04% (South) of span; fiber slack makes exact equality unlikely.
    cable = load_cable(name, csv)
    seg = np.sqrt(np.diff(cable.x) ** 2 + np.diff(cable.y) ** 2 + np.diff(cable.depth) ** 2)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    dm = cable.chan_m - cable.chan_m[0]
    assert np.abs(arc - dm).max() < 0.005 * dm[-1]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/physics/test_picks.py -v
```

Expected: `ModuleNotFoundError: No module named 'viz'`.

- [ ] **Step 3: Make `viz` importable and write the implementation**

Create `viz/__init__.py` (empty) and `viz/tests/__init__.py` (empty). Create `src/dclde_localization/physics/picks.py`:

```python
"""Load the annotated arrival picks and give every call a stable key.

Source is the committed snapshot fixtures/das_calls.csv.gz — all
11,117 picks, no network access needed.

The dataset writes the same window timestamp three ways (colons in the .nc
filename, dashes in the bucket object name, underscores in the annotation CSV
name), so keys are always underscore-normalised.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PICKS_SNAPSHOT = "fixtures/das_calls.csv.gz"

_TS = re.compile(r"(\d{4}-\d{2}-\d{2})[_ ](\d{2})[:_-](\d{2})[:_-](\d{2})")


def call_key(filename: str, call_id: int) -> str:
    found = _TS.findall(str(filename))
    if not found:
        raise ValueError(f"no timestamp in filename: {filename!r}")
    # Silently taking the first of several timestamps would assign a wrong key,
    # which is worse than failing: the key is the join axis for everything.
    if len({f for f in found}) > 1:
        raise ValueError(f"ambiguous timestamps in filename: {filename!r} -> {found}")
    date, hh, mm, ss = found[0]
    return f"{date}_{hh}_{mm}_{ss}/{int(call_id)}"


def load_picks(path: str | Path = PICKS_SNAPSHOT) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["cable"] = df["cable"].str.strip().str.capitalize()
    df["call"] = [call_key(f, c) for f, c in zip(df["filename"], df["call_id"])]
    df["window"] = df["call"].str.split("/").str[0]
    keep = ["call", "window", "call_id", "cable", "dist", "time", "snr", "call_type", "batch"]
    return df[keep].reset_index(drop=True)


def attach_xyz(picks: pd.DataFrame, cables: dict) -> pd.DataFrame:
    # Fail fast on a label we have no geometry for. Leaving those rows as NaN
    # would silently drop picks from every downstream join instead.
    unknown = set(picks["cable"].unique()) - set(cables)
    if unknown:
        raise ValueError(f"unknown cable label(s) in picks: {sorted(unknown)}")

    out = picks.copy()
    out[["px", "py", "pz"]] = np.nan
    for name, cable in cables.items():
        m = out["cable"] == name
        if not m.any():
            continue
        x, y, z = cable.chan_to_xyz(out.loc[m, "dist"].to_numpy())
        out.loc[m, "px"] = x
        out.loc[m, "py"] = y
        out.loc[m, "pz"] = z
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest tests/physics/test_picks.py -v
```

Expected: 5 passed. The counts 11,117 / 628 / 60 / 166 are verified facts about this dataset; if they change, the snapshot changed and that needs investigating before proceeding.

- [ ] **Step 5: Commit**

```bash
git add viz/__init__.py src/dclde_localization/physics/picks.py viz/tests/
git commit -m "Add pick loader with underscore-normalised call keys"
```

---

### Task 5: Bathymetry fetch, decode, reprojection and validation

**Files:**
- Create: `viz/bathymetry.py`
- Create: `viz/tests/test_bathymetry.py`

**Interfaces:**
- Consumes: `CableGeometry`, `FrameTransform` from Task 1.
- Produces:
  - `GMRT_BOUNDS = (-125.35, -123.90, 44.95, 45.40)` as `(min_lon, max_lon, min_lat, max_lat)`
  - `fetch_gmrt(bounds=GMRT_BOUNDS, dest=Path("fixtures/gmrt_bathy.nc")) -> Path`
  - `GeoGrid` frozen dataclass `(lon: np.ndarray, lat: np.ndarray, z: np.ndarray)` with `z` shaped `(nlat, nlon)`, ascending latitude
  - `decode_gmt_grid(path) -> GeoGrid`
  - `LocalGrid` frozen dataclass `(x0, y0, dx, dy, nx, ny, z: np.ndarray)` where `z` is int16 metres and `-32768` marks no-data
  - `reproject_to_local(grid: GeoGrid, transform: FrameTransform, nx=663, ny=294) -> LocalGrid`
  - `ValidationReport` frozen dataclass `(n, median_offset_m, correlation)`
  - `validate_against_cable(grid: GeoGrid, cable: CableGeometry) -> ValidationReport`

- [ ] **Step 1: Fetch the grid once and vendor it**

The tests need this file as an input, and `fetch_gmrt` does not exist yet, so fetch it directly:

```bash
mkdir -p fixtures
curl -sL --max-time 120 -o fixtures/gmrt_bathy.nc "https://www.gmrt.org/services/GridServer?minlongitude=-125.35&maxlongitude=-123.90&minlatitude=44.95&maxlatitude=45.40&format=netcdf&resolution=default&layer=topo"
ls -l fixtures/gmrt_bathy.nc
```

Expected: about 780 KB. This is a vendored input (GMRT is not guaranteed byte-stable over time, and the page must build offline), so it is committed despite the `*.nc` rule in `.gitignore` — add the exception:

```
!fixtures/gmrt_bathy.nc
```

- [ ] **Step 2: Write the failing tests**

Create `viz/tests/test_bathymetry.py`:

```python
import numpy as np
import pytest

from dclde_localization.physics.geometry import FrameTransform, load_cable
from viz.bathymetry import (
    NO_DATA,
    _sample,
    decode_gmt_grid,
    reproject_to_local,
    validate_against_cable,
)

GRID = "fixtures/gmrt_bathy.nc"
SOUTH_CSV = "fixtures/south_DAS_multicoord.csv"


@pytest.fixture(scope="module")
def grid():
    return decode_gmt_grid(GRID)


def test_decode_shape_and_orientation(grid):
    # Pinned dimensions rather than `z.shape == (len(lat), len(lon))`: that form
    # cannot fail, since both sides come from the same nx/ny. Pinning also
    # catches GMRT silently changing its response resolution, which would change
    # the bundle payload size.
    assert (len(grid.lon), len(grid.lat)) == (663, 294)
    assert grid.z.shape == (294, 663)
    assert grid.lat[0] < grid.lat[-1], "latitude must be ascending"
    assert grid.lon[0] < grid.lon[-1]
    assert -3000 < np.nanmin(grid.z) < 0 < np.nanmax(grid.z) < 2000


@pytest.mark.parametrize("k", [10, 100])
def test_sample_at_an_exact_grid_node_returns_that_node(grid, k):
    got = _sample(grid, np.array([grid.lat[k]]), np.array([grid.lon[k]]))
    assert np.array_equal(got, np.array([grid.z[k, k]]), equal_nan=True)


def test_sample_is_nearest_not_floor(grid):
    # 60% of the way from node 10 to node 11: a floor lookup returns node 10,
    # true nearest-neighbour returns node 11. Measured on the real grid, the two
    # conventions disagree on 74.9% of cable samples.
    dlat = grid.lat[1] - grid.lat[0]
    dlon = grid.lon[1] - grid.lon[0]
    got = _sample(
        grid,
        np.array([grid.lat[10] + 0.6 * dlat]),
        np.array([grid.lon[10] + 0.6 * dlon]),
    )
    assert got[0] == pytest.approx(grid.z[11, 11])
    assert got[0] != pytest.approx(grid.z[10, 10])


def test_sample_marks_out_of_bounds_as_nan(grid):
    got = _sample(grid, np.array([0.0, 89.0]), np.array([0.0, 179.0]))
    assert np.isnan(got).all()


def test_gmrt_agrees_with_the_cables_own_depth(grid):
    # Independent cross-validation of both the bathymetry and the cable
    # geolocation: these are different measurements of the same seafloor.
    cable = load_cable("South", SOUTH_CSV)
    rep = validate_against_cable(grid, cable)
    assert rep.n > 40000
    assert abs(rep.median_offset_m) < 15.0
    assert rep.correlation > 0.999


def test_reproject_to_local_preserves_depth_at_the_cable(grid):
    cable = load_cable("South", SOUTH_CSV)
    t = FrameTransform.from_cable(cable)
    lg = reproject_to_local(grid, t, nx=663, ny=294)

    assert lg.z.dtype == np.int16
    assert lg.nx == 663 and lg.ny == 294
    assert lg.z.shape == (lg.ny, lg.nx)

    i = ((cable.x - lg.x0) / lg.dx).round().astype(int)
    j = ((cable.y - lg.y0) / lg.dy).round().astype(int)
    inside = (i >= 0) & (i < lg.nx) & (j >= 0) & (j < lg.ny)
    sampled = lg.z[j[inside], i[inside]].astype(float)
    ok = sampled != NO_DATA
    diff = sampled[ok] - cable.depth[inside][ok]
    assert np.median(np.abs(diff)) < 30.0


def test_no_data_is_marked_not_zero(grid):
    cable = load_cable("South", SOUTH_CSV)
    lg = reproject_to_local(grid, FrameTransform.from_cable(cable))
    assert NO_DATA == -32768
    # The source grid has NaN cells; they must survive as the sentinel, because
    # letting them become 0 m would put fake sea level in the colour ramp.
    assert (lg.z == NO_DATA).sum() > 0
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
uv run pytest viz/tests/test_bathymetry.py -v
```

Expected: `ModuleNotFoundError: No module named 'viz.bathymetry'`.

- [ ] **Step 4: Write the implementation**

Create `viz/bathymetry.py`:

```python
"""GMRT bathymetry: fetch, decode the legacy GMT grid, reproject to local XY.

GMRT's GridServer returns a COARDS/legacy GMT netCDF where z is a flat array
stored top-down with x_range/y_range/dimension headers rather than coordinate
variables, so it needs decoding by hand.

Validated against the cables' own reported depth: median offset -0.6 m,
r = 0.9999 over 46,440 south-cable channels. That is an independent check on
both the bathymetry and the cable geolocation, so it is kept as a test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import xarray as xr

GMRT_BOUNDS = (-125.35, -123.90, 44.95, 45.40)
GMRT_URL = "https://www.gmrt.org/services/GridServer"
NO_DATA = -32768


def fetch_gmrt(bounds=GMRT_BOUNDS, dest: str | Path = "fixtures/gmrt_bathy.nc") -> Path:
    min_lon, max_lon, min_lat, max_lat = bounds
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(
        GMRT_URL,
        params={
            "minlongitude": min_lon, "maxlongitude": max_lon,
            "minlatitude": min_lat, "maxlatitude": max_lat,
            "format": "netcdf", "resolution": "default", "layer": "topo",
        },
        timeout=180,
    )
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


@dataclass(frozen=True)
class GeoGrid:
    lon: np.ndarray
    lat: np.ndarray
    z: np.ndarray  # (nlat, nlon), ascending latitude, NaN for no-data


def decode_gmt_grid(path: str | Path) -> GeoGrid:
    ds = xr.open_dataset(path)
    nx, ny = (int(v) for v in ds["dimension"].values)
    x_range = ds["x_range"].values
    y_range = ds["y_range"].values
    z = ds["z"].values.reshape(ny, nx)[::-1]  # stored top-down
    return GeoGrid(
        lon=np.linspace(float(x_range[0]), float(x_range[1]), nx),
        lat=np.linspace(float(y_range[0]), float(y_range[1]), ny),
        z=z.astype(float),
    )


@dataclass(frozen=True)
class LocalGrid:
    x0: float
    y0: float
    dx: float
    dy: float
    nx: int
    ny: int
    z: np.ndarray  # int16 metres, NO_DATA sentinel


def _sample(grid: GeoGrid, lat, lon) -> np.ndarray:
    """Nearest-neighbour sample; out-of-bounds and no-data return NaN.

    Index arithmetic on the regular grid, not searchsorted. `searchsorted(a, v) - 1`
    returns the LOWER neighbour for every interior query, never the nearer one,
    which biases every sample toward the grid's south-west by up to a full native
    cell (173 m here, ~87 m on average) and wrongly excludes a query landing
    exactly on the first row or column. Measured against the south cable, the two
    differ on 74.9% of samples. `grid.lon`/`grid.lat` are linspaces, so the
    rounding below is exact.
    """
    dlon = grid.lon[1] - grid.lon[0]
    dlat = grid.lat[1] - grid.lat[0]
    i = np.round((np.asarray(lon, float) - grid.lon[0]) / dlon).astype(int)
    j = np.round((np.asarray(lat, float) - grid.lat[0]) / dlat).astype(int)
    ok = (i >= 0) & (i < len(grid.lon)) & (j >= 0) & (j < len(grid.lat))
    out = np.full(np.shape(lat), np.nan, dtype=float)
    out[ok] = grid.z[j[ok], i[ok]]
    return out


def reproject_to_local(grid: GeoGrid, transform, nx: int = 663, ny: int = 294) -> LocalGrid:
    """Resample the geographic grid onto a regular local-XY grid.

    The output box is taken from the four geographic corners. That is only a
    rectangle in the local frame up to projection distortion; over this domain
    (~115 x 50 km, within 2.5 deg of UTM 10N's central meridian) the bowing is a
    few metres against ~173 m cells, so nothing clips. It would not hold for a
    much larger or more off-meridian domain.
    """
    corner_lat = grid.lat[[0, 0, -1, -1]]
    corner_lon = grid.lon[[0, -1, 0, -1]]
    cx, cy = transform.wgs84_to_local(corner_lat, corner_lon)
    x0, x1 = float(cx.min()), float(cx.max())
    y0, y1 = float(cy.min()), float(cy.max())

    dx = (x1 - x0) / (nx - 1)
    dy = (y1 - y0) / (ny - 1)
    gx, gy = np.meshgrid(x0 + dx * np.arange(nx), y0 + dy * np.arange(ny))
    lat, lon = transform.local_to_wgs84(gx.ravel(), gy.ravel())
    z = _sample(grid, lat, lon).reshape(ny, nx)

    zi = np.where(np.isfinite(z), np.clip(np.round(z), -32767, 32767), NO_DATA)
    return LocalGrid(x0, y0, dx, dy, nx, ny, zi.astype(np.int16))


@dataclass(frozen=True)
class ValidationReport:
    n: int
    median_offset_m: float
    correlation: float


def validate_against_cable(grid: GeoGrid, cable) -> ValidationReport:
    z = _sample(grid, cable.lat, cable.lon)
    ok = np.isfinite(z)
    return ValidationReport(
        n=int(ok.sum()),
        median_offset_m=float(np.median(z[ok] - cable.depth[ok])),
        correlation=float(np.corrcoef(z[ok], cable.depth[ok])[0, 1]),
    )
```

- [ ] **Step 5: Add `requests` to dependencies**

Add `requests` to `requirements.txt` under `# config, notebooks, tooling`, then `uv pip install -e .`.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
uv run pytest viz/tests/test_bathymetry.py -v
```

Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .gitignore viz/bathymetry.py viz/tests/test_bathymetry.py fixtures/gmrt_bathy.nc
git commit -m "Add GMRT bathymetry decode, local reprojection and cable cross-validation"
```

---

### Task 6: Results contract, frame guard and baseline importer

**Files:**
- Create: `viz/results.py`
- Create: `viz/tests/test_results.py`
- Create: `viz/results/goestchel_2026_batch1.json` (generated, committed)

**Interfaces:**
- Consumes: `FrameTransform`, `CableGeometry` from Task 1.
- Produces:
  - `VALID_KINDS = ("estimate", "ground_truth", "synthetic_truth")`
  - `FRAME_GUARD_THRESHOLD_M = 25_000.0`
  - `Solution` frozen dataclass `(call, x, y, z, z_fixed, t0, wrms, n_picks, cables_used, unc)`
  - `ResultSet` frozen dataclass `(name, kind, frame, produced_by, c_assumed, notes, solutions)`
  - `load_result_set(path) -> ResultSet` (validates, raises `ValueError` with a named reason)
  - `to_local(rs, transform) -> ResultSet` (returns a copy with `frame="local_xy"`)
  - `GuardReport` frozen dataclass `(median_offset_m, suspect)`
  - `frame_guard(rs_local, cables) -> GuardReport`
  - `baseline_csv_to_result_set(csv_path, name) -> ResultSet`

- [ ] **Step 1: Write the failing tests**

Create `viz/tests/test_results.py`:

```python
import json

import pytest

from dclde_localization.physics.geometry import FrameTransform, load_cable
from viz.results import (
    FRAME_GUARD_THRESHOLD_M,
    frame_guard,
    load_result_set,
    to_local,
)

NORTH_CSV = "fixtures/north_DAS_multicoord.csv"
SOUTH_CSV = "fixtures/south_DAS_multicoord.csv"


def _write(tmp_path, payload):
    p = tmp_path / "rs.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _minimal(**over):
    base = {
        "name": "test", "kind": "estimate", "frame": "local_xy",
        "produced_by": "unit test", "c_assumed": 1480.0,
        "solutions": [{"call": "2021-11-04_02_00_02/1", "x": 37000.0, "y": 22344.0,
                       "z": -30.0, "z_fixed": True, "n_picks": 41}],
    }
    base.update(over)
    return base


def test_load_minimal_result_set(tmp_path):
    rs = load_result_set(_write(tmp_path, _minimal()))
    assert rs.name == "test"
    assert rs.solutions[0].z_fixed is True
    assert rs.solutions[0].wrms is None


def test_missing_frame_is_rejected(tmp_path):
    payload = _minimal()
    del payload["frame"]
    with pytest.raises(ValueError, match="frame"):
        load_result_set(_write(tmp_path, payload))


def test_unknown_frame_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown frame"):
        load_result_set(_write(tmp_path, _minimal(frame="mercator")))


def test_unknown_kind_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown kind"):
        load_result_set(_write(tmp_path, _minimal(kind="guess")))


def test_non_finite_coordinate_is_rejected(tmp_path):
    payload = _minimal()
    payload["solutions"][0]["x"] = None
    with pytest.raises(ValueError, match="non-finite"):
        load_result_set(_write(tmp_path, payload))


@pytest.fixture(scope="module")
def cables():
    return {"North": load_cable("North", NORTH_CSV), "South": load_cable("South", SOUTH_CSV)}


def test_frame_guard_passes_a_correct_set(tmp_path, cables):
    rs = load_result_set(_write(tmp_path, _minimal()))
    t = FrameTransform.from_cable(cables["North"])
    rep = frame_guard(to_local(rs, t), cables)
    assert not rep.suspect
    assert rep.median_offset_m < 5_000.0


def test_frame_guard_flags_a_misframed_set(tmp_path, cables):
    # The reference implementation's published lat/lon land ~270 km west of the
    # cable; declaring those numbers as local metres must be caught.
    bad = _minimal()
    bad["solutions"][0]["x"] = 500_000.0
    bad["solutions"][0]["y"] = 500_000.0
    rs = load_result_set(_write(tmp_path, bad))
    t = FrameTransform.from_cable(cables["North"])
    rep = frame_guard(to_local(rs, t), cables)
    assert rep.suspect
    assert rep.median_offset_m > FRAME_GUARD_THRESHOLD_M


def test_to_local_roundtrips_via_utm(tmp_path, cables):
    t = FrameTransform.from_cable(cables["North"])
    ux, uy = t.local_to_utm(37000.0, 22344.0)
    payload = _minimal(frame="utm10n")
    payload["solutions"][0]["x"] = float(ux)
    payload["solutions"][0]["y"] = float(uy)
    rs = to_local(load_result_set(_write(tmp_path, payload)), t)
    assert rs.frame == "local_xy"
    assert rs.solutions[0].x == pytest.approx(37000.0, abs=1e-3)
    assert rs.solutions[0].y == pytest.approx(22344.0, abs=1e-3)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest viz/tests/test_results.py -v
```

Expected: `ModuleNotFoundError: No module named 'viz.results'`.

- [ ] **Step 3: Write the implementation**

Create `viz/results.py`:

```python
"""The pluggable results contract, plus the frame guard.

`frame` is mandatory and never inferred: the reference implementation shipped
lat/lon roughly 270 km west of the cable (its local coordinates were fine, its
local->UTM->geographic conversion was not). The guard turns that class of bug
into an automatic check on every contributed file.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from dclde_localization.physics.geometry import VALID_FRAMES

VALID_KINDS = ("estimate", "ground_truth", "synthetic_truth")
FRAME_GUARD_THRESHOLD_M = 25_000.0


@dataclass(frozen=True)
class Solution:
    call: str
    x: float
    y: float
    z: float | None = None
    z_fixed: bool | None = None
    t0: float | None = None
    wrms: float | None = None
    n_picks: int | None = None
    cables_used: tuple[str, ...] = ()
    unc: dict | None = None


@dataclass(frozen=True)
class ResultSet:
    name: str
    kind: str
    frame: str
    produced_by: str
    c_assumed: float | None
    notes: str
    solutions: tuple[Solution, ...]


def _finite(value, label):
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"non-finite {label}: {value!r}")
    return float(value)


def load_result_set(path: str | Path) -> ResultSet:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    for field in ("name", "kind", "frame", "solutions"):
        if field not in raw:
            raise ValueError(f"result set is missing required field: {field}")
    if raw["frame"] not in VALID_FRAMES:
        raise ValueError(f"unknown frame: {raw['frame']!r}; expected one of {VALID_FRAMES}")
    if raw["kind"] not in VALID_KINDS:
        raise ValueError(f"unknown kind: {raw['kind']!r}; expected one of {VALID_KINDS}")

    sols = []
    for s in raw["solutions"]:
        if "call" not in s:
            raise ValueError("solution is missing its call key")
        sols.append(
            Solution(
                call=str(s["call"]),
                x=_finite(s.get("x"), "x"),
                y=_finite(s.get("y"), "y"),
                z=None if s.get("z") is None else float(s["z"]),
                z_fixed=s.get("z_fixed"),
                t0=s.get("t0"),
                wrms=s.get("wrms"),
                n_picks=s.get("n_picks"),
                cables_used=tuple(s.get("cables_used", ())),
                unc=s.get("unc"),
            )
        )

    return ResultSet(
        name=str(raw["name"]),
        kind=raw["kind"],
        frame=raw["frame"],
        produced_by=str(raw.get("produced_by", "")),
        c_assumed=raw.get("c_assumed"),
        notes=str(raw.get("notes", "")),
        solutions=tuple(sols),
    )


def to_local(rs: ResultSet, transform) -> ResultSet:
    if rs.frame == "local_xy":
        return rs
    xs = np.array([s.x for s in rs.solutions], dtype=float)
    ys = np.array([s.y for s in rs.solutions], dtype=float)
    lx, ly = transform.to_local(rs.frame, xs, ys)
    sols = tuple(
        replace(s, x=float(a), y=float(b)) for s, a, b in zip(rs.solutions, lx, ly)
    )
    return replace(rs, frame="local_xy", solutions=sols)


@dataclass(frozen=True)
class GuardReport:
    median_offset_m: float
    suspect: bool


def frame_guard(rs_local: ResultSet, cables: dict) -> GuardReport:
    """Median distance from each solution to the nearest cable channel."""
    if rs_local.frame != "local_xy":
        raise ValueError("frame_guard expects a set already converted to local_xy")

    cx = np.concatenate([c.x for c in cables.values()])
    cy = np.concatenate([c.y for c in cables.values()])
    # Decimate the cable for the distance scan; 40 m resolution is far below
    # the 25 km threshold this guard tests.
    cx, cy = cx[::20], cy[::20]

    d = []
    for s in rs_local.solutions:
        d.append(np.sqrt(((cx - s.x) ** 2 + (cy - s.y) ** 2).min()))
    med = float(np.median(d)) if d else float("inf")
    return GuardReport(med, med > FRAME_GUARD_THRESHOLD_M)


def baseline_csv_to_result_set(csv_path: str | Path, name: str) -> ResultSet:
    """Import the reference implementation's released localization CSV.

    Its x_local/y_local are trustworthy (median 1.4 km from the nearest cable
    channel); its lat/lon and x_utm/y_utm are not, so they are ignored.
    Every row has z = -30.0, hence z_fixed=True.
    """
    df = pd.read_csv(csv_path)
    sols = tuple(
        Solution(
            call=str(r.get("call", f"row{i}")),
            x=float(r["x_local"]),
            y=float(r["y_local"]),
            z=float(r["z_local"]),
            z_fixed=True,
            wrms=float(r["wrms"]),
            unc={"type": "circle", "radius": float(r["deltax"])},
        )
        for i, r in df.iterrows()
    )
    return ResultSet(
        name=name,
        kind="estimate",
        frame="local_xy",
        produced_by="Goestchel, Wilcock & Abadi 2026, released code",
        c_assumed=1480.0,
        notes=(
            "Imported from batch1_localizations_with_coords.csv. Published lat/lon and "
            "x_utm/y_utm are discarded: they place calls ~270 km west of the cable. "
            "z is fixed at -30 m for all rows."
        ),
        solutions=sols,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest viz/tests/test_results.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Import the baseline result set**

The released CSV has no `call` column, so the imported set carries `row<i>` keys and will not join to picks. That is expected and acceptable for v1 — it renders as a comparison point cloud. Record it explicitly in `notes`.

```bash
mkdir -p viz/results
curl -sL -o /tmp/b1.csv https://raw.githubusercontent.com/Ocean-Data-Lab/Goestchel_JASA_2025b/main/batch1_localizations_with_coords.csv
uv run python -c "
import json, dataclasses
from viz.results import baseline_csv_to_result_set
rs = baseline_csv_to_result_set('/tmp/b1.csv', 'goestchel-2026-batch1')
d = dataclasses.asdict(rs)
open('viz/results/goestchel_2026_batch1.json','w').write(json.dumps(d, indent=1))
print(len(rs.solutions), 'solutions')
"
```

Expected: `176 solutions`.

- [ ] **Step 6: Commit**

```bash
git add viz/results.py viz/tests/test_results.py viz/results/goestchel_2026_batch1.json
git commit -m "Add results contract, frame guard and reference-implementation importer"
```

---

### Task 7: Bundle encoder and build CLI

**Files:**
- Create: `viz/bundle.py`
- Create: `viz/build_bundle.py`
- Create: `viz/template/app.html`
- Create: `viz/template/app.css`
- Create: `viz/template/app.js`
- Create: `viz/tests/test_bundle.py`

**Interfaces:**
- Consumes: everything from Tasks 1, 4, 5, 6.
- Produces:
  - `BUNDLE_VERSION = 1`
  - `encode_f32(a: np.ndarray) -> str`, `encode_i16(a: np.ndarray) -> str` (base64 of little-endian bytes)
  - `build_payload(cables, transform, picks, local_grid, result_sets, git_sha) -> dict`
  - `inline_module(path: Path) -> str` (strips `export ` / `import` lines for inlining)
  - `render_html(payload: dict, template_dir: Path) -> str`
  - CLI: `uv run python -m viz.build_bundle --out viz/out/index.html`

- [ ] **Step 1: Write the failing tests**

Create `viz/tests/test_bundle.py`:

```python
import base64
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from viz.bundle import BUNDLE_VERSION, encode_f32, encode_i16, inline_module, render_html


def test_encode_f32_roundtrips():
    a = np.array([1.5, -2.25, 1e6], dtype=np.float32)
    back = np.frombuffer(base64.b64decode(encode_f32(a)), dtype="<f4")
    assert np.allclose(back, a)


def test_encode_i16_roundtrips():
    a = np.array([0, -32768, 32767], dtype=np.int16)
    back = np.frombuffer(base64.b64decode(encode_i16(a)), dtype="<i2")
    assert np.array_equal(back, a)


def test_inline_module_strips_module_syntax(tmp_path):
    src = tmp_path / "m.mjs"
    src.write_text(
        "import { x } from './other.mjs';\n"
        "export const A = 1;\n"
        "export function f() { return A; }\n",
        encoding="utf-8",
    )
    out = inline_module(src)
    assert "import" not in out
    assert not re.search(r"^\s*export\s", out, re.M)
    assert "const A = 1;" in out
    assert "function f()" in out


def test_render_html_is_self_contained(tmp_path):
    payload = {"bundleVersion": BUNDLE_VERSION, "meta": {"c": 1480.0}, "picks": {}}
    html = render_html(payload, Path("viz/template"))

    assert html.lstrip().startswith("<title") or "<title" in html[:2000]
    # No runtime fetching of any kind: the Artifact CSP blocks it.
    for forbidden in ("fetch(", "XMLHttpRequest", "import("):
        assert forbidden not in html, f"template must not reference {forbidden}"
    # Relative asset references, in either quoting style.
    assert not re.search(r"""(?:src|href)\s*=\s*['"]\.?/""", html), "no relative asset references"
    # No external origin at all: the page carries no third-party library. Covers
    # protocol-relative URLs too, which a bare https?:// pattern would miss.
    assert re.findall(r'https?://[^"\'\s)]+', html) == [], "page must reference no external URL"
    assert not re.search(r"""(?:src|href)\s*=\s*['"]//""", html), "no protocol-relative URLs"
    assert "__PAYLOAD__" not in html, "payload placeholder was not substituted"


def test_render_html_stays_under_the_page_limit():
    from viz.bundle import PAGE_LIMIT_BYTES

    payload = {"bundleVersion": BUNDLE_VERSION, "meta": {}, "big": encode_f32(np.zeros(100, np.float32))}
    html = render_html(payload, Path("viz/template"))
    assert len(html.encode("utf-8")) < PAGE_LIMIT_BYTES


def test_render_html_refuses_an_oversized_page(monkeypatch):
    # Exercise the raise, not just the happy path.
    import viz.bundle as bundle_mod

    monkeypatch.setattr(bundle_mod, "PAGE_LIMIT_BYTES", 1000)
    with pytest.raises(ValueError, match="over the"):
        render_html({"bundleVersion": BUNDLE_VERSION}, Path("viz/template"))


def test_payload_cannot_break_out_of_the_script_block():
    # A result set's free-text field is the realistic carrier here, since the
    # contract is meant to accept third-party files.
    payload = {"bundleVersion": BUNDLE_VERSION, "notes": "</script><script>alert(1)</script>"}
    html = render_html(payload, Path("viz/template"))
    assert "</script><script>" not in html
    assert "\\u003c/script>" in html or "\\u003c/script\\u003e" in html
    # Exactly the script tags the template itself declares, and no more.
    assert html.count("<script") == html.count("</script>")


def test_page_declares_charset_and_viewport():
    # The Artifact host injects both of these, but the spec promises this same
    # file also runs from GitHub Pages and from disk, where nothing does.
    # Observed without them: served with no charset header the browser guessed
    # Latin-1 and rendered "filter callsâ€¦"; and with no viewport meta a phone
    # lays the page out at a forced 980px and scales the whole thing down to
    # illegibility. The charset must be inside the first 1024 bytes to take effect.
    html = render_html({"bundleVersion": BUNDLE_VERSION, "meta": {}}, Path("viz/template"))
    head = html[:1024].lower()
    assert 'charset="utf-8"' in head or "charset=utf-8" in head
    assert 'name="viewport"' in head
    assert "width=device-width" in head


def test_layout_does_not_let_the_canvas_force_overflow():
    # A plain `1fr` is `minmax(auto, 1fr)`, so the canvas's intrinsic width can
    # push the middle column past its share and scroll the body horizontally,
    # which the page must never do. Measured in a browser before this was pinned.
    css = (Path("viz/template") / "app.css").read_text(encoding="utf-8")
    grid = re.search(r"#app\s*\{[^}]*grid-template-columns:\s*([^;]+);", css)
    assert grid, "#app must set grid-template-columns"
    assert "minmax(0" in grid.group(1), f"middle column must be minmax(0, 1fr): {grid.group(1)}"
    # And it must stack rather than squeeze on a narrow viewport.
    assert re.search(r"@media\s*\(max-width:\s*\d+px\)", css), "needs a narrow-viewport stack"


# The validated palette. Cable hues are categorical slots 2 and 3; slot 1 (blue)
# is reserved for the sequential misfit ramp so the two encodings never share a
# hue. Checked all-pairs in both modes: CVD dE 9.2 light / 9.4 dark against a >=8
# target, normal-vision 27.6 / 26.5 against a >=15 floor. Pinned here because
# nothing else would notice the CSS drifting away from the validated values --
# which is exactly what happened once already.
VALIDATED_CABLE_HUES = {
    "light": {"--cable-n": "#eb6834", "--cable-s": "#1baf7a"},
    "dark": {"--cable-n": "#d95926", "--cable-s": "#199e70"},
}


def test_cable_hues_match_the_validated_palette():
    css = (Path("viz/template") / "app.css").read_text(encoding="utf-8")
    found = {}
    for token in ("--cable-n", "--cable-s"):
        found[token] = re.findall(rf"{token}:\s*(#[0-9a-fA-F]{{6}})", css)
        assert found[token], f"{token} not declared in app.css"

    light = {t: v[0].lower() for t, v in found.items()}
    assert light == VALIDATED_CABLE_HUES["light"], f"light cable hues drifted: {light}"

    # Dark values are stepped for the dark surface, not flipped, and are declared
    # twice (media query + data-theme scope) so the toggle wins both ways.
    for token, values in found.items():
        rest = {v.lower() for v in values[1:]}
        assert rest == {VALIDATED_CABLE_HUES["dark"][token]}, (
            f"{token} dark steps drifted or are inconsistent across the two scopes: {rest}"
        )


def test_template_app_version_matches_bundle_version():
    # Two independent literals in two files; without this they drift silently and
    # the page's own mismatch guard becomes the thing that is wrong.
    html = (Path("viz/template") / "app.html").read_text(encoding="utf-8")
    m = re.search(r"APP_BUNDLE_VERSION\s*=\s*(\d+)", html)
    assert m, "app.html must declare APP_BUNDLE_VERSION"
    assert int(m.group(1)) == BUNDLE_VERSION


def test_assembled_module_script_parses_as_javascript(tmp_path):
    # inline_module strips module syntax with line-anchored regexes. They cope with
    # today's single module, but Tasks 8-13 add six more, and a multi-line import,
    # an `export default`, or the text "export " inside a template literal would be
    # mangled into something that still builds and still passes every other test
    # while shipping a dead page. Parsing the assembled script closes that gap.
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available")

    html = render_html({"bundleVersion": BUNDLE_VERSION, "meta": {}}, Path("viz/template"))
    m = re.search(r'<script type="module">(.*?)</script>', html, re.S)
    assert m, "template must contain an inline module script"

    script = tmp_path / "assembled.mjs"
    script.write_text(m.group(1), encoding="utf-8")
    proc = subprocess.run([node, "--check", str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, f"assembled module does not parse:\n{proc.stderr}"


def test_unlisted_lib_module_is_rejected(tmp_path):
    # A module added under lib/ but forgotten in LIB_ORDER would silently not be
    # inlined, and the page would fail at runtime with an undefined function.
    import shutil

    tpl = tmp_path / "template"
    shutil.copytree("viz/template", tpl)
    (tpl / "lib" / "orphan.mjs").write_text("export const ORPHAN = 1;\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not in LIB_ORDER"):
        render_html({"bundleVersion": BUNDLE_VERSION}, tpl)


def test_every_listed_lib_module_is_inlined():
    from viz.bundle import LIB_ORDER

    html = render_html({"bundleVersion": BUNDLE_VERSION}, Path("viz/template"))
    for name in LIB_ORDER:
        p = Path("viz/template/lib") / name
        if not p.exists():
            continue
        lines = [ln for ln in inline_module(p).splitlines() if len(ln.strip()) > 12]
        assert lines, f"{name} produced no inlinable content"
        for ln in lines[:5]:
            assert ln in html, f"{name} was not inlined (missing {ln.strip()[:40]!r})"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest viz/tests/test_bundle.py -v
```

Expected: `ModuleNotFoundError: No module named 'viz.bundle'`.

- [ ] **Step 3: Write the template files**

Create `viz/template/app.css` with the theme tokens (the page must work in both themes):

```css
/* Cable hues are categorical slots 2 and 3 of the validated palette; slot 1
   (blue) is reserved for the sequential misfit ramp. Dark values are stepped for
   the dark surface, not flipped. */
:root {
  --bg: #f7f7f5; --panel: #ffffff; --surface: #ffffff;
  --ink: #0b0b0b; --ink-dim: #52514e; --line: #d8d8d4;
  --cable-n: #eb6834; --cable-s: #1baf7a; --accent: #2a78d6;
  --shadow: 0 1px 3px rgba(0,0,0,.08);
}
:root:not([data-theme="light"]) {
  @media (prefers-color-scheme: dark) {
    --bg: #17181a; --panel: #1f2023; --surface: #1f2023;
    --ink: #ffffff; --ink-dim: #c3c2b7; --line: #34363a;
    --cable-n: #d95926; --cable-s: #199e70; --accent: #3987e5;
    --shadow: 0 1px 3px rgba(0,0,0,.5);
  }
}
:root[data-theme="dark"] {
  --bg: #17181a; --panel: #1f2023; --surface: #1f2023;
  --ink: #ffffff; --ink-dim: #c3c2b7; --line: #34363a;
  --cable-n: #d95926; --cable-s: #199e70; --accent: #3987e5;
  --shadow: 0 1px 3px rgba(0,0,0,.5);
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font: 14px/1.5 ui-sans-serif, system-ui, sans-serif; }
/* minmax(0, 1fr), not 1fr: a plain 1fr is minmax(auto, 1fr), and the canvas's
   intrinsic size then forces the column wider than its share, overflowing the
   body horizontally. */
#app { display: grid; grid-template-columns: 240px minmax(0, 1fr) 320px; gap: 12px; height: 100vh; padding: 12px; }
/* Below three-column width, stack instead of squeezing the map to a sliver. */
@media (max-width: 900px) {
  #app { grid-template-columns: minmax(0, 1fr); height: auto; min-height: 100vh; }
  #map-wrap { min-height: 55vh; }
  #rail, #side { max-height: none; }
}
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; box-shadow: var(--shadow); overflow: auto; }
#map-wrap { position: relative; }
canvas { display: block; width: 100%; height: 100%; border-radius: 8px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; border: 1px solid var(--line); }
.badge.warn { border-color: var(--cable-s); color: var(--cable-s); }
.readout { display: flex; justify-content: space-between; font-variant-numeric: tabular-nums; }
.readout dt { color: var(--ink-dim); }
label { display: block; margin: 10px 0 2px; color: var(--ink-dim); font-size: 12px; }
input[type="range"] { width: 100%; }
```

Create `viz/template/app.html`:

```html
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DAS Fin-Whale Localization Explorer</title>
<style>/*__CSS__*/</style>
<div id="app">
  <aside class="panel" id="rail">
    <h1 style="font-size:15px;margin:0 0 8px">Localization Explorer</h1>
    <div id="version-error" hidden style="color:var(--cable-s)"></div>
    <div id="dataset-info"></div>
    <label for="result-select">Result sets</label>
    <div id="result-select"></div>
    <label for="call-search">Call</label>
    <input id="call-search" type="search" placeholder="filter calls…" style="width:100%">
    <div id="call-list" style="margin-top:8px;max-height:40vh;overflow:auto"></div>
  </aside>

  <section id="map-wrap" class="panel">
    <canvas id="map"></canvas>
  </section>

  <aside class="panel" id="side">
    <div id="call-readout"></div>
    <label>Sound speed <span id="c-value"></span></label>
    <input id="c-slider" type="range" min="1440" max="1520" step="1" value="1480">
    <label>Assumed source depth <span id="z-value"></span></label>
    <input id="z-slider" type="range" min="-200" max="0" step="5" value="-30">
    <label>Cables</label>
    <div id="cable-toggle"></div>
    <label>Grid resolution</label>
    <select id="res-select"><option>130</option><option selected>260</option><option>520</option></select>
    <h2 style="font-size:13px;margin-top:16px">Moveout</h2>
    <canvas id="moveout" height="180"></canvas>
    <h2 style="font-size:13px">Misfit vs depth</h2>
    <canvas id="zprofile" height="140"></canvas>
    <div id="zprofile-context" style="font-size:11px;color:var(--ink-dim);margin-top:4px">
      Source depth is essentially unobservable from two cables at similar depth.
      Measured across 32 real calls, the best misfit varies by a median of just
      <strong>0.3%</strong> over the whole −200…0 m range (worst case 1.0%). That is why every
      published method for this dataset fixes source depth rather than estimating it —
      the reference implementation's 176 solutions all carry z = −30 m exactly.
    </div>
    <h2 style="font-size:13px">Basin in 3D</h2>
    <canvas id="inset3d" height="220" style="cursor:grab"></canvas>
    <div id="inset3d-caption" style="color:var(--ink-dim);font-size:11px"></div>
    <p style="color:var(--ink-dim);font-size:11px;margin-top:14px">
      Bathymetry: GMRT (Global Multi-Resolution Topography) synthesis,
      Ryan et al. 2009, doi:10.1029/2008GC002332.
      Picks and cable geometry: DCLDE 2027 localization dataset, doi:10.25921/v2vh-8w16.
    </p>
  </aside>
</div>
<script type="module">
const PAYLOAD = __PAYLOAD__;
const APP_BUNDLE_VERSION = 1;
if (PAYLOAD.bundleVersion !== APP_BUNDLE_VERSION) {
  const el = document.getElementById("version-error");
  el.hidden = false;
  el.textContent =
    `Bundle version ${PAYLOAD.bundleVersion} does not match app version ${APP_BUNDLE_VERSION}. ` +
    `Rebuild with viz/build_bundle.py.`;
  throw new Error("bundle/app version mismatch");
}
/*__LIB__*/
/*__APP__*/
</script>
```

The version check runs before any module code, so a stale bundle fails loudly with a stated reason rather than rendering something subtly wrong. `APP_BUNDLE_VERSION` here must be bumped in lockstep with `BUNDLE_VERSION` in `viz/bundle.py`.

Create `viz/template/app.js` as a placeholder that Task 8 fills in — for this task it only needs to prove the wiring works:

```javascript
// Wired up in Tasks 8-13. For now, prove the payload arrives intact.
const info = document.getElementById("dataset-info");
info.textContent = `bundle v${PAYLOAD.bundleVersion}`;
```

- [ ] **Step 4: Write the bundle module**

Create `viz/bundle.py`:

```python
"""Assemble the self-contained HTML bundle.

A published Artifact cannot fetch its own data files (the CSP blocks all
fetch/XHR), so every payload is inlined as base64 typed arrays and every module
is concatenated into one inline module script.
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dclde_localization.physics.geometry import decimate_indices

BUNDLE_VERSION = 1
PAGE_LIMIT_BYTES = 16 * 1024 * 1024
CABLE_STRIDE = 4

# Inline order matters: imports are stripped, so a module's dependencies must
# already be in scope. Dependency-first, not alphabetical. Every file added
# under viz/template/lib/ must be listed here or it will silently not ship.
LIB_ORDER = (
    "forward.mjs",
    "render.mjs",
    "contour.mjs",
    "moveout.mjs",
    "mirror.mjs",
    "zprofile.mjs",
    "isosurface.mjs",
)


def encode_f32(a: np.ndarray) -> str:
    return base64.b64encode(np.asarray(a, dtype="<f4").tobytes()).decode("ascii")


def encode_i16(a: np.ndarray) -> str:
    return base64.b64encode(np.asarray(a, dtype="<i2").tobytes()).decode("ascii")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def inline_module(path: Path) -> str:
    """Strip ES module syntax so the source can live in one inline script."""
    text = Path(path).read_text(encoding="utf-8")
    text = re.sub(r"^\s*import\s.*?;\s*$", "", text, flags=re.M)
    text = re.sub(r"^(\s*)export\s+", r"\1", text, flags=re.M)
    return text


def build_payload(cables, transform, picks_xyz, local_grid, result_sets, sources) -> dict:
    cable_payload = {}
    for name, cable in cables.items():
        idx = decimate_indices(len(cable.x), CABLE_STRIDE)
        # chanM is shipped explicitly rather than reconstructed from start+step:
        # decimate_indices keeps the final channel, so the decimated samples are
        # NOT uniformly spaced at the tail and index arithmetic would be wrong
        # there. ~104 KB base64 across both cables, against a ~1.3 MB page.
        cable_payload[name] = {
            "n": int(len(idx)),
            "chanM": encode_f32(cable.chan_m[idx]),
            "x": encode_f32(cable.x[idx]),
            "y": encode_f32(cable.y[idx]),
            "depth": encode_f32(cable.depth[idx]),
        }

    calls = sorted(picks_xyz["call"].unique())
    call_index = {c: i for i, c in enumerate(calls)}
    p = picks_xyz.sort_values("call").reset_index(drop=True)

    return {
        "bundleVersion": BUNDLE_VERSION,
        "meta": {
            "gitSha": git_sha(),
            "sources": sources,
            "defaultC": 1480.0,
            "frame": {"xOffset": transform.x_offset, "yOffset": transform.y_offset},
            "contourRatios": [1.05, 1.10, 1.25, 1.50],
        },
        "cables": cable_payload,
        "bathy": {
            "x0": local_grid.x0, "y0": local_grid.y0,
            "dx": local_grid.dx, "dy": local_grid.dy,
            "nx": local_grid.nx, "ny": local_grid.ny,
            "noData": -32768,
            "z": encode_i16(local_grid.z.ravel()),
        },
        "calls": calls,
        "picks": {
            "callIndex": [call_index[c] for c in p["call"]],
            "cable": [0 if c == "North" else 1 for c in p["cable"]],
            "dist": encode_f32(p["dist"].to_numpy()),
            "time": encode_f32(p["time"].to_numpy()),
            "snr": encode_f32(p["snr"].to_numpy()),
            "px": encode_f32(p["px"].to_numpy()),
            "py": encode_f32(p["py"].to_numpy()),
            "pz": encode_f32(p["pz"].to_numpy()),
        },
        "resultSets": [asdict(rs) for rs in result_sets],
    }


def render_html(payload: dict, template_dir: Path) -> str:
    template_dir = Path(template_dir)
    lib_dir = template_dir / "lib"

    present = {p.name for p in lib_dir.glob("*.mjs")} if lib_dir.exists() else set()
    unlisted = present - set(LIB_ORDER)
    if unlisted:
        raise ValueError(f"lib modules not in LIB_ORDER would not ship: {sorted(unlisted)}")

    lib = "\n".join(
        inline_module(lib_dir / name) for name in LIB_ORDER if (lib_dir / name).exists()
    )

    html = (template_dir / "app.html").read_text(encoding="utf-8")
    html = html.replace("/*__CSS__*/", (template_dir / "app.css").read_text(encoding="utf-8"))
    html = html.replace("/*__LIB__*/", lib)
    html = html.replace("/*__APP__*/", inline_module(template_dir / "app.js"))

    # Escape "<" so no payload string can close the inline <script> block. json.dumps
    # leaves "<" alone, so a result set whose notes contained "</script>" would end
    # the module early and turn the rest of the page into unrestricted markup. The
    # results contract is meant to accept third-party files, so this is a real path.
    payload_js = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    html = html.replace("__PAYLOAD__", payload_js)

    size = len(html.encode("utf-8"))
    if size > PAGE_LIMIT_BYTES:
        raise ValueError(f"page is {size/1e6:.1f} MB, over the {PAGE_LIMIT_BYTES/1e6:.0f} MB limit")
    return html
```

- [ ] **Step 5: Write the build CLI**

Create `viz/build_bundle.py`:

```python
"""Build the self-contained visualizer page.

    uv run python -m viz.build_bundle --out viz/out/index.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dclde_localization.physics.geometry import CABLES, FrameTransform, load_cable
from viz.bathymetry import decode_gmt_grid, reproject_to_local, validate_against_cable
from viz.bundle import build_payload, render_html
from dclde_localization.physics.picks import attach_xyz, load_picks
from viz.results import frame_guard, load_result_set, to_local

FIXTURES = Path("fixtures")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="viz/out/index.html")
    ap.add_argument("--results-dir", default="viz/results")
    args = ap.parse_args()

    cables = {n: load_cable(n, FIXTURES / f"{n.lower()}_DAS_multicoord.csv") for n in CABLES}
    transform = FrameTransform.from_cable(cables["North"])

    geo = decode_gmt_grid(FIXTURES / "gmrt_bathy.nc")
    report = validate_against_cable(geo, cables["South"])
    print(f"bathymetry check: n={report.n} median offset={report.median_offset_m:.1f} m "
          f"r={report.correlation:.4f}")
    if abs(report.median_offset_m) > 15.0 or report.correlation < 0.999:
        raise SystemExit("bathymetry disagrees with the cable's own depth; refusing to build")
    grid = reproject_to_local(geo, transform)

    picks = attach_xyz(load_picks(), cables)
    print(f"picks: {len(picks)} rows, {picks['call'].nunique()} calls")

    result_sets = []
    for path in sorted(Path(args.results_dir).glob("*.json")):
        rs = to_local(load_result_set(path), transform)
        guard = frame_guard(rs, cables)
        flag = "SUSPECT" if guard.suspect else "ok"
        print(f"result set {rs.name}: {len(rs.solutions)} solutions, "
              f"median {guard.median_offset_m/1000:.1f} km from cable [{flag}]")
        result_sets.append(rs)

    html = render_html(
        build_payload(cables, transform, picks, grid, result_sets,
                      sources={"picks": "fixtures/das_calls.csv.gz",
                               "geometry": "fixtures/*_DAS_multicoord.csv",
                               "bathymetry": "GMRT GridServer"}),
        Path("viz/template"),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html.encode('utf-8'))/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the tests and the build**

```bash
uv run pytest viz/tests/test_bundle.py -v
uv run python -m viz.build_bundle --out viz/out/index.html
```

Expected: 14 tests pass; the build prints the bathymetry check (`median offset ≈ -0.6 m, r ≈ 0.9999`), `picks: 11117 rows, 628 calls`, one result set line, and a page size of roughly 1.3 MB.

- [ ] **Step 7: Commit**

```bash
git add viz/bundle.py viz/build_bundle.py viz/template/ viz/tests/test_bundle.py
git commit -m "Add bundle encoder and self-contained page build CLI"
```

---

### Task 8: Map rendering — bathymetry, cables, solutions

**Files:**
- Create: `viz/template/lib/render.mjs`
- Create: `viz/tests/test_render.mjs`
- Modify: `viz/template/app.js` (replace the placeholder)

**Interfaces:**
- Consumes: `PAYLOAD` from Task 7.
- Produces:
  - `decodeF32(b64) -> Float32Array`, `decodeI16(b64) -> Int16Array`
  - `makeView({x0, y0, x1, y1, width, height}) -> {toScreen(x,y), toWorld(px,py), scale}`
  - `hillshade(z, nx, ny, dx, dy, {azimuth, altitude, noData}) -> Float32Array` in 0..1
  - `depthColor(z, zmin, zmax, dark) -> [r,g,b]`
  - `misfitColor(t, dark) -> [r,g,b]` for `t` in 0..1; `t = 0` is the best fit
  - `isDarkTheme() -> boolean`
  - `drawBathy(ctx, bathy, view, {noData, dark})`, `drawCables(ctx, cables, view, colors)` (casing + core), `drawSolutions(ctx, solutions, view, {color, surface, radius})`

- [ ] **Step 1: Write the failing tests**

Create `viz/tests/test_render.mjs`:

```javascript
import test from "node:test";
import assert from "node:assert/strict";

import { decodeF32, decodeI16, makeView, hillshade, depthColor, misfitColor } from "../template/lib/render.mjs";

test("decodeF32 reverses the python encoder", () => {
  // base64 of Float32Array([1.5, -2.25]) little-endian.
  //   1.5  = 0x3FC00000 -> bytes 00 00 C0 3F
  //  -2.25 = 0xC0100000 -> bytes 00 00 10 C0
  // Generated, not hand-written:
  //   python -c "import base64,numpy as np; \
  //     print(base64.b64encode(np.array([1.5,-2.25],dtype='<f4').tobytes()).decode())"
  const a = decodeF32("AADAPwAAEMA=");
  assert.equal(a.length, 2);
  assert.ok(Math.abs(a[0] - 1.5) < 1e-6);
  assert.ok(Math.abs(a[1] + 2.25) < 1e-6);
});

test("decodeI16 reverses the python encoder", () => {
  const a = decodeI16("AAD//w=="); // [0, -1]
  assert.equal(a[0], 0);
  assert.equal(a[1], -1);
});

test("makeView maps corners and round-trips", () => {
  const v = makeView({ x0: 0, y0: 0, x1: 100, y1: 50, width: 200, height: 100 });
  const [sx, sy] = v.toScreen(0, 0);
  assert.ok(Math.abs(sx) < 1e-9);
  // world y is up, screen y is down
  assert.ok(Math.abs(sy - 100) < 1e-9);
  const [wx, wy] = v.toWorld(...v.toScreen(37, 22));
  assert.ok(Math.abs(wx - 37) < 1e-6 && Math.abs(wy - 22) < 1e-6);
});

test("makeView preserves aspect ratio", () => {
  const v = makeView({ x0: 0, y0: 0, x1: 100, y1: 100, width: 400, height: 200 });
  const [ax] = v.toScreen(1, 0);
  const [, ay] = v.toScreen(0, 1);
  const [bx] = v.toScreen(0, 0);
  const [, by] = v.toScreen(0, 0);
  assert.ok(Math.abs(Math.abs(ax - bx) - Math.abs(ay - by)) < 1e-9);
});

test("hillshade lights a slope and ignores no-data", () => {
  const nx = 3, ny = 3, noData = -32768;
  const flat = new Int16Array([0,0,0, 0,0,0, 0,0,0]);
  const hFlat = hillshade(flat, nx, ny, 1, 1, { azimuth: 315, altitude: 45, noData });
  const slope = new Int16Array([0,10,20, 0,10,20, 0,10,20]);
  const hSlope = hillshade(slope, nx, ny, 1, 1, { azimuth: 315, altitude: 45, noData });
  assert.ok(hFlat[4] > 0 && hFlat[4] <= 1);
  assert.notEqual(hSlope[4].toFixed(6), hFlat[4].toFixed(6));

  const holed = new Int16Array([noData,0,0, 0,0,0, 0,0,0]);
  const hHoled = hillshade(holed, nx, ny, 1, 1, { azimuth: 315, altitude: 45, noData });
  assert.ok(Number.isFinite(hHoled[4]), "no-data must not poison neighbours with NaN");
});

test("colour ramps stay in gamut", () => {
  for (const dark of [false, true]) {
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      for (const c of [misfitColor(t, dark), depthColor(t, 0, 1, dark)]) {
        assert.equal(c.length, 3);
        for (const v of c) assert.ok(v >= 0 && v <= 255, `channel out of gamut: ${v}`);
      }
    }
  }
  const shallow = depthColor(-10, -2000, 0);
  const deep = depthColor(-1900, -2000, 0);
  assert.notDeepEqual(shallow, deep);
});

test("sequential ramps are monotone in lightness in both modes", () => {
  // The palette validator's checks are for CATEGORICAL palettes; the requirement
  // for a sequential ramp is lightness monotonicity, which is what makes
  // magnitude readable. A ramp that reverses direction mid-way reads as two
  // different magnitudes at the same value.
  const lum = ([r, g, b]) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
  const ts = [0, 0.25, 0.5, 0.75, 1];

  for (const dark of [false, true]) {
    for (const [label, fn] of [
      ["misfit", (t) => misfitColor(t, dark)],
      ["depth", (t) => depthColor(t, 0, 1, dark)],
    ]) {
      const l = ts.map((t) => lum(fn(t)));
      const signs = l.slice(1).map((v, i) => Math.sign(v - l[i]));
      assert.ok(
        signs.every((s) => s === signs[0] && s !== 0),
        `${label} ramp (dark=${dark}) is not monotone in lightness: ${l.map((v) => v.toFixed(1))}`,
      );
    }
  }
});
```

- [ ] **Step 2: Run to verify failure**

```bash
cd viz && npm test
```

Expected: FAIL — cannot find `../template/lib/render.mjs`.

- [ ] **Step 3: Implement**

The palette in the code below is already settled and validated — do not substitute your own colours. The reasoning is in the comment block at the top of the palette section, and the numbers behind it are: cable hues are categorical slots 2 and 3 of a validated palette, checked all-pairs in both modes (CVD ΔE 9.2 light / 9.4 dark against a ≥8 target; normal-vision ΔE 27.6 / 26.5 against a ≥15 floor). Slot 1 (blue) is deliberately reserved for the sequential misfit ramp so the two encodings never share a hue, and bathymetry takes no hue at all.

Create `viz/template/lib/render.mjs`:

```javascript
// Pure render helpers. Everything here is unit-tested in Node; the canvas
// wiring in app.js stays a thin shell over these.

export function decodeF32(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

export function decodeI16(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Int16Array(buf);
}

export function makeView({ x0, y0, x1, y1, width, height }) {
  const scale = Math.min(width / (x1 - x0), height / (y1 - y0));
  const ox = (width - (x1 - x0) * scale) / 2;
  const oy = (height - (y1 - y0) * scale) / 2;
  return {
    scale,
    toScreen(x, y) {
      return [ox + (x - x0) * scale, height - oy - (y - y0) * scale];
    },
    toWorld(px, py) {
      return [x0 + (px - ox) / scale, y0 + (height - oy - py) / scale];
    },
  };
}

export function hillshade(z, nx, ny, dx, dy, { azimuth = 315, altitude = 45, noData = -32768 } = {}) {
  const az = (azimuth * Math.PI) / 180;
  const alt = (altitude * Math.PI) / 180;
  const out = new Float32Array(nx * ny);
  const at = (i, j) => {
    const ii = Math.min(nx - 1, Math.max(0, i));
    const jj = Math.min(ny - 1, Math.max(0, j));
    const v = z[jj * nx + ii];
    return v === noData ? 0 : v;
  };
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const dzdx = (at(i + 1, j) - at(i - 1, j)) / (2 * dx);
      const dzdy = (at(i, j + 1) - at(i, j - 1)) / (2 * dy);
      const slope = Math.atan(Math.hypot(dzdx, dzdy));
      const aspect = Math.atan2(dzdy, -dzdx);
      let v = Math.sin(alt) * Math.cos(slope) +
              Math.cos(alt) * Math.sin(slope) * Math.cos(az - aspect);
      out[j * nx + i] = Math.min(1, Math.max(0, v));
    }
  }
  return out;
}

function lerp3(a, b, t) {
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ];
}

function ramp(stops, t) {
  const u = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(u));
  return lerp3(stops[i], stops[i + 1], u - i);
}

// --- Palette -----------------------------------------------------------------
// Three encodings share this map, so hues are assigned by job, not by taste:
//
//   bathymetry  neutral grey, hillshaded. Pure context, so it takes no hue at
//               all and cannot compete with either of the other two.
//   misfit      the default sequential hue (blue), monotone in lightness. Low
//               misfit is DARK and opaque, high misfit recedes toward the
//               surface -- so the basin is what the eye lands on.
//   cables      categorical slots 2 and 3 (orange, aqua). Slot 1 is blue and is
//               claimed by the misfit ramp; the palette's own rule that a second
//               sequential context takes the next categorical hue implies hues
//               are not shared across encodings, so the cables start at slot 2.
//               Validated all-pairs in both modes: CVD dE 9.2 light / 9.4 dark
//               (>=8 target), normal-vision 27.6 / 26.5 (>=15 floor).
//
// Dark mode is stepped for the dark surface, not flipped.

const DEPTH_STOPS_LIGHT = [[74, 76, 78], [116, 118, 118], [154, 156, 154], [190, 191, 188], [222, 222, 218]];
const DEPTH_STOPS_DARK = [[26, 27, 28], [46, 48, 50], [70, 72, 74], [98, 100, 101], [130, 131, 130]];

// Blue sequential, reference steps 700 -> 100. t = 0 is the best fit.
const MISFIT_STOPS_LIGHT = [[13, 54, 107], [28, 92, 171], [42, 120, 214], [109, 167, 236], [205, 226, 251]];
// On a dark surface the bright end stands out and the dark end recedes.
const MISFIT_STOPS_DARK = [[205, 226, 251], [134, 182, 239], [57, 135, 229], [24, 79, 149], [16, 66, 129]];

export function depthColor(z, zmin, zmax, dark = false) {
  const stops = dark ? DEPTH_STOPS_DARK : DEPTH_STOPS_LIGHT;
  return ramp(stops, (z - zmin) / (zmax - zmin));
}

export function misfitColor(t, dark = false) {
  return ramp(dark ? MISFIT_STOPS_DARK : MISFIT_STOPS_LIGHT, t);
}

export function isDarkTheme() {
  const stamped = document.documentElement.getAttribute("data-theme");
  if (stamped === "dark") return true;
  if (stamped === "light") return false;
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
}

export function drawBathy(ctx, bathy, view, { noData = -32768, dark = false } = {}) {
  const { nx, ny, x0, y0, dx, dy, zArr, shade } = bathy;
  const img = ctx.createImageData(nx, ny);
  let zmin = Infinity, zmax = -Infinity;
  for (let k = 0; k < zArr.length; k++) {
    const v = zArr[k];
    if (v === noData) continue;
    if (v < zmin) zmin = v;
    if (v > zmax) zmax = v;
  }
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      const dst = ((ny - 1 - j) * nx + i) * 4; // world y up -> image y down
      const v = zArr[k];
      if (v === noData) {
        img.data[dst + 3] = 0;
        continue;
      }
      const c = depthColor(v, zmin, zmax, dark);
      const s = 0.55 + 0.45 * shade[k];
      img.data[dst] = c[0] * s;
      img.data[dst + 1] = c[1] * s;
      img.data[dst + 2] = c[2] * s;
      img.data[dst + 3] = 255;
    }
  }
  const off = new OffscreenCanvas(nx, ny);
  off.getContext("2d").putImageData(img, 0, 0);
  const [sx0, sy0] = view.toScreen(x0, y0 + dy * (ny - 1));
  const [sx1, sy1] = view.toScreen(x0 + dx * (nx - 1), y0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, sx0, sy0, sx1 - sx0, sy1 - sy0);
}

export function drawCables(ctx, cables, view, colors) {
  // Casing then core. Over a mid-grey relief both cable hues sit at ~1.1:1
  // contrast, so a bare stroke is genuinely hard to follow; a surface-coloured
  // casing underneath makes the line legible against any relief value and
  // discharges the palette's sub-3:1 relief obligation structurally rather than
  // relying on the legend.
  for (const pass of ["casing", "core"]) {
    for (const [name, c] of Object.entries(cables)) {
      ctx.beginPath();
      for (let k = 0; k < c.x.length; k++) {
        const [sx, sy] = view.toScreen(c.x[k], c.y[k]);
        k === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
      }
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      if (pass === "casing") {
        ctx.strokeStyle = colors.surface;
        ctx.lineWidth = 4;
      } else {
        ctx.strokeStyle = colors[name];
        ctx.lineWidth = 1.75;
      }
      ctx.stroke();
    }
  }
}

export function drawSolutions(ctx, solutions, view, { color, surface, radius = 2.5 } = {}) {
  // Ink, not a fifth hue. Adding a categorical colour here would collide with
  // the cables or the misfit ramp; solution marks are identified by shape and
  // position, and a surface ring keeps them readable wherever they land.
  for (const s of solutions) {
    const [sx, sy] = view.toScreen(s.x, s.y);
    ctx.beginPath();
    ctx.arc(sx, sy, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = surface;
    ctx.stroke();
  }
}
```

- [ ] **Step 4: Wire the map in `app.js`**

Replace `viz/template/app.js`:

```javascript
const $ = (id) => document.getElementById(id);

const bathy = (() => {
  const b = PAYLOAD.bathy;
  const zArr = decodeI16(b.z);
  return {
    nx: b.nx, ny: b.ny, x0: b.x0, y0: b.y0, dx: b.dx, dy: b.dy, noData: b.noData,
    zArr,
    shade: hillshade(zArr, b.nx, b.ny, b.dx, b.dy, { noData: b.noData }),
  };
})();

const cables = Object.fromEntries(
  Object.entries(PAYLOAD.cables).map(([name, c]) => [name, {
    chanM: decodeF32(c.chanM),
    x: decodeF32(c.x), y: decodeF32(c.y), depth: decodeF32(c.depth),
  }])
);

const state = {
  c: PAYLOAD.meta.defaultC,
  sz: -30,
  call: null,
  cablesOn: { North: true, South: true },
  res: 260,
  resultSet: PAYLOAD.resultSets[0] ?? null,
};

function themeColors() {
  const s = getComputedStyle(document.documentElement);
  const tok = (name, fallback) => s.getPropertyValue(name).trim() || fallback;
  return {
    North: tok("--cable-n", "#eb6834"),
    South: tok("--cable-s", "#1baf7a"),
    accent: tok("--accent", "#2a78d6"),
    surface: tok("--surface", "#ffffff"),
    ink: tok("--ink", "#0b0b0b"),
    inkDim: tok("--ink-dim", "#52514e"),
    dark: isDarkTheme(),
  };
}

// Repaint on a theme change: the ramps are selected per mode, not flipped, so a
// cached raster from the other mode would be wrong.
if (window.matchMedia) {
  window.matchMedia("(prefers-color-scheme: dark)")
    .addEventListener("change", () => resizeAndDraw());
}

let view;

function resizeAndDraw() {
  const canvas = $("map");
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  view = makeView({
    x0: bathy.x0, y0: bathy.y0,
    x1: bathy.x0 + bathy.dx * (bathy.nx - 1),
    y1: bathy.y0 + bathy.dy * (bathy.ny - 1),
    width: rect.width, height: rect.height,
  });
  draw();
}

// Later tasks add overlays by pushing onto `layers`, in draw order. Do not
// reassign `draw` itself — a chain of wrappers is hard to reason about and
// breaks silently if one task's ordering assumption is wrong.
const layers = [];

function draw() {
  const ctx = $("map").getContext("2d");
  const colors = themeColors();
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  drawBathy(ctx, bathy, view, { noData: bathy.noData, dark: colors.dark });
  drawCables(ctx, cables, view, colors);
  if (state.resultSet) {
    drawSolutions(ctx, state.resultSet.solutions, view,
      { color: colors.ink, surface: colors.surface });
  }
  for (const layer of layers) layer(ctx, colors);
}

$("dataset-info").innerHTML =
  `<div class="readout"><dt>calls</dt><dd>${PAYLOAD.calls.length}</dd></div>` +
  `<div class="readout"><dt>build</dt><dd>${PAYLOAD.meta.gitSha}</dd></div>`;

window.addEventListener("resize", resizeAndDraw);
resizeAndDraw();
```

- [ ] **Step 5: Run tests and rebuild**

```bash
cd viz && npm test && cd ..
uv run python -m viz.build_bundle --out viz/out/index.html
```

Expected: JS tests pass (now 10 across both files); the build succeeds. Open `viz/out/index.html` in a browser: the relief, both cables and 176 solution marks render, and the panel shows `calls 628`.

- [ ] **Step 6: Commit**

```bash
git add viz/template/lib/render.mjs viz/template/app.js viz/tests/test_render.mjs
git commit -m "Render bathymetry relief, cables and solution marks on the map"
```

---

### Task 9: Call selection, misfit field and contours

**Files:**
- Create: `viz/template/lib/contour.mjs`
- Create: `viz/tests/test_contour.mjs`
- Modify: `viz/template/app.js`

**Interfaces:**
- Consumes: `misfitField` (Task 3), `makeView`/`misfitColor` (Task 8).
- Produces:
  - `marchingSquares(field, nx, ny, level) -> Array<[x0,y0,x1,y1]>` in grid index space
  - `ratioLevels(best, ratios) -> number[]`
  - `gridForCall({px, py, res, bathyBounds}) -> {xs, ys}`
  - `fieldImage(field, nx, ny, best, cap, dark) -> Uint8ClampedArray` RGBA

- [ ] **Step 1: Write the failing tests**

Create `viz/tests/test_contour.mjs`:

```javascript
import test from "node:test";
import assert from "node:assert/strict";

import { marchingSquares, ratioLevels, gridForCall } from "../template/lib/contour.mjs";

test("ratioLevels scales the best misfit", () => {
  assert.deepEqual(ratioLevels(2, [1.05, 1.5]), [2.1, 3]);
});

test("marchingSquares finds a segment across a linear ramp", () => {
  // field increases left to right; the level-1.5 contour must cross once.
  const nx = 3, ny = 2;
  const f = new Float64Array([1, 2, 3, 1, 2, 3]);
  const segs = marchingSquares(f, nx, ny, 1.5);
  assert.ok(segs.length >= 1, "expected at least one segment");
  for (const [x0, , x1] of segs) {
    assert.ok(x0 >= 0 && x0 <= nx - 1 && x1 >= 0 && x1 <= nx - 1);
  }
});

test("marchingSquares returns nothing when the level is outside the range", () => {
  const f = new Float64Array([1, 1, 1, 1]);
  assert.equal(marchingSquares(f, 2, 2, 5).length, 0);
});

test("gridForCall covers both sides of the picks and respects resolution", () => {
  const { xs, ys } = gridForCall({
    px: [10000, 60000], py: [20000, 20000], res: 130,
    bathyBounds: { x0: -50000, y0: -50000, x1: 200000, y1: 150000 },
  });
  assert.equal(Math.max(xs.length, ys.length), 130);
  assert.ok(xs[0] <= 0, "must expand at least 10 km beyond the picks");
  assert.ok(ys[0] < 20000 && ys[ys.length - 1] > 20000, "must span both sides");
});
```

- [ ] **Step 2: Run to verify failure**

```bash
cd viz && npm test
```

Expected: FAIL — cannot find `../template/lib/contour.mjs`.

- [ ] **Step 3: Implement**

Create `viz/template/lib/contour.mjs`:

```javascript
import { misfitColor } from "./render.mjs";

export function ratioLevels(best, ratios) {
  return ratios.map((r) => best * r);
}

// Linear interpolation along cell edges; returns segments in grid index space.
export function marchingSquares(field, nx, ny, level) {
  const segs = [];
  const at = (i, j) => field[j * nx + i];
  const interp = (a, b) => (level - a) / (b - a);

  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const v = [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)];
      const above = v.map((x) => x > level);
      const idx = (above[0] ? 1 : 0) | (above[1] ? 2 : 0) | (above[2] ? 4 : 0) | (above[3] ? 8 : 0);
      if (idx === 0 || idx === 15) continue;

      const pts = [];
      if (above[0] !== above[1]) pts.push([i + interp(v[0], v[1]), j]);
      if (above[1] !== above[2]) pts.push([i + 1, j + interp(v[1], v[2])]);
      if (above[2] !== above[3]) pts.push([i + interp(v[3], v[2]), j + 1]);
      if (above[3] !== above[0]) pts.push([i, j + interp(v[0], v[3])]);
      for (let k = 0; k + 1 < pts.length; k += 2) {
        segs.push([pts[k][0], pts[k][1], pts[k + 1][0], pts[k + 1][1]]);
      }
    }
  }
  return segs;
}

export function gridForCall({ px, py, res, bathyBounds }) {
  const xmin = Math.min(...px), xmax = Math.max(...px);
  const ymin = Math.min(...py), ymax = Math.max(...py);
  const span = Math.max(xmax - xmin, ymax - ymin);
  const pad = Math.max(10000, 0.5 * span);

  const x0 = Math.max(bathyBounds.x0, xmin - pad);
  const x1 = Math.min(bathyBounds.x1, xmax + pad);
  const y0 = Math.max(bathyBounds.y0, ymin - pad);
  const y1 = Math.min(bathyBounds.y1, ymax + pad);

  const long = Math.max(x1 - x0, y1 - y0);
  const step = long / (res - 1);
  const nx = Math.max(2, Math.round((x1 - x0) / step) + 1);
  const ny = Math.max(2, Math.round((y1 - y0) / step) + 1);

  const xs = Array.from({ length: nx }, (_, i) => x0 + i * step);
  const ys = Array.from({ length: ny }, (_, j) => y0 + j * step);
  return { xs, ys };
}

export function fieldImage(field, nx, ny, best, cap = 2.0, dark = false) {
  const data = new Uint8ClampedArray(nx * ny * 4);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      const ratio = field[k] / best;
      const t = Math.min(1, Math.max(0, (ratio - 1) / (cap - 1)));
      const c = misfitColor(t, dark);
      const dst = ((ny - 1 - j) * nx + i) * 4;
      data[dst] = c[0];
      data[dst + 1] = c[1];
      data[dst + 2] = c[2];
      data[dst + 3] = Math.round(215 * (1 - t) + 40);
    }
  }
  return data;
}
```

- [ ] **Step 4: Wire selection and the field into `app.js`**

Append to `viz/template/app.js`:

```javascript
// ---- pick index: call -> row range in the flat pick arrays -------------------
const picksRaw = {
  callIndex: PAYLOAD.picks.callIndex,
  cable: PAYLOAD.picks.cable,
  dist: decodeF32(PAYLOAD.picks.dist),
  time: decodeF32(PAYLOAD.picks.time),
  snr: decodeF32(PAYLOAD.picks.snr),
  px: decodeF32(PAYLOAD.picks.px),
  py: decodeF32(PAYLOAD.picks.py),
  pz: decodeF32(PAYLOAD.picks.pz),
};

const callRows = new Map();
picksRaw.callIndex.forEach((ci, row) => {
  const key = PAYLOAD.calls[ci];
  if (!callRows.has(key)) callRows.set(key, []);
  callRows.get(key).push(row);
});

function picksFor(call, cablesOn) {
  const rows = (callRows.get(call) ?? []).filter((r) => {
    const name = picksRaw.cable[r] === 0 ? "North" : "South";
    return cablesOn[name];
  });
  return {
    rows,
    px: rows.map((r) => picksRaw.px[r]),
    py: rows.map((r) => picksRaw.py[r]),
    pz: rows.map((r) => picksRaw.pz[r]),
    obsT: rows.map((r) => picksRaw.time[r]),
    dist: rows.map((r) => picksRaw.dist[r]),
    cable: rows.map((r) => (picksRaw.cable[r] === 0 ? "North" : "South")),
  };
}

const bathyBounds = {
  x0: bathy.x0, y0: bathy.y0,
  x1: bathy.x0 + bathy.dx * (bathy.nx - 1),
  y1: bathy.y0 + bathy.dy * (bathy.ny - 1),
};

let currentField = null;

function computeField() {
  currentField = null;
  if (!state.call) return;
  const p = picksFor(state.call, state.cablesOn);
  if (p.rows.length < 5) return;

  const { xs, ys } = gridForCall({ px: p.px, py: p.py, res: state.res, bathyBounds });
  const { field, nx, ny } = misfitField({
    xs, ys, sz: state.sz, px: p.px, py: p.py, pz: p.pz,
    obsT: p.obsT, c: state.c, objective: "l1",
  });
  let best = Infinity, bi = 0;
  for (let k = 0; k < field.length; k++) if (field[k] < best) { best = field[k]; bi = k; }
  currentField = {
    xs, ys, field, nx, ny, best,
    bestXY: [xs[bi % nx], ys[Math.floor(bi / nx)]],
    nPicks: p.rows.length,
  };
}

function drawField(ctx) {
  if (!currentField) return;
  const { xs, ys, field, nx, ny, best } = currentField;
  const img = new ImageData(fieldImage(field, nx, ny, best, 2.0, themeColors().dark), nx, ny);
  const off = new OffscreenCanvas(nx, ny);
  off.getContext("2d").putImageData(img, 0, 0);
  const [sx0, sy0] = view.toScreen(xs[0], ys[ny - 1]);
  const [sx1, sy1] = view.toScreen(xs[nx - 1], ys[0]);
  ctx.drawImage(off, sx0, sy0, sx1 - sx0, sy1 - sy0);

  const colors = themeColors();
  ctx.lineWidth = 1;
  ctx.strokeStyle = colors.accent;
  for (const level of ratioLevels(best, PAYLOAD.meta.contourRatios)) {
    ctx.beginPath();
    for (const [gx0, gy0, gx1, gy1] of marchingSquares(field, nx, ny, level)) {
      const a = view.toScreen(xs[0] + gx0 * (xs[1] - xs[0]), ys[0] + gy0 * (ys[1] - ys[0]));
      const b = view.toScreen(xs[0] + gx1 * (xs[1] - xs[0]), ys[0] + gy1 * (ys[1] - ys[0]));
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
    }
    ctx.stroke();
  }
}

// Field raster + contours, then the selected call's pick marks on top.
layers.push((ctx, colors) => {
  drawField(ctx);
  if (!state.call) return;
  const p = picksFor(state.call, state.cablesOn);
  for (let k = 0; k < p.px.length; k++) {
    const [sx, sy] = view.toScreen(p.px[k], p.py[k]);
    ctx.fillStyle = colors[p.cable[k]];
    ctx.beginPath();
    ctx.arc(sx, sy, 2, 0, Math.PI * 2);
    ctx.fill();
  }
});
layers.push(() => renderReadout());

function renderReadout() {
  const el = $("call-readout");
  if (!state.call) { el.innerHTML = "<p style='color:var(--ink-dim)'>Select a call.</p>"; return; }
  const f = currentField;
  const warn = f && f.nPicks < 10
    ? "<span class='badge warn'>under-constrained</span>" : "";
  el.innerHTML = `<div style="font-weight:600">${state.call}</div>${warn}` +
    (f
      ? `<div class="readout"><dt>picks</dt><dd>${f.nPicks}</dd></div>` +
        `<div class="readout"><dt>best misfit</dt><dd>${f.best.toFixed(3)} s</dd></div>` +
        `<div class="readout"><dt>solution</dt><dd>${(f.bestXY[0]/1000).toFixed(2)}, ${(f.bestXY[1]/1000).toFixed(2)} km</dd></div>`
      : "<p style='color:var(--ink-dim)'>Too few picks on the selected cables.</p>");
}

function renderCallList(filter = "") {
  const list = $("call-list");
  list.innerHTML = "";
  for (const call of PAYLOAD.calls) {
    if (filter && !call.includes(filter)) continue;
    const b = document.createElement("button");
    b.textContent = `${call} (${(callRows.get(call) ?? []).length})`;
    b.style.cssText = "display:block;width:100%;text-align:left;border:0;background:none;color:inherit;cursor:pointer;padding:2px 0";
    b.onclick = () => { state.call = call; refresh(); };
    list.appendChild(b);
  }
}

// Single entry point for "something changed, recompute and redraw". Later
// tasks add overlays via `layers`, never by wrapping refresh or draw.
function refresh() {
  computeField();
  draw();
}

$("call-search").addEventListener("input", (e) => renderCallList(e.target.value.trim()));
renderCallList();
renderReadout();
```

- [ ] **Step 5: Run tests, rebuild, check in a browser**

```bash
cd viz && npm test && cd ..
uv run python -m viz.build_bundle --out viz/out/index.html
```

Expected: 14 JS tests pass. In the browser, clicking a call from the list renders its misfit field with contours over the relief, marks the picks on the cable, and the readout shows pick count, best misfit and the solution. Confirm a call with 8 picks shows the `under-constrained` badge.

- [ ] **Step 6: Commit**

```bash
git add viz/template/lib/contour.mjs viz/template/app.js viz/tests/test_contour.mjs
git commit -m "Add call selection, misfit field raster and ratio contours"
```

---

### Task 9b: Make the field legible — solution-sized grid, log ramp, zoom/pan, legend

**Why this exists.** Task 9 rendered the field but it is unreadable, and the defect is in the plan, not the implementation. Measured on real calls: only **0.04%** of grid cells fall within 1.5× of the best misfit, the median cell sits at **35×** the best and the worst at 50–95×, and the sub-1.5× region is a **median 1.8 km across inside a ~90 km grid box**. A linear ratio ramp capped at 2.0 therefore saturates over 99.9% of the field, and the 1.05–1.50 contours are sub-pixel. Confirmed in a browser: blue-minus-red is pinned at ~19 clear across the map with no structure. The first person to open the page described it, unprompted, as "a very precise but unappealing single image" and said they could not tell what the overlay meant.

Root cause: the grid is sized around the **picks** when the informative region is a neighbourhood of the **solution**. Four coupled fixes.

**Files:**
- Modify: `viz/template/lib/contour.mjs`
- Modify: `viz/template/app.js`
- Modify: `viz/template/app.html`, `viz/template/app.css`
- Modify: `viz/tests/test_contour.mjs`

**Interfaces:**
- Replaces `gridForCall` with `solutionGrid({px, py, pz, obsT, c, sz, res, bathyBounds, objective})` → `{xs, ys, coarseBest}`
- `fieldImage(field, nx, ny, best, cap, dark)` — `cap` now a **ratio for a log map**, default 8
- Adds `zoomBoxAt(box, wx, wy, factor)` and `fitBox(x0, y0, x1, y1, pad)` — pure, testable
- Adds `ratioLegendStops()` → the labelled ticks the legend draws

- [ ] **Step 1: Write the failing tests**

Add to `viz/tests/test_contour.mjs` (and drop the old `gridForCall` test):

```javascript
import { solutionGrid, fieldImage, zoomBoxAt, fitBox } from "../template/lib/contour.mjs";

function straightCallPicks() {
  // A source 4 km off a straight cable: the informative region is small and the
  // wide pick-bbox box would swamp it, which is the bug this task fixes.
  const n = 24;
  const px = Array.from({ length: n }, (_, i) => (i * 46000) / (n - 1));
  const py = px.map(() => 0);
  const pz = px.map(() => -400);
  const obsT = px.map((x) => Math.hypot(x - 23000, 4000, -370) / 1480);
  return { px, py, pz, obsT };
}

test("solutionGrid brackets the informative region, not the pick span", () => {
  const p = straightCallPicks();
  const g = solutionGrid({
    ...p, c: 1480, sz: -30, res: 120,
    bathyBounds: { x0: -60000, y0: -60000, x1: 160000, y1: 100000 },
  });
  const w = g.xs[g.xs.length - 1] - g.xs[0];
  const h = g.ys[g.ys.length - 1] - g.ys[0];

  // The old pick-sized box was 92 x 46 km. Measured, this fitted box is
  // 7.8 x 48.1 km -- a ~11x reduction in area. It is TALL on purpose: this
  // fixture uses a perfectly straight cable, so the mirror solution at
  // y = -4 km fits identically and the bracket must span both basins. Real
  // cables are curved, which is why real calls bracket to a median 8.8 km.
  // So the assertion is on area and on the constrained axis, not on height.
  const wideArea = 92000 * 46000;
  assert.ok(w * h < wideArea / 4, `box did not tighten: ${w} x ${h}`);
  assert.ok(w < 15000, `cross-cable axis should be tight, got ${w}`);
  assert.ok(g.xs[0] <= 23000 && g.xs[g.xs.length - 1] >= 23000, "must contain the source x");
  assert.ok(g.ys[0] <= 4000 && g.ys[g.ys.length - 1] >= 4000, "must contain the source y");
  assert.ok(g.ys[0] <= -4000, "must also contain the mirror basin");
});

test("solutionGrid stays wide when the call is genuinely unconstrained", () => {
  // Few picks over a short span: the good-fitting region really is huge, and
  // shrinking the box would hide that rather than reveal it.
  const px = [0, 500, 1000, 1500, 2000];
  const py = px.map(() => 0);
  const pz = px.map(() => -400);
  const obsT = px.map(() => 3.0);
  const g = solutionGrid({
    px, py, pz, obsT, c: 1480, sz: -30, res: 60,
    bathyBounds: { x0: -60000, y0: -60000, x1: 160000, y1: 100000 },
  });
  const w = g.xs[g.xs.length - 1] - g.xs[0];
  const h = g.ys[g.ys.length - 1] - g.ys[0];
  // Axis-agnostic: measured, this case brackets to 4.9 km wide by 160 km tall,
  // i.e. the good-fitting region spans the whole vertical extent. Asserting on
  // width alone checked the one axis that happens to be narrow.
  assert.ok(Math.max(w, h) > 20000, `expected a wide box, got ${w} x ${h}`);
});

test("fieldImage maps the ratio logarithmically so a wide range reads", () => {
  // ratio 1 -> darkest, ratio == cap -> lightest, and the midpoint of the LOG
  // range must land mid-ramp. A linear map put ratio 4 at the same place as
  // ratio 80, which is what made the whole field one flat wash.
  const field = Float64Array.from([1, Math.sqrt(8), 8, 80]);
  const rgba = fieldImage(field, 4, 1, 1, 8);
  const alpha = [0, 1, 2, 3].map((i) => rgba[i * 4 + 3]);
  assert.ok(alpha[0] > alpha[1], "best fit must be most opaque");
  assert.ok(alpha[1] > alpha[2], "mid-log must sit between");
  assert.equal(alpha[2], alpha[3], "beyond cap saturates");

  const lum = (i) => rgba[i * 4] + rgba[i * 4 + 1] + rgba[i * 4 + 2];
  const spread = Math.abs(lum(1) - lum(0));
  assert.ok(spread > 40, `ratio sqrt(8) must be visibly distinct from 1: ${spread}`);
});

test("zoomBoxAt keeps the cursor point fixed", () => {
  const box = { x0: 0, y0: 0, x1: 100, y1: 50 };
  const z = zoomBoxAt(box, 25, 10, 0.5);
  // The world point under the cursor maps to the same fractional position.
  const fx0 = (25 - box.x0) / (box.x1 - box.x0);
  const fx1 = (25 - z.x0) / (z.x1 - z.x0);
  assert.ok(Math.abs(fx0 - fx1) < 1e-9, `cursor drifted: ${fx0} vs ${fx1}`);
  assert.ok(z.x1 - z.x0 < box.x1 - box.x0, "zooming in must shrink the box");
});

test("fitBox pads and preserves the requested centre", () => {
  const b = fitBox(10, 20, 30, 40, 0.25);
  assert.ok(Math.abs((b.x0 + b.x1) / 2 - 20) < 1e-9);
  assert.ok(Math.abs((b.y0 + b.y1) / 2 - 30) < 1e-9);
  assert.ok(b.x1 - b.x0 > 20, "must pad outward");
});
```

- [ ] **Step 2: Run to verify failure**

```bash
cd viz && npm test
```

Expected: FAIL — `solutionGrid`, `zoomBoxAt` and `fitBox` are not exported.

- [ ] **Step 3: Rewrite the grid and ramp in `contour.mjs`**

Replace `gridForCall` and `fieldImage`:

```javascript
function boxGrid(x0, y0, x1, y1, res) {
  const long = Math.max(x1 - x0, y1 - y0);
  const step = long / (res - 1);
  const nx = Math.max(2, Math.round((x1 - x0) / step) + 1);
  const ny = Math.max(2, Math.round((y1 - y0) / step) + 1);
  return {
    xs: Array.from({ length: nx }, (_, i) => x0 + i * step),
    ys: Array.from({ length: ny }, (_, j) => y0 + j * step),
  };
}

export function fitBox(x0, y0, x1, y1, pad = 0.25) {
  const w = Math.max(x1 - x0, 1), h = Math.max(y1 - y0, 1);
  return {
    x0: x0 - w * pad, x1: x1 + w * pad,
    y0: y0 - h * pad, y1: y1 + h * pad,
  };
}

// Everything worse than this multiple of the best fit is uninteresting, so the
// second-pass box is the bracket of everything better. Chosen from measurement:
// the <5x region is a median 8.8 km across (p90 22.5 km), which puts the
// 1.05-1.50 contours at tens of metres per cell instead of hundreds.
const BRACKET_RATIO = 5.0;
const MIN_BOX_M = 2000;

export function solutionGrid({ px, py, pz, obsT, c, sz, res, bathyBounds, objective = "l1" }) {
  // Pass 1: coarse, over a box wide enough not to miss a mirror basin.
  const xmin = Math.min(...px), xmax = Math.max(...px);
  const ymin = Math.min(...py), ymax = Math.max(...py);
  const span = Math.max(xmax - xmin, ymax - ymin);
  const pad = Math.max(10000, 0.5 * span);
  const wide = {
    x0: Math.max(bathyBounds.x0, xmin - pad), x1: Math.min(bathyBounds.x1, xmax + pad),
    y0: Math.max(bathyBounds.y0, ymin - pad), y1: Math.min(bathyBounds.y1, ymax + pad),
  };
  const coarse = boxGrid(wide.x0, wide.y0, wide.x1, wide.y1, 90);
  const cf = misfitField({ ...coarse, sz, px, py, pz, obsT, c, objective });

  let best = Infinity, bi = 0;
  for (let k = 0; k < cf.field.length; k++) {
    if (cf.field[k] < best) { best = cf.field[k]; bi = k; }
  }
  const bx = coarse.xs[bi % cf.nx], by = coarse.ys[Math.floor(bi / cf.nx)];

  // Pass 2: bracket every coarse cell fitting within BRACKET_RATIO of the best.
  // This picks up a mirror basin automatically, with no need for cable geometry,
  // and stays wide when a call genuinely has no localised solution.
  let lx = bx, hx = bx, ly = by, hy = by;
  for (let k = 0; k < cf.field.length; k++) {
    if (cf.field[k] > best * BRACKET_RATIO) continue;
    const x = coarse.xs[k % cf.nx], y = coarse.ys[Math.floor(k / cf.nx)];
    if (x < lx) lx = x; if (x > hx) hx = x;
    if (y < ly) ly = y; if (y > hy) hy = y;
  }
  if (hx - lx < MIN_BOX_M) { const c0 = (lx + hx) / 2; lx = c0 - MIN_BOX_M / 2; hx = c0 + MIN_BOX_M / 2; }
  if (hy - ly < MIN_BOX_M) { const c0 = (ly + hy) / 2; ly = c0 - MIN_BOX_M / 2; hy = c0 + MIN_BOX_M / 2; }

  const fine = fitBox(lx, ly, hx, hy, 0.25);
  return { ...boxGrid(fine.x0, fine.y0, fine.x1, fine.y1, res), coarseBest: [bx, by] };
}

export const LEGEND_RATIOS = [1.0, 1.1, 1.25, 1.5, 2, 4, 8];
export function ratioLegendStops(cap = 8) {
  return LEGEND_RATIOS.filter((r) => r <= cap)
    .map((r) => ({ ratio: r, t: Math.log(r) / Math.log(cap) }));
}

export function fieldImage(field, nx, ny, best, cap = 8.0, dark = false) {
  // LOG map. Ratios here span 1 to ~80 with a median near 35, so a linear map
  // put almost every cell at the saturated end and the field read as one flat
  // wash. log(ratio)/log(cap) spreads the structure that actually matters.
  const data = new Uint8ClampedArray(nx * ny * 4);
  const lk = Math.log(cap);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      const t = Math.min(1, Math.max(0, Math.log(field[k] / best) / lk));
      const col = misfitColor(t, dark);
      const dst = ((ny - 1 - j) * nx + i) * 4;
      data[dst] = col[0];
      data[dst + 1] = col[1];
      data[dst + 2] = col[2];
      data[dst + 3] = Math.round(235 * (1 - t) + 20);
    }
  }
  return data;
}

export function zoomBoxAt(box, wx, wy, factor) {
  return {
    x0: wx - (wx - box.x0) * factor,
    x1: wx + (box.x1 - wx) * factor,
    y0: wy - (wy - box.y0) * factor,
    y1: wy + (box.y1 - wy) * factor,
  };
}
```

- [ ] **Step 4: Run to verify the tests pass**

```bash
cd viz && npm test
```

Expected: all pass, JS count rising to about 23.

- [ ] **Step 5: Zoom, pan and auto-fit in `app.js`**

Replace the fixed view derivation with a mutable view box, and frame each call as it is selected. Add near the top, after `bathyBounds` is defined:

```javascript
const homeBox = { ...bathyBounds };
state.viewBox = { ...homeBox };
```

Change `resizeAndDraw` to build the view from `state.viewBox` instead of the bathymetry extent:

```javascript
  view = makeView({
    x0: state.viewBox.x0, y0: state.viewBox.y0,
    x1: state.viewBox.x1, y1: state.viewBox.y1,
    width: rect.width, height: rect.height,
  });
```

Then append the interaction handlers:

```javascript
// --- zoom / pan ---------------------------------------------------------------
// A 1.8 km basin inside a 90 km survey box is not inspectable without this.
(() => {
  const canvas = $("map");
  canvas.style.cursor = "grab";
  let dragging = null;

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = canvas.getBoundingClientRect();
    const [wx, wy] = view.toWorld(e.clientX - r.left, e.clientY - r.top);
    const factor = e.deltaY > 0 ? 1.15 : 1 / 1.15;
    state.viewBox = zoomBoxAt(state.viewBox, wx, wy, factor);
    resizeAndDraw();
  }, { passive: false });

  canvas.addEventListener("pointerdown", (e) => {
    const r = canvas.getBoundingClientRect();
    dragging = { world: view.toWorld(e.clientX - r.left, e.clientY - r.top), box: { ...state.viewBox } };
    canvas.style.cursor = "grabbing";
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointerup", (e) => {
    dragging = null;
    canvas.style.cursor = "grab";
    canvas.releasePointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const r = canvas.getBoundingClientRect();
    const now = view.toWorld(e.clientX - r.left, e.clientY - r.top);
    const dx = dragging.world[0] - now[0], dy = dragging.world[1] - now[1];
    state.viewBox = {
      x0: dragging.box.x0 + dx, x1: dragging.box.x1 + dx,
      y0: dragging.box.y0 + dy, y1: dragging.box.y1 + dy,
    };
    resizeAndDraw();
  });
  canvas.addEventListener("dblclick", () => {
    state.viewBox = { ...homeBox };
    resizeAndDraw();
  });
})();
```

And in `computeField`, after the field is computed, frame it — selecting a call should take you to its basin rather than leaving you staring at the whole survey:

```javascript
  // Frame the call. Without this the basin is a handful of pixels.
  const padded = fitBox(xs[0], ys[0], xs[xs.length - 1], ys[ys.length - 1], 0.6);
  state.viewBox = padded;
```

Replace the `gridForCall` calls in `computeField` with `solutionGrid`:

```javascript
  const { xs, ys } = solutionGrid({
    px: p.px, py: p.py, pz: p.pz, obsT: p.obsT,
    c: state.c, sz: state.sz, res: state.res, bathyBounds,
  });
```

- [ ] **Step 6: The legend**

In `app.html`, immediately after the `#call-readout` div:

```html
    <div id="legend">
      <canvas id="legend-bar" height="14"></canvas>
      <div id="legend-ticks"></div>
      <div id="legend-caption">How much worse a position fits than the best fit.
        Darkest is the best-fitting region; contours mark 1.05, 1.10, 1.25 and 1.50×.</div>
    </div>
```

In `app.css`:

```css
#legend { margin: 10px 0 4px; }
#legend-bar { width: 100%; height: 14px; border-radius: 3px; display: block; }
#legend-ticks { position: relative; height: 14px; font-size: 10px; color: var(--ink-dim); }
#legend-ticks span { position: absolute; transform: translateX(-50%); white-space: nowrap; }
#legend-caption { font-size: 11px; color: var(--ink-dim); margin-top: 4px; }
```

In `app.js`, append and register as a layer:

```javascript
function drawLegend() {
  const bar = $("legend-bar");
  const rect = bar.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  bar.width = Math.max(1, Math.floor(rect.width * dpr));
  bar.height = Math.floor(14 * dpr);
  const ctx = bar.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const dark = themeColors().dark;
  for (let x = 0; x < rect.width; x++) {
    const c = misfitColor(x / Math.max(1, rect.width - 1), dark);
    ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
    ctx.fillRect(x, 0, 1, 14);
  }

  $("legend-ticks").innerHTML = ratioLegendStops()
    .map(({ ratio, t }) => {
      const label = ratio === 1 ? "best" : `${ratio}×`;
      return `<span style="left:${(t * 100).toFixed(1)}%">${label}</span>`;
    })
    .join("");
}

layers.push(() => drawLegend());
```

- [ ] **Step 7: Verify**

```bash
cd viz && npm test
uv run pytest paper/tests viz/tests -q
uv run python -m viz.build_bundle --out viz/out/index.html
```

All must pass and the build must succeed. Confirm `viz/out/index.html` contains `function solutionGrid`, `function zoomBoxAt` and `legend-ticks`.

- [ ] **Step 8: Commit**

```bash
git add viz/template viz/tests
git commit -m "Make the misfit field legible: solution-sized grid, log ramp, zoom/pan, legend"
```

- [ ] **Step 9: Cache the rasters, throttle panning, stop framing from fighting the user**

Review found that panning rebuilds the whole relief every mouse-move event, which will feel laggy — and that auto-framing, as written, will silently discard a deliberate pan as soon as Task 11 wires the sliders to `refresh()`.

The relief raster depends only on the elevation grid and the theme, never on the view; the field raster depends only on the computed field. Both can be built once and blitted.

In `render.mjs`, replace `drawBathy` with a build/blit pair:

```javascript
export function buildReliefRaster(bathy, dark = false) {
  const { nx, ny, zArr, shade, noData = -32768 } = bathy;
  const img = new ImageData(nx, ny);
  let zmin = Infinity, zmax = -Infinity;
  for (let k = 0; k < zArr.length; k++) {
    const v = zArr[k];
    if (v === noData) continue;
    if (v < zmin) zmin = v;
    if (v > zmax) zmax = v;
  }
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      const dst = ((ny - 1 - j) * nx + i) * 4; // world y up -> image y down
      const v = zArr[k];
      if (v === noData) { img.data[dst + 3] = 0; continue; }
      const c = depthColor(v, zmin, zmax, dark);
      const s = 0.55 + 0.45 * shade[k];
      img.data[dst] = c[0] * s;
      img.data[dst + 1] = c[1] * s;
      img.data[dst + 2] = c[2] * s;
      img.data[dst + 3] = 255;
    }
  }
  const off = new OffscreenCanvas(nx, ny);
  off.getContext("2d").putImageData(img, 0, 0);
  return off;
}

export function blitGrid(ctx, raster, geom, view) {
  const { x0, y0, dx, dy, nx, ny } = geom;
  const [sx0, sy0] = view.toScreen(x0, y0 + dy * (ny - 1));
  const [sx1, sy1] = view.toScreen(x0 + dx * (nx - 1), y0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(raster, sx0, sy0, sx1 - sx0, sy1 - sy0);
}
```

In `app.js`, cache both rasters and rebuild only when their inputs change:

```javascript
let reliefRaster = null, reliefDark = null;
function relief() {
  const dark = themeColors().dark;
  if (!reliefRaster || reliefDark !== dark) {
    reliefRaster = buildReliefRaster(bathy, dark);
    reliefDark = dark;
  }
  return reliefRaster;
}
```

Use `blitGrid(ctx, relief(), bathy, view)` in `draw()` in place of the old `drawBathy` call. Do the same for the field: build its `OffscreenCanvas` inside `computeField` (store it on `currentField.raster`) and blit it in the layer, so panning never re-colours it.

Throttle the pan to one repaint per frame:

```javascript
let rafPending = false;
function scheduleDraw() {
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => { rafPending = false; resizeAndDraw(); });
}
```

Call `scheduleDraw()` from `pointermove` and `wheel` instead of `resizeAndDraw()`.

Clamp the zoom so it cannot run away:

```javascript
const MIN_SPAN_M = 200;
const MAX_SPAN_M = 4 * Math.max(homeBox.x1 - homeBox.x0, homeBox.y1 - homeBox.y0);

function clampBox(b) {
  const w = b.x1 - b.x0, h = b.y1 - b.y0;
  const span = Math.max(w, h);
  if (span < MIN_SPAN_M || span > MAX_SPAN_M) return state.viewBox;
  return b;
}
```

Apply it in the wheel handler: `state.viewBox = clampBox(zoomBoxAt(state.viewBox, wx, wy, factor));`

Frame only when the selected call actually changes, so a slider drag leaves the view where the user put it:

```javascript
  // Frame on call change only. Re-framing on every recompute would yank the
  // view back every time a slider moves, discarding a deliberate pan.
  if (state.framedCall !== state.call) {
    state.viewBox = fitBox(xs[0], ys[0], xs[xs.length - 1], ys[ys.length - 1], 0.6);
    state.framedCall = state.call;
  }
```

Guard the ratio divide in `fieldImage`:

```javascript
  const denom = best > 0 && Number.isFinite(best) ? best : 1;
```

and use `denom` in place of `best`.

Add to `viz/tests/test_contour.mjs`:

```javascript
test("fieldImage survives a zero or non-finite best misfit", () => {
  // Unreachable with real noisy picks, but a degenerate synthetic set can fit
  // exactly, and NaN would propagate into every pixel of that raster.
  for (const best of [0, NaN, Infinity]) {
    const rgba = fieldImage(Float64Array.from([0, 1, 2, 3]), 4, 1, best, 8);
    assert.equal(rgba.length, 16);
    for (const v of rgba) assert.ok(Number.isFinite(v), `non-finite channel for best=${best}`);
  }
});
```

- [ ] **Step 10: Verify and commit**

```bash
cd viz && npm test
uv run pytest paper/tests viz/tests -q
uv run python -m viz.build_bundle --out viz/out/index.html
```

All must pass. Confirm the built page contains `function buildReliefRaster` and `function scheduleDraw`, and that `drawBathy` is gone.

```bash
git add viz/template viz/tests
git commit -m "Cache the relief and field rasters, throttle panning, frame only on call change"
```

---

### Task 9c: Make the map an index — per-call solutions, click-to-select, mark the current solution

**Why this exists.** The first person to use the page assumed the white dots on the map were the calls in the left list, and asked to click one to see its field. They are not: the dots are the **reference implementation's** 176 published solutions, and that CSV has no call-id column, so its rows carry synthetic `row0…row175` keys and **cannot be joined to the call list at all**. Meanwhile the map never marks the solution the current field actually refers to — the readout prints it numerically and nothing shows it in place. So the map looks like an index and isn't one.

This task makes it one, and labels what cannot be joined.

Cost was measured before committing to it: a coarse-only solve for each of the **608 calls with ≥5 picks** is ~66 M distance evaluations in total, about 0.3–0.8 s of scalar JavaScript. That is affordable once, deferred until after first paint.

**Files:**
- Modify: `viz/template/lib/contour.mjs` (export a coarse-solve helper)
- Modify: `viz/template/app.js`, `viz/template/app.html`, `viz/template/app.css`
- Modify: `viz/tests/test_contour.mjs`

**Interfaces:**
- Adds `coarseSolve({px, py, pz, obsT, c, sz, bathyBounds, res})` → `{x, y, best}` — the pass-1 solve alone, so the index and `solutionGrid` share one code path
- Adds `nearestMarker(markers, wx, wy, maxDist)` → index or −1 — pure, testable

- [ ] **Step 1: Write the failing tests**

Add to `viz/tests/test_contour.mjs`:

```javascript
import { coarseSolve, nearestMarker } from "../template/lib/contour.mjs";

test("coarseSolve recovers a synthetic source", () => {
  const n = 24;
  const px = Array.from({ length: n }, (_, i) => (i * 46000) / (n - 1));
  const py = px.map(() => 0);
  const pz = px.map(() => -400);
  const obsT = px.map((x) => Math.sqrt((x - 23000) ** 2 + 6000 ** 2 + 370 ** 2) / 1480);
  const s = coarseSolve({
    px, py, pz, obsT, c: 1480, sz: -30,
    bathyBounds: { x0: -60000, y0: -60000, x1: 160000, y1: 100000 },
  });
  assert.ok(Math.abs(s.x - 23000) < 1500, `x off: ${s.x}`);
  // A straight cable is bimodal, so either side is a correct answer.
  assert.ok(Math.abs(Math.abs(s.y) - 6000) < 1500, `|y| off: ${s.y}`);
  assert.ok(s.best >= 0 && Number.isFinite(s.best));
});

test("nearestMarker picks the closest inside the radius and rejects outside", () => {
  const m = [{ x: 0, y: 0 }, { x: 1000, y: 0 }, { x: 5000, y: 5000 }];
  assert.equal(nearestMarker(m, 900, 50, 500), 1);
  assert.equal(nearestMarker(m, 100, 0, 500), 0);
  assert.equal(nearestMarker(m, 3000, 0, 500), -1, "nothing within the radius");
  assert.equal(nearestMarker([], 0, 0, 500), -1, "empty list");
});
```

- [ ] **Step 2: Run to verify failure**

```bash
cd viz && npm test
```

Expected: FAIL — `coarseSolve` and `nearestMarker` are not exported.

- [ ] **Step 3: Factor the coarse pass out of `solutionGrid`**

In `contour.mjs`, extract what `solutionGrid`'s pass 1 already does so the index cannot drift from it:

```javascript
function wideBox({ px, py, bathyBounds }) {
  const xmin = Math.min(...px), xmax = Math.max(...px);
  const ymin = Math.min(...py), ymax = Math.max(...py);
  const pad = Math.max(10000, 0.5 * Math.max(xmax - xmin, ymax - ymin));
  return {
    x0: Math.max(bathyBounds.x0, xmin - pad), x1: Math.min(bathyBounds.x1, xmax + pad),
    y0: Math.max(bathyBounds.y0, ymin - pad), y1: Math.min(bathyBounds.y1, ymax + pad),
  };
}

export function coarseSolve({ px, py, pz, obsT, c, sz, bathyBounds, res = 90, objective = "l1" }) {
  const w = wideBox({ px, py, bathyBounds });
  const grid = boxGrid(w.x0, w.y0, w.x1, w.y1, res);
  const { field, nx } = misfitField({ ...grid, sz, px, py, pz, obsT, c, objective });
  let best = Infinity, bi = 0;
  for (let k = 0; k < field.length; k++) if (field[k] < best) { best = field[k]; bi = k; }
  return { x: grid.xs[bi % nx], y: grid.ys[Math.floor(bi / nx)], best, grid, field, nx };
}

export function nearestMarker(markers, wx, wy, maxDist) {
  let bi = -1, bd = maxDist * maxDist;
  for (let k = 0; k < markers.length; k++) {
    const dx = markers[k].x - wx, dy = markers[k].y - wy;
    const d = dx * dx + dy * dy;
    if (d <= bd) { bd = d; bi = k; }
  }
  return bi;
}
```

Rewrite `solutionGrid` to call `coarseSolve` rather than duplicating pass 1, keeping the rest of its bracket logic unchanged.

- [ ] **Step 4: Build the index, deferred, in `app.js`**

```javascript
// One coarse solution per call, so the map is an index of the dataset rather
// than a picture of somebody else's results. Measured at ~66M distance
// evaluations for 608 calls, so it runs after first paint, never before.
let callIndex = [];

function buildCallIndex() {
  const out = [];
  for (const call of PAYLOAD.calls) {
    const p = picksFor(call, { North: true, South: true });
    if (p.rows.length < 5) continue;
    const s = coarseSolve({
      px: p.px, py: p.py, pz: p.pz, obsT: p.obsT,
      c: PAYLOAD.meta.defaultC, sz: -30, bathyBounds,
    });
    out.push({ call, x: s.x, y: s.y, best: s.best, n: p.rows.length });
  }
  callIndex = out;
  $("index-status").textContent =
    `${out.length} of ${PAYLOAD.calls.length} calls located (coarse). Click a marker to open one.`;
  draw();
}

// After first paint, so the page is interactive immediately.
requestAnimationFrame(() => setTimeout(buildCallIndex, 0));
```

**Measured at ~2.1 s total, so it must be chunked rather than run in one go.** A single
synchronous pass would freeze the page for two seconds, which is worse than the delay itself.
Instead spend a bounded slice of each animation frame and yield, so the main thread is never
blocked for more than about a frame and the markers fill in progressively. The budget is
adaptive rather than a fixed chunk size, so a slower machine yields more often instead of
janking:

```javascript
const INDEX_FRAME_BUDGET_MS = 12;

function buildCallIndexIncremental() {
  const queue = PAYLOAD.calls.slice();
  const out = [];
  const t0 = performance.now();

  function step() {
    const frameStart = performance.now();
    while (queue.length && performance.now() - frameStart < INDEX_FRAME_BUDGET_MS) {
      const call = queue.shift();
      const p = picksFor(call, { North: true, South: true });
      if (p.rows.length < 5) continue;
      const s = coarseSolve({
        px: p.px, py: p.py, pz: p.pz, obsT: p.obsT,
        c: PAYLOAD.meta.defaultC, sz: -30, bathyBounds,
      });
      out.push({ call, x: s.x, y: s.y, best: s.best, n: p.rows.length });
    }
    callIndex = out;
    const done = PAYLOAD.calls.length - queue.length;
    if (queue.length) {
      $("index-status").textContent = `Locating calls… ${done} of ${PAYLOAD.calls.length}`;
      draw();
      requestAnimationFrame(step);
    } else {
      $("index-status").textContent =
        `${out.length} of ${PAYLOAD.calls.length} calls located (coarse, ` +
        `${((performance.now() - t0) / 1000).toFixed(1)} s). Click a marker to open one.`;
      draw();
    }
  }
  requestAnimationFrame(step);
}
```

Use `buildCallIndexIncremental` in place of the one-shot `buildCallIndex`.

- [ ] **Step 5: Draw the three mark types distinctly, and mark the current solution**

Identity comes from **shape**, not extra hue — the palette's categorical slots are already spent on the cables and its sequential slot on the field.

```javascript
layers.push((ctx, colors) => {
  // Per-call coarse solutions: small open circles.
  ctx.lineWidth = 1.25;
  ctx.strokeStyle = colors.accent;
  for (const m of callIndex) {
    if (m.call === state.call) continue;
    const [sx, sy] = view.toScreen(m.x, m.y);
    ctx.beginPath();
    ctx.arc(sx, sy, 3, 0, Math.PI * 2);
    ctx.stroke();
  }

  // The selected call's refined solution: bigger, ringed, with a crosshair.
  if (currentField) {
    const [sx, sy] = view.toScreen(currentField.bestXY[0], currentField.bestXY[1]);
    ctx.beginPath();
    ctx.arc(sx, sy, 7, 0, Math.PI * 2);
    ctx.strokeStyle = colors.surface;
    ctx.lineWidth = 3.5;
    ctx.stroke();
    ctx.strokeStyle = colors.accent;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(sx - 11, sy); ctx.lineTo(sx - 8, sy);
    ctx.moveTo(sx + 8, sy); ctx.lineTo(sx + 11, sy);
    ctx.moveTo(sx, sy - 11); ctx.lineTo(sx, sy - 8);
    ctx.moveTo(sx, sy + 8); ctx.lineTo(sx, sy + 11);
    ctx.stroke();
  }
});
```

- [ ] **Step 6: Click to select, and highlight the row**

Extend the pointer handling so a click that is not a drag selects the nearest indexed call:

```javascript
// In pointerdown, record the start position; in pointerup, if the pointer
// barely moved, treat it as a click rather than a pan.
canvas.addEventListener("pointerup", (e) => {
  const moved = dragStart
    ? Math.hypot(e.clientX - dragStart.sx, e.clientY - dragStart.sy)
    : 999;
  if (moved < 4 && callIndex.length) {
    const r = canvas.getBoundingClientRect();
    const [wx, wy] = view.toWorld(e.clientX - r.left, e.clientY - r.top);
    // 12 screen px, converted to world units at the current zoom.
    const tol = 12 / view.scale;
    const k = nearestMarker(callIndex, wx, wy, tol);
    if (k >= 0) selectCall(callIndex[k].call);
  }
  /* ...existing drag teardown... */
});
```

and centralise selection so the list and the map agree:

```javascript
function selectCall(call) {
  state.call = call;
  refresh();
  const row = $("call-list").querySelector(`[data-call="${CSS.escape(call)}"]`);
  if (row) {
    document.querySelectorAll("#call-list .sel").forEach((n) => n.classList.remove("sel"));
    row.classList.add("sel");
    row.scrollIntoView({ block: "nearest" });
  }
}
```

Give each list button `data-call` and make its handler call `selectCall(call)`.

- [ ] **Step 7: Say what cannot be joined**

In `app.html`, after the legend caption:

```html
      <div id="index-status" style="font-size:11px;color:var(--ink-dim);margin-top:6px"></div>
      <div id="marker-key" style="font-size:11px;color:var(--ink-dim);margin-top:6px">
        Open circles are one coarse solution per annotated call — click one to open it.
        The ringed crosshair is the selected call's refined solution.
        Filled dots are the reference implementation's published localizations; that CSV
        carries no call ids, so those cannot be linked to the list.
      </div>
```

In `app.css`:

```css
#call-list button.sel { background: var(--line); border-radius: 3px; font-weight: 600; }
```

- [ ] **Step 8: Verify and commit**

```bash
cd viz && npm test
uv run pytest paper/tests viz/tests -q
uv run python -m viz.build_bundle --out viz/out/index.html
```

All must pass. Report the total index build time, which `#index-status` prints when it finishes. About 2 s is expected and fine — what matters is that no single frame is blocked, which the adaptive per-frame budget guarantees. If a single frame ever exceeds ~30 ms, lower `INDEX_FRAME_BUDGET_MS` rather than reducing the grid resolution.

```bash
git add viz/template viz/tests
git commit -m "Make the map an index: per-call solutions, click-to-select, mark the current solution"
```

---

### Task 10: Moveout panel

**Files:**
- Modify: `viz/template/app.js`
- Create: `viz/template/lib/moveout.mjs`
- Create: `viz/tests/test_moveout.mjs`

**Interfaces:**
- Consumes: `travelTime` (Task 3), `picksFor` and `currentField` (Task 9).
- Produces: `moveoutScales({dist, obsT, width, height, pad}) -> {sx(d), sy(t), invert}`, `predictedCurve({sourceXY, sz, cable, distMin, distMax, c, n}) -> Array<[dist, t]>`

- [ ] **Step 1: Write the failing tests**

Create `viz/tests/test_moveout.mjs`:

```javascript
import test from "node:test";
import assert from "node:assert/strict";

import { moveoutScales, predictedCurve } from "../template/lib/moveout.mjs";
import { travelTime, bestT0 } from "../template/lib/forward.mjs";

test("moveoutScales maps data extremes inside the padded box", () => {
  const s = moveoutScales({ dist: [0, 1000], obsT: [5, 9], width: 100, height: 50, pad: 10 });
  assert.ok(s.sx(0) >= 10 && s.sx(1000) <= 90);
  assert.ok(s.sy(9) >= 10 && s.sy(5) <= 40);
  // time increases downward on screen
  assert.ok(s.sy(9) > s.sy(5));
});

test("moveoutScales handles a degenerate single-value range", () => {
  const s = moveoutScales({ dist: [500, 500], obsT: [3, 3], width: 100, height: 50, pad: 10 });
  assert.ok(Number.isFinite(s.sx(500)) && Number.isFinite(s.sy(3)));
});

test("predictedCurve is a V with its minimum nearest the source", () => {
  const cable = {
    chanM: Float32Array.from({ length: 101 }, (_, i) => i * 500),
    x: Float32Array.from({ length: 101 }, (_, i) => i * 500),
    y: new Float32Array(101),
    depth: new Float32Array(101),
  };
  const pts = predictedCurve({
    sourceXY: [25000, 4000], sz: 0, cable,
    distMin: 0, distMax: 50000, c: 1480, n: 51,
  });
  assert.equal(pts.length, 51);
  const times = pts.map(([, t]) => t);
  const kmin = times.indexOf(Math.min(...times));
  assert.ok(Math.abs(pts[kmin][0] - 25000) < 1200, "apex should sit near the source's along-cable position");
  assert.ok(times[0] > times[kmin] && times[times.length - 1] > times[kmin]);
});

test("a pooled t0 exposes inter-cable inconsistency that a per-cable t0 hides", () => {
  // This is the panel's whole reason for existing, so it gets a test rather than
  // a comment. Two cables' picks for one call, but cable B's arrival times are
  // offset by 2 s -- a solution that cannot explain both. With ONE pooled
  // emission time (what the misfit actually minimises) that offset shows up in
  // the residuals. With a t0 per cable, each cable is independently re-centred
  // on its own picks and the inconsistency vanishes from the panel entirely.
  const pxA = [0, 1000, 2000], pyA = [0, 0, 0], pzA = [-400, -400, -400];
  const pxB = [0, 1000, 2000], pyB = [8000, 8000, 8000], pzB = [-400, -400, -400];
  const sx = 1000, sy = 3000, sz = -30, c = 1480;

  const ttA = travelTime(sx, sy, sz, pxA, pyA, pzA, c);
  const ttB = travelTime(sx, sy, sz, pxB, pyB, pzB, c);
  const obsA = [...ttA].map((t) => t + 5.0);
  const obsB = [...ttB].map((t) => t + 7.0); // 2 s inconsistent

  const px = [...pxA, ...pxB], py = [...pyA, ...pyB], pz = [...pzA, ...pzB];
  const obs = [...obsA, ...obsB];
  const tt = travelTime(sx, sy, sz, px, py, pz, c);

  const pooled = bestT0(obs, tt, "l1");
  const worstPooled = Math.max(...obs.map((o, i) => Math.abs(o - tt[i] - pooled)));
  assert.ok(worstPooled > 0.9, `pooled t0 must surface the 2 s offset, got ${worstPooled}`);

  const t0A = bestT0(obsA, ttA, "l1"), t0B = bestT0(obsB, ttB, "l1");
  const worstSplit = Math.max(
    ...obsA.map((o, i) => Math.abs(o - ttA[i] - t0A)),
    ...obsB.map((o, i) => Math.abs(o - ttB[i] - t0B)),
  );
  assert.ok(worstSplit < 1e-9, `a per-cable t0 would hide it entirely, got ${worstSplit}`);
});

test("predictedCurve handles a non-uniform tail interval", () => {
  // The shipped cable keeps its final channel, so the last interval is shorter
  // than the rest. Index arithmetic from start+step would misplace points there.
  const cable = {
    chanM: Float32Array.from([0, 1000, 2000, 2100]),
    x: Float32Array.from([0, 1000, 2000, 2100]),
    y: Float32Array.from([0, 0, 0, 0]),
    depth: new Float32Array(4),
  };
  const pts = predictedCurve({
    sourceXY: [2100, 0], sz: 0, cable, distMin: 2000, distMax: 2100, c: 1000, n: 3,
  });
  // At d = 2100 the source is exactly on the cable, so travel time is 0.
  assert.ok(Math.abs(pts[2][1]) < 1e-6, `expected 0 at the tail, got ${pts[2][1]}`);
  // At d = 2050 the cable point is 50 m away.
  assert.ok(Math.abs(pts[1][1] - 0.05) < 1e-6, `expected 0.05 s, got ${pts[1][1]}`);
});
```

- [ ] **Step 2: Run to verify failure**

```bash
cd viz && npm test
```

Expected: FAIL — cannot find `../template/lib/moveout.mjs`.

- [ ] **Step 3: Implement**

Create `viz/template/lib/moveout.mjs`:

```javascript
export function moveoutScales({ dist, obsT, width, height, pad = 24 }) {
  const d0 = Math.min(...dist), d1 = Math.max(...dist);
  const t0 = Math.min(...obsT), t1 = Math.max(...obsT);
  const dspan = d1 - d0 || 1;
  const tspan = t1 - t0 || 1;
  return {
    sx: (d) => pad + ((d - d0) / dspan) * (width - 2 * pad),
    sy: (t) => pad + ((t - t0) / tspan) * (height - 2 * pad),
    invert: (px) => d0 + ((px - pad) / (width - 2 * pad)) * dspan,
  };
}

// Binary search on the shipped chanM array. The decimated samples are NOT
// uniformly spaced at the cable's tail (the final channel is always kept, which
// leaves a short last interval), so index arithmetic from start+step would be
// wrong there.
function locate(chanM, d) {
  let lo = 0, hi = chanM.length - 1;
  if (d <= chanM[0]) return [0, 0];
  if (d >= chanM[hi]) return [hi - 1, 1];
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (chanM[mid] <= d) lo = mid; else hi = mid;
  }
  const span = chanM[lo + 1] - chanM[lo];
  return [lo, span > 0 ? (d - chanM[lo]) / span : 0];
}

export function predictedCurve({ sourceXY, sz, cable, distMin, distMax, c, n = 120 }) {
  const [sx, sy] = sourceXY;
  const out = [];
  for (let k = 0; k < n; k++) {
    const d = distMin + ((distMax - distMin) * k) / (n - 1);
    const [i, f] = locate(cable.chanM, d);
    const cx = cable.x[i] + f * (cable.x[i + 1] - cable.x[i]);
    const cy = cable.y[i] + f * (cable.y[i + 1] - cable.y[i]);
    const cz = cable.depth[i] + f * (cable.depth[i + 1] - cable.depth[i]);
    const r = Math.hypot(cx - sx, cy - sy, cz - sz);
    out.push([d, r / c]);
  }
  return out;
}
```

- [ ] **Step 4: Draw the panel in `app.js`**

Append to `viz/template/app.js`:

```javascript
function drawMoveout() {
  const canvas = $("moveout");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(180 * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, 180);

  if (!state.call || !currentField) return;
  const p = picksFor(state.call, state.cablesOn);
  const colors = themeColors();
  const s = moveoutScales({ dist: p.dist, obsT: p.obsT, width: rect.width, height: 180 });

  // ONE emission time for the call, pooled across every active pick, computed
  // exactly as bestT0/misfitField do. A per-cable t0 would re-centre each cable
  // on its own picks and so hide inter-cable inconsistency -- the very "tiny
  // misfit but nonsense" case this panel exists to expose. Physically there is
  // one emission time, not one per cable, so pooling is also the honest model.
  // Uses travelTime per pick rather than the nearest of 120 curve samples,
  // which removes a discretisation error too.
  const ttAll = travelTime(
    currentField.bestXY[0], currentField.bestXY[1], state.sz,
    p.px, p.py, p.pz, state.c,
  );
  const t0 = bestT0(p.obsT, ttAll, "l1");

  // Predicted curve per active cable, all shifted by that single t0.
  for (const name of Object.keys(cables)) {
    if (!state.cablesOn[name]) continue;
    const idx = p.cable.map((cb, k) => (cb === name ? k : -1)).filter((k) => k >= 0);
    if (!idx.length) continue;
    const pts = predictedCurve({
      sourceXY: currentField.bestXY, sz: state.sz, cable: cables[name],
      distMin: Math.min(...idx.map((k) => p.dist[k])),
      distMax: Math.max(...idx.map((k) => p.dist[k])),
      c: state.c,
    });

    ctx.beginPath();
    pts.forEach(([d, t], k) => {
      const X = s.sx(d), Y = s.sy(t + t0);
      k === 0 ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y);
    });
    ctx.strokeStyle = colors[name];
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.85;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  for (let k = 0; k < p.dist.length; k++) {
    ctx.fillStyle = colors[p.cable[k]];
    ctx.beginPath();
    ctx.arc(s.sx(p.dist[k]), s.sy(p.obsT[k]), 2.5, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--ink-dim");
  ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
  ctx.fillText("along-cable distance →", 24, 174);
  ctx.save();
  ctx.translate(10, 100);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("arrival time →", 0, 0);
  ctx.restore();
}

layers.push(() => drawMoveout());
draw();
```

- [ ] **Step 5: Run tests, rebuild, check**

```bash
cd viz && npm test && cd ..
uv run python -m viz.build_bundle --out viz/out/index.html
```

Expected: the JS suite passes. In the browser, selecting a call draws its picks as dots in the moveout panel with the predicted V through them; residuals should look small for a well-constrained call and visibly poor for the 8-pick runaway.

- [ ] **Step 6: Commit**

```bash
git add viz/template/lib/moveout.mjs viz/template/app.js viz/tests/test_moveout.mjs
git commit -m "Add moveout panel with predicted arrival curve"
```

---

### Task 11: Sliders, cable toggles and the mirror badge

**Files:**
- Modify: `viz/template/app.js`
- Create: `viz/template/lib/mirror.mjs`
- Create: `viz/tests/test_mirror.mjs`

**Interfaces:**
- Consumes: `currentField`, `cables`, `picksFor`.
- Produces: `signedSide(cable, x, y) -> number` (single point, mirrors the Python `signed_side`), `mirrorPenalty({field, xs, ys, nx, ny, cable}) -> {best, mirror, ratio, bestXY, mirrorXY}`

- [ ] **Step 1: Write the failing tests**

Create `viz/tests/test_mirror.mjs`:

```javascript
import test from "node:test";
import assert from "node:assert/strict";

import { signedSide, mirrorPenalty } from "../template/lib/mirror.mjs";

const straight = {
  x: Float32Array.from({ length: 51 }, (_, i) => i * 1000),
  y: new Float32Array(51),
};

test("signedSide separates the two sides of a straight cable", () => {
  const a = signedSide(straight, 25000, 5000);
  const b = signedSide(straight, 25000, -5000);
  assert.equal(a, -b);
  assert.ok(a === 1 || a === -1);
});

test("mirrorPenalty ratio is 1 for a symmetric field", () => {
  const xs = [24000, 25000, 26000];
  const ys = [-5000, 0, 5000];
  const nx = 3, ny = 3;
  // symmetric about y = 0
  const field = new Float64Array([1, 0.5, 1, 9, 9, 9, 1, 0.5, 1]);
  const mp = mirrorPenalty({ field, xs, ys, nx, ny, cable: straight });
  assert.ok(Math.abs(mp.ratio - 1) < 1e-9);
});

test("mirrorPenalty reports the two candidate positions so separation is measurable", () => {
  // A low ratio alone does not mean the km-scale ambiguity: measured over 232
  // single-cable solves, 27% of the low-ratio cases have the two candidates under
  // 500 m apart, meaning the solution sits on the cable axis and the "two sides"
  // are the same place. The badge needs both positions to tell those apart, so
  // mirrorPenalty must return bestXY as well as mirrorXY.
  const xs = [24000, 25000, 26000];
  const ys = [-5000, 0, 5000];
  const field = new Float64Array([1, 0.5, 1, 9, 9, 9, 3, 2.0, 3]);
  const mp = mirrorPenalty({ field, xs, ys, nx: 3, ny: 3, cable: straight });
  assert.ok(Array.isArray(mp.bestXY) && mp.bestXY.length === 2, "bestXY must be reported");
  assert.ok(Array.isArray(mp.mirrorXY) && mp.mirrorXY.length === 2);
  const sep = Math.hypot(mp.bestXY[0] - mp.mirrorXY[0], mp.bestXY[1] - mp.mirrorXY[1]);
  assert.ok(sep > 0, "the two candidates must be distinct positions");
  // In this fixture the minima sit at y = -5000 and y = +5000, so 10 km apart.
  assert.ok(Math.abs(sep - 10000) < 1e-6, `expected 10 km separation, got ${sep}`);
});

test("mirrorPenalty ratio exceeds 1 when one side fits better", () => {
  const xs = [24000, 25000, 26000];
  const ys = [-5000, 0, 5000];
  const field = new Float64Array([1, 0.5, 1, 9, 9, 9, 3, 2.0, 3]);
  const mp = mirrorPenalty({ field, xs, ys, nx: 3, ny: 3, cable: straight });
  assert.ok(mp.ratio > 1);
  assert.ok(mp.best <= mp.mirror);
});
```

- [ ] **Step 2: Run to verify failure**

```bash
cd viz && npm test
```

Expected: FAIL — cannot find `../template/lib/mirror.mjs`.

- [ ] **Step 3: Implement**

Create `viz/template/lib/mirror.mjs`:

```javascript
// Mirrors signed_side / mirror_penalty in src/dclde_localization/physics/forward.py.

export function signedSide(cable, x, y) {
  let k = 0, bd = Infinity;
  for (let i = 0; i < cable.x.length; i++) {
    const d = (cable.x[i] - x) ** 2 + (cable.y[i] - y) ** 2;
    if (d < bd) { bd = d; k = i; }
  }
  const k0 = Math.max(0, k - 1);
  const k1 = Math.min(cable.x.length - 1, k + 1);
  const tx = cable.x[k1] - cable.x[k0];
  const ty = cable.y[k1] - cable.y[k0];
  const nx = x - cable.x[k];
  const ny = y - cable.y[k];
  const s = Math.sign(tx * ny - ty * nx);
  return s === 0 ? 1 : s;
}

export function mirrorPenalty({ field, xs, ys, nx, ny, cable }) {
  const best = { 1: Infinity, "-1": Infinity };
  const xy = { 1: null, "-1": null };
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const v = field[j * nx + i];
      const s = signedSide(cable, xs[i], ys[j]);
      if (v < best[s]) { best[s] = v; xy[s] = [xs[i], ys[j]]; }
    }
  }
  const pos = best[1], neg = best["-1"];
  const [lo, hi] = pos <= neg ? [pos, neg] : [neg, pos];
  const [loXY, hiXY] = pos <= neg ? [xy[1], xy["-1"]] : [xy["-1"], xy[1]];
  return { best: lo, mirror: hi, ratio: lo > 0 ? hi / lo : Infinity, bestXY: loXY, mirrorXY: hiXY };
}
```

- [ ] **Step 4: Wire the controls in `app.js`**

Append to `viz/template/app.js`:

```javascript
function activeCableNames() {
  return Object.keys(cables).filter((n) => state.cablesOn[n]);
}

function renderMirrorBadge() {
  const existing = $("mirror-badge");
  if (existing) existing.remove();
  if (!currentField) return;
  const host = document.createElement("div");
  host.id = "mirror-badge";

  const active = activeCableNames();
  const rows = [];
  if (active.length === 1) {
    const mp = mirrorPenalty({ ...currentField, cable: cables[active[0]] });
    const sep = Math.hypot(mp.bestXY[0] - mp.mirrorXY[0], mp.bestXY[1] - mp.mirrorXY[1]);

    // A low ratio alone does NOT mean the km-scale left-right ambiguity. Measured
    // over 232 single-cable solves: 14% have a ratio under 1.10, but 27% of those
    // have the two candidates less than 500 m apart, i.e. the solution sits
    // essentially on the cable axis and the "two sides" are the same place.
    // Only 8.2% of solves are both ratio < 1.10 AND more than 1 km apart, which
    // is the case that actually costs kilometres if you pick the wrong side.
    // Separation distribution: median 2.4 km, p10 0.47 km, p90 16.2 km.
    const AMBIGUOUS_RATIO = 1.10;
    const DISTINCT_SEP_M = 500;
    const lowRatio = mp.ratio < AMBIGUOUS_RATIO;
    const distinct = sep > DISTINCT_SEP_M;
    const cls = lowRatio && distinct ? "badge warn" : "badge";

    let note = "";
    if (lowRatio && distinct) {
      note = "<div style='color:var(--ink-dim);font-size:12px'>Both sides fit almost equally and " +
        "they are " + (sep / 1000).toFixed(1) + " km apart — a genuine left-right ambiguity. " +
        "Picking the wrong side costs that whole distance.</div>";
    } else if (lowRatio) {
      note = "<div style='color:var(--ink-dim);font-size:12px'>Both sides fit equally, but they are " +
        "only " + Math.round(sep) + " m apart: the solution sits essentially on the cable axis, " +
        "so the two sides are not distinct places.</div>";
    }

    rows.push(
      `<div style="margin:6px 0"><span class="${cls}">mirror +${((mp.ratio - 1) * 100).toFixed(0)}%</span></div>` +
      `<div class="readout"><dt>mirror at</dt><dd>${(mp.mirrorXY[0]/1000).toFixed(1)}, ${(mp.mirrorXY[1]/1000).toFixed(1)} km</dd></div>` +
      `<div class="readout"><dt>separation</dt><dd>${sep < 1000 ? Math.round(sep) + " m" : (sep/1000).toFixed(2) + " km"}</dd></div>` +
      note
    );
  } else {
    // With two cables there is no single mirror; report one ratio per cable.
    for (const name of active) {
      const mp = mirrorPenalty({ ...currentField, cable: cables[name] });
      // The grid is sized around the solution, so it can lie entirely on one
      // side of the OTHER cable — in which case there is no mirror-side cell to
      // compare against and the ratio is undefined, not infinite. Printing
      // "+Infinity%" told the user nothing.
      const cell = Number.isFinite(mp.ratio)
        ? `+${((mp.ratio - 1) * 100).toFixed(0)}%`
        : `<span title="the computed grid does not span both sides of ${name}">n/a</span>`;
      rows.push(`<div class="readout"><dt>mirror vs ${name}</dt><dd>${cell}</dd></div>`);
    }
  }
  host.innerHTML = rows.join("");
  $("call-readout").appendChild(host);
}

// The badge is a layer, appended after renderReadout has rebuilt the panel.
// `refresh` is defined in Task 9 and is not wrapped here.
layers.push(() => renderMirrorBadge());

// --- controls ---
const cableToggle = $("cable-toggle");
for (const name of Object.keys(cables)) {
  const id = `cable-${name}`;
  const wrap = document.createElement("label");
  wrap.style.cssText = "display:flex;gap:6px;align-items:center;margin:2px 0;color:var(--ink)";
  wrap.innerHTML = `<input type="checkbox" id="${id}" checked><span>${name}</span>`;
  cableToggle.appendChild(wrap);
  wrap.querySelector("input").addEventListener("change", (e) => {
    state.cablesOn[name] = e.target.checked;
    refresh();
  });
}

$("c-slider").addEventListener("input", (e) => {
  state.c = Number(e.target.value);
  $("c-value").textContent = `${state.c} m/s`;
  refresh();
});
$("z-slider").addEventListener("input", (e) => {
  state.sz = Number(e.target.value);
  $("z-value").textContent = `${state.sz} m`;
  refresh();
});
$("res-select").addEventListener("change", (e) => {
  state.res = Number(e.target.value);
  refresh();
});

$("c-value").textContent = `${state.c} m/s`;
$("z-value").textContent = `${state.sz} m`;

refresh();
```

**Say that the index does not follow the sliders.** The per-call index from Task 9c is computed once at the default sound speed and depth; rebuilding all 608 solutions on every slider tick would cost seconds. So the markers stay put while the selected call's field moves, and the UI must admit that rather than let a user infer the index is live. Append to `#marker-key` in `app.html`:

```html
        Index markers are computed once at the default sound speed and depth, so they
        do not move with the sliders; only the selected call's field and refined
        solution do.
```

Note there is deliberately no second `call-search` listener and no `call-list` click delegate here: Task 9's call buttons already call `refresh()`, and the badge updates because it is a layer.

- [ ] **Step 5: Run tests, rebuild, verify the centerpiece**

```bash
cd viz && npm test && cd ..
uv run python -m viz.build_bundle --out viz/out/index.html
```

Expected: 20 JS tests pass. In the browser, pick a call that appears on both cables, untick South: two basins appear and the badge reports a mirror penalty. Tick South again: the mirror basin lifts. Dragging the sound-speed slider should visibly translate the basins. Confirm at least one call shows the warn-styled badge (mirror within 10%).

- [ ] **Step 6: Commit**

```bash
git add viz/template/lib/mirror.mjs viz/template/app.js viz/tests/test_mirror.mjs
git commit -m "Add sound-speed and depth sliders, cable toggles and mirror-penalty badge"
```

- [ ] **Step 7: Make the badge cheap, and stop it crashing on a one-sided grid**

Two Critical findings, both measured.

**It crashes.** `mirrorPenalty` leaves one side's `xy` as `null` whenever every grid cell falls on the same side of the cable — which the solution-sized grid does routinely for a confidently-offset single-cable solve, because the bracket only keeps cells within 5× of the best misfit. The single-cable branch dereferences `mp.mirrorXY[0]` unguarded, so it throws inside the `layers` loop, aborting `draw()` mid-frame and taking `selectCall`'s row highlight with it. The two-cable branch was already guarded; this one was missed.

**It is unusably slow.** Measured in a browser: panning with a call selected runs at a **median 447 ms per frame** (worst 606 ms) — about 2 fps — against 0.1 ms with no call selected. `selectCall` takes 467 ms. The cause is that `renderMirrorBadge` is a `layers` entry, so it recomputes `mirrorPenalty` on every frame including every pan frame, and `mirrorPenalty` calls `signedSide` per grid cell while `signedSide` linearly scans the whole decimated cable: 33,800 cells × ~11,600 points ≈ 3.9 × 10⁸ iterations per badge.

Two independent fixes, both needed. Memoizing alone still leaves ~400 ms on every call selection; a cheaper side test alone still pays it every frame.

First, a coarse cable for side determination. Which side of a smooth cable a point lies on is a coarse-scale property — 200 m node spacing resolves it for anything not sitting within metres of the cable, and a point that close is genuinely ambiguous anyway. In `app.js`, next to `cables`:

```javascript
// Side determination does not need 8 m cable resolution; ~200 m is ample and
// cuts the mirrorPenalty scan by ~25x. Built once.
const SIDE_SPACING_M = 200;
const sideCables = Object.fromEntries(
  Object.entries(cables).map(([name, c]) => {
    const step = Math.max(1, Math.round(SIDE_SPACING_M / 8.17));
    const keep = [];
    for (let k = 0; k < c.x.length; k += step) keep.push(k);
    if (keep[keep.length - 1] !== c.x.length - 1) keep.push(c.x.length - 1);
    return [name, {
      x: Float32Array.from(keep, (k) => c.x[k]),
      y: Float32Array.from(keep, (k) => c.y[k]),
    }];
  })
);
```

Second, compute the penalties once per field rather than per frame. At the end of `computeField`, after `currentField` is assigned:

```javascript
  // Once per field, not once per frame: this depends on the field and the
  // active cables, neither of which changes while panning.
  currentField.mirror = {};
  for (const name of Object.keys(cables)) {
    if (!state.cablesOn[name]) continue;
    currentField.mirror[name] = mirrorPenalty({
      ...currentField, cable: sideCables[name],
    });
  }
```

and rewrite `renderMirrorBadge` to read `currentField.mirror[name]` instead of calling `mirrorPenalty`, guarding a one-sided grid in the single-cable branch too:

```javascript
  if (active.length === 1) {
    const mp = currentField.mirror[active[0]];
    if (!mp || !mp.mirrorXY || !mp.bestXY || !Number.isFinite(mp.ratio)) {
      // Every grid cell fell on one side of the cable, so there is no mirror
      // candidate to compare against. Undefined, not infinite.
      host.innerHTML =
        `<div class="readout"><dt>mirror</dt><dd>` +
        `<span title="the computed grid lies entirely on one side of ${active[0]}">n/a</span>` +
        `</dd></div>`;
      $("call-readout").appendChild(host);
      return;
    }
    /* ...existing separation / threshold / note logic... */
  }
```

Third, throttle the sliders to one recompute per frame, the way panning already is:

```javascript
let refreshPending = false;
function scheduleRefresh() {
  if (refreshPending) return;
  refreshPending = true;
  requestAnimationFrame(() => { refreshPending = false; refresh(); });
}
```

Have both slider `input` handlers call `scheduleRefresh()` rather than `refresh()`. The value readouts still update synchronously so the numbers track the thumb.

Add to `viz/tests/test_mirror.mjs`:

```javascript
test("mirrorPenalty reports a one-sided grid as unusable rather than crashing", () => {
  // The solution-sized grid routinely lies entirely on one side of the cable
  // for a confidently-offset solve. Consumers must be able to detect that.
  const xs = [24000, 25000, 26000];
  const ys = [5000, 6000, 7000]; // all strictly one side of a y = 0 cable
  const field = new Float64Array([3, 2, 3, 2, 1, 2, 3, 2, 3]);
  const mp = mirrorPenalty({ field, xs, ys, nx: 3, ny: 3, cable: straight });
  assert.ok(!Number.isFinite(mp.ratio) || mp.mirrorXY === null,
    "a one-sided grid must not report a finite mirror ratio with a real position");
  assert.ok(mp.bestXY, "the best side must still be reported");
});
```

- [ ] **Step 8: Verify the fix actually made it fast, and commit**

```bash
cd viz && npm test
uv run pytest paper/tests viz/tests -q
uv run python -m viz.build_bundle --out viz/out/index.html
```

All must pass. The performance claim needs measuring, not asserting: the reference numbers to beat are a 447 ms median pan frame and a 467 ms `selectCall`. Report what you can measure and say plainly if you cannot measure it in a browser.

```bash
git add viz/template viz/tests
git commit -m "Memoize the mirror penalty, coarsen side determination, guard a one-sided grid"
```

---

### Task 12: Misfit-versus-depth panel

**Files:**
- Modify: `viz/template/app.js`
- Create: `viz/template/lib/zprofile.mjs`
- Create: `viz/tests/test_zprofile.mjs`

**Interfaces:**
- Consumes: `misfitField` (Task 3).
- Produces: `zProfile({zs, xs, ys, px, py, pz, obsT, c, objective}) -> Array<{z, best}>`, `flatnessRatio(profile) -> number`

- [ ] **Step 1: Write the failing tests**

Create `viz/tests/test_zprofile.mjs`:

```javascript
import test from "node:test";
import assert from "node:assert/strict";

import { zProfile, flatnessRatio } from "../template/lib/zprofile.mjs";

function synthetic() {
  // Straight, flat cable: source depth is completely unobservable.
  const n = 30;
  const px = Array.from({ length: n }, (_, i) => (i * 50000) / (n - 1));
  const py = px.map(() => 0);
  const pz = px.map(() => 0);
  const obsT = px.map((x) => Math.hypot(x - 25000, 6000) / 1480);
  return { px, py, pz, obsT };
}

test("zProfile returns one best misfit per depth", () => {
  const { px, py, pz, obsT } = synthetic();
  const prof = zProfile({
    zs: [-200, -100, -30, 0], xs: [20000, 25000, 30000], ys: [4000, 6000, 8000],
    px, py, pz, obsT, c: 1480, objective: "l1",
  });
  assert.equal(prof.length, 4);
  for (const p of prof) assert.ok(Number.isFinite(p.best));
});

test("flatnessRatio is near 1 when depth is unobservable", () => {
  const { px, py, pz, obsT } = synthetic();
  const prof = zProfile({
    zs: [-150, -100, -50, -10], xs: [24000, 25000, 26000], ys: [5000, 6000, 7000],
    px, py, pz, obsT, c: 1480, objective: "l1",
  });
  const r = flatnessRatio(prof);
  assert.ok(r < 1.2, `expected a near-flat profile, got ratio ${r}`);
});

test("flatnessRatio handles a zero minimum without dividing by zero", () => {
  assert.ok(Number.isFinite(flatnessRatio([{ z: 0, best: 0 }, { z: -10, best: 1 }])));
});
```

- [ ] **Step 2: Run to verify failure**

```bash
cd viz && npm test
```

Expected: FAIL — cannot find `../template/lib/zprofile.mjs`.

- [ ] **Step 3: Implement**

Create `viz/template/lib/zprofile.mjs`:

```javascript
import { misfitField } from "./forward.mjs";

export function zProfile({ zs, xs, ys, px, py, pz, obsT, c, objective = "l1" }) {
  return zs.map((z) => {
    const { field } = misfitField({ xs, ys, sz: z, px, py, pz, obsT, c, objective });
    let best = Infinity;
    for (let k = 0; k < field.length; k++) if (field[k] < best) best = field[k];
    return { z, best };
  });
}

export function flatnessRatio(profile) {
  // An empty profile would give -Infinity / Infinity = NaN. Unreachable from the
  // current call site, but this is an exported primitive.
  if (!profile || profile.length === 0) return 1;
  let lo = Infinity, hi = -Infinity;
  for (const p of profile) {
    if (p.best < lo) lo = p.best;
    if (p.best > hi) hi = p.best;
  }
  return hi / Math.max(lo, 1e-12);
}
```

- [ ] **Step 4: Draw the panel in `app.js`**

Append to `viz/template/app.js`:

```javascript
function drawZProfile() {
  const canvas = $("zprofile");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(140 * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, 140);
  if (!currentField) return;

  const prof = currentField.zprof;
  if (!prof || prof.length < 2) return;

  const lo = Math.min(...prof.map((q) => q.best));
  const hi = Math.max(...prof.map((q) => q.best));
  const span = hi - lo || 1;
  const pad = 26;
  const sx = (z) => pad + ((z + 200) / 200) * (rect.width - 2 * pad);
  const sy = (v) => 12 + (1 - (v - lo) / span) * (140 - 2 * 12 - 14);

  const colors = themeColors();
  ctx.beginPath();
  prof.forEach((q, k) => (k === 0 ? ctx.moveTo(sx(q.z), sy(q.best)) : ctx.lineTo(sx(q.z), sy(q.best))));
  ctx.strokeStyle = colors.accent;
  ctx.lineWidth = 1.75;
  ctx.stroke();

  // Current assumed depth.
  ctx.beginPath();
  ctx.moveTo(sx(state.sz), 8);
  ctx.lineTo(sx(state.sz), 126);
  ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--ink-dim");
  ctx.setLineDash([3, 3]);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--ink-dim");
  ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
  const r = flatnessRatio(prof);
  ctx.fillText(`source depth −200…0 m · misfit varies ×${r.toFixed(3)}`, pad, 136);
}

layers.push(() => drawZProfile());
refresh();
```

**Compute the profile once per field, not once per frame.** `drawZProfile` is a `layers` entry, so it runs on every redraw including every pan frame. Task 11 shipped exactly that mistake with the mirror badge and it cost a measured 447 ms per frame. The profile depends only on the field and the sound speed, never on the view, so it belongs in `computeField`.

It also must reuse the field's own xy grid rather than building a new one — `gridForCall` no longer exists, and more importantly the profile is only meaningful as "misfit over the same area, at different assumed depths". Subsample that grid so 21 depths stay cheap:

`subsampleGrid` belongs in `contour.mjs` beside the other grid helpers, exported, so it can
actually be tested — and it must keep both endpoints. A plain modulo filter drops the last
element unless the length happens to divide evenly (300 elements at stride 7 drops index 299),
which would silently shrink the profile's footprint below the map's. That is the **third**
appearance of this same strand-the-tail bug in this plan, after the cable decimation cost 6.3 m
of position error, so it gets a guarantee and a test rather than another modulo filter:

```javascript
export function subsampleGrid(xs, ys, target = 40) {
  const pick = (arr) => {
    const step = Math.max(1, Math.floor(arr.length / target));
    const out = [];
    for (let i = 0; i < arr.length; i += step) out.push(arr[i]);
    // Always keep the far endpoint: without this the subsampled area is smaller
    // than the grid it is meant to represent.
    if (out[out.length - 1] !== arr[arr.length - 1]) out.push(arr[arr.length - 1]);
    return out;
  };
  return { xs: pick(xs), ys: pick(ys) };
}
```

Add to `viz/tests/test_contour.mjs`:

```javascript
test("subsampleGrid keeps both endpoints", () => {
  // A modulo filter drops the last element unless the length divides evenly.
  // The subsampled area must cover the same extent as the grid it represents.
  for (const n of [300, 301, 260, 41, 3]) {
    const xs = Array.from({ length: n }, (_, i) => i * 10);
    const ys = Array.from({ length: n }, (_, i) => i * 7);
    const g = subsampleGrid(xs, ys, 40);
    assert.equal(g.xs[0], xs[0], `n=${n}: lost the near endpoint`);
    assert.equal(g.xs[g.xs.length - 1], xs[n - 1], `n=${n}: lost the far endpoint`);
    assert.equal(g.ys[g.ys.length - 1], ys[n - 1], `n=${n}: lost the far y endpoint`);
    assert.ok(g.xs.length <= 42, `n=${n}: subsampled to ${g.xs.length}, expected near 40`);
    assert.ok(g.xs.length >= Math.min(n, 40), `n=${n}: subsampled too aggressively`);
  }
});
```

At the end of `computeField`, alongside the mirror memo:

```javascript
  // Same xy area as the map, at 21 assumed depths. Subsampled: the full grid
  // across 21 depths would be ~700k cells and stall the frame.
  // 28 rather than 40: the worst call in the dataset (61 picks) measured 84 ms
  // of z-profile at 40, pushing call selection to ~202 ms. The finding here is
  // flatness, which does not need fine spatial resolution, and (28/40)^2 roughly
  // halves the cost.
  const sub = subsampleGrid(xs, ys, 28);
  currentField.zprof = zProfile({
    zs: Array.from({ length: 21 }, (_, i) => -200 + i * 10),
    xs: sub.xs, ys: sub.ys,
    px: p.px, py: p.py, pz: p.pz, obsT: p.obsT, c: state.c,
  });
```

- [ ] **Step 5: Run tests, rebuild, verify**

```bash
cd viz && npm test && cd ..
uv run python -m viz.build_bundle --out viz/out/index.html
```

Expected: 23 JS tests pass. The panel should show an almost horizontal line with a ratio very close to 1.000 for most calls — that flatness *is* the finding, and the caption states it numerically.

- [ ] **Step 6: Commit**

```bash
git add viz/template/lib/zprofile.mjs viz/template/app.js viz/tests/test_zprofile.mjs
git commit -m "Add misfit-versus-depth panel showing the flat depth direction"
```

---

### Task 13: 3D basin inset

**Files:**
- Modify: `viz/template/app.js`
- Create: `viz/template/lib/isosurface.mjs`
- Create: `viz/tests/test_isosurface.mjs`

**Interfaces:**
- Consumes: `misfitField` (Task 3).
- Produces:
  - `misfitVolume({xs, ys, zs, px, py, pz, obsT, c, objective}) -> {vol: Float32Array, nx, ny, nz, best}`
  - `isoPoints({vol, nx, ny, nz, xs, ys, zs, level}) -> Float32Array` of xyz triples
  - `project3d({points, theta, phi, centre, half, width, height, zScale}) -> {xy: Float32Array, depth: Float32Array}`
  - `boxEdges() -> Array<[ax,ay,az,bx,by,bz]>` — the 12 edges of the unit cube

Two deliberate choices. **A point cloud, not marching cubes:** far less code, cannot produce a misleading watertight surface, and reads correctly for showing that the basin is a vertically extended tube. **A hand-rolled orthographic projection, not three.js:** the inset draws a few hundred points and a wireframe box, so a library is not warranted, and it keeps the page's no-external-resources property absolute. `three.min.js` is also the deprecated UMD build — it logs a removal warning in every viewer's console — and library rendering cannot be unit-tested, whereas `project3d` can.

- [ ] **Step 1: Write the failing tests**

Create `viz/tests/test_isosurface.mjs`:

```javascript
import test from "node:test";
import assert from "node:assert/strict";

import { misfitVolume, isoPoints, project3d, boxEdges } from "../template/lib/isosurface.mjs";

function synthetic() {
  const n = 24;
  const px = Array.from({ length: n }, (_, i) => (i * 40000) / (n - 1));
  const py = px.map((x) => 300 * Math.sin(x / 8000));
  const pz = px.map(() => -600);
  const obsT = px.map((x, i) => Math.hypot(x - 20000, py[i] - 5000, -600 + 30) / 1480);
  return { px, py, pz, obsT };
}

test("misfitVolume has the requested shape and a finite minimum", () => {
  const { px, py, pz, obsT } = synthetic();
  const out = misfitVolume({
    xs: [18000, 20000, 22000], ys: [3000, 5000, 7000], zs: [-100, -50, 0],
    px, py, pz, obsT, c: 1480,
  });
  assert.equal(out.nx, 3);
  assert.equal(out.ny, 3);
  assert.equal(out.nz, 3);
  assert.equal(out.vol.length, 27);
  assert.ok(Number.isFinite(out.best) && out.best >= 0);
});

test("isoPoints keeps only cells at or below the level", () => {
  const vol = Float32Array.from([1, 5, 5, 5, 5, 5, 5, 5]);
  const pts = isoPoints({
    vol, nx: 2, ny: 2, nz: 2,
    xs: [0, 1], ys: [0, 1], zs: [0, 1], level: 2,
  });
  assert.equal(pts.length, 3, "exactly one point, as xyz");
  assert.deepEqual([...pts], [0, 0, 0]);
});

test("isoPoints returns an empty array when nothing qualifies", () => {
  const vol = Float32Array.from([9, 9, 9, 9]);
  const pts = isoPoints({ vol, nx: 2, ny: 2, nz: 1, xs: [0, 1], ys: [0, 1], zs: [0], level: 1 });
  assert.equal(pts.length, 0);
});

test("project3d centres the volume and stays inside the viewport", () => {
  const points = Float32Array.from([0, 0, 0, 100, 100, 100, -100, -100, -100]);
  const { xy, depth } = project3d({
    points, theta: 0.7, phi: 1.0,
    centre: [0, 0, 0], half: 100, width: 200, height: 200, zScale: 1,
  });
  assert.equal(xy.length, 6);
  assert.equal(depth.length, 3);
  for (let k = 0; k < xy.length; k++) {
    assert.ok(xy[k] >= 0 && xy[k] <= 200, `projected coord ${xy[k]} outside viewport`);
  }
  // The centre point must land at the middle of the viewport.
  assert.ok(Math.abs(xy[0] - 100) < 1e-6 && Math.abs(xy[1] - 100) < 1e-6);
});

test("project3d rotation actually changes the projection", () => {
  const points = Float32Array.from([50, 0, 0]);
  const base = { points, phi: 1.0, centre: [0, 0, 0], half: 100, width: 200, height: 200, zScale: 1 };
  const a = project3d({ ...base, theta: 0 });
  const b = project3d({ ...base, theta: Math.PI / 2 });
  assert.ok(Math.abs(a.xy[0] - b.xy[0]) > 1, "theta must rotate the view");
});

test("project3d exaggerates depth by zScale so elongation is visible", () => {
  const points = Float32Array.from([0, 0, 50]);
  const base = { points, theta: 0, phi: 0.6, centre: [0, 0, 0], half: 100, width: 200, height: 200 };
  const flat = project3d({ ...base, zScale: 1 });
  const tall = project3d({ ...base, zScale: 3 });
  assert.ok(Math.abs(tall.xy[1] - 100) > Math.abs(flat.xy[1] - 100));
});

test("boxEdges returns the 12 edges of a unit cube", () => {
  const e = boxEdges();
  assert.equal(e.length, 12);
  for (const seg of e) {
    assert.equal(seg.length, 6);
    for (const v of seg) assert.ok(v === -1 || v === 1);
  }
});
```


- [ ] **Step 2: Run to verify failure**

```bash
cd viz && npm test
```

Expected: FAIL — cannot find `../template/lib/isosurface.mjs`.

- [ ] **Step 3: Implement**

Create `viz/template/lib/isosurface.mjs`:

```javascript
import { misfitField } from "./forward.mjs";

export function misfitVolume({ xs, ys, zs, px, py, pz, obsT, c, objective = "l1" }) {
  const nx = xs.length, ny = ys.length, nz = zs.length;
  const vol = new Float32Array(nx * ny * nz);
  let best = Infinity;
  for (let k = 0; k < nz; k++) {
    const { field } = misfitField({ xs, ys, sz: zs[k], px, py, pz, obsT, c, objective });
    for (let m = 0; m < field.length; m++) {
      const v = field[m];
      vol[k * nx * ny + m] = v;
      if (v < best) best = v;
    }
  }
  return { vol, nx, ny, nz, best };
}

export function isoPoints({ vol, nx, ny, nz, xs, ys, zs, level }) {
  const out = [];
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        if (vol[k * nx * ny + j * nx + i] <= level) out.push(xs[i], ys[j], zs[k]);
      }
    }
  }
  return Float32Array.from(out);
}

// Orthographic projection. No dependency, and unit-testable, unlike a library
// renderer. `zScale` exaggerates the vertical axis so the basin's elongation in
// depth stays visible even though its true extent is far smaller than its
// horizontal extent -- the axis label must therefore state the exaggeration.
export function project3d({ points, theta, phi, centre, half, width, height, zScale = 1 }) {
  const n = points.length / 3;
  const xy = new Float32Array(n * 2);
  const depth = new Float32Array(n);

  const ct = Math.cos(theta), st = Math.sin(theta);
  const cp = Math.cos(phi), sp = Math.sin(phi);
  // Fit the unit cube (plus the zScale'd vertical) inside the viewport.
  const fit = 0.5 * Math.min(width, height) / Math.max(1, Math.SQRT2 * Math.max(1, zScale));

  for (let k = 0; k < n; k++) {
    const ux = (points[k * 3] - centre[0]) / half;
    const uy = (points[k * 3 + 1] - centre[1]) / half;
    const uz = ((points[k * 3 + 2] - centre[2]) / half) * zScale;

    // Rotate about the vertical (z) axis, then tilt.
    const rx = ux * ct - uy * st;
    const ry = ux * st + uy * ct;

    xy[k * 2] = width / 2 + rx * fit;
    xy[k * 2 + 1] = height / 2 - (uz * sp + ry * cp) * fit;
    depth[k] = ry * sp - uz * cp;
  }
  return { xy, depth };
}

export function boxEdges() {
  const c = [];
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) c.push([sx, sy, sz]);
  const out = [];
  for (let a = 0; a < c.length; a++) {
    for (let b = a + 1; b < c.length; b++) {
      // Cube edges connect corners differing in exactly one coordinate.
      let diff = 0;
      for (let d = 0; d < 3; d++) if (c[a][d] !== c[b][d]) diff++;
      if (diff === 1) out.push([...c[a], ...c[b]]);
    }
  }
  return out;
}
```

- [ ] **Step 4: Wire the inset in `app.js`**

Append to `viz/template/app.js`:

```javascript
const insetView = { theta: 0.7, phi: 0.9 };

(() => {
  const canvas = $("inset3d");
  let dragging = false, lastX = 0, lastY = 0;
  canvas.addEventListener("pointerdown", (e) => {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    canvas.style.cursor = "grabbing";
    canvas.setPointerCapture(e.pointerId);
  });
  const endDrag = (e) => {
    dragging = false;
    canvas.style.cursor = "grab";
    try { canvas.releasePointerCapture(e.pointerId); } catch { /* already released */ }
  };
  canvas.addEventListener("pointerup", endDrag);
  // Without this, losing capture without a pointerup (a tab switch mid-drag, a
  // touch interruption) leaves `dragging` true, so ordinary hover then keeps
  // rotating the view with no button held.
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("lostpointercapture", () => { dragging = false; canvas.style.cursor = "grab"; });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    insetView.theta += (e.clientX - lastX) * 0.012;
    insetView.phi = Math.min(1.5, Math.max(0.08, insetView.phi - (e.clientY - lastY) * 0.012));
    lastX = e.clientX; lastY = e.clientY;
    drawInset();
  });
})();

// The vertical axis is exaggerated: the depth range we search (~280 m) is two
// orders of magnitude smaller than the horizontal grid, so at true scale the
// basin would collapse to a flat sheet and the point being made — that it is a
// tube in z — would be invisible. The caption states the factor.
const INSET_Z_SCALE = 3;

let insetCache = null;

function computeInset() {
  insetCache = null;
  if (!currentField) return;
  const p = picksFor(state.call, state.cablesOn);
  // Reuse the field's own xy grid, heavily subsampled. `gridForCall` no longer
  // exists, and a volume over a freshly-built grid would not correspond to what
  // the map is showing.
  const { xs, ys } = subsampleGrid(currentField.xs, currentField.ys, 24);
  const zs = Array.from({ length: 15 }, (_, i) => -280 + i * 20);
  const vol = misfitVolume({ xs, ys, zs, px: p.px, py: p.py, pz: p.pz, obsT: p.obsT, c: state.c });
  const points = isoPoints({ ...vol, xs, ys, zs, level: vol.best * 1.10 });
  insetCache = {
    points,
    centre: [(xs[0] + xs[xs.length - 1]) / 2, (ys[0] + ys[ys.length - 1]) / 2, (zs[0] + zs[zs.length - 1]) / 2],
    half: Math.max(xs[xs.length - 1] - xs[0], ys[ys.length - 1] - ys[0]) / 2 || 1,
  };
}

function drawInset() {
  const canvas = $("inset3d");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const h = 220;
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, h);
  $("inset3d-caption").textContent = "";
  if (!insetCache) return;

  const colors = themeColors();
  const common = {
    theta: insetView.theta, phi: insetView.phi,
    centre: insetCache.centre, half: insetCache.half,
    width: rect.width, height: h, zScale: INSET_Z_SCALE,
  };

  // Wireframe box first, in unit-cube coordinates.
  const edges = boxEdges();
  const boxPts = new Float32Array(edges.length * 6);
  edges.forEach((seg, k) => boxPts.set(seg, k * 6));
  const scaledBox = new Float32Array(boxPts.length);
  for (let k = 0; k < boxPts.length; k += 3) {
    scaledBox[k] = insetCache.centre[0] + boxPts[k] * insetCache.half;
    scaledBox[k + 1] = insetCache.centre[1] + boxPts[k + 1] * insetCache.half;
    scaledBox[k + 2] = insetCache.centre[2] + boxPts[k + 2] * insetCache.half;
  }
  const bp = project3d({ ...common, points: scaledBox });
  ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--line");
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let k = 0; k < bp.xy.length; k += 4) {
    ctx.moveTo(bp.xy[k], bp.xy[k + 1]);
    ctx.lineTo(bp.xy[k + 2], bp.xy[k + 3]);
  }
  ctx.stroke();

  if (insetCache.points.length === 0) {
    $("inset3d-caption").textContent = "No cells within 10% of the best fit.";
    return;
  }

  // Points, painted back to front so nearer points sit on top.
  const { xy, depth } = project3d({ ...common, points: insetCache.points });
  const order = Array.from({ length: depth.length }, (_, k) => k).sort((a, b) => depth[b] - depth[a]);
  // Hoisted: computing these inside the loop made the whole normalisation
  // O(n^2) with three full array scans per point, and this runs on every
  // pointermove during an orbit drag. A flat misfit surface can fill the whole
  // 24x24x15 volume (~8,640 points), which would be ~2e8 redundant operations
  // per frame -- defeating the reason orbiting is supposed to be cheap.
  let dmin = Infinity, dmax = -Infinity;
  for (let k = 0; k < depth.length; k++) {
    if (depth[k] < dmin) dmin = depth[k];
    if (depth[k] > dmax) dmax = depth[k];
  }
  const dspan = dmax - dmin || 1;
  for (const k of order) {
    const t = 0.35 + 0.65 * (1 - (depth[k] - dmin) / dspan);
    ctx.globalAlpha = t;
    ctx.fillStyle = colors.accent;
    ctx.beginPath();
    ctx.arc(xy[k * 2], xy[k * 2 + 1], 2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  const nPts = insetCache.points.length / 3;
  $("inset3d-caption").textContent =
    `${nPts} cells within 10% of best · depth axis exaggerated ${INSET_Z_SCALE}× · drag to orbit`;
}

layers.push(() => drawInset());
refresh();
```

**`computeInset` must not run per frame.** It is a 15-depth volume solve; running it inside the layer would stall every pan frame, which is the mistake Task 11 shipped and paid 447 ms for. Call it once at the end of `computeField`, right after the z-profile memo:

```javascript
  computeInset();
```

The layer then only draws. `drawInset` is additionally called directly by the orbit handler, which is correct — orbiting re-projects the cached points without re-solving anything.

- [ ] **Step 5: Run tests, rebuild, verify**

```bash
cd viz && npm test && cd ..
uv run pytest viz/tests paper/tests -q
uv run python -m viz.build_bundle --out viz/out/index.html
```

Expected: all JS tests and all Python tests pass. In the browser, the inset shows a point cloud visibly elongated along the vertical axis for most calls — the depth tube — with a caption stating the cell count and the exaggeration factor. Dragging orbits it. Confirm no `https://` URL appears anywhere in the built page:

```bash
grep -c 'https://' viz/out/index.html
```

Expected: `0`. The page is now fully self-contained with no external origins at all.

- [ ] **Step 6: Also apply the depth-flatness caption, which was never applied**

`#zprofile-context` was added to the plan during Task 12 but after that task's brief had already
been generated, so it never reached `app.html`. It carries the measured population figure, which is
this tool's headline negative finding and belongs on screen. Add it to `app.html` immediately after
the `#zprofile` canvas — the markup is in Task 12's section of the plan.

- [ ] **Step 7: Commit**

```bash
git add viz/template/app.html viz/template/app.js viz/template/lib/isosurface.mjs viz/tests/test_isosurface.mjs
git commit -m "Add orbitable 3D inset showing the misfit basin as a depth tube"
```

---

## Verification before declaring done

- [ ] `uv run pytest paper/tests viz/tests -q` — all Python tests pass
- [ ] `cd viz && npm test` — all JS tests pass, parity included
- [ ] `uv run python -m viz.build_bundle --out viz/out/index.html` prints the bathymetry check (median offset ≈ −0.6 m, r ≈ 0.9999), `11117 rows, 628 calls`, and a page under 2 MB
- [ ] Open the page: relief, both cables, solutions, and a selected call's field all render
- [ ] Untick one cable on a two-cable call: a second basin appears and the mirror badge reports a penalty; re-tick: it lifts
- [ ] Drag the sound-speed slider end to end: the basins move continuously with no stalls
- [ ] The depth panel reads a ratio near 1.000 for a well-constrained call
- [ ] Toggle OS dark mode: the page remains legible, with no unstyled white panels
- [ ] Page is reported as self-contained by `test_render_html_is_self_contained`

## Out of scope for this plan

Phase 2 from the spec — the `db` contribution inbox, ground-truth comparison, the calibration diagram and the `z`-error panel — gets its own plan once we know whether ground truth is coming. Also excluded: raw `.nc` waterfall rendering, detection/picking, track formation, and the aggregate all-calls worker view (add it only if the per-call view proves too slow to explore with).
