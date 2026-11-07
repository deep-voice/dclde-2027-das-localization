# DAS fin-whale localization visualizer — design

_2026-09-10. Design for a shareable, interactive visualizer for DCLDE 2027 Challenge L
(fin-whale localization from two DAS fibers off Oregon)._

## Purpose

Two goals, in priority order:

1. **Let the team see what the data and our results actually mean** — where calls localize, how well
   the fit is constrained, and what "ambiguity" concretely looks like.
2. **Be useful infrastructure for the wider community** even if our own estimators never reach
   state of the art. The DAS whale-localization literature currently publishes point estimates with
   Gaussian error ellipses; none of it shows the objective function, which is where the interesting
   failure modes live.

Success criteria:

- A teammate with no DAS background can open one link and understand, without explanation, why a
  single cable cannot always tell which side a whale is on.
- Every quantity on screen is computed from real data, not illustrated.
- A new set of results — ours, the reference implementation's, or a third party's — can be displayed
  by dropping in one small file that conforms to a documented contract.
- It looks good enough that we would put it in a talk.

## Key decisions

| Decision | Rationale |
|---|---|
| **Misfit field is the primary object**, error ellipses only an optional overlay | An ellipse is a single-mode Gaussian summary and *structurally cannot* represent a bimodal solution. The reference implementation's published median uncertainty is 57 m, which says nothing about which side of the cable the whale was on. The field shows both basins and their relative depth. |
| **Fields are computed in the browser, not shipped** | Precomputing fields for ~628 calls is ~33 M floats — impossible in a 16 MB page. The inputs (picks, decimated geometry, bathymetry) are under 1 MB, and a 200×260 grid against ~25 picks is ~1.3 M operations, a few ms in JS. |
| **Single self-contained HTML file, all data inlined as base64** | A published Artifact cannot fetch its own data files — the CSP blocks fetch/XHR to every host. Inlining is required, and it also makes one file that runs as an Artifact, on GitHub Pages, or from disk. |
| **2D plan view + one narrow 3D inset** | A misfit iso-surface is hard to read quantitatively and occlusion fights the viewer. But the depth story *is* inherently 3D: the point is that the basin is a vertically extended tube. So 3D is used for exactly that one job. |
| **Standalone `viz/` module, exempt from the repo's seaborn convention** | `CLAUDE.md` mandates seaborn for all plots. An interactive web page is a deliberate exception. Static figures in `docs/` still use seaborn. |
| **Our own forward model, not DAS4Whales** | DAS4Whales is CC BY-NC-SA 4.0 — viral and non-commercial. The travel-time model is ten lines of arithmetic; reimplementing avoids a licence entanglement in a tool we want others to reuse. |

## Architecture

```
viz/
  build_bundle.py       # reads S3/GCS annotations + geometry + GMRT -> emits one HTML file
  template/app.html     # page source: markup, CSS, JS (data injected at build)
  fixtures/             # committed golden fixtures for the parity + transform tests
  tests/
    test_bundle.py      # bundle writer, transform fit, interpolation error
    test_parity.mjs     # JS misfit == Python misfit
  out/                  # generated HTML (gitignored)

src/dclde/das_finwhale/
  geometry.py           # load cables, chan_m -> (x,y,depth), frame transforms
  forward.py            # travel-time model, misfit objectives  (single source of truth)
```

The physics lives in the package because our inversion pipeline needs it too and it must be tested.
The visualizer consumes it. Task packages must not cross-import (`CLAUDE.md`); nothing here does.

### Data flow

```
S3/GCS annotations (120 CSV)  ─┐
GCS geometry (2 CSV)          ─┼─> build_bundle.py ─> app.html (self-contained, ~1.3 MB)
GMRT bathymetry (GridServer)  ─┤        │
results/*.json (repo)         ─┘        └─ injects base64 typed arrays + JSON
                                                        │
                                        browser: decode -> compute misfit on demand -> render
                                                        │
                                        db capability (phase 2) -> provisional contributed sets
```

## Component 1 — physics (`src/dclde/das_finwhale/`)

`geometry.py`:

- `load_cable(cable)` → table of `chan_idx, chan_m, lat, lon, depth, x, y`.
- `chan_to_xy(cable, dist)` → interpolated `(x, y, depth)` for a pick's along-cable distance.
- `FrameTransform` — conversions between `local_xy`, `utm10n` and `wgs84`. The local frame turns out
  to be **UTM zone 10N with the x axis flipped**, exactly:

  ```
  utm_x = 435226.0809 − x       utm_y = 4985111.5099 + y
  ```

  Both constants have *zero* variance across all 77,958 channels of both cables, so this is an
  identity rather than an approximation. The constants are still **derived from the geometry table
  at load time and asserted to be constant**, never hardcoded blindly — that keeps the check alive if
  the dataset is ever revised. UTM10N ↔ WGS84 goes through `pyproj` (EPSG:32610 ↔ EPSG:4326), so
  `local ↔ wgs84` composes exactly.

  An affine fit of `lat/lon → x/y` was measured and rejected: 42 m RMS and 124 m maximum error in
  `y`, which is the same order as the differences between methods we want to visualize.

`forward.py`:

- `travel_time(source_xyz, pick_xyz, c)` → straight-ray 3-D Euclidean, `t = ‖S − C‖ / c`.
- `misfit(grid, picks, c, objective)` → scalar field. Default objective **L1 with `t0` set to the
  median residual** (matches the team's existing notebook and the spirit of Horne et al.'s
  trimmed-L1); L2 selectable. `t0` is always eliminated analytically, never gridded.
- `mirror_penalty(picks, cable, c)` → best misfit on each side of the reference cable and their
  ratio.

Sound speed `c` is an explicit argument everywhere with no default baked into a call site. Given
the feasibility finding that every published method silently fixes `c` (1440/1480/1500 m/s) and
that nobody fits it, making it an exposed parameter is the point.

## Component 2 — bundle builder (`viz/build_bundle.py`)

Emits one HTML file. Contents, all injected into the template:

| Payload | Encoding | Raw size |
|---|---|---|
| `meta` — bundle version, frame offsets, default `c`, contour ratios, git SHA, source URIs, build date | JSON | <1 KB |
| `cables` — decimated channels (~8 m): `chanM, x, y, depth` per cable | Float32 base64 | ~312 KB |
| `bathy` — 663×294 elevation grid reprojected into local XY | Int16 base64 + header | ~390 KB |
| `picks` — all 11,117 picks: call key, cable, `dist`, `time`, `snr`, `call_type`, `batch` | Float32/Int16 base64 | ~267 KB |
| `results` — each canonical result set | JSON | ~30 KB each |
| **Total** | after base64 (+33%) | **~1.3 MB** |

**Decimation keeps the cable's final channel.** A plain `[::4]` strands the last 1–3 channels beyond
the final kept sample, where interpolation clamps instead of interpolating: 1.8 m of position error
on North and 6.3 m on South. Including the final channel brings both to ~0.6 m, and a test
parametrised over both cables asserts the sub-metre bound.

Because that final interval is therefore shorter than the rest, **`chanM` is shipped explicitly**
rather than reconstructed from start + step — index arithmetic would be wrong at the tail. It costs
about 104 KB of base64 against a ~1.3 MB page. Full-resolution channel spacing genuinely is exactly
uniform (`2.041904688` m, max deviation 7e-12 m across all 77,958 channels) and a test asserts that
against the raw CSV column, since it is what validates the dataset; it is only the *decimated*
samples that are non-uniform.

`meta` records the git SHA and source URIs so any published page states exactly what produced it.

### Bathymetry

Source: GMRT GridServer, e.g.
`https://www.gmrt.org/services/GridServer?minlongitude=-125.35&maxlongitude=-123.90&minlatitude=44.95&maxlatitude=45.40&format=netcdf&resolution=default&layer=topo`
— returns a 663×294 legacy GMT grid at ~245 m spacing, 780 KB, covering both cables.

Validated: GMRT elevation at every south-cable channel agrees with the cable's own reported depth to
a median of −0.6 m with r = 0.99995 (n = 46,440). This cross-validates both the bathymetry and the
cable geolocation, and the check is retained as a test.

The grid is reprojected into local XY at build time so the raster, cables, misfit grid and solutions
share one metric frame — no runtime projection, true on-screen distances.

**Action item:** confirm GMRT's attribution terms and put the required citation in the page footer.

## Component 3 — results contract

One file per result set. `frame` is mandatory and never inferred.

```json
{
  "name": "goestchel-2026-batch1",
  "kind": "estimate",
  "frame": "local_xy",
  "produced_by": "Goestchel, Wilcock & Abadi 2026 (released code)",
  "c_assumed": 1480,
  "notes": "free text",
  "solutions": [
    { "call": "2021-11-04_02:00:02/1",
      "x": 37000.5, "y": 22344.4, "z": -30.0, "z_fixed": true,
      "t0": 6.515, "wrms": 0.126, "n_picks": 41,
      "cables_used": ["North", "South"],
      "unc": { "type": "ellipse", "semi_major": 8.8, "semi_minor": 4.1, "theta_deg": 31 } }
  ]
}
```

- `kind` ∈ `estimate` | `ground_truth` | `synthetic_truth`. Same schema for all three, so the
  ground-truth and simulator paths cost nothing extra.
- `call` is the `(file-window, call_id)` key. Verified to be a single global catalog shared across
  both cables: 628 keys over 60 windows, 166 on both cables, each cable's own ID set gappy in 44/60
  windows while the union is contiguous in 60/60. Everything joins on this key.
- `frame` ∈ `local_xy` | `utm10n` | `wgs84`. Mandatory because the reference implementation shipped
  lat/lon roughly 270 km west of the cable (its local coordinates were fine; its
  local→UTM→geographic conversion was not). See the frame guard below.
- `z_fixed` makes "this method assumed a depth rather than estimating one" visible in the UI. It is
  `true` for every currently published method, including the reference implementation's two-cable
  routine, whose released output has `z = −30.0` for all 176 solutions.
- `unc` is optional and typed.

### Frame guard

On load, for each result set: convert its solutions to local XY according to its declared `frame`,
then compute the median distance from those solutions to the nearest cable channel. Badge the set as
suspect above **25 km**. Reference values: the baseline's correct local coordinates give a median of
1.4 km; its broken lat/lon would give ~270 km. A median is used so a single runaway solution does
not trip the guard.

## Component 4 — the page

### Layout

- **Map** (primary): hillshaded bathymetry as a recessive base; both cables as thin high-contrast
  lines; solutions from selected result sets as small precise marks; the selected call's misfit
  field as a raster with contours.
- **Moveout panel**: the selected call's picks in (distance, time) space with the predicted curve
  for the current solution. Non-negotiable — it is the only view that shows whether a fit
  corresponds to real data.
- **Misfit-vs-`z` panel**: a 1-D slice showing the near-flat depth direction.
- **3D inset**: the set of cells within 1.10× of the best misfit over a `z` range, drawn as an
  orbitable point cloud. Its single job is to show the basin is a vertically extended tube.
  Implemented as a hand-rolled orthographic projection on a canvas — no third-party library. The
  inset draws a few hundred points and a wireframe box, `three.min.js` is the deprecated UMD build
  (it logs a removal warning in every viewer's console), and a library renderer cannot be
  unit-tested whereas a projection function can. The page therefore references no external origin
  at all. The vertical axis is deliberately exaggerated, with the factor stated in the caption,
  because the searched depth range is two orders of magnitude smaller than the horizontal grid and
  would otherwise collapse to a flat sheet.
- **Control rail**: result-set selector (multi), batch/cable filters, call search.
- **Readouts**: `n_picks`, `wrms`, standoff from nearest cable, mirror-side penalty ratio.
- **Sliders**: `c` (1440–1520 m/s), assumed source depth, cable subset, grid resolution.

### The centerpiece interaction

Select **north only** → the field shows two basins, and the badge reports the mirror-side penalty.
Select **both** → the mirror basin lifts.

Presented as a *ratio*, not a binary "ambiguous → resolved", because measurement shows curvature
already partly breaks the symmetry: on a batch-1 sample (n = 18 calls with ≥8 picks, single-cable
solves, search covering both sides) the wrong-side solution cost a median 27–29% higher misfit, with
only 3–4 of 18 calls having a mirror within 10% of the best. When it does flip, the error is
kilometres — two candidates 7 km apart on south-cable call 1. The ratio is the honest summary and it
lets a user hunt the thin cases.

*(Those numbers are indicative, from a quick reproduction on one sample, and are grid-resolution
sensitive at the ±1-count level. They motivate the design; they are not results.)*

### Grid and contours

- Default grid: bounding box of the call's pick coordinates in local XY, expanded in each direction
  by `max(10 km, 0.5 × along-cable pick span)`, clipped to the bathymetry extent, always covering
  both sides of the reference cable. Resolution is set by a long-axis cell count with aspect ratio
  preserved: **260 by default**, selectable 130 / 260 / 520.
- Contours at **1.05×, 1.10×, 1.25×, 1.50× the best misfit**. Ratio levels are interpretable
  ("solutions here fit 10% worse"); absolute summed-L1 seconds are not.

### Mirror-side convention

The mirror comparison is defined **per reference cable**. When only one cable is selected, that is
the reference and one ratio is shown. When both are selected there is no single mirror, so the
combined badge is suppressed and the two single-cable ratios are shown instead. This is stated
explicitly because it is the obvious place to hand-wave.

### Visual design

Palette selection happens at implementation time with the `dataviz` skill loaded. Principles:
muted desaturated relief that recedes; one sequential ramp for the field with the low end
emphasized; thin crisp contours; small precise solution marks rather than blobs; correct rendering
in both light and dark themes.

### Performance

- Per-call field (200×260 × ~25 picks ≈ 1.3 M ops): a few ms. Slider drags recompute live.
- Aggregate all-calls view would be ~800 M ops, so it runs coarse (50×65) in a Web Worker with
  progress (~1 s), falling back to points-only if workers are unavailable.

## Error handling

| Condition | Behaviour |
|---|---|
| `n_picks < 10` | Field still computed and shown, badged **under-constrained**. Not hidden — the 8-pick call that ran away 38 km from the cable at a *better* misfit than well-constrained calls is exactly what a user should see. |
| Result set fails the frame guard | Loads, badged **suspect frame**, with the measured median offset shown. |
| Unknown `frame` value | Set is rejected with a named reason; the rest of the page works. |
| Bathymetry no-data (1,989 NaN cells) | Rendered as no-data; excluded from the colour ramp's domain so it cannot skew the scale. |
| `claude.use("db")` returns `null` | Page is fully functional read-only; contribution UI hidden. |
| No WebGL | 3D inset hidden; the 2D map is canvas and unaffected. |
| Bundle/app version mismatch | Refuses to render, states the mismatch. |

## Testing

| Test | Guards against |
|---|---|
| `test_parity.mjs` — JS field vs Python field on a committed fixture (one call, fixed grid, fixed `c`), max relative difference over all cells < 1e-6 | The two implementations of the same equation silently drifting. This is the most important test in the module. |
| Channel spacing is exactly uniform on both cables (`chanM` is shipped explicitly, but `chan_to_xyz` still depends on `CHAN_SPACING_M`) | A non-uniform cable silently corrupting every pick position. |
| Transform round-trip: fitted local↔WGS84 applied to all 77,958 channel pairs, residual within tolerance | A wrong projection producing a plausible-looking but wrong map. |
| Cable interpolation error at every-4th-channel decimation < 1 m vs full resolution | Decimation quietly degrading pick positions. |
| GMRT vs cable-reported depth: median offset and correlation within tolerance (n = 46,440) | Bathymetry misalignment or a bad reprojection. |
| Frame guard fires on a deliberately mis-framed fixture and passes a correct one | The guard silently never firing. |
| Bundle writer golden test — byte-stable payload sizes and JSON shape | Accidental payload bloat past the page limit. |
| Unit tests on pure render helpers (colour mapping, hillshade, grid↔screen, contour extraction) | Rendering bugs, without resorting to screenshot tests. |

Per `CLAUDE.md`, every reported number states its denominator and `n`, and the page shows `n_picks`
for every solution rather than only aggregate statistics.

## Phase 2

### Contributed results

The `user` capability is unavailable to this account, so the page has no submitter identity and no
per-viewer private storage; everything in `db` is anonymous and writable by any viewer. Therefore
`db` is an **inbox**, not the record:

- Canonical result sets live in the repo and are inlined at build time. Anything we report comes
  from git, preserving the repo's "reproducible from a committed config" rule.
- `db` holds provisional contributions under `contrib/<slug>` with a random suffix, append-only, so
  one contributor cannot clobber another's set. Chunk at 200 solutions per document (conservative;
  verify the per-document limit against the `db` type definitions at implementation).
- Contributed content is untrusted: numbers coerced and range-checked, text escaped and never
  interpolated as HTML, nothing eval'd, and the frame guard applied on ingest.
- We review, promote into the repo, and rebuild.

`db` does not bar public sharing. `assets` would (a declaring page is organization-internal), which
is why contributions are point estimates only — the page computes the fields itself.

### Ground-truth comparison

A result set with `kind: "ground_truth"` enables:

- Error distribution and a residual-vector map.
- Error against the three candidate predictors of failure: standoff distance, pick count, and
  mirror-side penalty ratio. Whether the ambiguity margin predicts error is the most interesting of
  these — it would make the margin a usable confidence signal.
- **A calibration diagram.** A method claiming 95% ellipses should contain the truth 95% of the
  time. Nobody in this literature publishes that, uncertainty quantification came back as genuinely
  open, and Challenge L explicitly requires uncertainty estimates with every submission.
- **A `z`-error panel.** If ground-truth depth turns out to be a constant −30 m, this shows it
  immediately as a flat line — the fastest way to communicate that L-2's depth component may be
  unscoreable.

Scoring stays modular (median error, 90th percentile, fraction within a threshold, calibration
coverage) because the 2027 metric is unpublished.

## Non-goals for v1

- **Raw `.nc` waterfall rendering.** Files are 0.5–1 GB each; cannot be inlined and cannot be
  fetched. The moveout panel shows the picks, which is what the challenge scores.
- **Detection and picking.** The test set supplies picks.
- **Track formation** across calls. Valuable — Rørstadbotnen's state-space filter is the thing to
  reimplement eventually — but a separate concern.
- Pick editing, full volumetric 3D, contribution moderation beyond the frame guard, live S3 access
  from the page.

## Sequencing

Steps 1–5 are the scope of the first implementation plan. Phase 2 (step 6) gets its own plan once
we know whether ground truth is coming.

1. `geometry.py` + `forward.py` + the parity fixture and transform tests.
2. `build_bundle.py` producing a self-contained page, loaded with the reference implementation's 176
   batch-1 solutions and our own localizer's output on the batch-1 sample.
3. Map + misfit field + moveout panel + cable-subset toggles.
4. `c` and depth sliders + misfit-vs-`z` panel.
5. 3D inset.
6. Phase 2: `db` inbox, ground-truth and calibration panels.

## Open risks

- **Two implementations of the misfit.** Mitigated by the parity test; the reason `forward.py`
  should stay boringly explicit rather than clever.
- **Page size.** Adding many canonical result sets erodes the inline budget. Keep canonical sets
  small; push the long tail to `db`.
- **GMRT attribution terms** unverified — action item before publishing.
- **No ground truth exists yet**, so v1 can only compare methods against each other and against fit
  quality. This is a real limitation, not a temporary one: it may never be resolved for depth, which
  is what makes the synthetic-truth path in phase 2 load-bearing.
