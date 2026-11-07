# DAS fin-whale localization visualizer

> **There is no dev server — `npm run dev` does not exist.** This is a build step, not a web app.
> It compiles to a single static HTML file that you open directly. `npm` is used only to run the
> JavaScript tests.

## Quick start

From the **repo root** (not from `viz/`):

```bash
uv run python -m viz.build_bundle --out viz/out/index.html
```

Then open `viz/out/index.html` in a browser. That one file is the whole deliverable — about 1.4 MB,
with every byte of data inlined and **zero external requests**, so it works from disk, from a static
host like GitHub Pages, or published as an Artifact. If your browser restricts local files, serve the
directory: `python -m http.server 8749 --directory viz/out`.

The build reads only committed inputs, so it needs **no network access and no S3 credentials**.

## Commands

| What | Command |
|---|---|
| Build the page | `uv run python -m viz.build_bundle --out viz/out/index.html` |
| Build with extra result sets | `uv run python -m viz.build_bundle --results-dir viz/results` |
| Python tests | `uv run pytest viz/tests -q` (the shared core it uses: `uv run pytest tests/physics -q`) |
| JavaScript tests | `cd viz && npm test` |
| Regenerate the cross-language parity fixture | `uv run python viz/make_parity_fixture.py` |

## Dependencies

- **Python** via `uv`: from the repo root, `uv pip install -e . -r requirements.txt -r viz/requirements.txt`. The visualizer specifically needs
  `numpy`, `pandas`, `scipy`, `pyproj`, `xarray`, `netcdf4` and `requests`.
- **Node 18 or newer**, for the JavaScript tests only. There are **no npm dependencies** — the tests
  use Node's built-in runner, and the page itself loads no libraries at all.

## What the page shows

Every published method in this field reports a point estimate with a Gaussian error ellipse. That is
misleading here: a single cable often cannot tell which side of itself a whale is on, so the solution
space is frequently *bimodal* — two candidate positions kilometres apart, which an ellipse renders as
one blob centred on neither. This tool draws the actual objective function instead.

- **Map** — shaded-relief bathymetry, both cables, and one open circle per annotated call (608 of 628
  have enough picks). Click a marker or a row in the list to open that call. Scroll to zoom, drag to
  pan, double-click to reset.
- **Misfit field** — for the selected call, how badly each candidate position disagrees with the
  observed arrival times. Darkest fits best; contours mark 1.05, 1.10, 1.25 and 1.50× the best fit.
  The ringed crosshair is the refined solution.
- **Mirror badge** — with one cable active, how much worse the mirror-image candidate fits, *and how
  far away it is*. Both matter: 14% of single-cable solves have a near-equal mirror, but 27% of those
  are within 500 m, meaning the solution sits on the cable axis and the "two sides" are the same
  place. Only ~5–8% have the genuine kilometre-scale ambiguity.
- **Moveout** — the raw picks in (along-cable distance, time) space with the predicted curve through
  them. This is the only view that shows whether a fit corresponds to real data: a solution can have
  a tiny misfit and still be nonsense.
- **Misfit vs depth** — nearly flat, which is the point. Across 32 real calls the best misfit varies
  by a median 0.3% over the whole −200…0 m range, so source depth is effectively unobservable from
  two cables at similar depth. Every published method fixes it rather than estimating it.
- **3-D inset** — the same finding made visceral: the set of cells fitting within 10% of the best is
  a vertical *tube*, not a ball. Its depth axis is heavily exaggerated and the caption states by how
  much.

Sound-speed and assumed-depth sliders re-solve the selected call live, so you can see how much (or
how little) either changes the answer.

## Using it with our own results

The page displays any number of **result sets**. Drop a JSON file into `viz/results/` and rebuild —
the build picks it up automatically and reports it:

```
result set example-our-solver: 7 solutions, median 2.9 km from cable [ok]
result set goestchel-2026-batch1: 176 solutions, median 1.4 km from cable [ok]
```

The format:

```json
{
  "name": "our-solver-v1",
  "kind": "estimate",
  "frame": "local_xy",
  "produced_by": "who or what generated this",
  "c_assumed": 1480.0,
  "notes": "free text",
  "solutions": [
    { "call": "2021-11-04_02_00_02/1",
      "x": 37000.5, "y": 22344.4, "z": -30.0, "z_fixed": true,
      "wrms": 0.126, "n_picks": 41, "cables_used": ["North", "South"],
      "unc": { "type": "ellipse", "semi_major": 8.8, "semi_minor": 4.1, "theta_deg": 31 } }
  ]
}
```

Four things to get right:

1. **`frame` is mandatory and never inferred** — one of `local_xy`, `utm10n`, `wgs84`. This exists
   because the reference implementation published lat/lon that place calls ~270 km west of the cable
   while its local coordinates were fine. On load, every set is checked: if the median distance from
   its solutions to the nearest cable channel exceeds 25 km, it is flagged.
2. **`call` is the join key**, formatted `<window>/<call_id>` with the timestamp underscore-separated
   — `2021-11-04_02_00_02/1`. The dataset writes that timestamp three different ways (colons, dashes,
   underscores); always emit underscores. Without a valid key a solution still plots, but it cannot be
   linked to its picks, its moveout, or its list row. The bundled reference set has this problem: its
   source CSV carries no call ids, so its rows use synthetic `row0…row175` keys and are display-only.
3. **`kind`** is `estimate`, `ground_truth`, or `synthetic_truth`. The latter two are for when we
   have positions to score against — the page will then be able to show error distributions and an
   uncertainty-calibration diagram. Nothing consumes them yet.
4. **`z_fixed`** records whether depth was estimated or merely assumed. Set it honestly; the UI shows
   it, and for every published method so far it is `true`.

`viz/results.py` validates all of this and raises with a named reason, so a malformed file fails the
build rather than rendering something wrong.

## Where the physics lives

`src/dclde_localization/physics/forward.py` is the single source of truth for the travel-time model and the
misfit. `viz/template/lib/forward.mjs` mirrors it in JavaScript so the page can compute fields in the
browser, and `viz/tests/test_parity.mjs` holds the two together to within 1e-6 relative — they
currently agree bit-for-bit.

**If you change one, change both and run the JavaScript tests.** If you change the Python, also
regenerate the parity fixture. The parity test is sensitive enough to catch a 1-part-in-10⁷ change in
sound speed, so it will tell you.

## Further reading

- Design and rationale: `viz/docs/specs/2026-09-10-das-localization-visualizer-design.md`
- Every judgement call made while building it, with cost-if-wrong:
  `viz/docs/specs/2026-09-11-visualizer-execution-rulings.md`
- Why sound-speed structure matters for depth but not for horizontal position:
  `paper/docs/sound-speed-feasibility.md`
