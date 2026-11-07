import test from "node:test";
import assert from "node:assert/strict";

import { marchingSquares, ratioLevels, solutionGrid, fieldImage, zoomBoxAt, fitBox, coarseSolve, nearestMarker, subsampleGrid } from "../template/lib/contour.mjs";

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

test("fieldImage survives a zero or non-finite best misfit", () => {
  // Unreachable with real noisy picks, but a degenerate synthetic set can fit
  // exactly, and NaN would propagate into every pixel of that raster.
  //
  // fieldImage always returns a Uint8ClampedArray, so every element is
  // already, by construction, an integer in [0, 255] -- Number.isFinite over
  // it can never fail and was not testing anything. The load-bearing checks
  // are that the raster is not fully transparent (alpha clamped away from a
  // NaN-poisoned 0) and that the four cells are not all identical (NaN
  // propagating would collapse every pixel to the same clamped value).
  for (const best of [0, NaN, Infinity]) {
    const rgba = fieldImage(Float64Array.from([0, 1, 2, 3]), 4, 1, best, 8);
    assert.equal(rgba.length, 16);
    const alpha = [0, 1, 2, 3].map((i) => rgba[i * 4 + 3]);
    for (const a of alpha) assert.ok(a > 0, `alpha channel collapsed to 0 for best=${best}`);
    const pixels = [0, 1, 2, 3].map((i) => rgba.slice(i * 4, i * 4 + 4).join(","));
    assert.ok(new Set(pixels).size > 1, `all four pixels identical for best=${best}`);
  }
});
