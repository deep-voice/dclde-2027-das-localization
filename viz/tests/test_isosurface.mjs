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

test("project3d normalises z by its own half-span so a shallow volume is not flattened", () => {
  // Mirrors app.js's computeInset/drawInset: a horizontal half-extent of
  // several km against a vertical half-span of ~100 m (the source-depth
  // search range), the exact disparity that produced a 3.5%-of-wireframe
  // sliver before z got its own zHalf normalisation.
  const half = 5000, zHalf = 100, zScale = 3;
  const centre = [0, 0, 0];
  const common = { theta: 0.7, phi: 0.6, centre, half, zHalf, width: 200, height: 200, zScale };

  // Wireframe box corners, built the same way drawInset builds them: `half`
  // horizontally, `zHalf` vertically.
  const edges = boxEdges();
  const boxPts = new Float32Array(edges.length * 6);
  edges.forEach((seg, k) => boxPts.set(seg, k * 6));
  const scaledBox = new Float32Array(boxPts.length);
  for (let k = 0; k < boxPts.length; k += 3) {
    scaledBox[k] = boxPts[k] * half;
    scaledBox[k + 1] = boxPts[k + 1] * half;
    scaledBox[k + 2] = boxPts[k + 2] * zHalf;
  }
  const box = project3d({ ...common, points: scaledBox });
  let boxYmin = Infinity, boxYmax = -Infinity;
  for (let k = 1; k < box.xy.length; k += 2) {
    boxYmin = Math.min(boxYmin, box.xy[k]);
    boxYmax = Math.max(boxYmax, box.xy[k]);
  }
  const boxSpan = boxYmax - boxYmin;

  // A point cloud spanning the full vertical range (a flat misfit surface
  // routinely fills the whole searched depth -- see zprofile.mjs) but with
  // only modest horizontal spread near the centre.
  const zs = [-100, -50, 0, 50, 100];
  const pts = [];
  for (const z of zs) pts.push(300 * Math.sin(z), 300 * Math.cos(z), z);
  const points = Float32Array.from(pts);
  const proj = project3d({ ...common, points });
  let ptYmin = Infinity, ptYmax = -Infinity;
  for (let k = 1; k < proj.xy.length; k += 2) {
    ptYmin = Math.min(ptYmin, proj.xy[k]);
    ptYmax = Math.max(ptYmax, proj.xy[k]);
  }
  const ptSpan = ptYmax - ptYmin;

  assert.ok(
    ptSpan / boxSpan >= 0.25,
    `points occupied only ${(100 * ptSpan / boxSpan).toFixed(1)}% of the wireframe's projected height`,
  );
});

test("project3d keeps every cube corner inside the viewport at zScale: 3", () => {
  // The old containment test only ran at zScale: 1. The tightened
  // sqrt(zScale*zScale + 2) bound is derived generally, but was never
  // exercised at the value the app actually uses for the inset
  // (INSET_Z_SCALE = 3), so a regression there would not have been caught.
  const corners = [];
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) corners.push(sx * 100, sy * 100, sz * 100);
  const points = Float32Array.from(corners);
  for (const theta of [0, 0.3, 0.7, 1.2, Math.PI / 2, 2.4]) {
    for (const phi of [0.08, 0.4, 0.9, 1.5]) {
      const { xy } = project3d({
        points, theta, phi,
        centre: [0, 0, 0], half: 100, width: 200, height: 200, zScale: 3,
      });
      for (let k = 0; k < xy.length; k++) {
        assert.ok(
          xy[k] >= -1e-6 && xy[k] <= 200 + 1e-6,
          `theta=${theta} phi=${phi}: projected coord ${xy[k]} outside viewport at zScale 3`,
        );
      }
    }
  }
});

test("boxEdges returns the 12 edges of a unit cube", () => {
  const e = boxEdges();
  assert.equal(e.length, 12);
  for (const seg of e) {
    assert.equal(seg.length, 6);
    for (const v of seg) assert.ok(v === -1 || v === 1);
  }
});
