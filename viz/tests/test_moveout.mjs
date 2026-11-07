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
