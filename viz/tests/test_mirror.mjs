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
