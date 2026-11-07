import { readFileSync } from "node:fs";
import test from "node:test";
import assert from "node:assert/strict";

import { misfitField, travelTime, bestT0, misfitValue } from "../template/lib/forward.mjs";

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
