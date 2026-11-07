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
