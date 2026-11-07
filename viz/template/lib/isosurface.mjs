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
export function project3d({ points, theta, phi, centre, half, zHalf = half, width, height, zScale = 1 }) {
  const n = points.length / 3;
  const xy = new Float32Array(n * 2);
  const depth = new Float32Array(n);

  const ct = Math.cos(theta), st = Math.sin(theta);
  const cp = Math.cos(phi), sp = Math.sin(phi);
  // Fit the unit cube (plus the zScale'd vertical) inside the viewport.
  //
  // Deviates from the brief's `Math.SQRT2 * Math.max(1, zScale)` denominator:
  // that under-corrects at zScale=1 and fails
  // "project3d centres the volume and stays inside the viewport" (a corner
  // point lands at y=-13.3, outside [0,200]) and, for the same reason,
  // "project3d exaggerates depth by zScale". A horizontal cube corner has
  // (rx, ry) of norm at most sqrt(2) regardless of theta (rotation preserves
  // norm); the vertical offset is uz*sin(phi) + ry*cos(phi) with |uz| <=
  // zScale and |ry| <= sqrt(2), two independent quantities, so by
  // Cauchy-Schwarz its max over phi is sqrt(zScale^2 + 2), not
  // sqrt(2)*max(1, zScale) -- the two only coincide at zScale=0. This is the
  // exact (tight, achievable) worst case, so it is the smallest denominator
  // that still guarantees every cube corner stays inside the viewport for
  // every theta/phi, including mid-orbit.
  const fit = 0.5 * Math.min(width, height) / Math.max(1, Math.sqrt(zScale * zScale + 2));

  for (let k = 0; k < n; k++) {
    const ux = (points[k * 3] - centre[0]) / half;
    const uy = (points[k * 3 + 1] - centre[1]) / half;
    // z is normalised by its OWN half-span, not the horizontal one: the
    // search volume is ~280 m deep against a horizontal extent of several km,
    // so sharing `half` would flatten the cloud to a sliver (measured: 3.5%
    // of the wireframe's height) and inflate the wireframe box to claim a
    // depth extent it does not have. zHalf defaults to `half` so callers that
    // pass an isotropic volume (and existing tests) keep their meaning.
    const uz = ((points[k * 3 + 2] - centre[2]) / zHalf) * zScale;

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
