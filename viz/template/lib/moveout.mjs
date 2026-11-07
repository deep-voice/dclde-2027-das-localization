import { travelTime } from "./forward.mjs";

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
    // Via travelTime (forward.mjs), not a local hypot/c re-implementation --
    // identical arithmetic, and it inherits the cross-language parity
    // coverage rather than being a third, untested copy of the physics.
    out.push([d, travelTime(sx, sy, sz, [cx], [cy], [cz], c)[0]]);
  }
  return out;
}
