// Mirror of src/dclde_localization/physics/forward.py. Held together by viz/tests/test_parity.mjs.
// Keep this boringly explicit; a clever rewrite here silently corrupts every picture.

export const DEFAULT_C = 1480.0;

export function travelTime(sx, sy, sz, px, py, pz, c) {
  const n = px.length;
  const out = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const dx = px[k] - sx;
    const dy = py[k] - sy;
    const dz = pz[k] - sz;
    out[k] = Math.sqrt(dx * dx + dy * dy + dz * dz) / c;
  }
  return out;
}

function median(values) {
  const a = Float64Array.from(values).sort();
  const n = a.length;
  if (n === 0) return NaN;
  const mid = n >> 1;
  return n % 2 ? a[mid] : 0.5 * (a[mid - 1] + a[mid]);
}

function mean(values) {
  let s = 0;
  for (let k = 0; k < values.length; k++) s += values[k];
  return s / values.length;
}

export function bestT0(obsT, tt, objective = "l1") {
  const n = obsT.length;
  const r = new Float64Array(n);
  for (let k = 0; k < n; k++) r[k] = obsT[k] - tt[k];
  if (objective === "l1") return median(r);
  if (objective === "l2") return mean(r);
  throw new Error(`unknown objective: ${objective}`);
}

export function misfitValue(obsT, tt, objective = "l1") {
  const n = obsT.length;
  const t0 = bestT0(obsT, tt, objective);
  if (objective === "l1") {
    let s = 0;
    for (let k = 0; k < n; k++) s += Math.abs(obsT[k] - tt[k] - t0);
    return s;
  }
  let s = 0;
  for (let k = 0; k < n; k++) {
    const d = obsT[k] - tt[k] - t0;
    s += d * d;
  }
  return Math.sqrt(s / n);
}

export function misfitField({ xs, ys, sz, px, py, pz, obsT, c, objective = "l1" }) {
  const nx = xs.length;
  const ny = ys.length;
  const np_ = px.length;
  const field = new Float64Array(nx * ny);

  const dz2 = new Float64Array(np_);
  for (let k = 0; k < np_; k++) {
    const dz = pz[k] - sz;
    dz2[k] = dz * dz;
  }

  const r = new Float64Array(np_);
  for (let j = 0; j < ny; j++) {
    const wy = ys[j];
    for (let i = 0; i < nx; i++) {
      const wx = xs[i];
      for (let k = 0; k < np_; k++) {
        const dx = px[k] - wx;
        const dy = py[k] - wy;
        r[k] = obsT[k] - Math.sqrt(dx * dx + dy * dy + dz2[k]) / c;
      }
      let acc = 0;
      if (objective === "l1") {
        const t0 = median(r);
        for (let k = 0; k < np_; k++) acc += Math.abs(r[k] - t0);
      } else if (objective === "l2") {
        const t0 = mean(r);
        for (let k = 0; k < np_; k++) {
          const d = r[k] - t0;
          acc += d * d;
        }
        acc = Math.sqrt(acc / np_);
      } else {
        throw new Error(`unknown objective: ${objective}`);
      }
      field[j * nx + i] = acc;
    }
  }
  return { field, nx, ny };
}
