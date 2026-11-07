// Mirrors signed_side / mirror_penalty in src/dclde_localization/physics/forward.py.

export function signedSide(cable, x, y) {
  let k = 0, bd = Infinity;
  for (let i = 0; i < cable.x.length; i++) {
    const d = (cable.x[i] - x) ** 2 + (cable.y[i] - y) ** 2;
    if (d < bd) { bd = d; k = i; }
  }
  const k0 = Math.max(0, k - 1);
  const k1 = Math.min(cable.x.length - 1, k + 1);
  const tx = cable.x[k1] - cable.x[k0];
  const ty = cable.y[k1] - cable.y[k0];
  const nx = x - cable.x[k];
  const ny = y - cable.y[k];
  const s = Math.sign(tx * ny - ty * nx);
  return s === 0 ? 1 : s;
}

export function mirrorPenalty({ field, xs, ys, nx, ny, cable }) {
  const best = { 1: Infinity, "-1": Infinity };
  const xy = { 1: null, "-1": null };
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const v = field[j * nx + i];
      const s = signedSide(cable, xs[i], ys[j]);
      if (v < best[s]) { best[s] = v; xy[s] = [xs[i], ys[j]]; }
    }
  }
  const pos = best[1], neg = best["-1"];
  const [lo, hi] = pos <= neg ? [pos, neg] : [neg, pos];
  const [loXY, hiXY] = pos <= neg ? [xy[1], xy["-1"]] : [xy["-1"], xy[1]];
  return { best: lo, mirror: hi, ratio: lo > 0 ? hi / lo : Infinity, bestXY: loXY, mirrorXY: hiXY };
}
