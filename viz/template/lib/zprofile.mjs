import { misfitField } from "./forward.mjs";

export function zProfile({ zs, xs, ys, px, py, pz, obsT, c, objective = "l1" }) {
  return zs.map((z) => {
    const { field } = misfitField({ xs, ys, sz: z, px, py, pz, obsT, c, objective });
    let best = Infinity;
    for (let k = 0; k < field.length; k++) if (field[k] < best) best = field[k];
    return { z, best };
  });
}

export function flatnessRatio(profile) {
  // An empty profile would give -Infinity / Infinity = NaN. Unreachable from the
  // current call site, but this is an exported primitive.
  if (!profile || profile.length === 0) return 1;
  let lo = Infinity, hi = -Infinity;
  for (const p of profile) {
    if (p.best < lo) lo = p.best;
    if (p.best > hi) hi = p.best;
  }
  return hi / Math.max(lo, 1e-12);
}
