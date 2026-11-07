import { misfitColor } from "./render.mjs";
import { misfitField } from "./forward.mjs";

export function ratioLevels(best, ratios) {
  return ratios.map((r) => best * r);
}

// Linear interpolation along cell edges; returns segments in grid index space.
export function marchingSquares(field, nx, ny, level) {
  const segs = [];
  const at = (i, j) => field[j * nx + i];
  const interp = (a, b) => (level - a) / (b - a);

  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const v = [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)];
      const above = v.map((x) => x > level);
      const idx = (above[0] ? 1 : 0) | (above[1] ? 2 : 0) | (above[2] ? 4 : 0) | (above[3] ? 8 : 0);
      if (idx === 0 || idx === 15) continue;

      const pts = [];
      if (above[0] !== above[1]) pts.push([i + interp(v[0], v[1]), j]);
      if (above[1] !== above[2]) pts.push([i + 1, j + interp(v[1], v[2])]);
      if (above[2] !== above[3]) pts.push([i + interp(v[3], v[2]), j + 1]);
      if (above[3] !== above[0]) pts.push([i, j + interp(v[0], v[3])]);
      for (let k = 0; k + 1 < pts.length; k += 2) {
        segs.push([pts[k][0], pts[k][1], pts[k + 1][0], pts[k + 1][1]]);
      }
    }
  }
  return segs;
}

function boxGrid(x0, y0, x1, y1, res) {
  const long = Math.max(x1 - x0, y1 - y0);
  const step = long / (res - 1);
  const nx = Math.max(2, Math.round((x1 - x0) / step) + 1);
  const ny = Math.max(2, Math.round((y1 - y0) / step) + 1);
  return {
    xs: Array.from({ length: nx }, (_, i) => x0 + i * step),
    ys: Array.from({ length: ny }, (_, j) => y0 + j * step),
  };
}

export function fitBox(x0, y0, x1, y1, pad = 0.25) {
  const w = Math.max(x1 - x0, 1), h = Math.max(y1 - y0, 1);
  return {
    x0: x0 - w * pad, x1: x1 + w * pad,
    y0: y0 - h * pad, y1: y1 + h * pad,
  };
}

// Reusable: subsample a solved xy grid down to ~target points per axis.
// Used by the z-profile panel (Task 12) to reuse the field's own xy grid
// cheaply, and by Task 13.
//
// Deviates from the brief's fixed-stride snippet (step = floor(n/target), then
// force the far endpoint): that always keeps ceil(n/step) elements plus one for
// the endpoint, which overshoots target by more than the "near 40" bound the
// brief's own test asserts (e.g. n=300 -> 44 elements, n=260 -> 45, both over
// the test's <=42 cap). This picks `target` indices evenly spaced across the
// array by rounding, which lands on the target count exactly and still
// guarantees both endpoints by construction (i=0 -> index 0, i=target-1 ->
// index length-1) -- loop-based, no modulo filter.
export function subsampleGrid(xs, ys, target = 40) {
  const pick = (arr) => {
    if (arr.length <= target) return arr.slice();
    const last = arr.length - 1;
    const denom = Math.max(target - 1, 1);
    const out = [];
    for (let i = 0; i < target; i++) out.push(arr[Math.round((i * last) / denom)]);
    // Always keep the far endpoint explicitly, guarding any rounding edge case.
    if (out[out.length - 1] !== arr[last]) out[out.length - 1] = arr[last];
    return out;
  };
  return { xs: pick(xs), ys: pick(ys) };
}

// Everything worse than this multiple of the best fit is uninteresting, so the
// second-pass box is the bracket of everything better. Chosen from measurement:
// the <5x region is a median 8.8 km across (p90 22.5 km), which puts the
// 1.05-1.50 contours at tens of metres per cell instead of hundreds.
const BRACKET_RATIO = 5.0;
const MIN_BOX_M = 2000;

function wideBox({ px, py, bathyBounds }) {
  const xmin = Math.min(...px), xmax = Math.max(...px);
  const ymin = Math.min(...py), ymax = Math.max(...py);
  const pad = Math.max(10000, 0.5 * Math.max(xmax - xmin, ymax - ymin));
  return {
    x0: Math.max(bathyBounds.x0, xmin - pad), x1: Math.min(bathyBounds.x1, xmax + pad),
    y0: Math.max(bathyBounds.y0, ymin - pad), y1: Math.min(bathyBounds.y1, ymax + pad),
  };
}

// Pass 1 alone: a coarse solve over a box wide enough not to miss a mirror
// basin. Shared by solutionGrid (pass 1 of the fine field) and the per-call
// index (Task 9c), so the index and the field it labels cannot drift apart.
export function coarseSolve({ px, py, pz, obsT, c, sz, bathyBounds, res = 90, objective = "l1" }) {
  const w = wideBox({ px, py, bathyBounds });
  const grid = boxGrid(w.x0, w.y0, w.x1, w.y1, res);
  const { field, nx } = misfitField({ ...grid, sz, px, py, pz, obsT, c, objective });
  let best = Infinity, bi = 0;
  for (let k = 0; k < field.length; k++) if (field[k] < best) { best = field[k]; bi = k; }
  return { x: grid.xs[bi % nx], y: grid.ys[Math.floor(bi / nx)], best, grid, field, nx };
}

// Nearest marker to (wx, wy) within maxDist, or -1 if none qualify.
export function nearestMarker(markers, wx, wy, maxDist) {
  let bi = -1, bd = maxDist * maxDist;
  for (let k = 0; k < markers.length; k++) {
    const dx = markers[k].x - wx, dy = markers[k].y - wy;
    const d = dx * dx + dy * dy;
    if (d <= bd) { bd = d; bi = k; }
  }
  return bi;
}

export function solutionGrid({ px, py, pz, obsT, c, sz, res, bathyBounds, objective = "l1" }) {
  // Pass 1: coarse, over a box wide enough not to miss a mirror basin.
  const { x: bx, y: by, best, field, nx, grid: coarse } =
    coarseSolve({ px, py, pz, obsT, c, sz, bathyBounds, objective });

  // Pass 2: bracket every coarse cell fitting within BRACKET_RATIO of the best.
  // This picks up a mirror basin automatically, with no need for cable geometry,
  // and stays wide when a call genuinely has no localised solution.
  let lx = bx, hx = bx, ly = by, hy = by;
  for (let k = 0; k < field.length; k++) {
    if (field[k] > best * BRACKET_RATIO) continue;
    const x = coarse.xs[k % nx], y = coarse.ys[Math.floor(k / nx)];
    if (x < lx) lx = x; if (x > hx) hx = x;
    if (y < ly) ly = y; if (y > hy) hy = y;
  }
  if (hx - lx < MIN_BOX_M) { const c0 = (lx + hx) / 2; lx = c0 - MIN_BOX_M / 2; hx = c0 + MIN_BOX_M / 2; }
  if (hy - ly < MIN_BOX_M) { const c0 = (ly + hy) / 2; ly = c0 - MIN_BOX_M / 2; hy = c0 + MIN_BOX_M / 2; }

  const fine = fitBox(lx, ly, hx, hy, 0.25);
  return { ...boxGrid(fine.x0, fine.y0, fine.x1, fine.y1, res), coarseBest: [bx, by] };
}

export const LEGEND_RATIOS = [1.0, 1.1, 1.25, 1.5, 2, 4, 8];
export function ratioLegendStops(cap = 8) {
  return LEGEND_RATIOS.filter((r) => r <= cap)
    .map((r) => ({ ratio: r, t: Math.log(r) / Math.log(cap) }));
}

export function fieldImage(field, nx, ny, best, cap = 8.0, dark = false) {
  // LOG map. Ratios here span 1 to ~80 with a median near 35, so a linear map
  // put almost every cell at the saturated end and the field read as one flat
  // wash. log(ratio)/log(cap) spreads the structure that actually matters.
  const data = new Uint8ClampedArray(nx * ny * 4);
  const lk = Math.log(cap);
  const denom = best > 0 && Number.isFinite(best) ? best : 1;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      const t = Math.min(1, Math.max(0, Math.log(field[k] / denom) / lk));
      const col = misfitColor(t, dark);
      const dst = ((ny - 1 - j) * nx + i) * 4;
      data[dst] = col[0];
      data[dst + 1] = col[1];
      data[dst + 2] = col[2];
      // Taper to 0 at the cap (t=1), not a floor of 20: a non-zero floor made
      // the raster's own extent visible as a lighter rectangle over the
      // relief on every call, since even the worst cell in the grid still
      // painted some opaque colour right up to the computed box's edge.
      data[dst + 3] = Math.round(255 * (1 - t));
    }
  }
  return data;
}

export function zoomBoxAt(box, wx, wy, factor) {
  return {
    x0: wx - (wx - box.x0) * factor,
    x1: wx + (box.x1 - wx) * factor,
    y0: wy - (wy - box.y0) * factor,
    y1: wy + (box.y1 - wy) * factor,
  };
}
