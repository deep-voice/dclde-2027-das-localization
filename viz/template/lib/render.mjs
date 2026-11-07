// Pure render helpers. Everything here is unit-tested in Node; the canvas
// wiring in app.js stays a thin shell over these.

export function decodeF32(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

export function decodeI16(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Int16Array(buf);
}

export function makeView({ x0, y0, x1, y1, width, height }) {
  const scale = Math.min(width / (x1 - x0), height / (y1 - y0));
  const ox = (width - (x1 - x0) * scale) / 2;
  const oy = (height - (y1 - y0) * scale) / 2;
  return {
    scale,
    toScreen(x, y) {
      return [ox + (x - x0) * scale, height - oy - (y - y0) * scale];
    },
    toWorld(px, py) {
      return [x0 + (px - ox) / scale, y0 + (height - oy - py) / scale];
    },
  };
}

export function hillshade(z, nx, ny, dx, dy, { azimuth = 315, altitude = 45, noData = -32768 } = {}) {
  const az = (azimuth * Math.PI) / 180;
  const alt = (altitude * Math.PI) / 180;
  const out = new Float32Array(nx * ny);
  // A no-data NEIGHBOUR falls back to the centre pixel's own value, not a
  // substituted 0: swapping in 0 (usually far from the surrounding
  // elevations) manufactures a slope -- and so a bright synthetic rim -- around
  // every one of the grid's no-data holes, on the one layer whose job is to
  // recede into the background. Falling back to the centre keeps the local
  // gradient at 0 there instead, which reads as flat/no-relief.
  const at = (i, j, centre) => {
    const ii = Math.min(nx - 1, Math.max(0, i));
    const jj = Math.min(ny - 1, Math.max(0, j));
    const v = z[jj * nx + ii];
    return v === noData ? centre : v;
  };
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const centre = at(i, j, 0);
      const dzdx = (at(i + 1, j, centre) - at(i - 1, j, centre)) / (2 * dx);
      const dzdy = (at(i, j + 1, centre) - at(i, j - 1, centre)) / (2 * dy);
      const slope = Math.atan(Math.hypot(dzdx, dzdy));
      const aspect = Math.atan2(dzdy, -dzdx);
      let v = Math.sin(alt) * Math.cos(slope) +
              Math.cos(alt) * Math.sin(slope) * Math.cos(az - aspect);
      out[j * nx + i] = Math.min(1, Math.max(0, v));
    }
  }
  return out;
}

function lerp3(a, b, t) {
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ];
}

function ramp(stops, t) {
  const u = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(u));
  return lerp3(stops[i], stops[i + 1], u - i);
}

// --- Palette -----------------------------------------------------------------
// Three encodings share this map, so hues are assigned by job, not by taste:
//
//   bathymetry  neutral grey, hillshaded. Pure context, so it takes no hue at
//               all and cannot compete with either of the other two.
//   misfit      the default sequential hue (blue), monotone in lightness. Low
//               misfit is DARK and opaque, high misfit recedes toward the
//               surface -- so the basin is what the eye lands on.
//   cables      categorical slots 2 and 3 (orange, aqua). Slot 1 is blue and is
//               claimed by the misfit ramp; the palette's own rule that a second
//               sequential context takes the next categorical hue implies hues
//               are not shared across encodings, so the cables start at slot 2.
//               Validated all-pairs in both modes: CVD dE 9.2 light / 9.4 dark
//               (>=8 target), normal-vision 27.6 / 26.5 (>=15 floor).
//
// Dark mode is stepped for the dark surface, not flipped.

const DEPTH_STOPS_LIGHT = [[74, 76, 78], [116, 118, 118], [154, 156, 154], [190, 191, 188], [222, 222, 218]];
const DEPTH_STOPS_DARK = [[26, 27, 28], [46, 48, 50], [70, 72, 74], [98, 100, 101], [130, 131, 130]];

// Blue sequential, reference steps 700 -> 100. t = 0 is the best fit.
const MISFIT_STOPS_LIGHT = [[13, 54, 107], [28, 92, 171], [42, 120, 214], [109, 167, 236], [205, 226, 251]];
// On a dark surface the bright end stands out and the dark end recedes.
const MISFIT_STOPS_DARK = [[205, 226, 251], [134, 182, 239], [57, 135, 229], [24, 79, 149], [16, 66, 129]];

export function depthColor(z, zmin, zmax, dark = false) {
  const stops = dark ? DEPTH_STOPS_DARK : DEPTH_STOPS_LIGHT;
  return ramp(stops, (z - zmin) / (zmax - zmin));
}

export function misfitColor(t, dark = false) {
  return ramp(dark ? MISFIT_STOPS_DARK : MISFIT_STOPS_LIGHT, t);
}

export function isDarkTheme() {
  const stamped = document.documentElement.getAttribute("data-theme");
  if (stamped === "dark") return true;
  if (stamped === "light") return false;
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
}

// A neutral, partially-transparent fill for no-data cells -- distinct from
// both a fully transparent hole (which lets the panel background show through
// and makes a solution landing there look unmoored) and from any real depth
// colour in DEPTH_STOPS_LIGHT/DARK (so it cannot be mistaken for measured
// relief). Grey, not a hue: the palette's bathymetry slot takes no hue at
// all. Translucent, not opaque: it reads as "no relief data here" rather than
// as confidently-rendered terrain.
const NO_DATA_FILL_LIGHT = [188, 188, 184];
const NO_DATA_FILL_DARK = [54, 56, 59];
const NO_DATA_ALPHA = 140;

export function buildReliefRaster(bathy, dark = false) {
  const { nx, ny, zArr, shade, noData = -32768 } = bathy;
  const img = new ImageData(nx, ny);
  const noDataFill = dark ? NO_DATA_FILL_DARK : NO_DATA_FILL_LIGHT;
  let zmin = Infinity, zmax = -Infinity;
  for (let k = 0; k < zArr.length; k++) {
    const v = zArr[k];
    if (v === noData) continue;
    if (v < zmin) zmin = v;
    if (v > zmax) zmax = v;
  }
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      const dst = ((ny - 1 - j) * nx + i) * 4; // world y up -> image y down
      const v = zArr[k];
      if (v === noData) {
        img.data[dst] = noDataFill[0];
        img.data[dst + 1] = noDataFill[1];
        img.data[dst + 2] = noDataFill[2];
        img.data[dst + 3] = NO_DATA_ALPHA;
        continue;
      }
      const c = depthColor(v, zmin, zmax, dark);
      const s = 0.55 + 0.45 * shade[k];
      img.data[dst] = c[0] * s;
      img.data[dst + 1] = c[1] * s;
      img.data[dst + 2] = c[2] * s;
      img.data[dst + 3] = 255;
    }
  }
  const off = new OffscreenCanvas(nx, ny);
  off.getContext("2d").putImageData(img, 0, 0);
  return off;
}

export function blitGrid(ctx, raster, geom, view) {
  const { x0, y0, dx, dy, nx, ny } = geom;
  const [sx0, sy0] = view.toScreen(x0, y0 + dy * (ny - 1));
  const [sx1, sy1] = view.toScreen(x0 + dx * (nx - 1), y0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(raster, sx0, sy0, sx1 - sx0, sy1 - sy0);
}

export function drawCables(ctx, cables, view, colors) {
  // Casing then core. Over a mid-grey relief both cable hues sit at ~1.1:1
  // contrast, so a bare stroke is genuinely hard to follow; a surface-coloured
  // casing underneath makes the line legible against any relief value and
  // discharges the palette's sub-3:1 relief obligation structurally rather than
  // relying on the legend.
  for (const pass of ["casing", "core"]) {
    for (const [name, c] of Object.entries(cables)) {
      ctx.beginPath();
      for (let k = 0; k < c.x.length; k++) {
        const [sx, sy] = view.toScreen(c.x[k], c.y[k]);
        k === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
      }
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      if (pass === "casing") {
        ctx.strokeStyle = colors.surface;
        ctx.lineWidth = 4;
      } else {
        ctx.strokeStyle = colors[name];
        ctx.lineWidth = 1.75;
      }
      ctx.stroke();
    }
  }
}

export function drawSolutions(ctx, solutions, view, { color, surface, radius = 2.5 } = {}) {
  // Ink, not a fifth hue. Adding a categorical colour here would collide with
  // the cables or the misfit ramp; solution marks are identified by shape and
  // position, and a surface ring keeps them readable wherever they land.
  for (const s of solutions) {
    const [sx, sy] = view.toScreen(s.x, s.y);
    ctx.beginPath();
    ctx.arc(sx, sy, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = surface;
    ctx.stroke();
  }
}
