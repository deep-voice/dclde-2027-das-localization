const $ = (id) => document.getElementById(id);

const bathy = (() => {
  const b = PAYLOAD.bathy;
  const zArr = decodeI16(b.z);
  return {
    nx: b.nx, ny: b.ny, x0: b.x0, y0: b.y0, dx: b.dx, dy: b.dy, noData: b.noData,
    zArr,
    shade: hillshade(zArr, b.nx, b.ny, b.dx, b.dy, { noData: b.noData }),
  };
})();

const cables = Object.fromEntries(
  Object.entries(PAYLOAD.cables).map(([name, c]) => [name, {
    chanM: decodeF32(c.chanM),
    x: decodeF32(c.x), y: decodeF32(c.y), depth: decodeF32(c.depth),
  }])
);

// Side determination does not need 8 m cable resolution; ~200 m is ample and
// cuts the mirrorPenalty scan by ~25x. Built once.
const SIDE_SPACING_M = 200;
const sideCables = Object.fromEntries(
  Object.entries(cables).map(([name, c]) => {
    const step = Math.max(1, Math.round(SIDE_SPACING_M / 8.17));
    const keep = [];
    for (let k = 0; k < c.x.length; k += step) keep.push(k);
    if (keep[keep.length - 1] !== c.x.length - 1) keep.push(c.x.length - 1);
    return [name, {
      x: Float32Array.from(keep, (k) => c.x[k]),
      y: Float32Array.from(keep, (k) => c.y[k]),
    }];
  })
);

const state = {
  c: PAYLOAD.meta.defaultC,
  sz: -30,
  call: null,
  cablesOn: { North: true, South: true },
  res: 260,
  resultSet: PAYLOAD.resultSets[0] ?? null,
};

const bathyBounds = {
  x0: bathy.x0, y0: bathy.y0,
  x1: bathy.x0 + bathy.dx * (bathy.nx - 1),
  y1: bathy.y0 + bathy.dy * (bathy.ny - 1),
};

const homeBox = { ...bathyBounds };
state.viewBox = { ...homeBox };
state.framedCall = null;

function themeColors() {
  const s = getComputedStyle(document.documentElement);
  const tok = (name, fallback) => s.getPropertyValue(name).trim() || fallback;
  return {
    North: tok("--cable-n", "#eb6834"),
    South: tok("--cable-s", "#1baf7a"),
    accent: tok("--accent", "#2a78d6"),
    surface: tok("--surface", "#ffffff"),
    ink: tok("--ink", "#0b0b0b"),
    inkDim: tok("--ink-dim", "#52514e"),
    dark: isDarkTheme(),
  };
}

// Repaint on a theme change: the ramps are selected per mode, not flipped, so a
// cached raster from the other mode would be wrong.
if (window.matchMedia) {
  window.matchMedia("(prefers-color-scheme: dark)")
    .addEventListener("change", () => resizeAndDraw());
}

let view;

function resizeAndDraw() {
  const canvas = $("map");
  // Measure the canvas itself, not its parent (#map-wrap has 12px padding + a
  // 0.8px border, ~3.5% wider than the canvas): every pointer handler below
  // measures canvas.getBoundingClientRect() too, and view.toWorld must be
  // built from the same box those handlers convert screen coordinates against.
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  view = makeView({
    x0: state.viewBox.x0, y0: state.viewBox.y0,
    x1: state.viewBox.x1, y1: state.viewBox.y1,
    width: rect.width, height: rect.height,
  });
  draw();
}

// Later tasks add overlays by pushing onto `layers`, in draw order. Do not
// reassign `draw` itself — a chain of wrappers is hard to reason about and
// breaks silently if one task's ordering assumption is wrong.
const layers = [];

// The relief raster depends only on the elevation grid and the theme, never on
// the view, so it is built once (per theme) and blitted -- not rebuilt on
// every pan/zoom repaint.
let reliefRaster = null, reliefDark = null;
function relief() {
  const dark = themeColors().dark;
  if (!reliefRaster || reliefDark !== dark) {
    reliefRaster = buildReliefRaster(bathy, dark);
    reliefDark = dark;
  }
  return reliefRaster;
}

function draw() {
  const ctx = $("map").getContext("2d");
  const colors = themeColors();
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  blitGrid(ctx, relief(), bathy, view);
  drawCables(ctx, cables, view, colors);
  if (state.resultSet) {
    drawSolutions(ctx, state.resultSet.solutions, view,
      { color: colors.ink, surface: colors.surface });
  }
  for (const layer of layers) layer(ctx, colors);
}

$("dataset-info").innerHTML =
  `<div class="readout"><dt>calls</dt><dd>${PAYLOAD.calls.length}</dd></div>` +
  `<div class="readout"><dt>build</dt><dd>${PAYLOAD.meta.gitSha}</dd></div>`;

window.addEventListener("resize", resizeAndDraw);

// Throttle pan/zoom repaints to one per animation frame -- an unthrottled
// pointermove was rebuilding the entire relief and field rasters on every
// mouse-move event.
let rafPending = false;
function scheduleDraw() {
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => { rafPending = false; resizeAndDraw(); });
}

const MIN_SPAN_M = 200;
const MAX_SPAN_M = 4 * Math.max(homeBox.x1 - homeBox.x0, homeBox.y1 - homeBox.y0);

function clampBox(b) {
  const w = b.x1 - b.x0, h = b.y1 - b.y0;
  const span = Math.max(w, h);
  if (span < MIN_SPAN_M || span > MAX_SPAN_M) return state.viewBox;
  return b;
}

// --- zoom / pan ---------------------------------------------------------------
// A 1.8 km basin inside a 90 km survey box is not inspectable without this.
(() => {
  const canvas = $("map");
  canvas.style.cursor = "grab";
  let dragging = null;
  // Recorded on pointerdown, screen coordinates, so pointerup can tell a click
  // (barely moved) from the end of a pan without touching `dragging`'s world
  // coordinates.
  let dragStart = null;

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = canvas.getBoundingClientRect();
    const [wx, wy] = view.toWorld(e.clientX - r.left, e.clientY - r.top);
    const factor = e.deltaY > 0 ? 1.15 : 1 / 1.15;
    state.viewBox = clampBox(zoomBoxAt(state.viewBox, wx, wy, factor));
    scheduleDraw();
  }, { passive: false });

  canvas.addEventListener("pointerdown", (e) => {
    const r = canvas.getBoundingClientRect();
    dragging = { world: view.toWorld(e.clientX - r.left, e.clientY - r.top), box: { ...state.viewBox } };
    dragStart = { sx: e.clientX, sy: e.clientY };
    canvas.style.cursor = "grabbing";
    canvas.setPointerCapture(e.pointerId);
  });
  // Same convention as the 3-D inset's endDrag: releasePointerCapture is
  // wrapped in try/catch because a pointerup can arrive after capture was
  // already lost (see lostpointercapture below), in which case it throws.
  const endDrag = (e) => {
    dragging = null;
    dragStart = null;
    canvas.style.cursor = "grab";
    try { canvas.releasePointerCapture(e.pointerId); } catch { /* already released */ }
  };
  canvas.addEventListener("pointerup", (e) => {
    // A click must not fire at the end of a pan: only treat this as a
    // selection when the pointer barely moved since pointerdown.
    const moved = dragStart
      ? Math.hypot(e.clientX - dragStart.sx, e.clientY - dragStart.sy)
      : 999;
    if (moved < 4 && callIndex.length) {
      const r = canvas.getBoundingClientRect();
      const [wx, wy] = view.toWorld(e.clientX - r.left, e.clientY - r.top);
      // 12 screen px, converted to world units at the current zoom.
      const tol = 12 / view.scale;
      const k = nearestMarker(callIndex, wx, wy, tol);
      if (k >= 0) selectCall(callIndex[k].call);
    }
    endDrag(e);
  });
  // Without this, losing capture without a pointerup (a tab switch mid-drag, a
  // touch interruption) leaves `dragging` true, so ordinary hover then keeps
  // panning the map with no button held.
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("lostpointercapture", () => {
    dragging = null;
    dragStart = null;
    canvas.style.cursor = "grab";
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const r = canvas.getBoundingClientRect();
    const now = view.toWorld(e.clientX - r.left, e.clientY - r.top);
    const dx = dragging.world[0] - now[0], dy = dragging.world[1] - now[1];
    state.viewBox = {
      x0: dragging.box.x0 + dx, x1: dragging.box.x1 + dx,
      y0: dragging.box.y0 + dy, y1: dragging.box.y1 + dy,
    };
    scheduleDraw();
  });
  canvas.addEventListener("dblclick", () => {
    state.viewBox = { ...homeBox };
    resizeAndDraw();
  });
})();

// ---- pick index: call -> row range in the flat pick arrays -------------------
const picksRaw = {
  callIndex: PAYLOAD.picks.callIndex,
  cable: PAYLOAD.picks.cable,
  dist: decodeF32(PAYLOAD.picks.dist),
  time: decodeF32(PAYLOAD.picks.time),
  px: decodeF32(PAYLOAD.picks.px),
  py: decodeF32(PAYLOAD.picks.py),
  pz: decodeF32(PAYLOAD.picks.pz),
};

const callRows = new Map();
picksRaw.callIndex.forEach((ci, row) => {
  const key = PAYLOAD.calls[ci];
  if (!callRows.has(key)) callRows.set(key, []);
  callRows.get(key).push(row);
});

function picksFor(call, cablesOn) {
  const rows = (callRows.get(call) ?? []).filter((r) => {
    const name = picksRaw.cable[r] === 0 ? "North" : "South";
    return cablesOn[name];
  });
  return {
    rows,
    px: rows.map((r) => picksRaw.px[r]),
    py: rows.map((r) => picksRaw.py[r]),
    pz: rows.map((r) => picksRaw.pz[r]),
    obsT: rows.map((r) => picksRaw.time[r]),
    dist: rows.map((r) => picksRaw.dist[r]),
    cable: rows.map((r) => (picksRaw.cable[r] === 0 ? "North" : "South")),
  };
}

// One coarse solution per call, so the map is an index of the dataset rather
// than a picture of somebody else's results. Measured at ~66M distance
// evaluations for 608 calls (~2.1 s total in scalar JS), so it runs after
// first paint and is chunked against a per-frame time budget rather than run
// synchronously in one go, which would freeze the page for two seconds.
let callIndex = [];

const INDEX_FRAME_BUDGET_MS = 12;

function buildCallIndexIncremental() {
  const queue = PAYLOAD.calls.slice();
  const out = [];
  const t0 = performance.now();

  function step() {
    const frameStart = performance.now();
    while (queue.length && performance.now() - frameStart < INDEX_FRAME_BUDGET_MS) {
      const call = queue.shift();
      const p = picksFor(call, { North: true, South: true });
      if (p.rows.length < 5) continue;
      const s = coarseSolve({
        px: p.px, py: p.py, pz: p.pz, obsT: p.obsT,
        c: PAYLOAD.meta.defaultC, sz: -30, bathyBounds,
      });
      out.push({ call, x: s.x, y: s.y, best: s.best, n: p.rows.length });
    }
    callIndex = out;
    const done = PAYLOAD.calls.length - queue.length;
    if (queue.length) {
      $("index-status").textContent = `Locating calls… ${done} of ${PAYLOAD.calls.length}`;
      draw();
      requestAnimationFrame(step);
    } else {
      $("index-status").textContent =
        `${out.length} of ${PAYLOAD.calls.length} calls located (coarse, ` +
        `${((performance.now() - t0) / 1000).toFixed(1)} s). Click a marker to open one.`;
      draw();
    }
  }
  requestAnimationFrame(step);
}

// After first paint, so the page is interactive immediately.
requestAnimationFrame(() => setTimeout(buildCallIndexIncremental, 0));

function selectCall(call) {
  state.call = call;
  refresh();
  const row = $("call-list").querySelector(`[data-call="${CSS.escape(call)}"]`);
  if (row) {
    document.querySelectorAll("#call-list .sel").forEach((n) => n.classList.remove("sel"));
    row.classList.add("sel");
    row.scrollIntoView({ block: "nearest" });
  }
}

let currentField = null;
// Declared here (not beside computeInset/drawInset below) because computeField
// touches it unconditionally and is first invoked by the refresh() call before
// the inset section of this file runs -- a `let` declared down there would
// leave this reference in the temporal dead zone on that first call.
let insetCache = null;

// The one searched source-depth range. The z-profile's zs, its axis mapping,
// and the 3-D inset's zs all read this rather than each hardcoding their own
// bounds -- they drifted apart before (the inset used -280..0 while the
// z-profile and the sliders used -200..0), and a mapping like `(z + 200) /
// 200` silently mis-maps every point if the range ever changes without it
// being updated too.
const SOURCE_DEPTH_MIN = -200;
const SOURCE_DEPTH_MAX = 0;
const SOURCE_DEPTH_SPAN = SOURCE_DEPTH_MAX - SOURCE_DEPTH_MIN;
function depthSteps(n) {
  const step = SOURCE_DEPTH_SPAN / (n - 1);
  return Array.from({ length: n }, (_, i) => SOURCE_DEPTH_MIN + i * step);
}

function computeField() {
  currentField = null;
  insetCache = null;
  if (!state.call) return;
  const p = picksFor(state.call, state.cablesOn);
  if (p.rows.length < 5) return;

  const { xs, ys } = solutionGrid({
    px: p.px, py: p.py, pz: p.pz, obsT: p.obsT,
    c: state.c, sz: state.sz, res: state.res, bathyBounds,
  });
  const { field, nx, ny } = misfitField({
    xs, ys, sz: state.sz, px: p.px, py: p.py, pz: p.pz,
    obsT: p.obsT, c: state.c, objective: "l1",
  });
  let best = Infinity, bi = 0;
  for (let k = 0; k < field.length; k++) if (field[k] < best) { best = field[k]; bi = k; }

  // Build the field's raster once here, not on every pan/zoom repaint -- it
  // depends only on the computed field, never on the view.
  const img = new ImageData(fieldImage(field, nx, ny, best, 8.0, themeColors().dark), nx, ny);
  const raster = new OffscreenCanvas(nx, ny);
  raster.getContext("2d").putImageData(img, 0, 0);

  currentField = {
    xs, ys, field, nx, ny, best, raster,
    bestXY: [xs[bi % nx], ys[Math.floor(bi / nx)]],
    nPicks: p.rows.length,
  };

  // Frame on call change only. Re-framing on every recompute would yank the
  // view back every time a slider moves, discarding a deliberate pan.
  if (state.framedCall !== state.call) {
    state.viewBox = fitBox(xs[0], ys[0], xs[xs.length - 1], ys[ys.length - 1], 0.6);
    state.framedCall = state.call;
  }

  // Once per field, not once per frame: this depends on the field and the
  // active cables, neither of which changes while panning.
  currentField.mirror = {};
  for (const name of Object.keys(cables)) {
    if (!state.cablesOn[name]) continue;
    currentField.mirror[name] = mirrorPenalty({
      ...currentField, cable: sideCables[name],
    });
  }

  // Same xy area as the map, at 21 assumed depths. Subsampled: the full grid
  // across 21 depths would be ~700k cells and stall the frame.
  // 28 rather than 40: the worst call in the dataset (61 picks) measured 84 ms
  // of z-profile at 40, pushing call selection to ~202 ms. The finding here is
  // flatness, which does not need fine spatial resolution, and (28/40)^2 roughly
  // halves the cost.
  const sub = subsampleGrid(xs, ys, 28);
  currentField.zprof = zProfile({
    zs: depthSteps(21),
    xs: sub.xs, ys: sub.ys,
    px: p.px, py: p.py, pz: p.pz, obsT: p.obsT, c: state.c,
  });

  computeInset();
}

function drawField(ctx) {
  if (!currentField) return;
  const { xs, ys, field, nx, ny, best, raster } = currentField;
  const [sx0, sy0] = view.toScreen(xs[0], ys[ny - 1]);
  const [sx1, sy1] = view.toScreen(xs[nx - 1], ys[0]);
  ctx.drawImage(raster, sx0, sy0, sx1 - sx0, sy1 - sy0);

  const colors = themeColors();
  ctx.lineWidth = 1;
  ctx.strokeStyle = colors.accent;
  for (const level of ratioLevels(best, PAYLOAD.meta.contourRatios)) {
    ctx.beginPath();
    for (const [gx0, gy0, gx1, gy1] of marchingSquares(field, nx, ny, level)) {
      const a = view.toScreen(xs[0] + gx0 * (xs[1] - xs[0]), ys[0] + gy0 * (ys[1] - ys[0]));
      const b = view.toScreen(xs[0] + gx1 * (xs[1] - xs[0]), ys[0] + gy1 * (ys[1] - ys[0]));
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
    }
    ctx.stroke();
  }
}

// Field raster + contours, then the selected call's pick marks on top.
layers.push((ctx, colors) => {
  drawField(ctx);
  if (!state.call) return;
  const p = picksFor(state.call, state.cablesOn);
  for (let k = 0; k < p.px.length; k++) {
    const [sx, sy] = view.toScreen(p.px[k], p.py[k]);
    ctx.fillStyle = colors[p.cable[k]];
    ctx.beginPath();
    ctx.arc(sx, sy, 2, 0, Math.PI * 2);
    ctx.fill();
  }
});

// Per-call index markers and the selected call's refined solution. Identity
// is by shape, not colour: the palette's categorical slots are already spent
// on the two cables, and its sequential slot on the misfit field.
layers.push((ctx, colors) => {
  // Per-call coarse solutions: small open circles.
  ctx.lineWidth = 1.25;
  ctx.strokeStyle = colors.accent;
  for (const m of callIndex) {
    if (m.call === state.call) continue;
    const [sx, sy] = view.toScreen(m.x, m.y);
    ctx.beginPath();
    ctx.arc(sx, sy, 3, 0, Math.PI * 2);
    ctx.stroke();
  }

  // The selected call's refined solution: bigger, ringed, with a crosshair.
  if (currentField) {
    const [sx, sy] = view.toScreen(currentField.bestXY[0], currentField.bestXY[1]);
    ctx.beginPath();
    ctx.arc(sx, sy, 7, 0, Math.PI * 2);
    ctx.strokeStyle = colors.surface;
    ctx.lineWidth = 3.5;
    ctx.stroke();
    ctx.strokeStyle = colors.accent;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(sx - 11, sy); ctx.lineTo(sx - 8, sy);
    ctx.moveTo(sx + 8, sy); ctx.lineTo(sx + 11, sy);
    ctx.moveTo(sx, sy - 11); ctx.lineTo(sx, sy - 8);
    ctx.moveTo(sx, sy + 8); ctx.lineTo(sx, sy + 11);
    ctx.stroke();
  }
});
layers.push(() => renderReadout());

function drawLegend() {
  const bar = $("legend-bar");
  const rect = bar.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  bar.width = Math.max(1, Math.floor(rect.width * dpr));
  bar.height = Math.floor(14 * dpr);
  const ctx = bar.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const dark = themeColors().dark;
  for (let x = 0; x < rect.width; x++) {
    const c = misfitColor(x / Math.max(1, rect.width - 1), dark);
    ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
    ctx.fillRect(x, 0, 1, 14);
  }

  $("legend-ticks").innerHTML = ratioLegendStops()
    .map(({ ratio, t }) => {
      const label = ratio === 1 ? "best" : `${ratio}×`;
      return `<span style="left:${(t * 100).toFixed(1)}%">${label}</span>`;
    })
    .join("");
}

layers.push(() => drawLegend());

function renderReadout() {
  const el = $("call-readout");
  if (!state.call) { el.innerHTML = "<p style='color:var(--ink-dim)'>Select a call.</p>"; return; }
  const f = currentField;
  const warn = f && f.nPicks < 10
    ? "<span class='badge warn'>under-constrained</span>" : "";
  el.innerHTML = `<div style="font-weight:600">${state.call}</div>${warn}` +
    (f
      ? `<div class="readout"><dt>picks</dt><dd>${f.nPicks}</dd></div>` +
        `<div class="readout"><dt>best misfit</dt><dd>${f.best.toFixed(3)} s</dd></div>` +
        `<div class="readout"><dt>solution</dt><dd>${(f.bestXY[0]/1000).toFixed(2)}, ${(f.bestXY[1]/1000).toFixed(2)} km</dd></div>`
      : "<p style='color:var(--ink-dim)'>Too few picks on the selected cables.</p>");
}

function renderCallList(filter = "") {
  const list = $("call-list");
  list.innerHTML = "";
  for (const call of PAYLOAD.calls) {
    if (filter && !call.includes(filter)) continue;
    const b = document.createElement("button");
    b.textContent = `${call} (${(callRows.get(call) ?? []).length})`;
    b.dataset.call = call;
    b.className = call === state.call ? "sel" : "";
    b.style.cssText = "display:block;width:100%;text-align:left;border:0;background:none;color:inherit;cursor:pointer;padding:2px 0";
    b.onclick = () => selectCall(call);
    list.appendChild(b);
  }
}

// Single entry point for "something changed, recompute and redraw". Later
// tasks add overlays via `layers`, never by wrapping refresh or draw.
// Goes through resizeAndDraw (not a bare draw()) because computeField can
// move state.viewBox to auto-frame the selected call's basin, and `view`
// itself is only rebuilt there.
function refresh() {
  computeField();
  resizeAndDraw();
}

$("call-search").addEventListener("input", (e) => renderCallList(e.target.value.trim()));
renderCallList();
renderReadout();

// All layers (field, picks, readout, legend) are registered by this point, so
// the first paint is the only call to resizeAndDraw that needs to happen here.
resizeAndDraw();

function drawMoveout() {
  const canvas = $("moveout");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(180 * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, 180);

  if (!state.call || !currentField) return;
  const p = picksFor(state.call, state.cablesOn);
  const colors = themeColors();
  const s = moveoutScales({ dist: p.dist, obsT: p.obsT, width: rect.width, height: 180 });

  // ONE emission time for the call, pooled across every active pick, computed
  // exactly as bestT0/misfitField do. A per-cable t0 would re-centre each cable
  // on its own picks and so hide inter-cable inconsistency -- the very "tiny
  // misfit but nonsense" case this panel exists to expose. Physically there is
  // one emission time, not one per cable, so pooling is also the honest model.
  // Uses travelTime per pick rather than the nearest of 120 curve samples,
  // which removes a discretisation error too.
  const ttAll = travelTime(
    currentField.bestXY[0], currentField.bestXY[1], state.sz,
    p.px, p.py, p.pz, state.c,
  );
  const t0 = bestT0(p.obsT, ttAll, "l1");

  // Predicted curve per active cable, all shifted by that single t0.
  for (const name of Object.keys(cables)) {
    if (!state.cablesOn[name]) continue;
    const idx = p.cable.map((cb, k) => (cb === name ? k : -1)).filter((k) => k >= 0);
    if (!idx.length) continue;
    const pts = predictedCurve({
      sourceXY: currentField.bestXY, sz: state.sz, cable: cables[name],
      distMin: Math.min(...idx.map((k) => p.dist[k])),
      distMax: Math.max(...idx.map((k) => p.dist[k])),
      c: state.c,
    });

    ctx.beginPath();
    pts.forEach(([d, t], k) => {
      const X = s.sx(d), Y = s.sy(t + t0);
      k === 0 ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y);
    });
    ctx.strokeStyle = colors[name];
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.85;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  for (let k = 0; k < p.dist.length; k++) {
    ctx.fillStyle = colors[p.cable[k]];
    ctx.beginPath();
    ctx.arc(s.sx(p.dist[k]), s.sy(p.obsT[k]), 2.5, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--ink-dim");
  ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
  ctx.fillText("along-cable distance →", 24, 174);
  ctx.save();
  ctx.translate(10, 100);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("arrival time →", 0, 0);
  ctx.restore();
}

layers.push(() => drawMoveout());
draw();

function activeCableNames() {
  return Object.keys(cables).filter((n) => state.cablesOn[n]);
}

function renderMirrorBadge() {
  const existing = $("mirror-badge");
  if (existing) existing.remove();
  if (!currentField) return;
  const host = document.createElement("div");
  host.id = "mirror-badge";

  const active = activeCableNames();
  const rows = [];
  if (active.length === 1) {
    const mp = currentField.mirror[active[0]];
    if (!mp || !mp.mirrorXY || !mp.bestXY || !Number.isFinite(mp.ratio)) {
      // Every grid cell fell on one side of the cable, so there is no mirror
      // candidate to compare against. Undefined, not infinite. The
      // solution-sized grid does this routinely for a confidently-offset
      // single-cable solve, since the bracket only keeps cells within 5x of
      // the best misfit.
      host.innerHTML =
        `<div class="readout"><dt>mirror</dt><dd>` +
        `<span title="the computed grid lies entirely on one side of ${active[0]}">n/a</span>` +
        `</dd></div>`;
      $("call-readout").appendChild(host);
      return;
    }
    const sep = Math.hypot(mp.bestXY[0] - mp.mirrorXY[0], mp.bestXY[1] - mp.mirrorXY[1]);

    // A low ratio alone does NOT mean the km-scale left-right ambiguity. Measured
    // over 232 single-cable solves: 14% have a ratio under 1.10, but 27% of those
    // have the two candidates less than 500 m apart, i.e. the solution sits
    // essentially on the cable axis and the "two sides" are the same place.
    // Only 8.2% of solves are both ratio < 1.10 AND more than 1 km apart, which
    // is the case that actually costs kilometres if you pick the wrong side.
    // Separation distribution: median 2.4 km, p10 0.47 km, p90 16.2 km.
    const AMBIGUOUS_RATIO = 1.10;
    const DISTINCT_SEP_M = 500;
    const lowRatio = mp.ratio < AMBIGUOUS_RATIO;
    const distinct = sep > DISTINCT_SEP_M;
    const cls = lowRatio && distinct ? "badge warn" : "badge";

    let note = "";
    if (lowRatio && distinct) {
      note = "<div style='color:var(--ink-dim);font-size:12px'>Both sides fit almost equally and " +
        "they are " + (sep / 1000).toFixed(1) + " km apart — a genuine left-right ambiguity. " +
        "Picking the wrong side costs that whole distance.</div>";
    } else if (lowRatio) {
      note = "<div style='color:var(--ink-dim);font-size:12px'>Both sides fit equally, but they are " +
        "only " + Math.round(sep) + " m apart: the solution sits essentially on the cable axis, " +
        "so the two sides are not distinct places.</div>";
    }

    rows.push(
      `<div style="margin:6px 0"><span class="${cls}">mirror +${((mp.ratio - 1) * 100).toFixed(0)}%</span></div>` +
      `<div class="readout"><dt>mirror at</dt><dd>${(mp.mirrorXY[0]/1000).toFixed(1)}, ${(mp.mirrorXY[1]/1000).toFixed(1)} km</dd></div>` +
      `<div class="readout"><dt>separation</dt><dd>${sep < 1000 ? Math.round(sep) + " m" : (sep/1000).toFixed(2) + " km"}</dd></div>` +
      note
    );
  } else {
    // With two cables there is no single mirror; report one ratio per cable.
    for (const name of active) {
      const mp = currentField.mirror[name];
      // The grid is sized around the solution, so it can lie entirely on one
      // side of the OTHER cable — in which case there is no mirror-side cell to
      // compare against and the ratio is undefined, not infinite. Printing
      // "+Infinity%" told the user nothing.
      const cell = Number.isFinite(mp.ratio)
        ? `+${((mp.ratio - 1) * 100).toFixed(0)}%`
        : `<span title="the computed grid does not span both sides of ${name}">n/a</span>`;
      rows.push(`<div class="readout"><dt>mirror vs ${name}</dt><dd>${cell}</dd></div>`);
    }
  }
  host.innerHTML = rows.join("");
  $("call-readout").appendChild(host);
}

// The badge is a layer, appended after renderReadout has rebuilt the panel.
// `refresh` is defined in Task 9 and is not wrapped here.
layers.push(() => renderMirrorBadge());

// --- controls ---
const cableToggle = $("cable-toggle");
for (const name of Object.keys(cables)) {
  const id = `cable-${name}`;
  const wrap = document.createElement("label");
  wrap.style.cssText = "display:flex;gap:6px;align-items:center;margin:2px 0;color:var(--ink)";
  wrap.innerHTML = `<input type="checkbox" id="${id}" checked><span>${name}</span>`;
  cableToggle.appendChild(wrap);
  wrap.querySelector("input").addEventListener("change", (e) => {
    state.cablesOn[name] = e.target.checked;
    refresh();
  });
}

// Throttle slider drags to one recompute per animation frame, the way
// panning already is -- an unthrottled `input` was recomputing the field
// (and the mirror penalty) on every pixel of thumb movement.
let refreshPending = false;
function scheduleRefresh() {
  if (refreshPending) return;
  refreshPending = true;
  requestAnimationFrame(() => { refreshPending = false; refresh(); });
}

$("c-slider").addEventListener("input", (e) => {
  state.c = Number(e.target.value);
  $("c-value").textContent = `${state.c} m/s`;
  scheduleRefresh();
});
$("z-slider").addEventListener("input", (e) => {
  state.sz = Number(e.target.value);
  $("z-value").textContent = `${state.sz} m`;
  scheduleRefresh();
});
$("res-select").addEventListener("change", (e) => {
  state.res = Number(e.target.value);
  refresh();
});

$("c-value").textContent = `${state.c} m/s`;
$("z-value").textContent = `${state.sz} m`;

function drawZProfile() {
  const canvas = $("zprofile");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(140 * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, 140);
  if (!currentField) return;

  const prof = currentField.zprof;
  if (!prof || prof.length < 2) return;

  const lo = Math.min(...prof.map((q) => q.best));
  const hi = Math.max(...prof.map((q) => q.best));
  const span = hi - lo || 1;
  const pad = 26;
  const sx = (z) => pad + ((z - SOURCE_DEPTH_MIN) / SOURCE_DEPTH_SPAN) * (rect.width - 2 * pad);
  const sy = (v) => 12 + (1 - (v - lo) / span) * (140 - 2 * 12 - 14);

  const colors = themeColors();
  ctx.beginPath();
  prof.forEach((q, k) => (k === 0 ? ctx.moveTo(sx(q.z), sy(q.best)) : ctx.lineTo(sx(q.z), sy(q.best))));
  ctx.strokeStyle = colors.accent;
  ctx.lineWidth = 1.75;
  ctx.stroke();

  // Current assumed depth.
  ctx.beginPath();
  ctx.moveTo(sx(state.sz), 8);
  ctx.lineTo(sx(state.sz), 126);
  ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--ink-dim");
  ctx.setLineDash([3, 3]);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--ink-dim");
  ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
  const r = flatnessRatio(prof);
  ctx.fillText(`source depth −200…0 m · misfit varies ×${r.toFixed(3)}`, pad, 136);
}

layers.push(() => drawZProfile());

refresh();

const insetView = { theta: 0.7, phi: 0.9 };

(() => {
  const canvas = $("inset3d");
  let dragging = false, lastX = 0, lastY = 0;
  canvas.addEventListener("pointerdown", (e) => {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    canvas.style.cursor = "grabbing";
    canvas.setPointerCapture(e.pointerId);
  });
  const endDrag = (e) => {
    dragging = false;
    canvas.style.cursor = "grab";
    try { canvas.releasePointerCapture(e.pointerId); } catch { /* already released */ }
  };
  canvas.addEventListener("pointerup", endDrag);
  // Without this, losing capture without a pointerup (a tab switch mid-drag, a
  // touch interruption) leaves `dragging` true, so ordinary hover then keeps
  // rotating the view with no button held.
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("lostpointercapture", () => { dragging = false; canvas.style.cursor = "grab"; });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    insetView.theta += (e.clientX - lastX) * 0.012;
    insetView.phi = Math.min(1.5, Math.max(0.08, insetView.phi - (e.clientY - lastY) * 0.012));
    lastX = e.clientX; lastY = e.clientY;
    drawInset();
  });
})();

// The vertical axis is exaggerated: the depth range we search (~200 m) is two
// orders of magnitude smaller than the horizontal grid, so at true scale the
// basin would collapse to a flat sheet and the point being made — that it is a
// tube in z — would be invisible. This constant alone is no longer the whole
// story once z is normalised by its own half-span (see drawInset): the
// caption states the true vertical-to-horizontal exaggeration, which also
// folds in the half/zHalf disparity.
const INSET_Z_SCALE = 3;

function computeInset() {
  insetCache = null;
  if (!currentField) return;
  const p = picksFor(state.call, state.cablesOn);
  // Reuse the field's own xy grid, heavily subsampled. `gridForCall` no longer
  // exists, and a volume over a freshly-built grid would not correspond to what
  // the map is showing.
  const { xs, ys } = subsampleGrid(currentField.xs, currentField.ys, 24);
  const zs = depthSteps(15);
  const vol = misfitVolume({ xs, ys, zs, px: p.px, py: p.py, pz: p.pz, obsT: p.obsT, c: state.c });
  const points = isoPoints({ ...vol, xs, ys, zs, level: vol.best * 1.10 });
  insetCache = {
    points,
    centre: [(xs[0] + xs[xs.length - 1]) / 2, (ys[0] + ys[ys.length - 1]) / 2, (zs[0] + zs[zs.length - 1]) / 2],
    half: Math.max(xs[xs.length - 1] - xs[0], ys[ys.length - 1] - ys[0]) / 2 || 1,
    // z's own half-span, NOT the horizontal `half` -- the depth search range
    // is ~200 m against a horizontal extent of kilometres, so sharing `half`
    // flattened the point cloud to a sliver (measured 3.5% of the wireframe's
    // height) and made the wireframe box visually claim a depth extent it did
    // not have.
    zHalf: (zs[zs.length - 1] - zs[0]) / 2 || 1,
  };
}

function drawInset() {
  const canvas = $("inset3d");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const h = 220;
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, h);
  $("inset3d-caption").textContent = "";
  // Guard on currentField, not just insetCache: currentField is what
  // computeField nulls on its early returns, so this is the same invariant
  // the readout and moveout panels already honour. insetCache is nulled too
  // (see computeField), but guarding on it alone would silently draw a stale
  // cloud again if some future code path ever left it populated.
  if (!currentField || !insetCache) return;

  const colors = themeColors();
  const common = {
    theta: insetView.theta, phi: insetView.phi,
    centre: insetCache.centre, half: insetCache.half, zHalf: insetCache.zHalf,
    width: rect.width, height: h, zScale: INSET_Z_SCALE,
  };

  // Wireframe box first, in unit-cube coordinates. Scaled by `half`
  // horizontally and `zHalf` vertically so the box bounds the actual searched
  // volume instead of a cube claiming the same extent on every axis.
  const edges = boxEdges();
  const boxPts = new Float32Array(edges.length * 6);
  edges.forEach((seg, k) => boxPts.set(seg, k * 6));
  const scaledBox = new Float32Array(boxPts.length);
  for (let k = 0; k < boxPts.length; k += 3) {
    scaledBox[k] = insetCache.centre[0] + boxPts[k] * insetCache.half;
    scaledBox[k + 1] = insetCache.centre[1] + boxPts[k + 1] * insetCache.half;
    scaledBox[k + 2] = insetCache.centre[2] + boxPts[k + 2] * insetCache.zHalf;
  }
  const bp = project3d({ ...common, points: scaledBox });
  ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--line");
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let k = 0; k < bp.xy.length; k += 4) {
    ctx.moveTo(bp.xy[k], bp.xy[k + 1]);
    ctx.lineTo(bp.xy[k + 2], bp.xy[k + 3]);
  }
  ctx.stroke();

  if (insetCache.points.length === 0) {
    $("inset3d-caption").textContent = "No cells within 10% of the best fit.";
    return;
  }

  // Points, painted back to front so nearer points sit on top.
  const { xy, depth } = project3d({ ...common, points: insetCache.points });
  const order = Array.from({ length: depth.length }, (_, k) => k).sort((a, b) => depth[b] - depth[a]);
  // Hoisted: computing these inside the loop made the whole normalisation
  // O(n^2) with three full array scans per point, and this runs on every
  // pointermove during an orbit drag. A flat misfit surface can fill the whole
  // 24x24x15 volume (~8,640 points), which would be ~2e8 redundant operations
  // per frame -- defeating the reason orbiting is supposed to be cheap.
  let dmin = Infinity, dmax = -Infinity;
  for (let k = 0; k < depth.length; k++) {
    if (depth[k] < dmin) dmin = depth[k];
    if (depth[k] > dmax) dmax = depth[k];
  }
  const dspan = dmax - dmin || 1;
  for (const k of order) {
    const t = 0.35 + 0.65 * (1 - (depth[k] - dmin) / dspan);
    ctx.globalAlpha = t;
    ctx.fillStyle = colors.accent;
    ctx.beginPath();
    ctx.arc(xy[k * 2], xy[k * 2 + 1], 2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  const nPts = insetCache.points.length / 3;
  // With z normalised by its own half-span (zHalf), INSET_Z_SCALE alone is no
  // longer the displayed exaggeration -- the half/zHalf disparity (horizontal
  // extent vs. the ~200 m depth range) folds in too. This is the honest,
  // true vertical-to-horizontal exaggeration factor being drawn.
  const exaggeration = (insetCache.half / insetCache.zHalf) * INSET_Z_SCALE;
  const exagLabel = exaggeration >= 10 ? Math.round(exaggeration) : exaggeration.toFixed(1);
  $("inset3d-caption").textContent =
    `${nPts} cells within 10% of best · depth axis exaggerated ${exagLabel}× · drag to orbit`;
}

layers.push(() => drawInset());
refresh();
