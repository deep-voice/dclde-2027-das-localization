"""Reproduce the measured facts in section 2 of the sound-speed research design.

Two quantities, both load-bearing for that design:

1. Per-call span of seafloor depth across the channels that picked a call. This is
   the vertical aperture the whole depth-observability hypothesis depends on: the
   travel-time sensitivity to source depth, d t / d z_s = (z_s - z_c) / (c R), is
   non-zero only because channel depth varies along the cable.

2. The distribution of arrival-time residuals under the published constant-c model.
   This calibrates the simulator's noise model. Note the residuals contain pick
   error *plus* constant-c model error, so they are an upper bound on pick noise.

Reads only committed inputs. No network, no credentials.

    uv run python paper/scripts/recon_aperture_and_residuals.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dclde_localization.physics.forward import DEFAULT_C, best_t0, travel_time
from dclde_localization.physics.geometry import load_cables
from dclde_localization.physics.picks import attach_xyz, load_picks

GEOMETRY_DIR = "fixtures"
MIN_PICKS = 8
ASSUMED_SOURCE_Z = -30.0  # the baseline's own assumption, so residuals are comparable


def solve_l1(sub, c=DEFAULT_C, sz=ASSUMED_SOURCE_Z, half=60_000.0, n=161, rounds=4):
    """Zooming L1 grid search over (x, y) with t0 eliminated analytically.

    Deliberately a plain grid rather than a local optimiser: the misfit is bimodal
    for a single cable, and a gradient method would silently pick whichever lobe it
    started nearest. The zoom keeps the first pass wide enough to see both.
    """
    px, py, pz, t = (sub[k].to_numpy() for k in ("px", "py", "pz", "time"))
    cx, cy = px.mean(), py.mean()
    for _ in range(rounds):
        xs = np.linspace(cx - half, cx + half, n)
        ys = np.linspace(cy - half, cy + half, n)
        gx, gy = np.meshgrid(xs, ys)
        r = np.sqrt((gx[..., None] - px) ** 2 + (gy[..., None] - py) ** 2 + (sz - pz) ** 2)
        resid = t - r / c
        resid -= np.median(resid, axis=-1, keepdims=True)
        f = np.abs(resid).sum(-1)
        iy, ix = np.unravel_index(f.argmin(), f.shape)
        cx, cy = xs[ix], ys[iy]
        half /= 8.0
    tt = travel_time(cx, cy, sz, px, py, pz, c)
    return cx, cy, t - tt - best_t0(t, tt)


def main() -> None:
    cables = load_cables(GEOMETRY_DIR)
    for name, cab in cables.items():
        print(f"{name}: {len(cab.x)} channels, "
              f"chan_m {cab.chan_m[0]:.0f}..{cab.chan_m[-1]:.0f} m, "
              f"depth {cab.depth.min():.0f}..{cab.depth.max():.0f} m")

    picks = attach_xyz(load_picks(), cables)
    print(f"\npicks {len(picks)}, calls {picks['call'].nunique()}, "
          f"along-cable {picks['dist'].min():.0f}..{picks['dist'].max():.0f} m")

    # --- 1. vertical aperture ------------------------------------------------
    g = picks.groupby("call")
    span = (g["pz"].max() - g["pz"].min()).rename("zspan")
    ncab = g["cable"].nunique().rename("ncab")
    ap = pd.concat([span, ncab], axis=1)

    print(f"\nper-call channel-depth span (m), n = {len(ap)}")
    print(f"  median {ap.zspan.median():.0f}  p90 {ap.zspan.quantile(.9):.0f}  "
          f"max {ap.zspan.max():.0f}")
    for thr in (100, 200, 400):
        k = int((ap.zspan >= thr).sum())
        print(f"  span >= {thr:>3} m: {k}/{len(ap)} = {100 * k / len(ap):.0f}%")
    for n_cab, sub in ap.groupby("ncab"):
        label = "two-cable" if n_cab == 2 else "one-cable"
        print(f"  {label} calls (n={len(sub)}): median {sub.zspan.median():.0f} m, "
              f"max {sub.zspan.max():.0f} m")

    # --- 2. residuals under the published constant-c model -------------------
    per_call, pooled = [], []
    for call, sub in picks.groupby("call"):
        if len(sub) < MIN_PICKS:
            continue
        _, _, res = solve_l1(sub)
        pooled.append(res)
        per_call.append(np.std(res))

    r = np.concatenate(pooled)
    pc = np.asarray(per_call)
    print(f"\nresiduals at c={DEFAULT_C}, z_s={ASSUMED_SOURCE_Z} m: "
          f"{len(pc)} calls (>= {MIN_PICKS} picks), {len(r)} picks")
    print(f"  per-call std (s):  p10 {np.percentile(pc, 10):.3f}  "
          f"p50 {np.median(pc):.3f}  p90 {np.percentile(pc, 90):.3f}")
    print(f"  pooled |r| (s):    p50 {np.median(np.abs(r)):.3f}  "
          f"p90 {np.percentile(np.abs(r), 90):.3f}  max {np.abs(r).max():.1f}")
    print(f"  p50 |r| as range error at c={DEFAULT_C}: "
          f"{np.median(np.abs(r)) * DEFAULT_C:.0f} m")
    print("  (the max is mis-association, not measurement noise -- a Gaussian "
          "noise model would be wrong on its face)")


if __name__ == "__main__":
    main()
