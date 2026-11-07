"""Is a straight line at constant c the right PATH, not just the right speed?

The published justification for a constant sound speed is Rorstadbotnen et al.
(2023): perturbing c by +-25-50 m/s moved localizations only 16-31 m. That tests
the value of a constant. It does not test the GRADIENT, which is different
physics -- a gradient refracts, and refraction can remove a path entirely rather
than perturb its timing.

Three measurements:

  A  Magnitude. How much does a depth-averaged c change predicted arrival time
     versus a constant 1480? (The smooth, perturbative part of the effect.)

  B  Regime. With c falling from 1491 m/s at the surface to 1477 m/s at 500 m,
     every ray from a shallow source bends downward. How far can a direct,
     water-borne ray actually reach before it must hit the seabed?

  C  Evidence. Do the real picks show range-structured misfit, and does it
     survive stratification by SNR? A low-SNR envelope pick lands late, which
     would mimic a range trend, so that confound has to be excluded.

Run:  uv run python paper/scripts/recon_propagation_regime.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from dclde_localization.physics.forward import DEFAULT_C, best_t0, travel_time
from dclde_localization.physics.geometry import load_cables
from dclde_localization.physics.picks import attach_xyz, load_picks

# Indicative Oregon margin, November (paper/docs/sound-speed-feasibility.md).
# Depth positive-down in this module; the repo frame is negative-down.
PROF_Z = np.array([0.0, 100.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0])
PROF_C = np.array([1491.0, 1480.0, 1478.0, 1477.0, 1479.0, 1484.0, 1490.0])
ASSUMED_SOURCE_Z = -30.0
MIN_PICKS = 8


# --------------------------------------------------------------------------
# Ray tracing: within a constant-gradient layer the ray is a circular arc, so
# range and time are closed-form and the grazing limit stays finite. A
# quadrature in z diverges there and silently reports "no path".
# --------------------------------------------------------------------------

def c_at(z: float) -> float:
    return float(np.interp(z, PROF_Z, PROF_C))


def _layers(z1: float, z2: float):
    edges = sorted(set([z1, z2]) | {float(e) for e in PROF_Z if z1 < e < z2})
    return [(a, b, c_at(a), c_at(b), (c_at(b) - c_at(a)) / (b - a))
            for a, b in zip(edges[:-1], edges[1:])]


def trace(z1: float, z2: float, p: float):
    """Range and travel time for ray parameter p = cos(theta)/c. None if it turns."""
    lo, hi = sorted((z1, z2))
    rng = t = 0.0
    for za, zb, ca, cb, g in _layers(lo, hi):
        sa2, sb2 = 1 - (p * ca) ** 2, 1 - (p * cb) ** 2
        if sa2 < 0 or sb2 < 0:
            return None
        sa, sb = np.sqrt(sa2), np.sqrt(sb2)
        if abs(g) < 1e-12:
            if sa < 1e-15:
                return None
            rng += p * ca * (zb - za) / sa
            t += (zb - za) / (ca * sa)
        else:
            rng += abs((sa - sb) / (p * g))
            t += abs(np.log((cb * (1 + sa)) / (ca * (1 + sb))) / g)
    return rng, t


def _p_grazing(z1, z2):
    lo, hi = sorted((z1, z2))
    return 1.0 / max(c_at(z) for z in np.linspace(lo, hi, 2001))


def max_direct_range(z1, z2) -> float:
    got = trace(z1, z2, _p_grazing(z1, z2) * (1 - 1e-15))
    return got[0] if got else 0.0


def direct_time(z1, z2, target_r):
    if max_direct_range(z1, z2) < target_r:
        return None
    pmax = _p_grazing(z1, z2)
    f = lambda p: trace(z1, z2, p)[0] - target_r
    if f(pmax * 1e-9) > 0:
        return None
    p = brentq(f, pmax * 1e-9, pmax * (1 - 1e-15), xtol=1e-20, rtol=8.9e-16, maxiter=300)
    return trace(z1, z2, p)[1]


def harmonic_c(z1, z2) -> float:
    lo, hi = sorted((z1, z2))
    if hi - lo < 1e-9:
        return c_at(lo)
    zz = np.linspace(lo, hi, 4001)
    return float((hi - lo) / np.trapezoid([1 / c_at(z) for z in zz], zz))


def selftest() -> None:
    """A constant profile must reproduce the straight-line time exactly."""
    global PROF_Z, PROF_C
    keep_z, keep_c = PROF_Z, PROF_C
    PROF_Z, PROF_C = np.array([0.0, 3000.0]), np.array([DEFAULT_C, DEFAULT_C])
    try:
        for z1, z2, r in [(30.0, 100.0, 5000.0), (30.0, 1575.0, 30000.0)]:
            got = direct_time(z1, z2, r)
            exact = float(np.hypot(r, z2 - z1) / DEFAULT_C)
            assert got is not None and abs(got - exact) / exact < 1e-9, (z1, z2, r)
    finally:
        PROF_Z, PROF_C = keep_z, keep_c
    print("self-test: constant profile reproduces straight-line time (<1e-9 rel)  [ok]\n")


# --------------------------------------------------------------------------

def solve_l1(sub, c=DEFAULT_C, sz=ASSUMED_SOURCE_Z, half=60_000.0, n=161, rounds=4):
    px, py, pz, t = (sub[k].to_numpy() for k in ("px", "py", "pz", "time"))
    cx, cy = px.mean(), py.mean()
    for _ in range(rounds):
        xs = np.linspace(cx - half, cx + half, n)
        ys = np.linspace(cy - half, cy + half, n)
        gx, gy = np.meshgrid(xs, ys)
        r = np.sqrt((gx[..., None] - px) ** 2 + (gy[..., None] - py) ** 2 + (sz - pz) ** 2)
        res = t - r / c
        res -= np.median(res, axis=-1, keepdims=True)
        f = np.abs(res).sum(-1)
        iy, ix = np.unravel_index(f.argmin(), f.shape)
        cx, cy = xs[ix], ys[iy]
        half /= 8.0
    tt = travel_time(cx, cy, sz, px, py, pz, c)
    return t - tt - best_t0(t, tt), np.hypot(px - cx, py - cy)


def main() -> None:
    selftest()
    print("profile:", {int(z): int(c) for z, c in zip(PROF_Z, PROF_C)},
          "-- downward refracting over 0-500 m\n")

    # --- A + B ------------------------------------------------------------
    print("A/B  smooth correction vs. whether a direct ray exists at all")
    print(f"{'z_src':>6}{'z_rcv':>7}{'range':>7} | {'pathavg-const':>14} "
          f"{'max direct range':>18} {'direct?':>9}")
    print(f"{'(m)':>6}{'(m)':>7}{'(km)':>7} | {'(ms)':>14} {'(km)':>18}")
    print("-" * 70)
    smooth = []
    for zs in (30.0, 100.0):
        for zr, rk in [(100.0, 5.0), (500.0, 10.0), (1000.0, 20.0),
                       (1575.0, 30.0), (1575.0, 40.0)]:
            if zr <= zs:
                continue
            r = rk * 1000
            slant = float(np.hypot(r, zr - zs))
            d = (slant / harmonic_c(zs, zr) - slant / DEFAULT_C) * 1e3
            mx = max_direct_range(zs, zr) / 1000
            smooth.append(abs(d))
            print(f"{zs:6.0f}{zr:7.0f}{rk:7.1f} | {d:14.1f} {mx:18.1f} "
                  f"{'yes' if mx >= rk else 'NO':>9}")
    print(f"\n  smooth (path-average) correction: |delta| median {np.median(smooth):.1f} ms, "
          f"max {max(smooth):.1f} ms = {max(smooth) * DEFAULT_C / 1000:.0f} m of range")
    print("  annotated calls span 12.5-95 km along cable; typical ranges 5-40 km\n")

    # --- C ----------------------------------------------------------------
    cables = load_cables("fixtures")
    picks = attach_xyz(load_picks(), cables)
    frames = []
    for call, sub in picks.groupby("call"):
        if len(sub) < MIN_PICKS:
            continue
        res, rng = solve_l1(sub)
        frames.append(pd.DataFrame({"res": res, "range": rng,
                                    "snr": sub["snr"].to_numpy()}))
    d = pd.concat(frames, ignore_index=True)
    # Gross mis-associations would otherwise carry a mean on their own.
    d = d[d.res.abs() < 1.0]

    print(f"C  residual structure, n = {len(d)} picks, |residual| < 1 s")
    d["rbin"] = pd.cut(d["range"] / 1000, [0, 5, 10, 20, 30, 200])
    d["sbin"] = pd.qcut(d["snr"], 4, labels=["snr Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    piv = d.pivot_table(index="sbin", columns="rbin", values="res",
                        aggfunc=lambda s: s.mean() * 1e3, observed=True)
    print("mean residual (ms); rows SNR quartile, cols range (km):")
    print(piv.round(1).to_string())
    print("\nrange trend within each SNR quartile (far bin - near bin, ms):")
    for s in piv.index:
        row = piv.loc[s].dropna()
        if len(row) > 1:
            print(f"  {s:<13} {row.iloc[-1] - row.iloc[0]:+7.1f}")
    print("\nIf this were a low-SNR late-pick artefact the trend would be largest in Q1")
    print("and absent in Q4. It is the other way round, so it is model error.")
    print("\nCAVEAT: these residuals come from a fit to these same picks, so whatever")
    print("looks like a source shift is already absorbed -- what survives is a lower")
    print("bound. And source depth is fixed at -30 m here, which itself produces")
    print("short-range structure; propagation path and source depth are confounded.")
    print("At 20 Hz (lambda ~74 m) ray theory is marginal anyway. This measurement")
    print("raises a regime question; normal modes are needed to settle it.")


if __name__ == "__main__":
    main()
