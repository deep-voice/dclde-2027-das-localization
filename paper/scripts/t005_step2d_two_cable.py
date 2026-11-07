"""T-016: two-cable solves with dense picks on paired North/South windows.

For every call annotated on both cables in a window, dense-pick each cable
(step-2 picker), then solve under the ConstantC(1480), z = -30 m null four ways:

  north   North dense picks alone
  south   South dense picks alone
  both    both cables' dense picks together (one t0, as in gate_table)
  annot   both cables' annotation picks (reference)

Reported per call: basin ambiguity (two basins > 500 m apart) for each solve,
the one-cable vs two-cable position shift, and the per-cable residual split
(smooth ~1 km running median vs the rest) under the one- and two-cable solves.

Question: does the ~61 ms smooth term of the one-cable solves survive when both
cables constrain the position?

Usage: python paper/scripts/t005_step2d_two_cable.py <north.nc> <south.nc> <out_prefix>
"""
import re
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

from dclde_localization.physics.dense_picks import PickerConfig, pick_call
from dclde_localization.physics.forward import travel_time
from dclde_localization.physics.geometry import load_cables
from dclde_localization.physics.picks import load_picks
from dclde_localization.physics.solve import solve
from dclde_localization.physics.sound_speed_models import ConstantC

C, ZS, FLIP_M = 1480.0, -30.0, 500.0
REPO = Path(__file__).resolve().parents[2]


def rsig(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else np.nan


def smooth(r, n=21):
    return pd.Series(r).rolling(n, center=True, min_periods=5).median().to_numpy()


def window_key(path):
    s = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", Path(path).name).group(0)
    return s[:10] + "_" + s[11:].replace("-", "_")


def dense_for_call(ds, sub, cfg):
    dist, t_ax = ds["dist"][:].data, ds["time"][:].data
    band = "SNR_hf" if (sub.call_type == "HF").mean() >= 0.5 else "SNR_lf"
    sd, st = sub.dist.to_numpy(float), sub.time.to_numpy(float)
    i0 = max(int(np.searchsorted(dist, sd.min() - cfg.station_m)), 0)
    i1 = int(np.searchsorted(dist, sd.max() + cfg.station_m)) + 1
    j0 = max(int(np.searchsorted(t_ax, st.min() - 2.0)), 0)
    j1 = int(np.searchsorted(t_ax, st.max() + 2.0)) + 1
    var = ds[band]
    var.set_auto_mask(False)
    block = np.asarray(var[i0:i1, j0:j1], float)
    dp = pick_call(block, dist[i0:i1], t_ax[j0:j1], sd, st, cfg)
    return dp[dp["keep"]].sort_values("dist")


def solve_set(sets):
    """sets: list of (cable, dist, time). Returns (solution, residual per set) or (None, None)."""
    xyz = [c.chan_to_xyz(d) for c, d, _ in sets]
    px = np.concatenate([a[0] for a in xyz])
    py = np.concatenate([a[1] for a in xyz])
    pz = np.concatenate([a[2] for a in xyz])
    t = np.concatenate([s[2] for s in sets])
    sol = solve(px, py, pz, t, ConstantC(C), sz=ZS)
    if not sol.basins:
        return None, None
    b = sol.best
    r = t - (travel_time(b.x, b.y, ZS, px, py, pz, C) + b.t0)
    out, i = [], 0
    for _, d, _ in sets:
        out.append(r[i:i + len(d)])
        i += len(d)
    return sol, out


def sep(sol):
    if sol is None:
        return np.inf
    return sol.separation_m if sol.separation_m is not None else 0.0


def main(north_nc, south_nc, out_prefix, cfg=PickerConfig()):
    w = window_key(north_nc)
    assert w == window_key(south_nc), "the two files must be the same window"
    dsn, dss = netCDF4.Dataset(north_nc), netCDF4.Dataset(south_nc)
    cables = load_cables(REPO / "fixtures")
    picks = load_picks(REPO / "fixtures/das_calls.csv.gz")
    picks = picks[picks.window == w]
    two = picks.groupby("call").cable.nunique()
    rows = []
    for call in two[two == 2].index:
        sub = picks[picks.call == call]
        an, as_ = sub[sub.cable == "North"], sub[sub.cable == "South"]
        dn, dsd = dense_for_call(dsn, an, cfg), dense_for_call(dss, as_, cfg)
        row = {"window": w, "call": call, "n_dense_north": len(dn), "n_dense_south": len(dsd),
               "n_annot_north": len(an), "n_annot_south": len(as_)}
        if len(dn) < 30 or len(dsd) < 30:
            rows.append(row)
            continue
        N = (cables["North"], dn.dist.to_numpy(), dn.time.to_numpy())
        S = (cables["South"], dsd.dist.to_numpy(), dsd.time.to_numpy())
        sn, rn = solve_set([N])
        ss, rs = solve_set([S])
        rn1 = rn[0] if rn is not None else None
        rs1 = rs[0] if rs is not None else None
        sb, rb = solve_set([N, S])
        sa, ra = solve_set([(cables["North"], an.dist.to_numpy(), an.time.to_numpy()),
                            (cables["South"], as_.dist.to_numpy(), as_.time.to_numpy())])
        row.update(sep_north_m=sep(sn), sep_south_m=sep(ss), sep_both_m=sep(sb), sep_annot_m=sep(sa))
        if sb is not None:
            bx, by = sb.best.x, sb.best.y
            for lab, s in (("north", sn), ("south", ss), ("annot", sa)):
                row[f"shift_{lab}_vs_both_m"] = (float(np.hypot(s.best.x - bx, s.best.y - by))
                                                 if s is not None else np.nan)
            row["x_both"], row["y_both"] = bx, by
            row["sig_both_ms"] = rsig(np.concatenate(rb)) * 1e3
            for lab, r1, r2 in (("north", rn1, rb[0]), ("south", rs1, rb[1])):
                if r1 is not None:
                    row[f"smooth_{lab}_one_ms"] = rsig(smooth(r1)) * 1e3
                row[f"smooth_{lab}_two_ms"] = rsig(smooth(r2)) * 1e3
                row[f"rough_{lab}_two_ms"] = rsig(r2 - smooth(r2)) * 1e3
                # offset between cables under the shared t0 (median residual per cable)
                row[f"median_{lab}_two_ms"] = float(np.median(r2)) * 1e3
        if ra is not None:
            row["sig_annot_ms"] = rsig(np.concatenate(ra)) * 1e3
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(f"{out_prefix}_calls.csv", index=False)
    pd.set_option("display.width", 250)
    cols = [c for c in ["call", "n_dense_north", "n_dense_south", "sep_north_m", "sep_south_m", "sep_both_m",
                        "shift_north_vs_both_m", "shift_south_vs_both_m", "smooth_north_one_ms",
                        "smooth_north_two_ms", "smooth_south_one_ms", "smooth_south_two_ms",
                        "median_north_two_ms", "median_south_two_ms", "sig_both_ms", "sig_annot_ms"] if c in res]
    print(f"window {w}: {len(res)} two-cable calls")
    print(res[cols].round(0).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
