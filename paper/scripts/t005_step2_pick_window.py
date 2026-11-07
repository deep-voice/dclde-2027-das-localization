"""T-005 step 2: dense-pick every annotated call in one .nc window and compare residuals.

For each call on this window's cable, three pick sets are solved under the same
ConstantC(1480), z = -30 m null (one cable, since one window holds one cable):

  annot   the annotation picks
  same    dense picks at the annotation stations (nearest kept station <= 30 m)
  dense   all kept dense stations (one per gauge length)

`annot` vs `same` holds pick count and geometry fixed, so any change in residual
sigma is due to picking alone.

Usage: python paper/scripts/t005_step2_pick_window.py <window.nc> <out_prefix>
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

C, ZS, MATCH_M = 1480.0, -30.0, 30.0
REPO = Path(__file__).resolve().parents[2]


def rsig(x):
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else np.nan


def residuals(cable, d, t):
    px, py, pz = cable.chan_to_xyz(d)
    sol = solve(px, py, pz, t, ConstantC(C), sz=ZS)
    if not sol.basins:  # no usable basin: report NaN rather than drop the call
        return np.full(len(t), np.nan), (np.nan, np.nan)
    b = sol.best
    r = t - (travel_time(b.x, b.y, ZS, px, py, pz, C) + b.t0)
    return r, (b.x, b.y)


def main(nc_path: Path, out_prefix: Path, cfg=PickerConfig()):
    ds = netCDF4.Dataset(nc_path)
    cab_name = ds.getncattr("cablename").capitalize()
    dist = ds["dist"][:].data
    t_ax = ds["time"][:].data
    stamp = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", nc_path.name).group(0)
    window = stamp[:10] + "_" + stamp[11:].replace("-", "_")

    picks = load_picks(REPO / "fixtures/das_calls.csv.gz")
    picks = picks[(picks.window == window) & (picks.cable == cab_name)]
    cable = load_cables(REPO / "fixtures")[cab_name]

    all_dense, rows = [], []
    for call, sub in picks.groupby("call"):
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
        dp["call"] = call
        all_dense.append(dp)
        k = dp[dp.keep]

        # nearest kept dense station to each annotation pick
        idx = np.searchsorted(k.dist.to_numpy(), sd).clip(1, max(len(k) - 1, 1))
        kd = k.dist.to_numpy()
        near = np.where(np.abs(kd[idx - 1] - sd) < np.abs(kd[idx.clip(max=len(kd) - 1)] - sd),
                        idx - 1, idx.clip(max=len(kd) - 1)) if len(kd) else idx
        ok = np.abs(kd[near] - sd) <= MATCH_M if len(kd) else np.zeros(len(sd), bool)
        row = {"call": call, "band": band, "n_annot": len(sd), "n_dense_kept": len(k),
               "n_dense_total": len(dp), "n_same": int(ok.sum()),
               "cc_median": float(k.cc.median()) if len(k) else np.nan}
        if ok.sum() >= 6:
            same_t = k.time.to_numpy()[near[ok]]
            r_a, xy_a = residuals(cable, sd[ok], st[ok])
            r_s, xy_s = residuals(cable, sd[ok], same_t)
            row.update(sig_annot_ms=rsig(r_a) * 1e3, sig_same_ms=rsig(r_s) * 1e3,
                       dense_minus_annot_rsig_ms=rsig(same_t - st[ok]) * 1e3,
                       dense_minus_annot_median_ms=float(np.median(same_t - st[ok])) * 1e3,
                       xy_shift_same_m=float(np.hypot(xy_a[0] - xy_s[0], xy_a[1] - xy_s[1])))
        if len(k) >= 6:
            r_d, xy_d = residuals(cable, kd, k.time.to_numpy())  # kd is ascending
            # split the dense residual: smooth (~1 km running median, 21 stations) vs rough,
            # and a pick-noise estimate from neighbouring stations (<60 m apart). Adjacent
            # stations share part of the 51 m gauge, so the neighbour estimate is a lower bound.
            smooth = pd.Series(r_d).rolling(21, center=True, min_periods=5).median().to_numpy()
            adj = np.diff(kd) < 60.0
            row.update(sig_dense_ms=rsig(r_d) * 1e3,
                       noise_neighbour_ms=rsig(np.diff(r_d)[adj]) / np.sqrt(2) * 1e3,
                       sig_smooth_1km_ms=rsig(smooth[np.isfinite(smooth)]) * 1e3,
                       sig_rough_ms=rsig((r_d - smooth)[np.isfinite(smooth)]) * 1e3)
        rows.append(row)

    res = pd.DataFrame(rows)
    pd.concat(all_dense).to_csv(f"{out_prefix}_picks.csv", index=False)
    res.to_csv(f"{out_prefix}_calls.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"{nc_path.name}: cable {cab_name}, {len(res)} calls, {len(picks)} annotation picks")
    print(res.round(1).to_string(index=False))
    good = res.dropna(subset=["sig_same_ms"])
    print(f"\nmedian over {len(good)} calls: sig_annot {good.sig_annot_ms.median():.1f} ms -> "
          f"sig_same {good.sig_same_ms.median():.1f} ms (same stations); "
          f"sig_dense {res.sig_dense_ms.median():.1f} ms (all kept stations)")
    print(f"dense residual split, median over calls: neighbour noise {res.noise_neighbour_ms.median():.1f} ms, "
          f"smooth (1 km) {res.sig_smooth_1km_ms.median():.1f} ms, rough {res.sig_rough_ms.median():.1f} ms")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
