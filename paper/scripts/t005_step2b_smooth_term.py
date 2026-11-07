"""T-005 step 2b: is the km-scale smooth residual model error or picker drift?

For every kept dense station of every call in one window, the arrival is measured
four ways on the same gauge-averaged SNR trace:

  xcorr     the step-2 cross-correlation pick
  peak      the envelope maximum within +/- 0.4 s of the xcorr pick
  centroid  first moment of the envelope above half its peak, same window
  onset     leading-edge crossing of 50% of the peak (searching back from the peak)

plus the envelope FWHM. Each set is solved under the same ConstantC(1480),
z = -30 m null, and its residual smoothed (~1 km running median).

Reading:
  * If the smooth residual is the same for all four definitions -- including the
    onset, which ignores how the envelope's tail broadens -- the delay is in the
    arrival itself: model error (or real propagation the model lacks).
  * If it shrinks for the onset and the peak-minus-onset difference tracks the
    FWHM, the "structure" is envelope-shape drift that moves the pick.

Usage: python paper/scripts/t005_step2b_smooth_term.py <window.nc> <step2_picks.csv> <out_prefix>
"""
import re
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

from dclde_localization.physics.dense_picks import PickerConfig, station_traces
from dclde_localization.physics.forward import travel_time
from dclde_localization.physics.geometry import load_cables
from dclde_localization.physics.picks import load_picks
from dclde_localization.physics.solve import solve
from dclde_localization.physics.sound_speed_models import ConstantC

C, ZS, SEARCH = 1480.0, -30.0, 0.4
REPO = Path(__file__).resolve().parents[2]


def rsig(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else np.nan


def solve_resid(cable, d, t):
    px, py, pz = cable.chan_to_xyz(d)
    sol = solve(px, py, pz, t, ConstantC(C), sz=ZS)
    if not sol.basins:
        return np.full(len(t), np.nan), (np.nan, np.nan)
    b = sol.best
    return t - (travel_time(b.x, b.y, ZS, px, py, pz, C) + b.t0), (b.x, b.y)


def smooth(r, n=21):
    return pd.Series(r).rolling(n, center=True, min_periods=5).median().to_numpy()


def measures(tr, t_ax, t_pick):
    """peak, centroid, onset (50% leading edge), FWHM around t_pick on one trace."""
    m = (t_ax >= t_pick - SEARCH) & (t_ax <= t_pick + SEARCH)
    idx = np.where(m)[0]
    if idx.size < 5:
        return (np.nan,) * 4
    k = idx[np.argmax(tr[idx])]
    pk = tr[k]
    if pk <= 0:
        return (np.nan,) * 4
    half = pk / 2
    a = k
    while a > 0 and tr[a - 1] >= half:
        a -= 1
    b = k
    while b < len(tr) - 1 and tr[b + 1] >= half:
        b += 1
    # onset: linear interpolation of the half-peak crossing between a-1 and a
    if a > 0 and tr[a] != tr[a - 1]:
        onset = t_ax[a - 1] + (half - tr[a - 1]) / (tr[a] - tr[a - 1]) * (t_ax[a] - t_ax[a - 1])
    else:
        onset = t_ax[a]
    w = tr[a:b + 1] - half
    cen = float((t_ax[a:b + 1] * w).sum() / w.sum()) if w.sum() > 0 else np.nan
    return t_ax[k], cen, onset, (b - a + 1) * (t_ax[1] - t_ax[0])


def main(nc_path, picks_csv, out_prefix, cfg=PickerConfig()):
    ds = netCDF4.Dataset(nc_path)
    cab_name = ds.getncattr("cablename").capitalize()
    dist, t_ax = ds["dist"][:].data, ds["time"][:].data
    cable = load_cables(REPO / "fixtures")[cab_name]
    dense = pd.read_csv(picks_csv)
    dense = dense[dense["keep"]]
    stamp = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", Path(nc_path).name).group(0)
    window = stamp[:10] + "_" + stamp[11:].replace("-", "_")
    ann = load_picks(REPO / "fixtures/das_calls.csv.gz")
    ann = ann[(ann.window == window) & (ann.cable == cab_name)]

    per_station, per_call = [], []
    for call, k in dense.groupby("call"):
        k = k.sort_values("dist").reset_index(drop=True)
        band = "SNR_hf" if (ann[ann.call == call].call_type == "HF").mean() >= 0.5 else "SNR_lf"
        sd = k.dist.to_numpy()
        i0 = max(int(np.searchsorted(dist, sd.min() - cfg.station_m)), 0)
        i1 = int(np.searchsorted(dist, sd.max() + cfg.station_m)) + 1
        j0 = max(int(np.searchsorted(t_ax, k.time.min() - 2.0)), 0)
        j1 = int(np.searchsorted(t_ax, k.time.max() + 2.0)) + 1
        var = ds[band]
        var.set_auto_mask(False)
        block = np.asarray(var[i0:i1, j0:j1], float)
        tr = station_traces(block, dist[i0:i1], sd, cfg.station_m)
        m = np.array([measures(tr[i], t_ax[j0:j1], t) for i, t in enumerate(k.time)])
        k["peak"], k["centroid"], k["onset"], k["fwhm"] = m.T
        k = k.dropna(subset=["peak", "centroid", "onset"]).reset_index(drop=True)
        if len(k) < 30:
            continue
        row = {"call": call, "n": len(k)}
        sm = {}
        for name in ("time", "peak", "centroid", "onset"):
            r, xy = solve_resid(cable, k.dist.to_numpy(), k[name].to_numpy())
            s = smooth(r)
            sm[name] = s
            label = "xcorr" if name == "time" else name
            k[f"res_{label}"] = r
            row[f"sig_{label}_ms"] = rsig(r) * 1e3
            row[f"smooth_{label}_ms"] = rsig(s) * 1e3
            row[f"x_{label}"], row[f"y_{label}"] = xy
        ok = np.isfinite(sm["time"]) & np.isfinite(sm["onset"])
        row["corr_smooth_xcorr_onset"] = float(np.corrcoef(sm["time"][ok], sm["onset"][ok])[0, 1]) if ok.sum() > 2 else np.nan
        row["corr_smooth_xcorr_peak"] = float(np.corrcoef(sm["time"][ok], sm["peak"][ok])[0, 1]) if ok.sum() > 2 else np.nan
        # does the pick's lag behind the onset track envelope width?
        lag = (k.time - k.onset).to_numpy()
        row["sd_xcorr_minus_onset_ms"] = rsig(lag) * 1e3
        row["corr_lag_fwhm"] = float(np.corrcoef(smooth(lag)[ok], smooth(k.fwhm.to_numpy())[ok])[0, 1]) if ok.sum() > 2 else np.nan
        row["smooth_lag_ms"] = rsig(smooth(lag)[ok]) * 1e3
        row["median_fwhm_ms"] = float(np.median(k.fwhm)) * 1e3
        per_call.append(row)
        k["call"] = call
        per_station.append(k)

    res = pd.DataFrame(per_call)
    pd.concat(per_station).to_csv(f"{out_prefix}_stations.csv", index=False)
    res.to_csv(f"{out_prefix}_calls.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["call", "n", "sig_xcorr_ms", "smooth_xcorr_ms", "smooth_peak_ms", "smooth_centroid_ms",
            "smooth_onset_ms", "corr_smooth_xcorr_onset", "smooth_lag_ms", "corr_lag_fwhm", "median_fwhm_ms"]
    print(res[cols].round(2).to_string(index=False))
    print("\nmedians:")
    print(res[cols[1:]].median().round(2).to_string())

    # Does the smooth residual repeat for calls from (nearly) the same source position?
    # Pairs with >= 3 km of shared cable; curves compared on a 200 m grid, medians removed.
    st = pd.concat(per_station)
    curves = {c: (g.sort_values("dist").dist.to_numpy(), smooth(g.sort_values("dist").res_xcorr.to_numpy()))
              for c, g in st.groupby("call")}
    pos = res.set_index("call")[["x_xcorr", "y_xcorr"]]
    pairs = []
    calls = list(res.call)
    for i, a in enumerate(calls):
        for b in calls[i + 1:]:
            (da, ra), (db, rb) = curves[a], curves[b]
            lo, hi = max(da.min(), db.min()), min(da.max(), db.max())
            if hi - lo < 3000:
                continue
            g = np.arange(lo, hi, 200.0)
            ia, ib = np.interp(g, da, ra), np.interp(g, db, rb)
            ok = np.isfinite(ia) & np.isfinite(ib)
            if ok.sum() < 10:
                continue
            ia, ib = ia[ok] - np.median(ia[ok]), ib[ok] - np.median(ib[ok])
            pairs.append({"a": a, "b": b,
                          "sep_km": float(np.hypot(*(pos.loc[a] - pos.loc[b]))) / 1e3,
                          "overlap_km": (hi - lo) / 1e3, "corr": float(np.corrcoef(ia, ib)[0, 1]),
                          "rms_diff_ms": float(np.sqrt(np.mean((ia - ib) ** 2))) * 1e3,
                          "rms_each_ms": float(np.sqrt((np.mean(ia**2) + np.mean(ib**2)) / 2)) * 1e3})
    pr = pd.DataFrame(pairs, columns=["a", "b", "sep_km", "overlap_km", "corr", "rms_diff_ms", "rms_each_ms"]).sort_values("sep_km")
    pr.to_csv(f"{out_prefix}_pairs.csv", index=False)
    near, far = pr[pr.sep_km < 1.0], pr[pr.sep_km >= 1.0]
    print(f"\npairs < 1 km apart: n={len(near)}, smooth-curve corr median {near["corr"].median():.2f} "
          f"(range {near["corr"].min():.2f}..{near["corr"].max():.2f})")
    print(f"pairs >= 1 km apart: n={len(far)}, corr median {far["corr"].median():.2f} "
          f"(range {far["corr"].min():.2f}..{far["corr"].max():.2f})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
