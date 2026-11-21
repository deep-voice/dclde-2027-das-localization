"""T-023: the baseline find_peaks picker against the T-005 dense picker, on the same calls.

Baseline picker (Goestchel_JASA_2025 main_section5.py; parameters from the file names in
Goestchel_JASA_2025b/data/detections): on every channel of the denoised SNR image,
scipy.signal.find_peaks(row, distance=IPI_S * fs, height=TH_DB[cable]). On the 2021-11-04
02:00:02 window this reproduces his published HF picks exactly (101239/101239 North,
110907/110907 South), so it is his picker, not an approximation.

For every call annotated on both cables in the window pairs on disk:
1. Station grid every 51 m inside the annotated span (as in pick_call); seed = linear
   interpolation of the annotation picks.
2. find_peaks picks: on each channel within +/- 25.5 m of a station, the peak nearest to the
   seed within +/- 0.4 s; the station time is the median over those channels, kept when at
   least half of the channels have a peak.
3. Dense picks: the T-005 picker (dense_for_call), kept stations only.
4. Per cable: coverage, neighbour noise (robust sigma of station-to-station differences / sqrt 2),
   per-channel find_peaks noise, agreement fp - dense (median removed), and whether the
   annotation picks sit on find_peaks peaks.
5. Two-cable solves (ConstantC 1480, z = -30 m): find_peaks stations, dense stations, and both
   restricted to their common stations. Residual sigma (total and smooth) and position offsets.

Usage: python paper/scripts/t023_findpeaks_vs_dense.py <samples_dir> <out_prefix>
"""
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import scipy.signal as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t005_step2d_two_cable import dense_for_call, rsig, sep, smooth, solve_set, window_key  # noqa: E402

from dclde_localization.physics.dense_picks import PickerConfig
from dclde_localization.physics.geometry import load_cables
from dclde_localization.physics.picks import load_picks

IPI_S = 3.0
TH_DB = {"North": 4.0, "South": 5.0}
MIN_STATIONS = 30
REPO = Path(__file__).resolve().parents[2]


def fp_rows(ds, band, i0, i1, cable):
    """find_peaks on full rows i0:i1 (full rows, so edge effects match the baseline)."""
    var = ds[band]
    var.set_auto_mask(False)
    block = np.nan_to_num(np.asarray(var[i0:i1, :], float))
    fs = 1.0 / float(np.median(np.diff(ds["time"][:20].data)))
    return [sp.find_peaks(r, distance=int(IPI_S * fs), height=TH_DB[cable])[0] for r in block]


def fp_for_call(ds, sub, cable, cfg):
    dist, t_ax = ds["dist"][:].data, ds["time"][:].data
    band = "SNR_hf" if (sub.call_type == "HF").mean() >= 0.5 else "SNR_lf"
    o = np.argsort(sub.dist.to_numpy(float))
    sd, st = sub.dist.to_numpy(float)[o], sub.time.to_numpy(float)[o]
    if len(sd) < 2 or sd[-1] - sd[0] < 2 * cfg.station_m:
        return pd.DataFrame(columns=["dist", "time"]), {}
    i0 = max(int(np.searchsorted(dist, sd[0] - cfg.station_m)), 0)
    i1 = int(np.searchsorted(dist, sd[-1] + cfg.station_m)) + 1
    peaks = fp_rows(ds, band, i0, i1, cable)
    dd = dist[i0:i1]
    # per-channel nearest peak to the seed
    seed_ch = np.interp(dd, sd, st)
    t_ch = np.full(len(dd), np.nan)
    for k, pk in enumerate(peaks):
        if len(pk):
            tt = t_ax[pk]
            j = int(np.argmin(np.abs(tt - seed_ch[k])))
            if abs(tt[j] - seed_ch[k]) <= cfg.max_shift_s:
                t_ch[k] = tt[j]
    stations = np.arange(sd[0], sd[-1] + 1e-9, cfg.station_m)
    rows = []
    for s in stations:
        m = np.abs(dd - s) <= cfg.station_m / 2
        v = t_ch[m]
        if m.sum() and np.isfinite(v).sum() >= 0.5 * m.sum():
            rows.append((s, float(np.nanmedian(v))))
    # do the annotation picks sit on find_peaks peaks? (nearest channel, within one sample)
    on = []
    for d_, t_ in zip(sd, st):
        k = int(np.argmin(np.abs(dd - d_)))
        pk = peaks[k]
        on.append(bool(len(pk)) and float(np.min(np.abs(t_ax[pk] - t_))) <= 0.0051)
    ok = np.isfinite(t_ch)
    ch_noise = rsig(np.diff(t_ch[ok])) / np.sqrt(2) if ok.sum() > 10 else np.nan
    extra = {"n_grid": len(stations), "annot_on_fp": float(np.mean(on)), "ch_noise_ms": ch_noise * 1e3,
             "ch_cover": float(ok.mean())}
    return pd.DataFrame(rows, columns=["dist", "time"]), extra


def nb_noise(df):
    """Neighbour noise on consecutive stations only."""
    d, t = df.dist.to_numpy(), df.time.to_numpy()
    if len(d) < 10:
        return np.nan
    step = np.diff(d)
    good = np.abs(step - np.median(step)) < 1.0
    return rsig(np.diff(t)[good]) / np.sqrt(2) * 1e3


def solve_pair(cables, dn, ds_):
    sets = [(cables["North"], dn.dist.to_numpy(), dn.time.to_numpy()),
            (cables["South"], ds_.dist.to_numpy(), ds_.time.to_numpy())]
    sol, res = solve_set(sets)
    if sol is None or sep(sol) > 500.0:
        return None
    r = np.concatenate(res)
    sm = [smooth(x - np.median(x)) for x in res]
    rough = np.concatenate([x - np.median(x) - s for x, s in zip(res, sm)])
    return {"x": sol.best.x, "y": sol.best.y, "sigma_ms": rsig(r) * 1e3,
            "smooth_ms": rsig(np.concatenate(sm)) * 1e3, "rough_ms": rsig(rough) * 1e3,
            "ns_offset_ms": float(np.median(res[0]) - np.median(res[1])) * 1e3}


def main(samples: Path, out_prefix: str, cfg=PickerConfig()):
    cables = load_cables(REPO / "fixtures")
    picks = load_picks(REPO / "fixtures/das_calls.csv.gz")
    rows = []
    for north in sorted(samples.rglob("Denoised_SNR_North_*.nc")):
        south = north.with_name(north.name.replace("North", "South"))
        if not south.exists():
            continue
        w = window_key(north)
        p = picks[picks.window == w]
        two = p.groupby("call").cable.nunique()
        dsets = {"North": netCDF4.Dataset(north), "South": netCDF4.Dataset(south)}
        for call in two[two == 2].index:
            sub = p[p.call == call]
            row = {"call": call, "window": w, "batch": int(sub.batch.iloc[0]),
                   "call_type": sub.call_type.mode().iloc[0]}
            per = {}
            for cab in ("North", "South"):
                s = sub[sub.cable == cab]
                dense = dense_for_call(dsets[cab], s, cfg)
                fp, extra = fp_for_call(dsets[cab], s, cab, cfg)
                per[cab] = (dense, fp)
                c = cab.lower()
                row[f"{c}_n_annot"] = len(s)
                row[f"{c}_n_dense"] = len(dense)
                row[f"{c}_n_fp"] = len(fp)
                row.update({f"{c}_{k}": v for k, v in extra.items()})
                row[f"{c}_noise_dense_ms"] = nb_noise(dense)
                row[f"{c}_noise_fp_ms"] = nb_noise(fp)
                m = pd.merge(dense[["dist", "time"]], fp, on="dist", suffixes=("_d", "_f"))
                row[f"{c}_n_common"] = len(m)
                if len(m) >= 10:
                    diff = m.time_f - m.time_d
                    row[f"{c}_fp_minus_dense_median_ms"] = float(np.median(diff)) * 1e3
                    row[f"{c}_fp_vs_dense_rsig_ms"] = rsig(diff) * 1e3
                    row[f"{c}_noise_dense_common_ms"] = nb_noise(m.rename(columns={"time_d": "time"}))
                    row[f"{c}_noise_fp_common_ms"] = nb_noise(m.rename(columns={"time_f": "time"}))
            (dn, fn), (dsx, fsx) = per["North"], per["South"]
            if min(len(dn), len(dsx)) >= MIN_STATIONS and min(len(fn), len(fsx)) >= MIN_STATIONS:
                sd_ = solve_pair(cables, dn, dsx)
                sf_ = solve_pair(cables, fn, fsx)
                cn = pd.merge(dn[["dist", "time"]], fn, on="dist", suffixes=("", "_f"))
                cs = pd.merge(dsx[["dist", "time"]], fsx, on="dist", suffixes=("", "_f"))
                sdc = sfc = None
                if min(len(cn), len(cs)) >= MIN_STATIONS:
                    sdc = solve_pair(cables, cn[["dist", "time"]], cs[["dist", "time"]])
                    sfc = solve_pair(cables, cn[["dist", "time_f"]].rename(columns={"time_f": "time"}),
                                     cs[["dist", "time_f"]].rename(columns={"time_f": "time"}))
                for tag, s in (("dense", sd_), ("fp", sf_), ("dense_common", sdc), ("fp_common", sfc)):
                    if s:
                        row.update({f"solve_{tag}_{k}": v for k, v in s.items()})
                if sdc and sfc:
                    row["common_pos_diff_m"] = float(np.hypot(sdc["x"] - sfc["x"], sdc["y"] - sfc["y"]))
                if sd_ and sf_:
                    row["pos_diff_m"] = float(np.hypot(sd_["x"] - sf_["x"], sd_["y"] - sf_["y"]))
            rows.append(row)
        print(f"{w}: {len(two[two == 2])} two-cable calls", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{out_prefix}_calls.csv", index=False)
    summarize(df)


def summarize(df):
    print(f"\ncalls: {len(df)} (HF {(df.call_type == 'HF').sum()}, LF {(df.call_type == 'LF').sum()})")
    for c in ("north", "south"):
        print(f"\n[{c}] medians")
        print(f"  stations: grid {df[f'{c}_n_grid'].median():.0f}, dense kept {df[f'{c}_n_dense'].median():.0f}, "
              f"find_peaks {df[f'{c}_n_fp'].median():.0f}, common {df[f'{c}_n_common'].median():.0f}")
        print(f"  per-channel find_peaks coverage {df[f'{c}_ch_cover'].median():.2f}")
        print(f"  annotation picks on a find_peaks peak: {df[f'{c}_annot_on_fp'].median():.2f} "
              f"(mean {df[f'{c}_annot_on_fp'].mean():.2f})")
        print(f"  neighbour noise (51 m; correlated by the ~46 m SNR blur), common stations: dense {df[f'{c}_noise_dense_common_ms'].median():.1f} ms, "
              f"find_peaks {df[f'{c}_noise_fp_common_ms'].median():.1f} ms "
              f"(dense lower in {(df[f'{c}_noise_dense_common_ms'] < df[f'{c}_noise_fp_common_ms']).sum()}"
              f"/{df[f'{c}_noise_fp_common_ms'].notna().sum()})")
        print(f"  fp - dense: median offset {df[f'{c}_fp_minus_dense_median_ms'].median():.1f} ms, "
              f"robust sigma {df[f'{c}_fp_vs_dense_rsig_ms'].median():.1f} ms")
    for tag in ("dense", "fp", "dense_common", "fp_common"):
        col = f"solve_{tag}_sigma_ms"
        if col in df:
            ok = df[col].notna()
            print(f"solve {tag:13s}: n={ok.sum():3d}  sigma {df.loc[ok, col].median():.1f} ms, "
                  f"smooth {df.loc[ok, f'solve_{tag}_smooth_ms'].median():.1f} ms, "
                  f"rough {df.loc[ok, f'solve_{tag}_rough_ms'].median():.1f} ms, "
                  f"N-S offset median {df.loc[ok, f'solve_{tag}_ns_offset_ms'].median():+.1f} ms "
                  f"(|.| {df.loc[ok, f'solve_{tag}_ns_offset_ms'].abs().median():.1f})")
    if "solve_fp_common_rough_ms" in df:
        ok = df.solve_fp_common_rough_ms.notna() & df.solve_dense_common_rough_ms.notna()
        print(f"common stations: dense rough lower in {(df.solve_dense_common_rough_ms < df.solve_fp_common_rough_ms)[ok].sum()}/{ok.sum()}, "
              f"dense total sigma lower in {(df.solve_dense_common_sigma_ms < df.solve_fp_common_sigma_ms)[ok].sum()}/{ok.sum()}")
    for col in ("pos_diff_m", "common_pos_diff_m"):
        if col in df:
            print(f"{col}: median {df[col].median():.0f} m, 90th pct {df[col].quantile(0.9):.0f} m (n={df[col].notna().sum()})")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
