"""T-017: can a source-position-dependent travel-time correction be learned from repeat calls?

Steps
1. Every two-cable call in the downloaded window pairs is dense-picked on both
   cables (step-2 picker) and solved on both cables under the ConstantC(1480),
   z = -30 m null. Calls are dropped if unsolved, basin-ambiguous (> 500 m), or
   gross misfits (overall residual sigma > GROSS_MS, e.g. a mis-associated track).
2. Calls in the same batch are clustered by two-cable position (single linkage,
   CLUSTER_M). Repeat calls from one whale should fall in one cluster.
3. Leave one call out. For each cable, the correction is the median of the
   other cluster members' smooth residual curves (per-cable median removed,
   ~1 km running median), interpolated onto the held-out call's stations.
   Only stations covered by at least one other member are scored.
4. Scores on the held-out call, per cable, offset removed, covered stations only:
     smooth sigma and total sigma, before vs after subtracting the correction;
   plus a re-solve of the held-out call with the corrected times (both cables).
5. Control: the same, but the correction comes from the nearest other cluster
   in the same batch at >= CONTROL_MIN_KM. If the term is position-dependent,
   the own-cluster correction helps and the control does not.

With --leave-window-out, the own-cluster correction also excludes every call from the held-out
call's window (a stricter test against any window-specific artifact).

Usage: python paper/scripts/t017_position_correction.py <samples_dir> <out_prefix> [--leave-window-out]
"""
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t005_step2d_two_cable import dense_for_call, rsig, sep, smooth, solve_set, window_key  # noqa: E402

from dclde_localization.physics.dense_picks import PickerConfig
from dclde_localization.physics.geometry import load_cables
from dclde_localization.physics.picks import load_picks

GROSS_MS = 500.0
CLUSTER_M = 1000.0
CONTROL_MIN_KM = 3.0
MIN_COVER = 30  # stations
REPO = Path(__file__).resolve().parents[2]


def collect(samples: Path, cfg=PickerConfig()):
    cables = load_cables(REPO / "fixtures")
    picks = load_picks(REPO / "fixtures/das_calls.csv.gz")
    calls, dropped = [], []
    for north in sorted(samples.rglob("Denoised_SNR_North_*.nc")):
        south = north.with_name(north.name.replace("North", "South"))
        if not south.exists():
            continue
        w = window_key(north)
        p = picks[picks.window == w]
        two = p.groupby("call").cable.nunique()
        dsn, dss = netCDF4.Dataset(north), netCDF4.Dataset(south)
        for call in two[two == 2].index:
            sub = p[p.call == call]
            dn = dense_for_call(dsn, sub[sub.cable == "North"], cfg)
            ds_ = dense_for_call(dss, sub[sub.cable == "South"], cfg)
            if len(dn) < MIN_COVER or len(ds_) < MIN_COVER:
                dropped.append((call, "short track"))
                continue
            sets = [(cables["North"], dn.dist.to_numpy(), dn.time.to_numpy()),
                    (cables["South"], ds_.dist.to_numpy(), ds_.time.to_numpy())]
            sol, res = solve_set(sets)
            if sol is None or sep(sol) > 500.0:
                dropped.append((call, "unsolved/ambiguous"))
                continue
            if rsig(np.concatenate(res)) * 1e3 > GROSS_MS:
                dropped.append((call, "gross misfit"))
                continue
            calls.append({"call": call, "window": w, "batch": int(sub.batch.iloc[0]),
                          "x": sol.best.x, "y": sol.best.y, "sets": sets, "res": res})
    return calls, dropped


def cluster(calls):
    lab = -np.ones(len(calls), int)
    k = 0
    for i in range(len(calls)):
        if lab[i] >= 0:
            continue
        lab[i] = k
        stack = [i]
        while stack:
            a = stack.pop()
            for b in range(len(calls)):
                if lab[b] < 0 and calls[b]["batch"] == calls[a]["batch"] and \
                        np.hypot(calls[a]["x"] - calls[b]["x"], calls[a]["y"] - calls[b]["y"]) <= CLUSTER_M:
                    lab[b] = k
                    stack.append(b)
        k += 1
    return lab


def curve(call, ci):
    d = call["sets"][ci][1]
    r = call["res"][ci] - np.median(call["res"][ci])
    return d, smooth(r)


def correction(members, ci, d_target):
    """Median of the members' smooth curves at d_target; NaN where none covers."""
    vals = []
    for m in members:
        d, s = curve(m, ci)
        ok = np.isfinite(s)
        if ok.sum() < 5:
            continue
        v = np.interp(d_target, d[ok], s[ok], left=np.nan, right=np.nan)
        vals.append(v)
    if not vals:
        return np.full(len(d_target), np.nan)
    return np.nanmedian(np.vstack(vals), axis=0)


def score(call, members):
    """Per-cable before/after on covered stations, plus a corrected re-solve."""
    out, corr_sets = {}, []
    for ci, name in ((0, "north"), (1, "south")):
        cab, d, t = call["sets"][ci]
        c = correction(members, ci, d)
        cov = np.isfinite(c)
        out[f"cover_{name}"] = int(cov.sum())
        r = call["res"][ci] - np.median(call["res"][ci])
        if cov.sum() >= MIN_COVER:
            s_before = smooth(r)[cov]
            r_after = r[cov] - c[cov]
            out[f"smooth_before_{name}_ms"] = rsig(s_before) * 1e3
            out[f"smooth_after_{name}_ms"] = rsig(smooth(r_after)) * 1e3
            out[f"sigma_before_{name}_ms"] = rsig(r[cov]) * 1e3
            out[f"sigma_after_{name}_ms"] = rsig(r_after) * 1e3
        corr_sets.append((cab, d, t - np.where(cov, c, 0.0)))
    sol, res = solve_set(corr_sets)
    if sol is not None:
        out["resolve_sigma_ms"] = rsig(np.concatenate(res)) * 1e3
        out["resolve_shift_m"] = float(np.hypot(sol.best.x - call["x"], sol.best.y - call["y"]))
    out["base_sigma_ms"] = rsig(np.concatenate(call["res"])) * 1e3
    return out


def main(samples: Path, out_prefix: str, leave_window_out: bool = False):
    calls, dropped = collect(samples)
    lab = cluster(calls)
    for c, l in zip(calls, lab):
        c["cluster"] = int(l)
    sizes = pd.Series(lab).value_counts()
    print(f"two-cable calls kept: {len(calls)}; dropped: {len(dropped)} "
          f"({pd.Series([d[1] for d in dropped]).value_counts().to_dict()})")
    print(f"clusters: {len(sizes)}; sizes: {sorted(sizes.tolist(), reverse=True)}")

    rows = []
    cent = {k: (np.mean([c['x'] for c in calls if c['cluster'] == k]),
                np.mean([c['y'] for c in calls if c['cluster'] == k]),
                [c for c in calls if c['cluster'] == k][0]['batch']) for k in sizes.index}
    for i, c in enumerate(calls):
        own = [m for j, m in enumerate(calls) if j != i and m["cluster"] == c["cluster"]
               and not (leave_window_out and m["window"] == c["window"])]
        if not own:
            continue
        row = {"call": c["call"], "batch": c["batch"], "cluster": c["cluster"], "n_others": len(own)}
        row.update({f"own_{k}": v for k, v in score(c, own).items()})
        # control: nearest other cluster in the same batch, at least CONTROL_MIN_KM away
        far = [(np.hypot(cent[k][0] - c["x"], cent[k][1] - c["y"]), k) for k in cent
               if k != c["cluster"] and cent[k][2] == c["batch"]]
        far = sorted(f for f in far if f[0] >= CONTROL_MIN_KM * 1e3)
        for dist_m, k in far:
            members = [m for m in calls if m["cluster"] == k]
            s = score(c, members)
            if any(s.get(f"cover_{n}", 0) >= MIN_COVER for n in ("north", "south")):
                row["control_cluster"], row["control_km"] = k, dist_m / 1e3
                row.update({f"ctl_{kk}": v for kk, v in s.items()})
                break
        rows.append(row)

    res = pd.DataFrame(rows)
    pd.DataFrame([{k: v for k, v in c.items() if k not in ("sets", "res")} for c in calls]).to_csv(
        f"{out_prefix}_calls.csv", index=False)
    res.to_csv(f"{out_prefix}_loo.csv", index=False)

    print(f"\nheld-out calls scored: {len(res)} (in clusters of >= 2)")
    for who in ("own", "ctl"):
        parts = []
        for cab in ("north", "south"):
            b, a = res.get(f"{who}_smooth_before_{cab}_ms"), res.get(f"{who}_smooth_after_{cab}_ms")
            if b is None or a is None:
                continue
            ok = b.notna() & a.notna()
            if ok.any():
                parts.append(f"{cab} smooth {b[ok].median():.1f} -> {a[ok].median():.1f} ms "
                             f"(lower in {(a[ok] < b[ok]).sum()}/{ok.sum()}); total "
                             f"{res[f'{who}_sigma_before_{cab}_ms'][ok].median():.1f} -> "
                             f"{res[f'{who}_sigma_after_{cab}_ms'][ok].median():.1f} ms")
        label = "own cluster" if who == "own" else "control (other cluster >= 3 km)"
        print(f"{label}: " + "; ".join(parts))
        rs = res.get(f"{who}_resolve_sigma_ms")
        if rs is not None and rs.notna().any():
            ok = rs.notna()
            print(f"   re-solve with corrected times: sigma {res[f'{who}_base_sigma_ms'][ok].median():.1f} -> "
                  f"{rs[ok].median():.1f} ms (n={ok.sum()}), position shift median "
                  f"{res[f'{who}_resolve_shift_m'][ok].median():.0f} m")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], "--leave-window-out" in sys.argv[3:])
