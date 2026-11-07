"""T-005 step 2c: pool steps 2 and 2b over every window run so far.

Re-solves each call's dense xcorr picks to flag basin-ambiguous calls (two
basins > 500 m apart, Plan B's flip threshold); those positions are unreliable,
so they are reported but left out of the pooled statistics.

Usage: python paper/scripts/t005_step2c_pool.py <results_dir>
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dclde_localization.physics.geometry import load_cables
from dclde_localization.physics.solve import solve
from dclde_localization.physics.sound_speed_models import ConstantC

FLIP_M = 500.0
REPO = Path(__file__).resolve().parents[2]


def main(res_dir: Path):
    cables = load_cables(REPO / "fixtures")
    s2, s2b, pairs = [], [], []
    for f in sorted(res_dir.glob("t005_step2b_*_calls.csv")):
        tag = re.sub(r"^t005_step2b_|_calls\.csv$", "", f.name)
        cab = cables[tag.split("_")[0].capitalize()]
        c2 = pd.read_csv(res_dir / f"t005_step2_{tag}_calls.csv")
        cb = pd.read_csv(f)
        st = pd.read_csv(res_dir / f"t005_step2b_{tag}_stations.csv")
        sep = {}
        for call, k in st.groupby("call"):
            k = k.sort_values("dist")
            px, py, pz = cab.chan_to_xyz(k.dist.to_numpy())
            sol = solve(px, py, pz, k.time.to_numpy(), ConstantC(1480.0), sz=-30.0)
            if not sol.basins:
                sep[call] = np.inf  # unsolved: excluded like an ambiguous call
            else:
                sep[call] = sol.separation_m if sol.separation_m is not None else 0.0
        for d in (c2, cb):
            d["window"] = tag
            d["basin_sep_m"] = d.call.map(sep)
            d["ambiguous"] = d.basin_sep_m > FLIP_M
        s2.append(c2)
        s2b.append(cb)
        p = pd.read_csv(res_dir / f"t005_step2b_{tag}_pairs.csv")
        p["window"] = tag
        p["ambiguous"] = p.a.map(sep).gt(FLIP_M) | p.b.map(sep).gt(FLIP_M)
        pairs.append(p)
    s2, s2b, pairs = pd.concat(s2), pd.concat(s2b), pd.concat(pairs)
    out = res_dir / "t005_step2c_pooled"
    s2.to_csv(f"{out}_step2_calls.csv", index=False)
    s2b.to_csv(f"{out}_step2b_calls.csv", index=False)
    pairs.to_csv(f"{out}_pairs.csv", index=False)

    a2 = s2[~s2.ambiguous]
    ab = s2b[~s2b.ambiguous]
    print(f"windows {s2.window.nunique()}, calls {len(s2)} ({s2.ambiguous.sum()} basin-ambiguous, excluded)")
    g = a2.dropna(subset=["sig_same_ms"])
    print(f"same stations (n={len(g)}): annot {g.sig_annot_ms.median():.1f} -> dense {g.sig_same_ms.median():.1f} ms; "
          f"improved in {(g.sig_same_ms < g.sig_annot_ms).sum()}/{len(g)}")
    print(f"all dense stations: {a2.sig_dense_ms.median():.1f} ms; neighbour noise {a2.noise_neighbour_ms.median():.1f}; "
          f"rough {a2.sig_rough_ms.median():.1f}; smooth {a2.sig_smooth_1km_ms.median():.1f} ms")
    print(f"smooth by definition (n={len(ab)}): xcorr {ab.smooth_xcorr_ms.median():.1f}, peak {ab.smooth_peak_ms.median():.1f}, "
          f"centroid {ab.smooth_centroid_ms.median():.1f}, onset {ab.smooth_onset_ms.median():.1f} ms; "
          f"corr xcorr-onset {ab.corr_smooth_xcorr_onset.median():.2f}; xcorr-onset drift {ab.smooth_lag_ms.median():.1f} ms")
    pa = pairs[~pairs.ambiguous]
    for lab, m in (("< 1 km", pa.sep_km < 1), ("1-3 km", (pa.sep_km >= 1) & (pa.sep_km < 3)), (">= 3 km", pa.sep_km >= 3)):
        x = pa[m]
        if len(x):
            print(f"pairs {lab:>7}: n={len(x):3d} in {x.window.nunique()} windows, corr median {x['corr'].median():.2f} "
                  f"(IQR {x['corr'].quantile(.25):.2f}..{x['corr'].quantile(.75):.2f})")
    print("\nper window (unambiguous calls):")
    print(a2.groupby("window").agg(calls=("call", "size"), annot=("sig_annot_ms", "median"),
                                   same=("sig_same_ms", "median"), noise=("noise_neighbour_ms", "median"),
                                   smooth=("sig_smooth_1km_ms", "median")).round(1).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
