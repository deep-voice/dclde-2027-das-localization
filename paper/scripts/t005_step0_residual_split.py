"""T-005 step 0: how much of the per-call residual is smooth structure vs pick scatter?

Re-solves the 271 Plan B stratum calls under the ConstantC(1480), z=-30 m null
(the gate_table path), removes nested smooth models (per-cable offset, linear and
quadratic in range) from the residual, and reports the robust sigma left over,
its SNR dependence, and a split-half test of per-channel statics.

Usage: python paper/scripts/t005_step0_residual_split.py <repo_root> <out.csv>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "paper" / "src"))

from dclde_localization.physics.forward import travel_time
from dclde_localization.physics.geometry import load_cables
from dclde_localization.physics.picks import attach_xyz, load_picks
from dclde_localization.physics.solve import solve
from dclde_localization.physics.sound_speed_models import ConstantC
from dclde_soundspeed.experiments.moveout import robust_sigma

C, ZS = 1480.0, -30.0
CUT = 1.0  # s; Plan A's gross mis-association cut

gate = pd.read_csv(REPO / "paper/results/gate_moveout.csv", comment="#")
strat = gate[gate.in_stratum == True].set_index("call")
cables = load_cables(REPO / "fixtures")
picks = attach_xyz(load_picks(REPO / "fixtures/das_calls.csv.gz"), cables)
picks = picks[picks.call.isin(strat.index)]


def rsig_after(design, r):
    """Least-squares fit of `design` to r (gross tail excluded), robust sigma of what is left."""
    keep = np.abs(r - np.median(r)) < CUT
    if keep.sum() <= design.shape[1] + 2:
        return np.nan
    beta, *_ = np.linalg.lstsq(design[keep], r[keep], rcond=None)
    return robust_sigma((r - design @ beta)[keep]) * 1e3


def lag1(res, order):
    x = res[order]
    x = x - x.mean()
    d = (x * x).sum()
    return float((x[1:] * x[:-1]).sum() / d) if d > 0 else np.nan


rows, perpick = [], []
model = ConstantC(C)
for call, sub in picks.groupby("call"):
    px, py, pz = (sub[k].to_numpy(float) for k in ("px", "py", "pz"))
    t = sub["time"].to_numpy(float)
    sol = solve(px, py, pz, t, model, sz=ZS)
    b = sol.best
    r = t - (travel_time(b.x, b.y, ZS, px, py, pz, C) + b.t0)
    rng = np.sqrt((px - b.x) ** 2 + (py - b.y) ** 2 + (pz - ZS) ** 2) / 1e3  # km
    cab = sub["cable"].to_numpy()
    one = np.ones_like(r)
    offs = np.column_stack([(cab == c).astype(float) for c in np.unique(cab)])
    rz = (rng - rng.mean()) / (rng.std() or 1.0)
    keep = np.abs(r - np.median(r)) < CUT
    # lag-1 autocorrelation of the per-cable-offset residual, along the cable
    beta, *_ = np.linalg.lstsq(offs[keep], r[keep], rcond=None)
    res_off = r - offs @ beta
    ac = []
    for c in np.unique(cab):
        m = keep & (cab == c)
        if m.sum() >= 6:
            ac.append(lag1(res_off[m], np.argsort(sub["dist"].to_numpy()[m])))
    D=np.column_stack([offs, rz, rz ** 2]); bq,*_=np.linalg.lstsq(D[keep], r[keep], rcond=None)
    rq=r-D@bq
    perpick.append(pd.DataFrame({"call":call,"snr":sub["snr"].to_numpy()[keep],"res":rq[keep],"dist":sub["dist"].to_numpy()[keep],"cable":cab[keep],"batch":sub["batch"].to_numpy()[keep],"n_cables":len(np.unique(cab))}))
    rows.append({
        "call": call, "n_picks": len(r), "n_cables": len(np.unique(cab)),
        "sig_raw": robust_sigma(r) * 1e3,
        "sig_ref": strat.loc[call, "resid_sigma_ms"],
        "sig_cable_offset": rsig_after(offs, r),
        "sig_offset_lin": rsig_after(np.column_stack([offs, rz]), r),
        "sig_offset_quad": rsig_after(np.column_stack([offs, rz, rz ** 2]), r),
        "lag1_ac": np.nanmean(ac) if ac else np.nan,
        "range_km_med": float(np.median(rng)),
    })

df = pd.DataFrame(rows)
out = Path(sys.argv[2])
df.to_csv(out, index=False)

print(f"calls: {len(df)} (stratum 271); picks: {df.n_picks.sum()}")
print("reproduction: |sig_raw - sig_ref| median %.2f ms, max %.2f ms"
      % ((df.sig_raw - df.sig_ref).abs().median(), (df.sig_raw - df.sig_ref).abs().max()))
cols = ["sig_raw", "sig_cable_offset", "sig_offset_lin", "sig_offset_quad"]
for name, g in [("all", df), ("one cable", df[df.n_cables == 1]), ("two cables", df[df.n_cables == 2])]:
    q = g[cols].quantile([0.25, 0.5, 0.75]).round(1)
    print(f"\n{name} (n={len(g)}) quartiles, ms:\n{q.to_string()}")
    print(f"  lag-1 autocorrelation along cable, median: {g.lag1_ac.median():.2f} (n={g.lag1_ac.notna().sum()})")
for thr in (15, 30):
    print(f"calls with sig_offset_quad <= {thr} ms: {(df.sig_offset_quad <= thr).sum()} / {len(df)}")

pp=pd.concat(perpick); pp["snr_q"]=pd.qcut(pp.snr,4,labels=["Q1 low","Q2","Q3","Q4 high"])
print("per-pick residual after offset+quad, by pick SNR quartile:")
print(pp.groupby("snr_q",observed=True).agg(n=("res","size"),snr_med=("snr","median"),rsig_ms=("res",lambda x: robust_sigma(x)*1e3)).round(1).to_string())
pp.to_csv(out.with_name("t005_step0_perpick.csv"),index=False)

# channel statics: does the same stretch of cable carry the same residual in different calls?
rng_=np.random.default_rng(0)
for binm in (500, 1000, 2000):
    pp["bin"]=pp.cable+"_"+(pp.dist//binm).astype(int).astype(str)
    for split in ("calls", "batch"):
        res=[]
        for rep in range(20 if split=="calls" else 1):
            if split=="calls":
                calls=pp.call.unique(); A=set(rng_.choice(calls,len(calls)//2,replace=False)); isA=pp.call.isin(A)
            else:
                isA=pp.batch.isin([1,2])  # batches 1-2 vs 3-5: different days/hours
            for tr,te in ((isA,~isA),(~isA,isA)):
                st=pp[tr].groupby("bin").res.median()
                cnt=pp[tr].groupby("bin").res.size(); st=st[cnt>=5]
                t=pp[te]; m=t.bin.isin(st.index)
                before=robust_sigma(t.res[m])*1e3; after=robust_sigma((t.res[m]-t.bin[m].map(st)))*1e3
                res.append((before,after,m.mean()))
        b,a,cov=np.mean(res,axis=0)
        print(f"bin {binm:4d} m, split by {split:5s}: held-out sigma {b:.1f} -> {a:.1f} ms after statics (coverage {cov:.0%})")
