"""How big is the depth signal next to the pick noise, on the gate's own stratum?

Answers two team questions after the Plan B gate (paper/docs/das-finwhale-plan-b-gate-findings.md):
how many picks does a call carry, and does having more than 4 picks across two cables mean depth
is recoverable. For each of the gate's 271 stratum calls the source is moved from the null's
-30 m to -130 m, (x, y) is re-minimised at both depths (L2, `depth_profile`), and the change in
misfit is turned into a chi-squared with the module's own convention, `n * (m1**2 - m0**2) /
sigma**2`, sigma being the call's robust residual under the null.

Single-direction profiles only: the walk-consistency check of the gate is NOT applied here, so
the calls that reach 2 sigma are partly the direction-dependent artifacts the gate identified.
Read the median, not the tail.

    PYTHONPATH=. uv run python -m paper.scripts.depth_signal_vs_noise

Writes `depth_signal_vs_noise.csv` (one row per call), `depth_signal_vs_noise_summary.csv` and
`picks_per_call_summary.csv` under paper/results/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dclde_soundspeed.experiments.moveout import depth_profile
from dclde_soundspeed.experiments.runner import write_results
from dclde_localization.physics.geometry import CABLES, load_cable
from dclde_localization.physics.solve import solve
from dclde_localization.physics.sound_speed_models import ConstantC

Z_REF, Z_SHIFT, C = -30.0, -130.0, 1480.0
FIX = Path("fixtures")
GATE = Path("paper/results/gate_moveout.csv")
N_BINS = [7, 15, 25, 40, 100]


def delta_chi2(m_ref, m_shift, n_picks, sigma):
    """chi-squared change for a depth shift; positive means the shifted depth fits worse.
    `m_*` are RMS misfits and `sigma` the per-pick noise, all in seconds."""
    return n_picks * (np.asarray(m_shift) ** 2 - np.asarray(m_ref) ** 2) / np.asarray(sigma) ** 2


def per_call_table(picks: pd.DataFrame, gate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for call, sub in picks.groupby("call"):
        g = gate.loc[call]
        if not g.in_stratum:
            continue
        px, py, pz = (sub[k].to_numpy(float) for k in ("px", "py", "pz"))
        t = sub["time"].to_numpy(float)
        b = solve(px, py, pz, t, ConstantC(C), sz=Z_REF).best
        _, m = depth_profile(px, py, pz, t, z_grid=np.array([Z_SHIFT, Z_REF]), c=C, x0=b.x, y0=b.y)
        sigma = g.resid_sigma_ms / 1e3
        rows.append(dict(call=call, n_picks=len(sub), n_cables=int(g.n_cables), resid_sigma_ms=g.resid_sigma_ms,
                         misfit_ref=m[1], misfit_shift=m[0],
                         rms_equiv_ms=float(np.sqrt(abs(m[0] ** 2 - m[1] ** 2)) * 1e3),
                         delta_chi2=float(delta_chi2(m[1], m[0], len(sub), sigma))))
    return pd.DataFrame(rows)


def summarise(t: pd.DataFrame) -> pd.DataFrame:
    a = t["delta_chi2"].abs()

    def row(label, mask):
        x, ax = t[mask], a[mask]
        return dict(group=label, n_calls=int(mask.sum()), median_abs_dchi2=ax.median(),
                    n_ge_1sigma=int((ax >= 1).sum()), n_ge_2sigma=int((ax >= 4).sum()), n_ge_3sigma=int((ax >= 9).sum()),
                    median_rms_equiv_ms=x["rms_equiv_ms"].median(), median_noise_ms=x["resid_sigma_ms"].median())

    out = [row("all stratum", a.notna())]
    out += [row(f"n_cables={k}", t["n_cables"] == k) for k in (1, 2)]
    bins = pd.cut(t["n_picks"], N_BINS)
    out += [row(f"n_picks in {iv}", bins == iv) for iv in bins.cat.categories]
    return pd.DataFrame(out)


def picks_summary(picks: pd.DataFrame) -> pd.DataFrame:
    n = picks.groupby("call").size()
    ncab = picks.groupby("call")["cable"].nunique()
    per_cable = picks[picks["call"].isin(ncab[ncab == 2].index)].groupby(["call", "cable"]).size().groupby("call").min()
    return pd.DataFrame([dict(
        calls=len(n), picks=int(n.sum()), median_picks=n.median(), mean_picks=n.mean(), min_picks=n.min(),
        max_picks=n.max(), calls_ge_4=int((n >= 4).sum()), calls_ge_8=int((n >= 8).sum()), calls_ge_20=int((n >= 20).sum()),
        calls_two_cable=int((ncab == 2).sum()), median_picks_on_sparser_cable_two_cable_calls=per_cable.median())])


def main():
    from dclde_localization.physics.picks import attach_xyz, load_picks
    cables = {n: load_cable(n, FIX / f"{n.lower()}_DAS_multicoord.csv") for n in CABLES}
    picks = attach_xyz(load_picks(), cables)
    gate = pd.read_csv(GATE, comment="#").set_index("call")
    table = per_call_table(picks, gate)
    note = [f"depth shift {Z_REF:g} m -> {Z_SHIFT:g} m, (x, y) re-minimised at both (L2); c = {C:g} m/s",
            "denominator: the gate's 271-call stratum (n_picks >= 8, receiver-depth span >= 200 m, basin-stable)",
            "single-direction profiles; the gate's walk-consistency check is not applied, so the tail is partly artifact"]
    write_results(table, "depth_signal_vs_noise", notes=note)
    write_results(summarise(table), "depth_signal_vs_noise_summary", notes=note)
    write_results(picks_summary(picks), "picks_per_call_summary",
                  notes=["denominator: all 628 annotated calls / 11,117 picks"])
    print(summarise(table).to_string(index=False))
    print(picks_summary(picks).T.to_string(header=False))


if __name__ == "__main__":
    main()
