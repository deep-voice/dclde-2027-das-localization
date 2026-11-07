"""Figures for the sound-speed / depth-observability pitch deck (paper/docs/pitch/).

Run from the repo root (PYTHONPATH=. so `viz` imports):

    PYTHONPATH=. uv run python paper/scripts/make_pitch_figs.py [out_dir] [figure ...]

Default out_dir is paper/docs/figures/pitch. Every data figure reads a committed table
under paper/results/ or recomputes from the committed picks
snapshot + cable geometry. `depth_profiles` and `fields` rerun the gate's own
profiling code on three / one real calls (a minute or two).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dclde_localization.physics.geometry import CABLES, FrameTransform, load_cable
from dclde_localization.physics.bathymetry import decode_gmt_grid
from dclde_localization.physics.forward import misfit_field, travel_time
from dclde_localization.physics.solve import solve
from dclde_localization.physics.sound_speed_models import ConstantC
from dclde_soundspeed.experiments.moveout import depth_profile, robust_sigma
from dclde_localization.physics.picks import attach_xyz, load_picks

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("paper/docs/figures/pitch")
OUT.mkdir(parents=True, exist_ok=True)
R = Path("paper/results")
FIX = Path("fixtures")

BG = "#F6F4EE"      # light slide surface
DARK = "#0E1F2B"    # dark slide surface
INK = "#1B2A36"
MUTED = "#5E6B75"
TEAL, CORAL, INDIGO = "#0F9A92", "#D9713F", "#5B6CB0"
GREY = "#A7AFB5"

sns.set_theme(style="ticks", context="talk", rc={
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "font.size": 17,
    "axes.labelsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16,
    "legend.fontsize": 16, "legend.frameon": False, "axes.grid": False,
})


def rd(name):
    return pd.read_csv(R / name, comment="#")


def save(fig, name):
    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- shared data
cables = {n: load_cable(n, FIX / f"{n.lower()}_DAS_multicoord.csv") for n in CABLES}
tf = FrameTransform.from_cable(cables["North"])
picks = attach_xyz(load_picks(), cables)
geo = decode_gmt_grid(FIX / "gmrt_bathy.nc")
gate = rd("gate_moveout.csv")


def bathy_axes(ax, dark=True, lon=(-125.35, -123.95), lat=(44.95, 45.40)):
    i = (geo.lon >= lon[0]) & (geo.lon <= lon[1])
    j = (geo.lat >= lat[0]) & (geo.lat <= lat[1])
    z = geo.z[np.ix_(j, i)]
    z = np.where(z > 0, np.nan, z)
    ax.contourf(geo.lon[i], geo.lat[j], -z, levels=np.arange(0, 3100, 100),
                cmap="mako_r" if not dark else "mako_r", extend="max")
    ax.contour(geo.lon[i], geo.lat[j], -z, levels=[200, 500, 1000, 1500, 2000],
               colors="#ffffff", linewidths=0.6, alpha=0.35)
    land = np.where(geo.z[np.ix_(j, i)] > 0, 1.0, np.nan)
    ax.contourf(geo.lon[i], geo.lat[j], land, levels=[0.5, 1.5],
                colors=["#3A3F3A" if dark else "#D8D3C4"])
    ax.set_aspect(1 / np.cos(np.deg2rad(45.2)))
    ax.set_xlim(*lon)
    ax.set_ylim(*lat)


# ---------------------------------------------------------------- 1. map
def fig_map():
    fr = rd("ablation_free_c.csv")
    lat, lon = tf.local_to_wgs84(fr["fixed_x"].to_numpy(), fr["fixed_y"].to_numpy())
    fig, ax = plt.subplots(figsize=(14, 6.2), facecolor=DARK)
    ax.set_facecolor(DARK)
    bathy_axes(ax)
    for name, c in cables.items():
        ax.plot(c.lon, c.lat, color="#F2E6C9", lw=3.2, solid_capstyle="round")
    ax.scatter(lon, lat, s=26, facecolor="none", edgecolor=CORAL, lw=1.3, alpha=0.9)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(OUT / "map.png", dpi=150, facecolor=DARK)
    plt.close(fig)
    print("wrote map.png")


# ---------------------------------------------------------------- 2. profile
def fig_profile():
    p = rd("hycom_profile_south_cable_deep.csv")
    fig, ax = plt.subplots(figsize=(6.4, 7.6))
    pz = -picks["pz"].to_numpy()
    ax.axhspan(pz.min(), pz.max(), color=GREY, alpha=0.25, lw=0, label="cable channels")
    ax.axhspan(219, 573, color=TEAL, alpha=0.18, lw=0, label="checked vs. CTD")
    ax.plot(p["c"], p["z_down"], color=INK, lw=2.6)
    ax.axvline(1480, color=CORAL, lw=2, ls="--", label="c = 1480 (standard)")
    ax.invert_yaxis()
    ax.set_xlabel("sound speed (m/s)")
    ax.set_ylabel("depth (m)")
    ax.legend(loc="lower left", fontsize=14)
    save(fig, "profile.png")


# ---------------------------------------------------------------- 3. join
def fig_join():
    j = rd("repro_join_feasibility.csv")
    j = j[j["sweep_type"] == "tol_s"].pivot(index="param_value", columns="metric",
                                           values="value").reset_index()
    tolcol = "param_value"
    j = j.sort_values(tolcol, ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5.6))
    x = np.arange(len(j))
    bottom = np.zeros(len(j))
    for col, colr, lab in [("matched", TEAL, "matched"), ("ambiguous", GREY, "ambiguous"),
                           ("unmatched", CORAL, "unmatched")]:
        ax.bar(x, j[col], bottom=bottom, color=colr, width=0.72, label=lab,
               edgecolor=BG, linewidth=2)
        bottom += j[col].to_numpy()
    ax.set_xticks(x, [f"{t:g} s" for t in j[tolcol]])
    ax.set_xlabel("timestamp tolerance")
    ax.set_ylabel("baseline rows (n = 176)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    save(fig, "join.png")


# ---------------------------------------------------------------- 4. free c
def fig_free_c():
    fr = rd("ablation_free_c.csv")
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.axvspan(fr["profile_c_min"].iloc[0], fr["profile_c_max"].iloc[0], color=TEAL,
               alpha=0.2, lw=0, label="physically plausible")
    sns.histplot(fr["best_c"], bins=np.arange(1430, 1532.5, 2.5), color=INDIGO,
                 edgecolor=BG, linewidth=1, ax=ax, label="fitted c per call")
    ax.set_xlabel("fitted sound speed (m/s)")
    ax.set_ylabel("calls (n = 498)")
    ax.legend(loc="upper right")
    save(fig, "free_c.png")


# ---------------------------------------------------------------- 5. range trend vs depth
def fig_trend():
    t = rd("regime_depth_vs_path.csv")
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.errorbar(-t["source_z"], t["range_trend_ms"], yerr=1.96 * t["range_trend_se_ms"],
                fmt="o-", color=INDIGO, ms=10, lw=2, capsize=5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("assumed source depth (m)")
    ax.set_ylabel("far − near residual (ms)")
    save(fig, "trend.png")


# ---------------------------------------------------------------- 6. hypothesis test
def fig_hyp():
    h = rd("propagation_hypothesis_test.csv")
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    y = np.arange(len(h))
    ax.hlines(y, h["ci_lo"], h["ci_hi"], color=CORAL, lw=5, alpha=0.55)
    ax.plot(h["delta_loglik"], y, "o", color=CORAL, ms=13)
    ax.axvline(0, color=INK, lw=1.5)
    ax.set_yticks(y, [f"{c:g} m/s" for c in h["seabed_c"]])
    ax.set_xlim(-1600, 150)
    ax.set_xlabel("Δ log-likelihood (trapped − direct)")
    ax.set_ylabel("seabed sound speed")
    save(fig, "hyp.png")


# ---------------------------------------------------------------- 7. direct vs bounce
def fig_bounce():
    g = gate.dropna(subset=["delta_chi2"])
    d = g["delta_chi2"].clip(-12, 30)
    fig, ax = plt.subplots(figsize=(9, 5.6))
    bins = np.arange(-12, 31, 1)
    sns.histplot(d[d >= 0], bins=bins, color=TEAL, edgecolor=BG, ax=ax, label="favours direct")
    sns.histplot(d[d < 0], bins=bins, color=CORAL, edgecolor=BG, ax=ax, label="favours bounce")
    for v in (-4, 4):
        ax.axvline(v, color=MUTED, lw=1.2, ls=":")
    ax.set_xlabel("Δχ² (bounce − direct), clipped at 30")
    ax.set_ylabel(f"calls (n = {len(g)})")
    ax.legend(loc="upper right")
    save(fig, "bounce.png")


# ---------------------------------------------------------------- 8. sigma_z breakdown
def fig_sigma():
    s = gate[gate["in_stratum"]].copy()
    fin = s[np.isfinite(s["sigma_z_m"])]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    sns.histplot(fin["sigma_z_m"], bins=np.arange(0, 260, 10), color=INDIGO,
                 edgecolor=BG, ax=ax)
    ax.axvline(100, color=CORAL, lw=2.5, ls="--")
    ax.set_xlabel("marginal depth uncertainty σz (m)")
    ax.set_ylabel(f"calls with finite σz (n = {len(fin)})")
    save(fig, "sigma_hist.png")

    counts = {"no finite width": int((~np.isfinite(s["sigma_z_m"])).sum()),
              "finite, > 100 m": int((fin["sigma_z_m"] > 100).sum()),
              "≤ 100 m, artifact": int(((fin["sigma_z_m"] <= 100) & (s.loc[fin.index, "walk_consistent"] == False)).sum()),
              "≤ 100 m, survives": int(((fin["sigma_z_m"] <= 100) & (s.loc[fin.index, "walk_consistent"] == True)).sum())}
    print("sigma counts", counts, "stratum", len(s))


# ---------------------------------------------------------------- 9. depth profiles
def call_arrays(call):
    sub = picks[picks["call"] == call]
    return [sub[k].to_numpy(float) for k in ("px", "py", "pz", "time")], sub


def seed(px, py, pz, t):
    sol = solve(px, py, pz, t, ConstantC(1480.0), sz=-30.0)
    b = sol.best
    resid = t - (travel_time(b.x, b.y, -30.0, px, py, pz, 1480.0) + b.t0)
    return b, robust_sigma(resid)


def fig_depth_profiles():
    zg = np.linspace(-600.0, 0.0, 121)
    calls = [("2021-11-04_02_05_02/7", TEAL, "two cables · bounded (σz 78 m)"),
             ("2021-11-03_22_10_42/1", INDIGO, "two cables · never crosses 1"),
             ("2021-11-04_08_02_32/7", CORAL, "one cable · walked down")]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for call, col, lab in calls:
        (px, py, pz, t), _ = call_arrays(call)
        b, sig = seed(px, py, pz, t)
        z, m = depth_profile(px, py, pz, t, z_grid=zg, x0=b.x, y0=b.y)
        ref = m.min()
        if "walked" in lab:
            zr, mr = depth_profile(px, py, pz, t, z_grid=zg[::-1], x0=b.x, y0=b.y)
            ref = min(ref, mr.min())
            print("artifact fwd/rev min", m.min(), mr.min(), "rev chi max", (len(t)*(mr**2-ref**2)/sig**2).max())
        chi = len(t) * (m ** 2 - ref ** 2) / sig ** 2
        ax.plot(-z, chi, color=col, lw=2.8, label=lab)
        if "walked" in lab:
            chir = len(t) * (mr ** 2 - ref ** 2) / sig ** 2
            ax.plot(-zr, chir, color=col, lw=2.2, ls="--", label="same call, walked back up")
    ax.axhline(1, color=MUTED, lw=1.5, ls=":")
    ax.set_ylim(0, 12)
    ax.set_xlim(0, 600)
    ax.set_xlabel("source depth (m)")
    ax.set_ylabel("Δχ² from best depth")
    ax.legend(loc="upper left", bbox_to_anchor=(0.2, 1.0), fontsize=13)
    save(fig, "depth_profiles.png")


# ---------------------------------------------------------------- 10. misfit fields at two depths
def fig_fields():
    call = "2021-11-04_08_02_32/7"
    (px, py, pz, t), sub = call_arrays(call)
    b, _ = seed(px, py, pz, t)
    half = 3500.0
    xs = np.linspace(b.x - half, b.x + half, 281)
    ys = np.linspace(b.y - half, b.y + half, 281)
    X, Y = np.meshgrid(xs, ys)
    lat, lon = tf.local_to_wgs84(X.ravel(), Y.ravel())
    lat, lon = lat.reshape(X.shape), lon.reshape(X.shape)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), facecolor=DARK)
    cab = cables[sub["cable"].iloc[0]]
    for ax, sz, col in zip(axes, (-30.0, -450.0), (TEAL, CORAL)):
        ax.set_facecolor(DARK)
        f = misfit_field(xs, ys, sz, px, py, pz, t, 1480.0, "l2")
        rel = f / f.min()
        bathy_axes(ax, lon=(lon.min(), lon.max()), lat=(lat.min(), lat.max()))
        show = np.where(rel <= 3.0, np.log(rel), np.nan)
        ax.contourf(lon, lat, show, levels=np.linspace(0, np.log(3.0), 14),
                    cmap="rocket", alpha=0.8)
        ax.contour(lon, lat, rel, levels=[1.1, 1.5, 2.0], colors="#F6F4EE",
                   linewidths=[2.4, 1.3, 0.8])
        ax.plot(cab.lon, cab.lat, color="#F2E6C9", lw=3)
        sl, so = tf.local_to_wgs84(sub["px"].to_numpy(), sub["py"].to_numpy())
        ax.scatter(so, sl, s=18, color="#F2E6C9", zorder=5)
        k = np.unravel_index(np.argmin(f), f.shape)
        ax.plot(lon[k], lat[k], marker="P", ms=20, color=col, mec=DARK, mew=2, zorder=6)
        print(f"field z={sz}: min misfit {f.min()*1e3:.3f} ms at "
              f"({xs[k[1]]-b.x:.0f},{ys[k[0]]-b.y:.0f}) m from seed")
        ax.set_axis_off()
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99, wspace=0.04)
    fig.savefig(OUT / "fields.png", dpi=150, facecolor=DARK)
    plt.close(fig)
    print("wrote fields.png")


# ---------------------------------------------------------------- 11. exchangeability
def fig_exchange():
    d = pd.DataFrame({"geometry": ["one cable,\ndepth monotonic", "one cable,\nV-shaped depth",
                                   "two cables,\ndifferent bearings"],
                      "us": [23, 918, 3044]})
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.bar(d["geometry"], d["us"], color=[CORAL, INDIGO, TEAL], width=0.6, edgecolor=BG, lw=2)
    ax.set_yscale("log")
    ax.set_ylim(10, 10000)
    ax.set_ylabel("leftover signal from 400 m\nof depth error (µs, log)")
    for i, v in enumerate(d["us"]):
        ax.text(i, v * 1.25, f"{v:,} µs", ha="center", va="bottom", fontsize=17, color=INK)
    save(fig, "exchange.png")


if __name__ == "__main__":
    which = sys.argv[2:] or ["map", "profile", "join", "free_c", "trend", "hyp", "bounce",
                            "sigma", "depth_profiles", "fields", "exchange"]
    for w in which:
        {"map": fig_map, "profile": fig_profile, "join": fig_join, "free_c": fig_free_c,
         "trend": fig_trend, "hyp": fig_hyp, "bounce": fig_bounce, "sigma": fig_sigma,
         "depth_profiles": fig_depth_profiles, "fields": fig_fields,
         "exchange": fig_exchange}[w]()
