"""Re-pick a call on the dense DAS channels by envelope cross-correlation.

The annotation picks sit on the peak of a ~350 ms-wide SNR envelope, one pick
every ~1.1 km (T-005 step 1). Peak times of a broad, noisy envelope are jittery.
This module instead aligns the envelope SHAPE across stations:

1. Seed: linear interpolation of the call's own annotation picks along the
   cable. Stations are only placed inside the picked span; nothing is
   extrapolated.
2. Stations every `station_m` (default 51 m, one gauge length, so neighbouring
   stations are roughly independent). Each station's trace is the mean of the
   channels within +/- station_m / 2.
3. A template is the mean of all station windows aligned on the current
   times. Each station is cross-correlated with it (normalised), and the lag
   is refined to sub-sample precision with a parabola through the peak. The
   template is rebuilt on the new alignment and the step repeats.
4. The median lag is removed every iteration, so the picks keep the
   annotations' timing convention on average (the absolute time per cable
   matters for two-cable solves; t0 per call is free anyway).
5. A station is kept only if its correlation is >= `min_cc` and its lag sits
   within `max_jump_s` of the running median of its neighbours' lags.

Nothing here uses a propagation model, so the picks can be tested against one
without circularity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PickerConfig:
    station_m: float = 51.0
    half_window_s: float = 0.6     # template half-length around the pick
    max_shift_s: float = 0.4       # search range for each lag
    n_iter: int = 3
    min_cc: float = 0.6
    max_jump_s: float = 0.15       # allowed departure from the running-median lag
    median_stations: int = 11      # running-median length, in stations


def _parabolic(cc: np.ndarray, k: int) -> float:
    """Sub-sample offset of the peak at index k, from a parabola through k-1, k, k+1."""
    if k <= 0 or k >= len(cc) - 1:
        return 0.0
    a, b, c = cc[k - 1], cc[k], cc[k + 1]
    den = a - 2 * b + c
    return 0.0 if den == 0 else 0.5 * (a - c) / den


def _windows(traces: np.ndarray, t_axis: np.ndarray, centres: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Sample each station trace at centre + tau (linear interpolation, zero outside)."""
    out = np.empty((len(centres), len(tau)))
    for i, (tr, c) in enumerate(zip(traces, centres)):
        out[i] = np.interp(c + tau, t_axis, tr, left=0.0, right=0.0)
    return out


def _norm_xcorr(win: np.ndarray, tmpl: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation of tmpl against every offset of win (valid mode)."""
    m = len(tmpl)
    t = tmpl - tmpl.mean()
    t /= np.linalg.norm(t) or 1.0
    # sliding windows of length m
    s = np.lib.stride_tricks.sliding_window_view(win, m)
    s = s - s.mean(axis=1, keepdims=True)
    n = np.linalg.norm(s, axis=1)
    n[n == 0] = np.inf
    return (s @ t) / n


def station_traces(block: np.ndarray, dist_axis: np.ndarray, stations: np.ndarray, station_m: float) -> np.ndarray:
    """Mean trace of the channels within +/- station_m/2 of each station. block is (dist, time)."""
    out = np.zeros((len(stations), block.shape[1]))
    lo = np.searchsorted(dist_axis, stations - station_m / 2, side="left")
    hi = np.searchsorted(dist_axis, stations + station_m / 2, side="right")
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            out[i] = block[a:b].mean(axis=0)
    return out


def pick_call(block: np.ndarray, dist_axis: np.ndarray, t_axis: np.ndarray,
              seed_dist: np.ndarray, seed_time: np.ndarray,
              cfg: PickerConfig = PickerConfig()) -> pd.DataFrame:
    """Dense picks for one call on one cable.

    block: SNR image (dist, time) covering at least the seed span and the time
    window; dist_axis/t_axis: its coordinates (ascending). Returns one row per
    station: dist, time, cc, keep.
    """
    order = np.argsort(seed_dist)
    sd, st = np.asarray(seed_dist, float)[order], np.asarray(seed_time, float)[order]
    if len(sd) < 2 or sd[-1] - sd[0] < 2 * cfg.station_m:
        return pd.DataFrame(columns=["dist", "time", "cc", "keep"])
    stations = np.arange(sd[0], sd[-1] + 1e-9, cfg.station_m)
    seed = np.interp(stations, sd, st)
    traces = station_traces(block, dist_axis, stations, cfg.station_m)

    dt = float(np.median(np.diff(t_axis)))
    tau_t = np.arange(-cfg.half_window_s, cfg.half_window_s + dt / 2, dt)
    nshift = int(round(cfg.max_shift_s / dt))
    tau_w = np.arange(-cfg.half_window_s - nshift * dt, cfg.half_window_s + nshift * dt + dt / 2, dt)

    lag = np.zeros(len(stations))
    cc = np.zeros(len(stations))
    keep = np.ones(len(stations), bool)
    for _ in range(cfg.n_iter):
        cur = seed + lag
        aligned = _windows(traces, t_axis, cur, tau_t)
        tmpl = aligned[keep].mean(axis=0) if keep.any() else aligned.mean(axis=0)
        wins = _windows(traces, t_axis, seed, tau_w)  # always search around the seed
        new = np.zeros_like(lag)
        for i in range(len(stations)):
            c = _norm_xcorr(wins[i], tmpl)
            k = int(np.argmax(c))
            cc[i] = c[k]
            new[i] = (k - nshift + _parabolic(c, k)) * dt
        run = pd.Series(new).rolling(cfg.median_stations, center=True, min_periods=1).median().to_numpy()
        keep = (cc >= cfg.min_cc) & (np.abs(new - run) <= cfg.max_jump_s)
        if keep.any():
            new -= np.median(new[keep])
        lag = new

    return pd.DataFrame({"dist": stations, "time": seed + lag, "cc": cc, "keep": keep,
                         "seed_time": seed})
