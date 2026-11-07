"""OOI ERDDAP client for in-situ temperature and salinity.

Coverage for November 2021 is much thinner than the feasibility note assumed;
NOV_2021_STATUS records what was actually measured on 2026-09-15 so nobody
repeats the search. Exactly one instrument near the cables was profiling:
`ce04ospd_deep`. Its RAW mechanical range is z -575..-97 m (positive-down
97..575 m, 129,177 samples, 25 Oct - 15 Nov 2021) -- but that is what the
instrument swept, not what is trustworthy. After `filter_qartod` (QC flag
<= 2) only 93,901/129,177 rows (73%) survive, spanning -575..-154 m
(positive-down 154..575 m): the shallowest ~120-450 m of the raw sweep is
QARTOD-suspect for temperature and gets correctly dropped. **The entire
upper water column above ~150-220 m (window-dependent) has no QC-passing
in-situ measurement** -- do not read "97 m" out of this module and assume it
is usable; read NOV_2021_STATUS's "QC-passing" field instead. The Slope Base
deep profiler (`rs01sbpd_deep`, site RS01SBPD -- not RS01SBPS) returned zero
rows. The two shallow instruments were parked at a single depth for the
whole window, not profiling; `fetch_profile` refuses to build a profile from
a depth span too small to be a profile (see `MIN_PROFILE_SPAN_M`).

The raw variable ranges are polluted (salinity to 33929 PSU, temperature to
61 C). Always filter on the QARTOD aggregate flags before trusting a value.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile

BASE = "https://erddap.dataexplorer.oceanobservatories.org/erddap/tabledap"

DATASETS = {
    "ce04ospd_deep": "ooi-ce04ospd-dp01b-01-ctdpfl105",
    "rs01sbpd_deep": "ooi-rs01sbpd-dp01a-01-ctdpfl104",
    "rs01sbps_shallow": "ooi-rs01sbps-sf01a-2a-ctdpfa102",
    "ce04osps_shallow": "ooi-ce04osps-sf01b-2a-ctdpfa107",
}

# Measured 2026-09-15 for 25 Oct - 15 Nov 2021. Not guesswork -- see the
# module docstring and the Task 8 brief for how this was queried. Every entry
# states BOTH the raw instrument range and the QARTOD-passing (flag<=2)
# range with its denominator, so nobody mistakes the raw sweep for usable
# coverage (Task 8 review finding 1: 97 m alone is not the usable minimum).
NOV_2021_STATUS = {
    "ce04ospd_deep": (
        "Profiling: raw 129,177 samples, z -575..-97 m, 479 distinct depths "
        "(25 Oct-15 Nov 2021); QC-passing (flag<=2) 93,901/129,177 rows "
        "(73%), z -575..-154 m -- the shallowest ~150-450 m of the raw "
        "sweep is QARTOD-suspect for temperature and is dropped. A shorter "
        "5-day window (1-6 Nov) measured an even thinner QC-passing band, "
        "216..575 m (5,067/7,472 rows, 68%): the usable shallow edge is "
        "window-dependent, so re-check it for whatever window you fetch "
        "rather than assuming either number."
    ),
    "rs01sbpd_deep": "No data: HTTP 404, zero rows for 25 Oct - 15 Nov 2021",
    "rs01sbps_shallow": "Parked at z -195..-189 m over five days: a point, not a profile",
    "ce04osps_shallow": "Parked at z -197..-192 m",
}

TEMP = "sea_water_temperature_profiler_depth_enabled"
SALT = "sea_water_practical_salinity_profiler_depth_enabled"
QC_PASS_MAX = 2  # 1 pass, 2 not evaluated; 3 suspect and 4 fail are rejected

# A profile needs a real depth span. The parked shallow instruments
# (rs01sbps_shallow, ce04osps_shallow) move only ~5-6 m over five days --
# refuse those at construction time with a message naming the cause, rather
# than building a near-degenerate SoundSpeedProfile that fails later, three
# modules away, in path_average with an error that never mentions ERDDAP.
MIN_PROFILE_SPAN_M = 10.0


def build_url(dataset_id: str, start: str, end: str) -> str:
    """Build the ERDDAP tabledap CSV request URL, including QC columns.

    Fetching the data columns without their QC companions is how a garbage
    profile gets built silently: the raw ranges include salinity up to
    33929 PSU and temperature up to 61 C.
    """
    cols = ",".join(["time", "z", TEMP, SALT, f"{TEMP}_qc_agg", f"{SALT}_qc_agg"])
    return (f"{BASE}/{dataset_id}.csv?{cols}"
            f"&time%3E={start}T00:00:00Z&time%3C={end}T00:00:00Z")


def filter_qartod(df: pd.DataFrame, max_flag: int = QC_PASS_MAX) -> pd.DataFrame:
    """Keep only rows where every `_qc_agg` column is <= max_flag.

    QARTOD aggregate flags: 1 pass, 2 not evaluated, 3 suspect, 4 fail, 9
    missing. The default (max_flag=2) keeps pass and not-evaluated, and
    rejects suspect/fail/missing.
    """
    keep = pd.Series(True, index=df.index)
    for col in (c for c in df.columns if c.endswith("_qc_agg")):
        keep &= df[col].astype(float) <= max_flag
    return df.loc[keep].copy()


def _flag_counts(series: pd.Series) -> str:
    """Render a QC column's value counts as e.g. '1:5067,3:2405' (sorted by flag)."""
    vc = series.astype(float).value_counts()
    return ",".join(f"{int(flag)}:{int(count)}" for flag, count in sorted(vc.items()))


def fetch_profile(dataset_key: str, start: str, end: str, lat: float, lon: float,
                   *, session=None, n_bins: int = 60,
                   max_flag: int = QC_PASS_MAX) -> SoundSpeedProfile:
    """Fetch, QC-filter and depth-bin one instrument into a sound-speed profile.

    `session` is injectable (anything exposing `.get(url, timeout=...)`,
    e.g. `requests.Session()` or a test double) so the parse path can be
    exercised offline against a canned CSV response without hitting the
    network. Defaults to the `requests` module itself.

    `max_flag` is the QARTOD aggregate-flag cutoff passed to `filter_qartod`
    (default: pass and not-evaluated only). A caller building a profile that
    feeds a travel-time model needs to both control this and see what it
    discarded -- both the threshold and the kept/raw counts and per-flag
    breakdown are recorded in the returned profile's `source` string, since
    `SoundSpeedProfile` has no other field for provenance.
    """
    dataset_id = DATASETS[dataset_key]
    get = (session or requests).get
    resp = get(build_url(dataset_id, start, end), timeout=180)
    if resp.status_code == 404:
        raise ValueError(f"{dataset_key} ({dataset_id}) returned no rows for {start}..{end}. "
                          f"Known status: {NOV_2021_STATUS.get(dataset_key, 'unknown')}")
    resp.raise_for_status()
    # Row 0 (the pd.read_csv header) is the column names; ERDDAP inserts a
    # second row of units (e.g. "UTC", "m", "degree_C") immediately after --
    # skip it explicitly, or every column silently becomes dtype=object and
    # arithmetic downstream produces garbage instead of raising.
    df = pd.read_csv(io.StringIO(resp.text), skiprows=[1])
    n_raw = len(df)
    temp_flags = _flag_counts(df[f"{TEMP}_qc_agg"])
    salt_flags = _flag_counts(df[f"{SALT}_qc_agg"])

    df = filter_qartod(df, max_flag=max_flag).dropna(subset=["z", TEMP, SALT])
    n_kept = len(df)
    if df.empty:
        raise ValueError(
            f"{dataset_key}: every row failed QARTOD filtering "
            f"(0/{n_raw} kept at max_flag={max_flag}; "
            f"temp_flags={{{temp_flags}}}, salt_flags={{{salt_flags}}})")

    # ERDDAP's `z` is negative-down (height); SoundSpeedProfile.from_ts and
    # the rest of this module want positive-down depth. Flip sign once, here.
    z_down = -df["z"].to_numpy(dtype=float)

    span = z_down.max() - z_down.min()
    if span < MIN_PROFILE_SPAN_M:
        raise ValueError(
            f"{dataset_key}: QC-passing depth span is only {span:.1f} m "
            f"(< {MIN_PROFILE_SPAN_M:.0f} m minimum), too small to be a "
            f"profile -- this instrument is likely parked, not profiling. "
            f"Known status: {NOV_2021_STATUS.get(dataset_key, 'unknown')}")

    bins = np.linspace(z_down.min(), z_down.max(), n_bins + 1)
    idx = np.clip(np.digitize(z_down, bins) - 1, 0, n_bins - 1)
    grouped = pd.DataFrame({"bin": idx, "z": z_down,
                             "t": df[TEMP].to_numpy(float),
                             "s": df[SALT].to_numpy(float)}).groupby("bin").mean()

    pct = 100.0 * n_kept / n_raw if n_raw else 0.0
    qc_note = (f"qc<={max_flag} kept={n_kept}/{n_raw}({pct:.0f}%) "
               f"temp_flags={{{temp_flags}}} salt_flags={{{salt_flags}}}")
    return SoundSpeedProfile.from_ts(
        grouped["z"].to_numpy(), grouped["t"].to_numpy(), grouped["s"].to_numpy(),
        lat, lon, source=f"ooi:{dataset_key}:{start}..{end}:{qc_note}")
