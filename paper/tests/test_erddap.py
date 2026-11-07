import numpy as np
import pandas as pd
import pytest

from dclde_soundspeed.soundspeed.erddap import DATASETS, build_url, fetch_profile, filter_qartod

T = "sea_water_temperature_profiler_depth_enabled"
S = "sea_water_practical_salinity_profiler_depth_enabled"


def test_url_requests_the_qc_columns_alongside_the_data():
    # Fetching values without their QC flags is how you build a garbage profile:
    # the raw ranges include salinity 33929 PSU and temperature 61 C.
    url = build_url(DATASETS["ce04ospd_deep"], "2021-11-01", "2021-11-06")
    assert "ooi-ce04ospd-dp01b-01-ctdpfl105.csv" in url
    assert f"{T}_qc_agg" in url and f"{S}_qc_agg" in url
    assert "time%3E=2021-11-01" in url or "time>=2021-11-01" in url


def test_qartod_filter_rejects_suspect_and_fail_flags():
    df = pd.DataFrame({
        "z": [-100.0, -200.0, -300.0, -400.0],
        T: [8.0, 9.0, 61.0, 7.5], S: [33.8, 33.9, 33.9, 33929.0],
        f"{T}_qc_agg": [1, 2, 4, 1], f"{S}_qc_agg": [1, 2, 1, 3]})
    got = filter_qartod(df, max_flag=2)
    assert len(got) == 2
    assert 61.0 not in got[T].to_numpy()
    assert 33929.0 not in got[S].to_numpy()


def test_qartod_filter_keeps_everything_when_all_flags_pass():
    df = pd.DataFrame({"z": [-100.0], T: [8.0], S: [33.8],
                       f"{T}_qc_agg": [1], f"{S}_qc_agg": [1]})
    assert len(filter_qartod(df)) == 1


def test_known_gaps_are_recorded_in_the_module():
    # These were measured, not assumed. Anyone reaching for the Slope Base deep
    # profiler should find the note before spending an afternoon on a 404.
    from dclde_soundspeed.soundspeed import erddap
    assert "rs01sbpd_deep" in erddap.DATASETS
    assert "no data" in erddap.NOV_2021_STATUS["rs01sbpd_deep"].lower()
    assert "park" in erddap.NOV_2021_STATUS["rs01sbps_shallow"].lower()


# --- offline parse-path coverage (network is not exercised in this file) ---
#
# The brief's own tests only cover build_url and filter_qartod; the fetch_profile
# parse path (units-row skip, QC filtering, sign flip, depth binning) is exactly
# where a silent bug would live, so it gets a fake session here instead.

class _FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    """Records the requested URL and returns a canned response, no network."""

    def __init__(self, text, status_code=200):
        self._text = text
        self._status_code = status_code
        self.requested_url = None

    def get(self, url, timeout=None):
        self.requested_url = url
        return _FakeResponse(self._text, self._status_code)


def _erddap_csv(rows):
    """Build a canned ERDDAP-shaped CSV: header, units row, then data rows.

    `rows` is a list of (z_negative_down, temp_c, salt_psu, temp_qc, salt_qc).
    """
    header = f"time,z,{T},{S},{T}_qc_agg,{S}_qc_agg"
    units = "UTC,m,degree_C,1,1,1"
    lines = [header, units]
    for i, (z, t, s, tqc, sqc) in enumerate(rows):
        lines.append(f"2021-11-0{i + 1}T00:00:00Z,{z},{t},{s},{tqc},{sqc}")
    return "\n".join(lines) + "\n"


def test_fetch_profile_parses_units_row_filters_qc_and_flips_sign():
    # 8 real (QC-pass) samples spanning -100..-400 m, plus one polluted row
    # (temperature 61 C, QC flag 4) that must be dropped before it ever
    # reaches from_ts. Depths intentionally out of time order, matching a
    # profiler's up/down cast.
    rows = [
        (-100.0, 8.5, 33.6, 1, 1),
        (-150.0, 8.2, 33.7, 1, 1),
        (-200.0, 7.9, 33.8, 2, 1),
        (-250.0, 7.6, 33.85, 1, 2),
        (-300.0, 7.3, 33.9, 1, 1),
        (-350.0, 7.0, 33.95, 1, 1),
        (-400.0, 6.7, 34.0, 1, 1),
        (-100.0, 8.6, 33.6, 1, 1),
        (-300.0, 61.0, 33.9, 4, 1),  # polluted, must be filtered out
    ]
    session = _FakeSession(_erddap_csv(rows))

    profile = fetch_profile("ce04ospd_deep", "2021-11-01", "2021-11-06", 44.37, -124.95,
                             session=session, n_bins=4)

    # The fake session actually got hit, with the real dataset id in the URL.
    assert "ooi-ce04ospd-dp01b-01-ctdpfl105" in session.requested_url

    # Sign flip: ERDDAP z is negative-down; the profile is positive-down and
    # strictly ascending (SoundSpeedProfile enforces this in __post_init__).
    assert profile.z_down.min() > 0
    assert np.all(np.diff(profile.z_down) > 0)
    # Binned means fall within the sampled range; the shallowest bin averages
    # the two -100 m rows together with -150 m, and the deepest bin is -400 m
    # alone, so exact bin means depend on bin-edge placement rather than
    # matching a raw sample exactly.
    assert 95.0 <= profile.z_down.min() <= 150.0
    assert 350.0 <= profile.z_down.max() <= 405.0

    # The polluted 61 C / QC-4 row must not have survived into the profile;
    # if it leaked in, sound speed would be wildly outside the physical band.
    assert np.all((profile.c > 1400.0) & (profile.c < 1550.0))

    # source records provenance AND the QC denominator: threshold, kept/raw
    # counts, and the per-flag breakdown that fed that decision (review
    # finding 2 -- the caller must be able to see what was discarded without
    # reimplementing the fetch).
    assert profile.source.startswith("ooi:ce04ospd_deep:2021-11-01..2021-11-06:")
    assert "qc<=2" in profile.source
    assert "kept=8/9" in profile.source  # 8 of 9 raw rows pass; the 61 C row is the one reject
    assert "temp_flags={1:7,2:1,4:1}" in profile.source
    assert "salt_flags={1:8,2:1}" in profile.source


def test_fetch_profile_max_flag_is_a_caller_controlled_passthrough():
    # Same 9 rows as above, but loosen max_flag to 4 (accept "suspect"/"fail"
    # too) so the previously-rejected 61 C row survives. If max_flag were not
    # actually plumbed through to filter_qartod, this would be identical to
    # the default-threshold test above.
    rows = [
        (-100.0, 8.5, 33.6, 1, 1), (-150.0, 8.2, 33.7, 1, 1),
        (-200.0, 7.9, 33.8, 2, 1), (-250.0, 7.6, 33.85, 1, 2),
        (-300.0, 7.3, 33.9, 1, 1), (-350.0, 7.0, 33.95, 1, 1),
        (-400.0, 6.7, 34.0, 1, 1), (-100.0, 8.6, 33.6, 1, 1),
        (-300.0, 61.0, 33.9, 4, 1),
    ]
    default_profile = fetch_profile("ce04ospd_deep", "2021-11-01", "2021-11-06", 44.37, -124.95,
                                     session=_FakeSession(_erddap_csv(rows)), n_bins=4)
    loose_profile = fetch_profile("ce04ospd_deep", "2021-11-01", "2021-11-06", 44.37, -124.95,
                                   session=_FakeSession(_erddap_csv(rows)), n_bins=4, max_flag=4)

    assert "qc<=2" in default_profile.source and "kept=8/9" in default_profile.source
    assert "qc<=4" in loose_profile.source and "kept=9/9" in loose_profile.source

    # The 61 C outlier is excluded by default and included once max_flag=4 is
    # passed through, so it pulls up whichever bin's mean temperature it now
    # shares -- the two runs must produce measurably different sound speeds.
    # If max_flag were not actually plumbed through to filter_qartod, these
    # would be identical.
    assert loose_profile.c.max() > default_profile.c.max() + 20.0


def test_fetch_profile_raises_a_clear_error_on_404():
    session = _FakeSession("", status_code=404)
    with pytest.raises(ValueError, match="rs01sbpd_deep"):
        fetch_profile("rs01sbpd_deep", "2021-11-01", "2021-11-06", 44.5, -125.4,
                       session=session)


def test_fetch_profile_raises_when_every_row_fails_qartod():
    rows = [(-100.0, 8.0, 33.8, 4, 4), (-200.0, 7.5, 33.9, 3, 1)]
    session = _FakeSession(_erddap_csv(rows))
    with pytest.raises(ValueError, match="QARTOD"):
        fetch_profile("ce04ospd_deep", "2021-11-01", "2021-11-06", 44.37, -124.95,
                       session=session)


def test_fetch_profile_refuses_a_parked_instrument_depth_span():
    # rs01sbps_shallow was measured parked at z -195..-189 m: a ~6 m wobble,
    # not a profile. fetch_profile must refuse this at construction time
    # (naming the likely cause) rather than build a near-degenerate
    # SoundSpeedProfile that only fails three modules later in path_average
    # with an error that never mentions ERDDAP (review finding 3).
    rows = [
        (-195.0, 8.0, 33.8, 1, 1), (-193.0, 8.0, 33.8, 1, 1),
        (-191.0, 8.0, 33.8, 1, 1), (-189.0, 8.0, 33.8, 1, 1),
    ]
    session = _FakeSession(_erddap_csv(rows))
    with pytest.raises(ValueError, match="park"):
        fetch_profile("rs01sbps_shallow", "2021-11-01", "2021-11-06", 44.52, -124.95,
                       session=session)
