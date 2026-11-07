import numpy as np
import pandas as pd
import pytest

from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile
from dclde_soundspeed.soundspeed.reanalysis import (
    check_sound_speed_sanity, compare_profiles, coverage_stats,
    fetch_hycom_profile, load_profile, save_profile, validation_summary)


def test_comparison_is_restricted_to_the_overlapping_depth_band():
    # Comparing outside the overlap would silently compare a measurement with a
    # clamped extrapolation and call the agreement good.
    a = SoundSpeedProfile(np.array([0.0, 500.0, 1500.0]),
                          np.array([1490.0, 1477.0, 1484.0]), "reanalysis")
    b = SoundSpeedProfile(np.array([100.0, 300.0, 575.0]),
                          np.array([1481.0, 1478.5, 1477.2]), "in-situ")
    df = compare_profiles(a, b, 100.0, 575.0)
    assert df["z_down"].min() >= 100.0
    assert df["z_down"].max() <= 575.0


def test_identical_profiles_show_zero_disagreement():
    p = SoundSpeedProfile(np.array([100.0, 300.0, 575.0]),
                          np.array([1481.0, 1478.5, 1477.2]), "x")
    s = validation_summary(compare_profiles(p, p, 100.0, 575.0))
    assert s["median_abs_diff_ms"] == pytest.approx(0.0, abs=1e-9)


def test_summary_reports_its_denominator():
    a = SoundSpeedProfile(np.array([100.0, 575.0]), np.array([1481.0, 1477.0]), "a")
    b = SoundSpeedProfile(np.array([100.0, 575.0]), np.array([1483.0, 1479.0]), "b")
    s = validation_summary(compare_profiles(a, b, 100.0, 575.0))
    assert s["n"] > 0
    assert s["median_abs_diff_ms"] == pytest.approx(2.0, abs=1e-6)


def test_compare_profiles_raises_when_bands_do_not_overlap():
    a = SoundSpeedProfile(np.array([0.0, 100.0]), np.array([1490.0, 1480.0]), "a")
    b = SoundSpeedProfile(np.array([500.0, 600.0]), np.array([1477.0, 1478.0]), "b")
    with pytest.raises(ValueError, match="do not overlap"):
        compare_profiles(a, b, 0.0, 1000.0)


# --- offline parse-path coverage for fetch_hycom_profile ---
#
# The brief's own tests only cover compare_profiles/validation_summary; the
# fetch parse path is exactly where a silent bug would live. Note the header
# below (`vertCoord[unit="m"]`, not "depth") and the raw packed values are
# copied verbatim from a real NCSS response captured 2026-09-17 -- reading
# -6483.0 directly as degrees C is the bug this test exists to catch.

T = 'water_temp[unit="degC"]'
S = 'salinity[unit="psu"]'


class _FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    """Records the requested URL (with params expanded) and returns a canned
    response, no network -- mirrors the pattern in test_erddap.py."""

    def __init__(self, text, status_code=200):
        self._text = text
        self._status_code = status_code
        self.requested_url = None
        self.requested_params = None

    def get(self, url, params=None, timeout=None):
        self.requested_url = url
        self.requested_params = params
        return _FakeResponse(self._text, self._status_code)


def _ncss_csv(rows):
    """Build a canned NCSS-shaped CSV: real column names, real packing.

    `rows` is a list of (vertCoord, water_temp_raw_or_None, salinity_raw_or_None).
    None renders as an empty field (NaN once pandas parses it), matching how a
    real gap would show up rather than the fill sentinel.
    """
    header = ('time,latitude[unit="degrees_north"],longitude[unit="degrees_east"],'
              f'vertCoord[unit="m"],{T},{S}')
    lines = [header]
    for z, t_raw, s_raw in rows:
        t_field = "" if t_raw is None else str(t_raw)
        s_field = "" if s_raw is None else str(s_raw)
        lines.append(f'2021-11-04T12:00:00Z,44.37,-124.95,{z},{t_field},{s_field}')
    return "\n".join(lines) + "\n"


def test_fetch_hycom_profile_unpacks_scale_offset_and_matches_gsw_bounds():
    # Real raw values captured live at 44.37,-124.95 for 2021-11-04: depths
    # given out of time-series order (100 before 50) to exercise the sort,
    # plus a fill-value row (below this shallow shelf site's ~600 m seafloor)
    # and an explicit NaN row, both of which must be dropped.
    rows = [
        (0.0, -6483.0, 12319.0),
        (100.0, -8239.0, 13188.0),
        (50.0, -8012.0, 12763.0),
        (700.0, -30000.0, -30000.0),   # fill sentinel: below local seafloor
        (900.0, None, None),           # explicit NaN/empty field
    ]
    session = _FakeSession(_ncss_csv(rows))

    profile = fetch_hycom_profile(44.37, -124.95, "2021-11-04", session=session)

    assert session.requested_params["latitude"] == 44.37
    assert session.requested_params["time"] == "2021-11-04T12:00:00Z"

    # Sorted, strictly ascending, fill/NaN rows dropped: only 0, 50, 100 m survive.
    assert np.allclose(profile.z_down, [0.0, 50.0, 100.0])

    # Unpacking check: raw -6483 with scale 0.001 / offset 20.0 -> 13.517 degC,
    # a plausible November shelf-water temperature. If the raw value were used
    # directly (the bug this test exists to catch) sound speed would be wildly
    # unphysical instead of in the real ocean band.
    assert np.all((profile.c > 1400.0) & (profile.c < 1550.0))

    assert profile.source.startswith("hycom:2021-11-04:44.37,-124.95:")
    assert "kept=3/5" in profile.source


def test_fetch_hycom_profile_raises_when_every_row_is_fill_or_nan():
    rows = [(700.0, -30000.0, -30000.0), (900.0, None, None)]
    session = _FakeSession(_ncss_csv(rows))
    with pytest.raises(ValueError, match="fill value or NaN"):
        fetch_hycom_profile(44.37, -124.95, "2021-11-04", session=session)


# --- save_profile / load_profile: the committed, network-free artifact ---
#
# experiments.runner.PROFILES["hycom_oregon_nov"] loads a profile this way at
# import time, with no network -- these round-trip the exact mechanism it
# depends on, including the one thing a bare (z, c) CSV would otherwise lose:
# provenance.

def test_save_and_load_profile_round_trips_values_and_provenance(tmp_path):
    original = SoundSpeedProfile(
        np.array([0.0, 100.0, 500.0]), np.array([1490.0, 1480.0, 1477.0]),
        "hycom:2021-11-04:44.37,-124.95:kept=29/40(72%) z=0..600m")
    path = save_profile(original, tmp_path / "hycom_profile.csv")

    loaded = load_profile(path)

    assert np.allclose(loaded.z_down, original.z_down)
    assert np.allclose(loaded.c, original.c)
    assert loaded.source == original.source


def test_save_profile_writes_a_hash_prefixed_provenance_header_readable_by_pandas(tmp_path):
    # Same header convention as experiments.runner.write_results: '#'-prefixed
    # lines before the CSV body, so pd.read_csv(path, comment="#") is the one
    # documented way to read it back, and the source line is human-legible in
    # a plain diff without needing load_profile at all.
    p = SoundSpeedProfile(np.array([0.0, 10.0]), np.array([1480.0, 1481.0]), "src:x")
    path = save_profile(p, tmp_path / "p.csv")

    raw = path.read_text(encoding="utf-8")
    header_lines = [ln for ln in raw.splitlines() if ln.startswith("#")]
    assert any(ln == "# source: src:x" for ln in header_lines)

    got = pd.read_csv(path, comment="#")
    assert list(got.columns) == ["z_down", "c"]


def test_load_profile_refuses_a_file_with_no_provenance_header(tmp_path):
    path = tmp_path / "no_header.csv"
    pd.DataFrame({"z_down": [0.0, 10.0], "c": [1480.0, 1481.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="source"):
        load_profile(path)


# --- check_sound_speed_sanity / coverage_stats: review round 2 additions ---
#
# check_sound_speed_sanity is the "report, don't silently fall back" gate the
# coordinator asked for when fetching at a new (never-validated) location.
# coverage_stats is the pure-function version of the mechanism the review
# finding actually named: a profile that doesn't reach as deep as the picks
# it will be used on produces clamped extrapolation, not a real answer.

def test_check_sound_speed_sanity_accepts_a_physically_plausible_profile():
    p = SoundSpeedProfile(np.array([0.0, 500.0, 1500.0]),
                          np.array([1490.0, 1477.0, 1484.0]), "ok")
    check_sound_speed_sanity(p)  # must not raise


def test_check_sound_speed_sanity_rejects_an_implausible_profile():
    # Sound speeds like these would come from a corrupted fetch or a swapped
    # temperature/salinity argument (from_ts has no way to detect that
    # itself) -- this must be reported loudly, not silently accepted.
    p = SoundSpeedProfile(np.array([0.0, 500.0]), np.array([13.5, 32.3]), "corrupt")
    with pytest.raises(ValueError, match="sane band"):
        check_sound_speed_sanity(p)


def test_coverage_stats_reports_full_coverage_when_profile_reaches_the_deepest_pick():
    p = SoundSpeedProfile(np.array([0.0, 1500.0]), np.array([1490.0, 1484.0]), "deep")
    stats = coverage_stats(p, np.array([0.0, 500.0, 1500.0]))
    assert stats == {"n": 3, "n_covered": 3, "pct_covered": 100.0,
                      "profile_z_max_m": 1500.0, "max_extrapolation_m": 0.0}


def test_coverage_stats_reports_the_extrapolation_gap_with_its_denominator():
    # Mirrors the actual review finding: a profile shallower than the real
    # pick range leaves some picks beyond its deepest sample, where
    # path_average/c_at fall back to a clamped boundary value.
    p = SoundSpeedProfile(np.array([0.0, 600.0]), np.array([1490.0, 1477.0]), "shallow")
    stats = coverage_stats(p, np.array([100.0, 600.0, 900.0, 1575.0]))
    assert stats["n"] == 4
    assert stats["n_covered"] == 2
    assert stats["pct_covered"] == pytest.approx(50.0)
    assert stats["profile_z_max_m"] == 600.0
    assert stats["max_extrapolation_m"] == pytest.approx(1575.0 - 600.0)
