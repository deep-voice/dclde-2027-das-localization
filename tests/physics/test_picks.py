import numpy as np
import pytest

from dclde_localization.physics.geometry import load_cable
from dclde_localization.physics.picks import attach_xyz, call_key, load_picks


def test_call_key_normalises_colons_and_dashes():
    assert call_key("Denoised_SNR_North_2021-11-04_02:00:02.nc", 1) == "2021-11-04_02_00_02/1"
    assert call_key("Denoised_SNR_South_2021-11-04_02-00-02.nc", 7) == "2021-11-04_02_00_02/7"
    assert call_key("annotated_calls_north_2021-11-04_02_00_02.csv", 3) == "2021-11-04_02_00_02/3"


def test_call_key_rejects_a_filename_with_no_timestamp():
    with pytest.raises(ValueError, match="no timestamp"):
        call_key("Denoised_SNR_North.nc", 1)


@pytest.fixture(scope="module")
def picks():
    return load_picks()


def test_load_picks_matches_known_totals(picks):
    assert len(picks) == 11117
    assert picks["call"].nunique() == 628
    assert picks["window"].nunique() == 60
    assert set(picks["cable"].unique()) == {"North", "South"}


def test_keys_on_both_cables_match_the_verified_count(picks):
    per_key = picks.groupby("call")["cable"].nunique()
    assert int((per_key == 2).sum()) == 166


@pytest.fixture(scope="module")
def cables():
    return {n: load_cable(n, f"fixtures/{n.lower()}_DAS_multicoord.csv")
            for n in ("North", "South")}


def test_attach_xyz_fills_every_row(picks, cables):
    out = attach_xyz(picks, cables)
    assert {"px", "py", "pz"} <= set(out.columns)
    assert out[["px", "py", "pz"]].notna().all().all()
    assert len(out) == len(picks)


@pytest.mark.parametrize("name,other", [("North", "South"), ("South", "North")])
def test_attach_xyz_uses_each_cable_its_own_geometry(picks, cables, name, other):
    # The real failure mode is cross-wiring: giving South's picks positions from
    # the North cable. Comparing against the SAME interpolation attach_xyz used
    # would be circular and would report exactly zero error, so instead require
    # that each cable's picks are clearly NOT explained by the other cable.
    out = attach_xyz(picks, cables)
    sub = out[out.cable == name]
    wx, wy, _ = cables[other].chan_to_xyz(sub["dist"].to_numpy())
    d = np.hypot(sub["px"].to_numpy() - wx, sub["py"].to_numpy() - wy)
    assert d.max() > 1000.0, f"{name} picks are indistinguishable from {other} geometry"


def test_attach_xyz_raises_on_an_unknown_cable_label(picks, cables):
    bad = picks.head(50).copy()
    bad.loc[bad.index[0], "cable"] = "Souh"
    with pytest.raises(ValueError, match="unknown cable"):
        attach_xyz(bad, cables)


@pytest.mark.parametrize("name,csv", [
    ("North", "fixtures/north_DAS_multicoord.csv"),
    ("South", "fixtures/south_DAS_multicoord.csv"),
])
def test_chan_m_matches_the_cables_own_arc_length(name, csv):
    # An independent check on what `dist` MEANS, which attach_xyz depends on
    # entirely: the dataset's along-cable distance should equal the arc length of
    # its own coordinate polyline. This compares the chan_m column against the
    # x/y/depth columns, so unlike an interpolation round-trip it would catch a
    # wrong unit or offset in `dist`. Measured agreement is 0.19% (North) and
    # 0.04% (South) of span; fiber slack makes exact equality unlikely.
    cable = load_cable(name, csv)
    seg = np.sqrt(np.diff(cable.x) ** 2 + np.diff(cable.y) ** 2 + np.diff(cable.depth) ** 2)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    dm = cable.chan_m - cable.chan_m[0]
    assert np.abs(arc - dm).max() < 0.005 * dm[-1]
