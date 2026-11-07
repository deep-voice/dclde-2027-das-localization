import numpy as np
import pandas as pd
import pytest

from dclde_localization.physics.geometry import (
    CHAN_IDX0,
    CHAN_SPACING_M,
    CableGeometry,
    FrameTransform,
    decimate_indices,
    load_cable,
)

NORTH_CSV = "fixtures/north_DAS_multicoord.csv"


@pytest.fixture(scope="module")
def north() -> CableGeometry:
    return load_cable("North", NORTH_CSV)


def test_load_cable_shape_and_constants(north):
    assert north.name == "North"
    assert len(north.x) == 31518
    assert north.spacing_m == pytest.approx(CHAN_SPACING_M, abs=1e-9)
    assert north.chan_m0 == pytest.approx(1923.474216, abs=1e-6)


@pytest.mark.parametrize("csv", [
    "fixtures/north_DAS_multicoord.csv",
    "fixtures/south_DAS_multicoord.csv",
])
def test_raw_chan_m_column_is_linear_in_channel_index(csv):
    # The bundle ships chanM explicitly now (decimate_indices keeps the final
    # channel, so the decimated samples are not uniformly spaced at the tail
    # and start+step reconstruction would be wrong there). This test still
    # earns its keep: it pins the CSV's chan_m against CHAN_SPACING_M, which
    # chan_to_xyz depends on, so a non-uniform cable would silently corrupt
    # every pick position.
    #
    # This reads the CSV directly and compares the DATA against the documented
    # constant. Asserting on CableGeometry.chan_m instead would be circular:
    # that property IS the reconstruction, so it would be compared to its own
    # formula and could never fail.
    df = pd.read_csv(csv).sort_values("chan_idx")
    raw = df["chan_m"].to_numpy()
    expected = raw[0] + np.arange(len(raw)) * CHAN_SPACING_M
    assert np.abs(raw - expected).max() < 1e-3


def _corrupt_fixture(tmp_path, column, row, delta):
    df = pd.read_csv("fixtures/north_DAS_multicoord.csv", nrows=200)
    df.loc[row, column] = df.loc[row, column] + delta
    path = tmp_path / "corrupt.csv"
    df.to_csv(path, index=False)
    return path


def test_load_cable_rejects_non_uniform_spacing(tmp_path):
    bad = _corrupt_fixture(tmp_path, "chan_m", 100, 5.0)
    with pytest.raises(ValueError, match="not uniform"):
        load_cable("North", bad)


def test_load_cable_rejects_non_contiguous_chan_idx(tmp_path):
    bad = _corrupt_fixture(tmp_path, "chan_idx", 100, 7)
    with pytest.raises(ValueError, match="contiguous"):
        load_cable("North", bad)


def test_chan_to_xyz_matches_table_at_channel_centres(north):
    idx = np.array([0, 100, 5000, 31517])
    x, y, z = north.chan_to_xyz(north.chan_m[idx])
    assert np.allclose(x, north.x[idx], atol=1e-6)
    assert np.allclose(y, north.y[idx], atol=1e-6)
    assert np.allclose(z, north.depth[idx], atol=1e-6)


def test_chan_to_xyz_clamps_outside_range(north):
    x, y, _ = north.chan_to_xyz([-1e6, 1e9])
    assert x[0] == pytest.approx(north.x[0])
    assert x[1] == pytest.approx(north.x[-1])


def test_frame_transform_offsets_are_exact(north):
    t = FrameTransform.from_cable(north)
    assert t.x_offset == pytest.approx(435226.0809, abs=1e-3)
    assert t.y_offset == pytest.approx(4985111.5099, abs=1e-3)


def test_frame_transform_reproduces_table_utm(north):
    t = FrameTransform.from_cable(north)
    ux, uy = t.local_to_utm(north.x, north.y)
    assert np.abs(ux - north.utm_x).max() < 1e-6
    assert np.abs(uy - north.utm_y).max() < 1e-6


def test_frame_transform_reproduces_table_latlon(north):
    # pyproj UTM10N -> WGS84 must reproduce the dataset's own lat/lon.
    t = FrameTransform.from_cable(north)
    lat, lon = t.local_to_wgs84(north.x, north.y)
    assert np.abs(lat - north.lat).max() < 1e-6
    assert np.abs(lon - north.lon).max() < 1e-6


def test_frame_roundtrip_wgs84(north):
    t = FrameTransform.from_cable(north)
    lat, lon = t.local_to_wgs84(north.x, north.y)
    x2, y2 = t.wgs84_to_local(lat, lon)
    assert np.abs(x2 - north.x).max() < 0.01
    assert np.abs(y2 - north.y).max() < 0.01


def test_to_local_dispatches_on_frame(north):
    t = FrameTransform.from_cable(north)
    x, y = t.to_local("local_xy", north.x[:5], north.y[:5])
    assert np.allclose(x, north.x[:5])
    ux, uy = t.local_to_utm(north.x[:5], north.y[:5])
    x2, y2 = t.to_local("utm10n", ux, uy)
    assert np.allclose(x2, north.x[:5], atol=1e-6)
    with pytest.raises(ValueError, match="unknown frame"):
        t.to_local("mercator", 0, 0)


def test_decimate_indices_always_includes_both_ends():
    idx = decimate_indices(10, 4)
    assert idx[0] == 0
    assert idx[-1] == 9
    assert np.array_equal(idx, np.array([0, 4, 8, 9]))
    # Already-aligned lengths must not duplicate the final index.
    assert np.array_equal(decimate_indices(9, 4), np.array([0, 4, 8]))


@pytest.mark.parametrize("name,csv", [
    ("North", "fixtures/north_DAS_multicoord.csv"),
    ("South", "fixtures/south_DAS_multicoord.csv"),
])
def test_decimation_error_under_one_metre(name, csv):
    # The bundle ships a decimated cable; interpolating back must stay sub-metre.
    # Plain [::4] leaves 1-3 channels past the last kept sample, where np.interp
    # clamps rather than interpolating: that tail costs 1.8 m on North and 6.3 m
    # on South. decimate_indices always keeps the final channel, which is why
    # both the bundle and this test must go through it.
    cable = load_cable(name, csv)
    idx = decimate_indices(len(cable.x), 4)
    xi = np.interp(cable.chan_m, cable.chan_m[idx], cable.x[idx])
    yi = np.interp(cable.chan_m, cable.chan_m[idx], cable.y[idx])
    err = np.hypot(xi - cable.x, yi - cable.y)
    assert err.max() < 1.0, f"{name}: max decimation error {err.max():.3f} m"
