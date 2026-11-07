import numpy as np
import pytest

from dclde_localization.physics.geometry import FrameTransform, load_cable
from viz.bathymetry import (
    NO_DATA,
    _sample,
    decode_gmt_grid,
    reproject_to_local,
    validate_against_cable,
)

GRID = "fixtures/gmrt_bathy.nc"
SOUTH_CSV = "fixtures/south_DAS_multicoord.csv"


@pytest.fixture(scope="module")
def grid():
    return decode_gmt_grid(GRID)


def test_decode_shape_and_orientation(grid):
    # Pinned dimensions rather than `z.shape == (len(lat), len(lon))`: that form
    # cannot fail, since both sides come from the same nx/ny. Pinning also
    # catches GMRT silently changing its response resolution, which would change
    # the bundle payload size.
    assert (len(grid.lon), len(grid.lat)) == (663, 294)
    assert grid.z.shape == (294, 663)
    assert grid.lat[0] < grid.lat[-1], "latitude must be ascending"
    assert grid.lon[0] < grid.lon[-1]
    assert -3000 < np.nanmin(grid.z) < 0 < np.nanmax(grid.z) < 2000


@pytest.mark.parametrize("k", [10, 100])
def test_sample_at_an_exact_grid_node_returns_that_node(grid, k):
    got = _sample(grid, np.array([grid.lat[k]]), np.array([grid.lon[k]]))
    assert np.array_equal(got, np.array([grid.z[k, k]]), equal_nan=True)


def test_sample_is_nearest_not_floor(grid):
    # 60% of the way from node 10 to node 11: a floor lookup returns node 10,
    # true nearest-neighbour returns node 11. Measured on the real grid, the two
    # conventions disagree on 74.9% of cable samples.
    dlat = grid.lat[1] - grid.lat[0]
    dlon = grid.lon[1] - grid.lon[0]
    got = _sample(
        grid,
        np.array([grid.lat[10] + 0.6 * dlat]),
        np.array([grid.lon[10] + 0.6 * dlon]),
    )
    assert got[0] == pytest.approx(grid.z[11, 11])
    assert got[0] != pytest.approx(grid.z[10, 10])


def test_sample_marks_out_of_bounds_as_nan(grid):
    got = _sample(grid, np.array([0.0, 89.0]), np.array([0.0, 179.0]))
    assert np.isnan(got).all()


def test_gmrt_agrees_with_the_cables_own_depth(grid):
    # Independent cross-validation of both the bathymetry and the cable
    # geolocation: these are different measurements of the same seafloor.
    cable = load_cable("South", SOUTH_CSV)
    rep = validate_against_cable(grid, cable)
    assert rep.n > 40000
    assert abs(rep.median_offset_m) < 15.0
    assert rep.correlation > 0.999


def test_reproject_to_local_preserves_depth_at_the_cable(grid):
    cable = load_cable("South", SOUTH_CSV)
    t = FrameTransform.from_cable(cable)
    lg = reproject_to_local(grid, t, nx=663, ny=294)

    assert lg.z.dtype == np.int16
    assert lg.nx == 663 and lg.ny == 294
    assert lg.z.shape == (lg.ny, lg.nx)

    i = ((cable.x - lg.x0) / lg.dx).round().astype(int)
    j = ((cable.y - lg.y0) / lg.dy).round().astype(int)
    inside = (i >= 0) & (i < lg.nx) & (j >= 0) & (j < lg.ny)
    sampled = lg.z[j[inside], i[inside]].astype(float)
    ok = sampled != NO_DATA
    diff = sampled[ok] - cable.depth[inside][ok]
    assert np.median(np.abs(diff)) < 30.0


def test_no_data_is_marked_not_zero(grid):
    cable = load_cable("South", SOUTH_CSV)
    lg = reproject_to_local(grid, FrameTransform.from_cable(cable))
    assert NO_DATA == -32768
    # The source grid has NaN cells; they must survive as the sentinel, because
    # letting them become 0 m would put fake sea level in the colour ramp.
    assert (lg.z == NO_DATA).sum() > 0
