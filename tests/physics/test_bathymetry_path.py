import numpy as np
import pytest

from dclde_localization.physics.bathymetry import NO_DATA, LocalGrid, depth_along_path


def _grid(z):
    """A 1 km-spaced local grid holding `z` (int16 metres, negative = below sea level)."""
    z = np.asarray(z, dtype=np.int16)
    ny, nx = z.shape
    return LocalGrid(x0=0.0, y0=0.0, dx=1000.0, dy=1000.0, nx=nx, ny=ny, z=z)


def test_constant_depth_path_returns_that_depth_everywhere():
    grid = _grid(np.full((5, 5), -800))
    p = depth_along_path(grid, 0.0, 0.0, 4000.0, 0.0, n_steps=9)
    assert p.ok
    assert p.depth_m.shape == (9,)
    # Positive-DOWN water depth, so the -800 m elevation becomes +800 m of water.
    assert np.allclose(p.depth_m, 800.0)


def test_range_axis_spans_zero_to_the_horizontal_separation():
    grid = _grid(np.full((5, 5), -800))
    p = depth_along_path(grid, 0.0, 0.0, 3000.0, 4000.0, n_steps=5)
    assert p.ok
    assert p.range_m[0] == pytest.approx(0.0)
    assert p.range_m[-1] == pytest.approx(5000.0)  # 3-4-5 triangle
    assert np.all(np.diff(p.range_m) > 0)


def test_a_sloping_path_reports_the_slope():
    # Depth deepens with x: -100, -200, -300, -400, -500 m elevation.
    z = np.tile(np.array([-100, -200, -300, -400, -500], dtype=np.int16), (3, 1))
    p = depth_along_path(_grid(z), 0.0, 1000.0, 4000.0, 1000.0, n_steps=5)
    assert p.ok
    assert np.allclose(p.depth_m, [100.0, 200.0, 300.0, 400.0, 500.0])


def test_a_path_leaving_the_grid_fails_explicitly_and_does_not_extrapolate():
    grid = _grid(np.full((5, 5), -800))
    p = depth_along_path(grid, 0.0, 0.0, 99_000.0, 0.0, n_steps=9)
    assert not p.ok
    assert "outside" in p.reason
    # It must NOT quietly hand back a usable-looking profile.
    assert p.depth_m.size == 0


def test_a_path_crossing_a_no_data_hole_fails_rather_than_interpolating_across_it():
    z = np.full((5, 5), -800, dtype=np.int16)
    z[:, 2] = NO_DATA
    p = depth_along_path(_grid(z), 0.0, 1000.0, 4000.0, 1000.0, n_steps=5)
    assert not p.ok
    assert "no-data" in p.reason
    assert p.depth_m.size == 0


def test_land_is_rejected_because_there_is_no_water_column_to_propagate_in():
    z = np.full((5, 5), -800, dtype=np.int16)
    z[:, 2] = 150  # above sea level
    p = depth_along_path(_grid(z), 0.0, 1000.0, 4000.0, 1000.0, n_steps=5)
    assert not p.ok
    assert "land" in p.reason


def test_degenerate_path_is_rejected():
    grid = _grid(np.full((5, 5), -800))
    p = depth_along_path(grid, 1000.0, 1000.0, 1000.0, 1000.0, n_steps=5)
    assert not p.ok
    assert "zero-length" in p.reason


def test_nearest_neighbour_indexing_not_floor():
    # Regression: this repo shipped a floor-indexing bug once (commit 762488c).
    # Verify that depth_along_path uses round, not floor.
    # Grid has depth varying by x column: -100, -200, -300, -400, -500 m elevation
    z = np.zeros((5, 5), dtype=np.int16)
    for col in range(5):
        z[:, col] = -(100 * (col + 1))

    grid = _grid(z)
    # Path from x=750 to x=3750 with n_steps=5 hits off-node x-positions.
    # s = [0, 0.25, 0.5, 0.75, 1.0]
    # xs = [750, 1500, 2250, 3000, 3750]
    # With round: indices [1, 2, 2, 3, 4] → depths [200, 300, 300, 400, 500]
    # With floor: indices [0, 1, 2, 3, 3] → depths [100, 200, 300, 400, 400]
    p = depth_along_path(grid, 750.0, 2000.0, 3750.0, 2000.0, n_steps=5)
    assert p.ok
    assert np.allclose(p.depth_m, [200.0, 300.0, 300.0, 400.0, 500.0])
