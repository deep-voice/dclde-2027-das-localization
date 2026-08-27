"""TDOA-style whale localization from DAS arrival-time picks.

Refactored from the explainer notebook's `localize_whale()`: for each
candidate whale position on a grid, the optimal emission time is the median
residual (for L1 robustness), and the misfit is the sum of absolute
residuals. The grid search is vectorized (the notebook's version used
nested Python loops despite a comment claiming otherwise), then refined
with Nelder-Mead.
"""
import numpy as np
from scipy.optimize import minimize

DEFAULT_SOUND_SPEED = 1480.0  # m/s


def localize_2d(
    picks_x,
    picks_y,
    picks_t,
    c=DEFAULT_SOUND_SPEED,
    x_range=(10000, 80000),
    y_range=(10000, 35000),
    grid_size=80,
    refine=True,
):
    """Localize a whale's (x, y) position from arrival-time picks.

    Parameters:
        picks_x, picks_y: physical coordinates (m) of each pick point on the cable(s)
        picks_t: observed arrival time (s) at each pick point
        c: sound speed in water (m/s)
        x_range, y_range: search area (m)
        grid_size: number of grid points per axis
        refine: whether to refine the grid-search optimum with Nelder-Mead

    Returns a dict with 'x', 'y', 't0', 'residual', 'per_pick_residuals',
    'residual_map', 'grid_xs', 'grid_ys'.
    """
    picks_x = np.asarray(picks_x, dtype=float)
    picks_y = np.asarray(picks_y, dtype=float)
    picks_t = np.asarray(picks_t, dtype=float)

    xs = np.linspace(x_range[0], x_range[1], grid_size)
    ys = np.linspace(y_range[0], y_range[1], grid_size)
    wx_grid, wy_grid = np.meshgrid(xs, ys)  # shape (grid_size, grid_size)

    dx = wx_grid[..., None] - picks_x[None, None, :]
    dy = wy_grid[..., None] - picks_y[None, None, :]
    travel_times = np.sqrt(dx**2 + dy**2) / c  # shape (grid_size, grid_size, n_picks)

    t0_grid = np.median(picks_t[None, None, :] - travel_times, axis=-1)
    residual_map = np.sum(np.abs(picks_t[None, None, :] - (t0_grid[..., None] + travel_times)), axis=-1)

    best_i, best_j = np.unravel_index(np.argmin(residual_map), residual_map.shape)
    best_pos = (wx_grid[best_i, best_j], wy_grid[best_i, best_j])
    best_t0 = t0_grid[best_i, best_j]
    best_residual = residual_map[best_i, best_j]

    if refine:
        def objective(params):
            wx, wy, t0 = params
            dists = np.sqrt((wx - picks_x) ** 2 + (wy - picks_y) ** 2)
            return np.sum(np.abs(picks_t - (t0 + dists / c)))

        result = minimize(
            objective,
            [best_pos[0], best_pos[1], best_t0],
            method="Nelder-Mead",
            options={"xatol": 10, "fatol": 0.001, "maxiter": 10000},
        )
        wx, wy, t0 = result.x
        final_residual = result.fun
    else:
        wx, wy, t0 = best_pos[0], best_pos[1], best_t0
        final_residual = best_residual

    dists = np.sqrt((wx - picks_x) ** 2 + (wy - picks_y) ** 2)
    per_pick_res = picks_t - (t0 + dists / c)

    return {
        "x": wx,
        "y": wy,
        "t0": t0,
        "residual": final_residual,
        "per_pick_residuals": per_pick_res,
        "residual_map": residual_map,
        "grid_xs": xs,
        "grid_ys": ys,
    }


def localize_3d(
    picks_x,
    picks_y,
    picks_z,
    picks_t,
    c=DEFAULT_SOUND_SPEED,
    x_range=(10000, 80000),
    y_range=(10000, 35000),
    grid_size=80,
    z0=0.0,
):
    """Localize a whale's (x, y, z) position (the L-2, dual-fiber case).

    `picks_z` is each pick's channel depth (m) from the cable geometry.
    Seeds from a 2D grid search (ignoring depth, which is a good starting
    point since channel depth is a second-order effect on the misfit
    landscape's horizontal shape), then refines all four parameters
    (wx, wy, wz, t0) jointly with Nelder-Mead using the true 3D distance.
    """
    picks_z = np.asarray(picks_z, dtype=float)

    seed = localize_2d(
        picks_x, picks_y, picks_t, c=c, x_range=x_range, y_range=y_range, grid_size=grid_size, refine=False
    )

    picks_x = np.asarray(picks_x, dtype=float)
    picks_y = np.asarray(picks_y, dtype=float)
    picks_t = np.asarray(picks_t, dtype=float)

    def objective(params):
        wx, wy, wz, t0 = params
        dists = np.sqrt((wx - picks_x) ** 2 + (wy - picks_y) ** 2 + (wz - picks_z) ** 2)
        return np.sum(np.abs(picks_t - (t0 + dists / c)))

    result = minimize(
        objective,
        [seed["x"], seed["y"], z0, seed["t0"]],
        method="Nelder-Mead",
        options={"xatol": 10, "fatol": 0.001, "maxiter": 20000},
    )
    wx, wy, wz, t0 = result.x

    dists = np.sqrt((wx - picks_x) ** 2 + (wy - picks_y) ** 2 + (wz - picks_z) ** 2)
    per_pick_res = picks_t - (t0 + dists / c)

    return {
        "x": wx,
        "y": wy,
        "z": wz,
        "t0": t0,
        "residual": result.fun,
        "per_pick_residuals": per_pick_res,
    }


def jackknife_uncertainty(picks_x, picks_y, picks_t, x0, y0, t0, c=DEFAULT_SOUND_SPEED):
    """Leave-one-pick-out uncertainty estimate around an existing (x0, y0, t0) fit.

    Refines from (x0, y0, t0) with one pick held out at a time (no grid
    search -- reuses the full-data solution as the starting point for
    speed) and reports the spread of the resulting position estimates.
    Requires an uncertainty estimate per the challenge's submission format.
    """
    picks_x = np.asarray(picks_x, dtype=float)
    picks_y = np.asarray(picks_y, dtype=float)
    picks_t = np.asarray(picks_t, dtype=float)
    n = len(picks_x)
    if n < 4:
        return {"x_std": np.nan, "y_std": np.nan, "position_std": np.nan, "estimates": np.empty((0, 2))}

    estimates = np.empty((n, 2))
    for i in range(n):
        mask = np.arange(n) != i

        def objective(params, mask=mask):
            wx, wy, t0_ = params
            dists = np.sqrt((wx - picks_x[mask]) ** 2 + (wy - picks_y[mask]) ** 2)
            return np.sum(np.abs(picks_t[mask] - (t0_ + dists / c)))

        result = minimize(
            objective,
            [x0, y0, t0],
            method="Nelder-Mead",
            options={"xatol": 10, "fatol": 0.001, "maxiter": 5000},
        )
        estimates[i] = result.x[:2]

    x_std = estimates[:, 0].std(ddof=1)
    y_std = estimates[:, 1].std(ddof=1)
    return {
        "x_std": x_std,
        "y_std": y_std,
        "position_std": float(np.sqrt(x_std**2 + y_std**2)),
        "estimates": estimates,
    }
