"""Straight-ray travel-time forward model and misfit fields.

Deliberately explicit rather than clever: this arithmetic is mirrored in
viz/template/lib/forward.mjs and the two are held together by a parity test.
Every published DAS localization method fixes the sound speed (1440/1480/1500
m/s); here `c` is always an explicit argument so it can be varied and fitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_C = 1480.0
OBJECTIVES = ("l1", "l2")


def travel_time(sx, sy, sz, px, py, pz, c):
    """Straight-ray travel time from source (sx, sy, sz) to each pick point."""
    dx = np.asarray(px, float) - sx
    dy = np.asarray(py, float) - sy
    dz = np.asarray(pz, float) - sz
    return np.sqrt(dx * dx + dy * dy + dz * dz) / c


def best_t0(obs_t, tt, objective="l1"):
    """Emission time that minimises the objective, eliminated analytically."""
    r = np.asarray(obs_t, float) - np.asarray(tt, float)
    if objective == "l1":
        return float(np.median(r))
    if objective == "l2":
        return float(np.mean(r))
    raise ValueError(f"unknown objective: {objective!r}; expected one of {OBJECTIVES}")


def misfit_value(obs_t, tt, objective="l1"):
    r = np.asarray(obs_t, float) - np.asarray(tt, float) - best_t0(obs_t, tt, objective)
    if objective == "l1":
        return float(np.abs(r).sum())
    return float(np.sqrt(np.mean(r * r)))


def misfit_field(xs, ys, sz, px, py, pz, obs_t, c, objective="l1"):
    """Misfit over a grid. Returns shape (len(ys), len(xs))."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    px = np.asarray(px, float)
    py = np.asarray(py, float)
    pz = np.asarray(pz, float)
    obs = np.asarray(obs_t, float)

    out = np.empty((len(ys), len(xs)), dtype=float)
    dz2 = (pz - sz) ** 2
    for j, wy in enumerate(ys):
        # (npicks, nxs) travel times for this row of the grid.
        dx = px[:, None] - xs[None, :]
        dy = py - wy
        tt = np.sqrt(dx * dx + (dy * dy)[:, None] + dz2[:, None]) / c
        r = obs[:, None] - tt
        if objective == "l1":
            out[j] = np.abs(r - np.median(r, axis=0)).sum(axis=0)
        elif objective == "l2":
            rc = r - r.mean(axis=0)
            out[j] = np.sqrt((rc * rc).mean(axis=0))
        else:
            raise ValueError(f"unknown objective: {objective!r}")
    return out


def signed_side(cable, x, y):
    """+1 / -1 for which side of the cable each point lies on.

    Uses the local tangent at the nearest cable node, so it follows the cable's
    curvature rather than assuming a straight line.
    """
    cx = np.asarray(cable.x, float)
    cy = np.asarray(cable.y, float)
    x = np.atleast_1d(np.asarray(x, float))
    y = np.atleast_1d(np.asarray(y, float))

    k = np.argmin((cx[None, :] - x[:, None]) ** 2 + (cy[None, :] - y[:, None]) ** 2, axis=1)
    k0 = np.clip(k - 1, 0, len(cx) - 1)
    k1 = np.clip(k + 1, 0, len(cx) - 1)
    tx, ty = cx[k1] - cx[k0], cy[k1] - cy[k0]
    nx, ny = x - cx[k], y - cy[k]
    s = np.sign(tx * ny - ty * nx)
    return np.where(s == 0, 1.0, s)


@dataclass(frozen=True)
class MirrorPenalty:
    best_misfit: float
    mirror_misfit: float
    ratio: float
    best_xy: tuple[float, float]
    mirror_xy: tuple[float, float]


def mirror_penalty(cable, xs, ys, sz, px, py, pz, obs_t, c, objective="l1"):
    """Best misfit on each side of the cable, and the wrong-side penalty ratio.

    Defined per reference cable. With picks from two cables there is no single
    mirror, so callers should report one of these per cable instead.
    """
    field = misfit_field(xs, ys, sz, px, py, pz, obs_t, c, objective)
    gx, gy = np.meshgrid(np.asarray(xs, float), np.asarray(ys, float))
    side = signed_side(cable, gx.ravel(), gy.ravel()).reshape(field.shape)

    out = []
    for sgn in (1.0, -1.0):
        masked = np.where(side == sgn, field, np.inf)
        j, i = np.unravel_index(masked.argmin(), masked.shape)
        out.append((float(masked[j, i]), (float(gx[j, i]), float(gy[j, i]))))
    out.sort(key=lambda t: t[0])
    (m1, p1), (m2, p2) = out
    if m1 > 0:
        ratio = float(m2 / m1)
    elif m1 == m2:
        ratio = 1.0
    else:
        ratio = float("inf")
    return MirrorPenalty(m1, m2, ratio, p1, p2)
