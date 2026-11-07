"""Propagation models, all sharing one interface.

`c` cannot be precomputed per channel and handed to forward.travel_time: the
path average between source depth and channel depth depends on the source depth,
which is the unknown being solved for. So this is a model object, and its
interface takes positions rather than speeds.

All coordinates are in the REPO frame (negative down). Conversion to the
profile's positive-down frame happens here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from dclde_localization.physics.forward import DEFAULT_C, travel_time as _straight_ray
from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile, to_down


@runtime_checkable
class SoundSpeedModel(Protocol):
    name: str

    def travel_time(self, sx, sy, sz, px, py, pz) -> np.ndarray:
        """Travel time from source (sx, sy, sz) to each pick point.

        Broadcasting contract: sx, sy, sz may be scalars or arrays of any shape
        that broadcasts against px, py, pz (which are themselves broadcast
        against each other, typically as (n_picks,) arrays). This is what lets
        a caller evaluate a whole grid of candidate source positions against
        the same picks in one call, e.g. sx/sy of shape (ny, nx, 1) against
        picks of shape (n_picks,), yielding a result of shape (ny, nx, n_picks).
        No implementation may loop over picks or grid cells in Python; the
        whole point of this contract is that a 161x161 localization grid stays
        cheap.
        """
        ...


@dataclass(frozen=True)
class ConstantC:
    """The published baseline: one speed everywhere, straight rays.

    Delegates to the frozen forward.py so there is exactly one definition of
    this arithmetic in the codebase, enforced by test_models.py. forward.travel_time
    is already pure numpy broadcasting, so this satisfies the broadcasting
    contract with no extra work.
    """

    c: float = DEFAULT_C

    @property
    def name(self) -> str:
        return f"constant_c:{self.c:g}"

    def travel_time(self, sx, sy, sz, px, py, pz) -> np.ndarray:
        return _straight_ray(sx, sy, sz, px, py, pz, self.c)


@dataclass(frozen=True)
class PathAverage:
    """Straight geometric path, but traversed at the depth-averaged speed.

    Fermat's principle makes travel time stationary with respect to path
    perturbation, so to first order the refraction correction IS the change in
    path-averaged speed; the bending itself is second order. RayTrace (Task 3)
    measures what is left over.

    Fully vectorised: profile.path_average is itself O(1) and broadcasts, so
    c_eff is computed for every (source, pick) pair in one call with no Python
    loop over picks or grid cells. This matters because a localization search
    evaluates this over a full grid of candidate source positions.
    """

    profile: SoundSpeedProfile

    @property
    def name(self) -> str:
        return f"path_average:{self.profile.source}"

    def travel_time(self, sx, sy, sz, px, py, pz) -> np.ndarray:
        sx = np.asarray(sx, dtype=float)
        sy = np.asarray(sy, dtype=float)
        sz = np.asarray(sz, dtype=float)
        px = np.asarray(px, dtype=float)
        py = np.asarray(py, dtype=float)
        pz = np.asarray(pz, dtype=float)

        dist = np.sqrt((px - sx) ** 2 + (py - sy) ** 2 + (pz - sz) ** 2)
        zs_down = to_down(sz)
        zc_down = to_down(pz)
        c_eff = self.profile.path_average(zs_down, zc_down)
        return dist / c_eff
