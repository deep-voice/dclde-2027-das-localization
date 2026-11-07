"""Sound-speed research (paper) layer for DAS fin-whale localization.

The profile and the ConstantC/PathAverage models live in the core
(dclde_localization.physics.sound_speed_*) because solve() needs them; they are
re-exported here so older imports keep working.

Layers above the frozen physics/forward.py so the visualizer's Python/JavaScript
parity contract is untouched.
"""

from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile, to_down, to_repo
from dclde_localization.physics.sound_speed_models import ConstantC, PathAverage, SoundSpeedModel
from dclde_soundspeed.soundspeed.raytrace import (
    RayTrace, direct_time, max_direct_range, trace)
from dclde_soundspeed.soundspeed.modes import ModeSet, Seabed, modes
from dclde_soundspeed.soundspeed.tl import (
    AdiabaticResult, DegenerateGeometryError, adiabatic_tl, alpha_nepers_per_m,
    das_axial_response, excitation, modal_attenuation, mode_sum_tl)

__all__ = ["SoundSpeedProfile", "to_down", "to_repo",
           "SoundSpeedModel", "ConstantC", "PathAverage",
           "RayTrace", "trace", "max_direct_range", "direct_time",
           "Seabed", "ModeSet", "modes",
           "AdiabaticResult", "DegenerateGeometryError", "adiabatic_tl",
           "alpha_nepers_per_m", "das_axial_response",
           "excitation", "modal_attenuation", "mode_sum_tl"]
