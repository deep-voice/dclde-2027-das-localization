"""YAML config -> experiment, with the self-consistency guard.

The way this research fails silently is by simulating and inverting with the
same physics: truth comes back by construction and the experiment measures its
own conditioning rather than anything about the ocean. `truth_model` and
`inversion_model` are therefore separate REQUIRED keys, neither defaulting to
the other, and identical models are refused unless a config says
`allow_same_model` in so many words. This must be impossible by accident, not
merely discouraged, so the check lives in the loader rather than left to
reviewer discipline.

Note on `free_z`: an earlier draft of this module carried a `free_z` field on
`ExperimentConfig` on the theory that a later task would read it out and wire
it into `solve()`. That never happened -- `repro.py`, `regime.py` and
`ablation.py` all call `solve()` directly today, none of them pass a
`free_z`-like argument, and `solve()` itself still has no such parameter (Task
4 removed one for producing source depths above the sea surface). A config
setting `free_z: true` was therefore silently ignored rather than doing
anything. The field is removed rather than kept around unhonoured (Task 14
review) -- add it back, with a real consumer, only when some task actually
frees the source depth as an unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dclde_localization.physics.forward import DEFAULT_C, OBJECTIVES
from dclde_localization.physics.sound_speed_models import ConstantC, PathAverage
from dclde_localization.physics.sound_speed_profile import SoundSpeedProfile
from dclde_soundspeed.soundspeed.raytrace import RayTrace
from dclde_soundspeed.soundspeed.reanalysis import (
    HYCOM_EXPERIMENT_PROFILE_PATH, HYCOM_PROFILE_PATH, load_profile)

RESULTS_DIR = Path("paper/results")

# The three PROFILES' validation status, as plain prose (Task 14 review,
# Important 8). Set once, here, as a structured `SoundSpeedProfile.validated`
# field at construction -- never re-derived downstream by parsing `source` --
# so `experiments.regime._profile_validation_note` cannot drift out of sync
# with the real provenance the way its predecessor (matching "45.1492", a
# substring that occurs in none of the real `:.2f`-formatted source strings)
# already did once.
_OREGON_NOV_VALIDATED = "no -- hand-typed indicative numbers, not measured or reanalysis"
_HYCOM_OREGON_NOV_VALIDATED = (
    "yes -- validated against OOI in-situ CTD at this site "
    "(median 0.47 m/s, p90 1.09 m/s, n=400)")
_HYCOM_SOUTH_DEEP_VALIDATED = (
    "no -- inferred from hycom_oregon_nov's 0.47 m/s median agreement with "
    "in-situ CTD 87 km away at the shallower OOI site; not validated in-situ "
    "at this site")

# Indicative Oregon margin, November -- not measured, not reanalysis; a
# placeholder so early experiments have something to run against. `source`
# stays exactly "oregon_nov_indicative" so any result table built from it
# records, in the model name, that the profile behind it is indicative rather
# than measured. Kept alongside the measured/reanalysis profile below (Task
# 9): existing configs still name "oregon_nov" today, and a later task may
# want the contrast between indicative and reanalysis-based profiles.
_OREGON_NOV = SoundSpeedProfile(
    np.array([0.0, 100.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0]),
    np.array([1491.0, 1480.0, 1478.0, 1477.0, 1479.0, 1484.0, 1490.0]),
    "oregon_nov_indicative", validated=_OREGON_NOV_VALIDATED)

# Task 9: HYCOM GOFS 3.1 at the OOI mooring site, fetched once by
# paper/scripts/validate_hycom_profile.py and committed as
# paper/results/hycom_profile_oregon_nov2021.csv, so loading
# it here needs no network at experiment time. Validated against OOI in-situ
# CTD over 219-573 m (1-6 Nov 2021 window): median 0.47 m/s / p90 1.09 m/s
# disagreement (n=400 compare points).
#
# VALIDATION-SITE PROFILE ONLY -- do not use this for an experiment. It only
# reaches ~600 m (the OOI site's local seafloor on HYCOM's grid), so it would
# clamp-extrapolate 30.8% of the real picks (3,428/11,117 are deeper than
# -600 m). Use "hycom_south_deep" below for anything that needs the cables'
# full 0..-1575 m depth range. See soundspeed.reanalysis's module docstring
# for the two-profile split and the Task 9 report for the review finding that
# made this split necessary.
_HYCOM_OREGON_NOV = replace(load_profile(HYCOM_PROFILE_PATH), validated=_HYCOM_OREGON_NOV_VALIDATED)

# Task 9 (review round 2): HYCOM at the South cable's own deepest channel
# (45.1492, -125.1081, cable depth -1575 m), not the OOI validation site --
# the experiment profile. HYCOM resolves real data there to 1500 m, leaving
# only the last ~75 m of the cable's -1575 m deepest channel as clamped
# extrapolation: 1.64% of picks (182/11,117), versus 30.8% for the
# validation-site profile above.
#
# THIS PROFILE WAS NEVER VALIDATED AGAINST IN-SITU DATA. Trust in it is
# inferred from the 0.47 m/s agreement measured ~87 km away at the shallower
# OOI site, not measured here. See soundspeed.reanalysis's module docstring
# for why that distinction matters and must be carried into any conclusion
# drawn from an experiment that uses this profile.
_HYCOM_SOUTH_DEEP = replace(load_profile(HYCOM_EXPERIMENT_PROFILE_PATH), validated=_HYCOM_SOUTH_DEEP_VALIDATED)

# Keyed by short name, as configs reference profiles (`profile: oregon_nov`).
# Later tasks import PROFILES from this module by name -- keep the name and
# keys stable.
PROFILES = {
    "oregon_nov": _OREGON_NOV,
    "hycom_oregon_nov": _HYCOM_OREGON_NOV,
    "hycom_south_deep": _HYCOM_SOUTH_DEEP,
}


class SelfConsistencyError(ValueError):
    """Raised when an experiment would invert with the physics it simulated."""


def build_model(spec: dict):
    """Build a SoundSpeedModel from `{kind: constant_c|path_average|raytrace,
    c?: float, profile?: str}`."""
    kind = spec.get("kind")
    if kind == "constant_c":
        return ConstantC(float(spec.get("c", DEFAULT_C)))
    if kind in ("path_average", "raytrace"):
        key = spec.get("profile")
        if key not in PROFILES:
            raise ValueError(f"unknown or missing profile {key!r}; have {sorted(PROFILES)}")
        prof = PROFILES[key]
        return PathAverage(prof) if kind == "path_average" else RayTrace(prof)
    raise ValueError(f"unknown model kind {kind!r}")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    truth_model: object
    inversion_model: object
    objective: str = "l1"
    source_z: float = -30.0
    batches: tuple = ()
    cables: tuple = ("North", "South")
    min_picks: int = 8
    allow_same_model: bool = False
    notes: str = ""
    raw: dict = field(default_factory=dict)


def load_config(path) -> ExperimentConfig:
    """Load and validate an experiment config, enforcing the self-consistency
    guard: `truth_model` and `inversion_model` are both required and, unless
    `allow_same_model: true` appears explicitly, must name different models."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("name", "truth_model", "inversion_model"):
        if key not in raw:
            raise ValueError(f"config is missing required key {key!r}")

    truth, inversion = build_model(raw["truth_model"]), build_model(raw["inversion_model"])
    allow_same = bool(raw.get("allow_same_model", False))
    if truth.name == inversion.name and not allow_same:
        raise SelfConsistencyError(
            f"{raw['name']}: truth and inversion use the same model ({truth.name}). "
            "Inverting with the physics you simulated recovers truth by construction "
            "and measures nothing but the experiment's own conditioning. "
            "Set allow_same_model: true only for a deliberate null experiment.")

    objective = raw.get("objective", "l1")
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")

    return ExperimentConfig(
        name=raw["name"], truth_model=truth, inversion_model=inversion,
        objective=objective, source_z=float(raw.get("source_z", -30.0)),
        batches=tuple(raw.get("batches", ())),
        cables=tuple(raw.get("cables", ("North", "South"))),
        min_picks=int(raw.get("min_picks", 8)),
        allow_same_model=allow_same, notes=raw.get("notes", ""), raw=raw)


def write_results(df: pd.DataFrame, name: str, out_dir=RESULTS_DIR,
                  notes: list[str] | None = None) -> Path:
    """Write a result table with a provenance header, so a CSV explains itself
    even after it has been copied out of `results/` and separated from this
    codebase.

    The header is written as `#`-prefixed comment lines before the CSV body,
    so the file is still a valid CSV to any tool that understands leading
    comments; a consumer reading it back with pandas must pass
    `comment="#"`, e.g. `pandas.read_csv(path, comment="#")` -- without it,
    pandas would try to parse the header lines as data rows.

    `notes`: optional extra `#`-prefixed lines, written after the
    `experiment`/`generated`/`rows` lines -- for a caveat that must travel
    with the CSV itself (e.g. a unit or reference-scale mismatch between
    columns, or a co-varying-by-construction axis) and not live only in this
    module's source, which a reader of a copied-out CSV never sees. Omitting
    `notes` (the default) produces a header BYTE-IDENTICAL to before this
    parameter existed -- several Plan A result tables are committed with
    that exact header and their regeneration checks must keep passing.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.csv"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(f"# experiment: {name}\n# generated: {stamp}\n# rows: {len(df)}\n")
        for line in notes or ():
            fh.write(f"# {line}\n")
        df.to_csv(fh, index=False)
    return path
