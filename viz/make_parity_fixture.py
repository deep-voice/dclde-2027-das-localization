"""Generate the committed fixture that pins the JS misfit to the Python misfit."""

import json
from pathlib import Path

import numpy as np

from dclde_localization.physics.forward import DEFAULT_C, misfit_field, misfit_value, travel_time

OUT = Path("viz/fixtures/parity_fixture.json")


def main() -> None:
    rng = np.random.default_rng(20260910)
    n = 12
    px = np.linspace(14_000.0, 61_000.0, n)
    py = 20_000.0 + 300.0 * np.sin(px / 9_000.0)
    pz = -400.0 - 300.0 * np.cos(px / 15_000.0)
    obs = 6.5 + np.sqrt((px - 37_000.0) ** 2 + (py - 22_300.0) ** 2 + (pz + 30.0) ** 2) / DEFAULT_C
    obs = obs + rng.normal(0.0, 0.02, n)

    xs = np.linspace(30_000.0, 44_000.0, 9)
    ys = np.linspace(16_000.0, 28_000.0, 7)
    sz = -30.0

    payload = {
        "c": DEFAULT_C,
        "sz": sz,
        "objective": "l1",
        "px": px.tolist(),
        "py": py.tolist(),
        "pz": pz.tolist(),
        "obsT": obs.tolist(),
        "xs": xs.tolist(),
        "ys": ys.tolist(),
    }
    for objective in ("l1", "l2"):
        f = misfit_field(xs, ys, sz, px, py, pz, obs, DEFAULT_C, objective)
        payload[f"expected_{objective}"] = f.ravel().tolist()

    # Scalar-path parity too. Without this, misfitValue would be exported and
    # never exercised in either language against non-degenerate residuals.
    tt_ref = travel_time(37_000.0, 22_300.0, sz, px, py, pz, DEFAULT_C)
    payload["ttRef"] = tt_ref.tolist()
    for objective in ("l1", "l2"):
        payload[f"expected_value_{objective}"] = misfit_value(obs, tt_ref, objective)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
