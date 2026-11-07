"""Assemble the self-contained HTML bundle.

A published Artifact cannot fetch its own data files (the CSP blocks all
fetch/XHR), so every payload is inlined as base64 typed arrays and every module
is concatenated into one inline module script.
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dclde_localization.physics.geometry import decimate_indices

BUNDLE_VERSION = 1
PAGE_LIMIT_BYTES = 16 * 1024 * 1024
CABLE_STRIDE = 4

# Inline order matters: imports are stripped, so a module's dependencies must
# already be in scope. Dependency-first, not alphabetical. Every file added
# under viz/template/lib/ must be listed here or it will silently not ship.
LIB_ORDER = (
    "forward.mjs",
    "render.mjs",
    "contour.mjs",
    "moveout.mjs",
    "mirror.mjs",
    "zprofile.mjs",
    "isosurface.mjs",
)


def encode_f32(a: np.ndarray) -> str:
    return base64.b64encode(np.asarray(a, dtype="<f4").tobytes()).decode("ascii")


def encode_i16(a: np.ndarray) -> str:
    return base64.b64encode(np.asarray(a, dtype="<i2").tobytes()).decode("ascii")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def inline_module(path: Path) -> str:
    """Strip ES module syntax so the source can live in one inline script."""
    text = Path(path).read_text(encoding="utf-8")
    text = re.sub(r"^\s*import\s.*?;\s*$", "", text, flags=re.M)
    text = re.sub(r"^(\s*)export\s+", r"\1", text, flags=re.M)
    return text


def build_payload(cables, transform, picks_xyz, local_grid, result_sets, sources) -> dict:
    cable_payload = {}
    for name, cable in cables.items():
        idx = decimate_indices(len(cable.x), CABLE_STRIDE)
        # chanM is shipped explicitly rather than reconstructed from start+step:
        # decimate_indices keeps the final channel, so the decimated samples are
        # NOT uniformly spaced at the tail and index arithmetic would be wrong
        # there. ~104 KB base64 across both cables, against a ~1.3 MB page.
        cable_payload[name] = {
            "n": int(len(idx)),
            "chanM": encode_f32(cable.chan_m[idx]),
            "x": encode_f32(cable.x[idx]),
            "y": encode_f32(cable.y[idx]),
            "depth": encode_f32(cable.depth[idx]),
        }

    calls = sorted(picks_xyz["call"].unique())
    call_index = {c: i for i, c in enumerate(calls)}
    p = picks_xyz.sort_values("call").reset_index(drop=True)

    return {
        "bundleVersion": BUNDLE_VERSION,
        "meta": {
            "gitSha": git_sha(),
            "sources": sources,
            "defaultC": 1480.0,
            "frame": {"xOffset": transform.x_offset, "yOffset": transform.y_offset},
            "contourRatios": [1.05, 1.10, 1.25, 1.50],
        },
        "cables": cable_payload,
        "bathy": {
            "x0": local_grid.x0, "y0": local_grid.y0,
            "dx": local_grid.dx, "dy": local_grid.dy,
            "nx": local_grid.nx, "ny": local_grid.ny,
            "noData": -32768,
            "z": encode_i16(local_grid.z.ravel()),
        },
        "calls": calls,
        "picks": {
            "callIndex": [call_index[c] for c in p["call"]],
            "cable": [0 if c == "North" else 1 for c in p["cable"]],
            "dist": encode_f32(p["dist"].to_numpy()),
            "time": encode_f32(p["time"].to_numpy()),
            # snr is not shipped: nothing in app.js reads it, and it was
            # costing ~59 KB of base64 for a payload no code decoded.
            "px": encode_f32(p["px"].to_numpy()),
            "py": encode_f32(p["py"].to_numpy()),
            "pz": encode_f32(p["pz"].to_numpy()),
        },
        "resultSets": [asdict(rs) for rs in result_sets],
    }


def render_html(payload: dict, template_dir: Path) -> str:
    template_dir = Path(template_dir)
    lib_dir = template_dir / "lib"

    present = {p.name for p in lib_dir.glob("*.mjs")} if lib_dir.exists() else set()
    unlisted = present - set(LIB_ORDER)
    if unlisted:
        raise ValueError(f"lib modules not in LIB_ORDER would not ship: {sorted(unlisted)}")

    lib = "\n".join(
        inline_module(lib_dir / name) for name in LIB_ORDER if (lib_dir / name).exists()
    )

    html = (template_dir / "app.html").read_text(encoding="utf-8")
    html = html.replace("/*__CSS__*/", (template_dir / "app.css").read_text(encoding="utf-8"))
    html = html.replace("/*__LIB__*/", lib)
    html = html.replace("/*__APP__*/", inline_module(template_dir / "app.js"))

    # Escape "<" so no payload string can close the inline <script> block. json.dumps
    # leaves "<" alone, so a result set whose notes contained "</script>" would end
    # the module early and turn the rest of the page into unrestricted markup. The
    # results contract is meant to accept third-party files, so this is a real path.
    payload_js = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    html = html.replace("__PAYLOAD__", payload_js)

    size = len(html.encode("utf-8"))
    if size > PAGE_LIMIT_BYTES:
        raise ValueError(f"page is {size/1e6:.1f} MB, over the {PAGE_LIMIT_BYTES/1e6:.0f} MB limit")
    return html
