"""Build the self-contained visualizer page.

    uv run python -m viz.build_bundle --out viz/out/index.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dclde_localization.physics.geometry import CABLES, FrameTransform, load_cable
from viz.bathymetry import decode_gmt_grid, reproject_to_local, validate_against_cable
from viz.bundle import build_payload, render_html
from dclde_localization.physics.picks import attach_xyz, load_picks
from viz.results import frame_guard, load_result_set, to_local

FIXTURES = Path("fixtures")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="viz/out/index.html")
    ap.add_argument("--results-dir", default="viz/results")
    args = ap.parse_args()

    cables = {n: load_cable(n, FIXTURES / f"{n.lower()}_DAS_multicoord.csv") for n in CABLES}
    transform = FrameTransform.from_cable(cables["North"])

    geo = decode_gmt_grid(FIXTURES / "gmrt_bathy.nc")
    report = validate_against_cable(geo, cables["South"])
    print(f"bathymetry check: n={report.n} median offset={report.median_offset_m:.1f} m "
          f"r={report.correlation:.4f}")
    if abs(report.median_offset_m) > 15.0 or report.correlation < 0.999:
        raise SystemExit("bathymetry disagrees with the cable's own depth; refusing to build")
    grid = reproject_to_local(geo, transform)

    picks = attach_xyz(load_picks(), cables)
    print(f"picks: {len(picks)} rows, {picks['call'].nunique()} calls")

    result_sets = []
    for path in sorted(Path(args.results_dir).glob("*.json")):
        rs = to_local(load_result_set(path), transform)
        guard = frame_guard(rs, cables)
        flag = "SUSPECT" if guard.suspect else "ok"
        print(f"result set {rs.name}: {len(rs.solutions)} solutions, "
              f"median {guard.median_offset_m/1000:.1f} km from cable [{flag}]")
        result_sets.append(rs)

    html = render_html(
        build_payload(cables, transform, picks, grid, result_sets,
                      sources={"picks": "fixtures/das_calls.csv.gz",
                               "geometry": "fixtures/*_DAS_multicoord.csv",
                               "bathymetry": "GMRT GridServer"}),
        Path("viz/template"),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html.encode('utf-8'))/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
