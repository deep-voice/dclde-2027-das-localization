import json

import pytest

from dclde_localization.physics.geometry import FrameTransform, load_cable
from viz.results import (
    FRAME_GUARD_THRESHOLD_M,
    frame_guard,
    load_result_set,
    to_local,
)

NORTH_CSV = "fixtures/north_DAS_multicoord.csv"
SOUTH_CSV = "fixtures/south_DAS_multicoord.csv"


def _write(tmp_path, payload):
    p = tmp_path / "rs.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _minimal(**over):
    base = {
        "name": "test", "kind": "estimate", "frame": "local_xy",
        "produced_by": "unit test", "c_assumed": 1480.0,
        "solutions": [{"call": "2021-11-04_02_00_02/1", "x": 37000.0, "y": 22344.0,
                       "z": -30.0, "z_fixed": True, "n_picks": 41}],
    }
    base.update(over)
    return base


def test_load_minimal_result_set(tmp_path):
    rs = load_result_set(_write(tmp_path, _minimal()))
    assert rs.name == "test"
    assert rs.solutions[0].z_fixed is True
    assert rs.solutions[0].wrms is None


def test_missing_frame_is_rejected(tmp_path):
    payload = _minimal()
    del payload["frame"]
    with pytest.raises(ValueError, match="frame"):
        load_result_set(_write(tmp_path, payload))


def test_unknown_frame_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown frame"):
        load_result_set(_write(tmp_path, _minimal(frame="mercator")))


def test_unknown_kind_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown kind"):
        load_result_set(_write(tmp_path, _minimal(kind="guess")))


def test_non_finite_coordinate_is_rejected(tmp_path):
    payload = _minimal()
    payload["solutions"][0]["x"] = None
    with pytest.raises(ValueError, match="non-finite"):
        load_result_set(_write(tmp_path, payload))


def test_boolean_coordinate_is_rejected(tmp_path):
    # bool is a subclass of int in Python, so isinstance(True, int) and
    # math.isfinite(True) are both True: without an explicit bool check, a
    # contributed {"x": true} would silently load as x = 1.0 instead of being
    # refused as malformed input.
    payload = _minimal()
    payload["solutions"][0]["x"] = True
    with pytest.raises(ValueError, match="non-finite"):
        load_result_set(_write(tmp_path, payload))


@pytest.fixture(scope="module")
def cables():
    return {"North": load_cable("North", NORTH_CSV), "South": load_cable("South", SOUTH_CSV)}


def test_frame_guard_passes_a_correct_set(tmp_path, cables):
    rs = load_result_set(_write(tmp_path, _minimal()))
    t = FrameTransform.from_cable(cables["North"])
    rep = frame_guard(to_local(rs, t), cables)
    assert not rep.suspect
    assert rep.median_offset_m < 5_000.0


def test_frame_guard_flags_a_misframed_set(tmp_path, cables):
    # The reference implementation's published lat/lon land ~270 km west of the
    # cable; declaring those numbers as local metres must be caught.
    bad = _minimal()
    bad["solutions"][0]["x"] = 500_000.0
    bad["solutions"][0]["y"] = 500_000.0
    rs = load_result_set(_write(tmp_path, bad))
    t = FrameTransform.from_cable(cables["North"])
    rep = frame_guard(to_local(rs, t), cables)
    assert rep.suspect
    assert rep.median_offset_m > FRAME_GUARD_THRESHOLD_M


def test_to_local_roundtrips_via_utm(tmp_path, cables):
    t = FrameTransform.from_cable(cables["North"])
    ux, uy = t.local_to_utm(37000.0, 22344.0)
    payload = _minimal(frame="utm10n")
    payload["solutions"][0]["x"] = float(ux)
    payload["solutions"][0]["y"] = float(uy)
    rs = to_local(load_result_set(_write(tmp_path, payload)), t)
    assert rs.frame == "local_xy"
    assert rs.solutions[0].x == pytest.approx(37000.0, abs=1e-3)
    assert rs.solutions[0].y == pytest.approx(22344.0, abs=1e-3)
