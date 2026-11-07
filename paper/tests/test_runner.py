import textwrap

import pandas as pd
import pytest
import yaml

from dclde_soundspeed.experiments.runner import (
    PROFILES, SelfConsistencyError, build_model, load_config, write_results)


def _write(tmp_path, body):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_guard_rejects_identical_models(tmp_path):
    # The failure mode this whole guard exists for: simulate and invert with the
    # same physics and truth comes back by construction, measuring nothing.
    # It must be impossible by accident, not merely discouraged.
    cfg = _write(tmp_path, """
        name: bad
        truth_model: {kind: constant_c, c: 1480.0}
        inversion_model: {kind: constant_c, c: 1480.0}
        """)
    with pytest.raises(SelfConsistencyError, match="same model"):
        load_config(cfg)


def test_guard_can_be_overridden_only_explicitly(tmp_path):
    cfg = _write(tmp_path, """
        name: null-check
        truth_model: {kind: constant_c, c: 1480.0}
        inversion_model: {kind: constant_c, c: 1480.0}
        allow_same_model: true
        notes: deliberate null experiment
        """)
    got = load_config(cfg)
    assert got.allow_same_model is True


def test_accepts_differing_models(tmp_path):
    cfg = _write(tmp_path, """
        name: free-c
        truth_model: {kind: constant_c, c: 1480.0}
        inversion_model: {kind: constant_c, c: 1440.0}
        """)
    got = load_config(cfg)
    assert got.truth_model.name != got.inversion_model.name


def test_defaults_match_the_published_baseline(tmp_path):
    cfg = _write(tmp_path, """
        name: d
        truth_model: {kind: constant_c, c: 1480.0}
        inversion_model: {kind: path_average, profile: oregon_nov}
        """)
    got = load_config(cfg)
    assert got.source_z == -30.0
    assert got.objective == "l1"
    assert got.min_picks == 8


def test_build_model_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown model kind"):
        build_model({"kind": "telepathy"})


def test_build_model_rejects_a_missing_profile():
    with pytest.raises(ValueError, match="profile"):
        build_model({"kind": "path_average"})


def test_build_model_builds_a_raytrace_model():
    # Coverage gap (Task 14 review): build_model's raytrace branch
    # (`kind in ("path_average", "raytrace")`) was only ever exercised via
    # "path_average" in this file; nothing checked that "raytrace" actually
    # builds a RayTrace, not a PathAverage, from the same profile lookup.
    from dclde_soundspeed.soundspeed.raytrace import RayTrace

    model = build_model({"kind": "raytrace", "profile": "oregon_nov"})
    assert isinstance(model, RayTrace)
    assert model.profile.source == PROFILES["oregon_nov"].source


def test_load_config_raises_on_each_missing_required_key(tmp_path):
    # Coverage gap: load_config's `for key in ("name", "truth_model",
    # "inversion_model"): if key not in raw: raise ...` loop was only ever
    # exercised indirectly (every existing config happens to carry all
    # three) -- nothing pinned the missing-key message itself, which is
    # exactly the error `ablation_free_c.yaml` used to raise before Critical
    # 2's fix removed its (decorative, unread) truth_model/inversion_model
    # keys.
    full = {"name": "x", "truth_model": {"kind": "constant_c", "c": 1480.0},
            "inversion_model": {"kind": "constant_c", "c": 1440.0}}
    for missing_key in ("name", "truth_model", "inversion_model"):
        body = {k: v for k, v in full.items() if k != missing_key}
        p = tmp_path / f"missing_{missing_key}.yaml"
        p.write_text(yaml.safe_dump(body), encoding="utf-8")
        with pytest.raises(ValueError, match=f"missing required key '{missing_key}'"):
            load_config(p)


def test_load_config_rejects_an_unknown_objective(tmp_path):
    # Coverage gap: `objective must be one of OBJECTIVES` was never fired by
    # any existing test.
    cfg = _write(tmp_path, """
        name: bad-objective
        truth_model: {kind: constant_c, c: 1480.0}
        inversion_model: {kind: constant_c, c: 1440.0}
        objective: l3
        """)
    with pytest.raises(ValueError, match="objective must be one of"):
        load_config(cfg)


def test_hycom_profile_resolves_through_profiles_and_is_not_the_indicative_one():
    # Task 9 review finding: the validated HYCOM profile must actually be
    # reachable through the same PROFILES table configs use, not just sit in
    # reanalysis.py with no caller. It must also be a real, distinct profile
    # from the hand-typed "oregon_nov" placeholder, not an alias for it.
    from dclde_soundspeed.experiments.runner import PROFILES

    assert "hycom_oregon_nov" in PROFILES
    hycom = PROFILES["hycom_oregon_nov"]
    oregon_nov = PROFILES["oregon_nov"]
    assert hycom.source.startswith("hycom:")
    assert hycom.source != oregon_nov.source

    model = build_model({"kind": "path_average", "profile": "hycom_oregon_nov"})
    assert model.profile.source.startswith("hycom:")


def test_hycom_south_deep_is_a_distinct_deeper_profile_for_experiments():
    # Round 2 review finding: the validation-site profile (~600 m, the OOI
    # mooring's own local seafloor) is far short of the cables' -1575 m, so
    # it must not be the only HYCOM profile PROFILES offers. "hycom_south_deep"
    # is fetched at the South cable's own deepest channel instead and must
    # reach substantially deeper than the validation-site one.
    from dclde_soundspeed.experiments.runner import PROFILES

    assert "hycom_south_deep" in PROFILES
    south_deep = PROFILES["hycom_south_deep"]
    validation_site = PROFILES["hycom_oregon_nov"]
    assert south_deep.source.startswith("hycom:")
    assert south_deep.source != validation_site.source
    assert south_deep.z_down.max() > validation_site.z_down.max() + 500.0

    model = build_model({"kind": "path_average", "profile": "hycom_south_deep"})
    assert model.profile.source.startswith("hycom:")


def test_write_results_round_trips_through_pandas_with_comment_arg(tmp_path):
    # The provenance header is load-bearing: every committed result table in
    # this plan is read back with pd.read_csv(path, comment="#"). A refactor
    # that changes the header line count, drops the leading "#", or alters
    # the delimiter must fail this test rather than silently corrupt every
    # downstream table.
    df = pd.DataFrame({"x": [1, 2, 3], "label": ["a", "b", "c"]})
    path = write_results(df, "roundtrip-check", out_dir=tmp_path)

    raw = path.read_text(encoding="utf-8")
    header_lines = [ln for ln in raw.splitlines() if ln.startswith("#")]
    assert any("roundtrip-check" in ln for ln in header_lines)
    assert any("generated" in ln for ln in header_lines)
    assert any("rows" in ln for ln in header_lines)

    got = pd.read_csv(path, comment="#")
    assert list(got.columns) == ["x", "label"]
    pd.testing.assert_frame_equal(got, df)


def test_write_results_without_notes_is_byte_identical_to_before(tmp_path):
    # Additive-only guarantee: several Plan A result tables are committed with
    # today's header format and their regeneration checks must keep passing.
    # Omitting `notes` must not change a single byte of the header.
    df = pd.DataFrame({"x": [1, 2, 3]})
    path = write_results(df, "no-notes-check", out_dir=tmp_path)
    raw = path.read_text(encoding="utf-8")
    header_lines = [ln for ln in raw.splitlines() if ln.startswith("#")]
    assert header_lines == [
        "# experiment: no-notes-check",
        header_lines[1],  # generated timestamp varies; only its prefix is checked
        "# rows: 3",
    ]
    assert header_lines[1].startswith("# generated: ")


def test_write_results_with_notes_appends_lines_after_the_provenance_header(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3]})
    notes = ["first caveat travels with the data", "second caveat also travels"]
    path = write_results(df, "with-notes-check", out_dir=tmp_path, notes=notes)

    raw = path.read_text(encoding="utf-8")
    header_lines = [ln for ln in raw.splitlines() if ln.startswith("#")]
    assert header_lines == [
        "# experiment: with-notes-check",
        header_lines[1],
        "# rows: 3",
        "# first caveat travels with the data",
        "# second caveat also travels",
    ]


def test_write_results_with_notes_still_round_trips_with_comment_arg(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3], "label": ["a", "b", "c"]})
    path = write_results(df, "with-notes-roundtrip", out_dir=tmp_path,
                         notes=["a caveat", "another caveat"])
    got = pd.read_csv(path, comment="#")
    assert list(got.columns) == ["x", "label"]
    pd.testing.assert_frame_equal(got, df)
