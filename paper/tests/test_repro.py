import numpy as np
import pandas as pd
import pytest

from dclde_soundspeed.experiments.repro import (
    join_feasibility_sweep, join_to_calls, parse_baseline_csv, run_repro)

BASELINE_HEAD = (
    "utc,sensor,call_type,x_local,y_local,z_local,x_utm,y_utm,z_utm,lat,lon,wrms,deltax\n"
    "2021-11-04 02:00:08,hf,pair,37000.55,22344.41,-30.0,68607.0,5007455.9,-30.0,"
    "45.0888,-128.4819,0.1259,8.796\n"
    "2021-11-04 02:00:51,hf,pair,37011.21,22382.38,-30.0,68596.3,5007493.9,-30.0,"
    "45.0891,-128.4821,0.1271,10.809\n")


def test_parse_keeps_local_coordinates_and_discards_published_latlon(tmp_path):
    # The published lat/lon place calls ~270 km west of the cable while the local
    # coordinates are sound. Carrying both invites using the wrong one.
    p = tmp_path / "b.csv"
    p.write_text(BASELINE_HEAD, encoding="utf-8")
    df = parse_baseline_csv(p)
    assert list(df.columns) == ["utc", "sensor", "call_type", "x_local", "y_local",
                                "z_local", "wrms", "deltax"]
    assert df["x_local"].iloc[0] == pytest.approx(37000.55)
    assert pd.api.types.is_datetime64_any_dtype(df["utc"])


def test_join_matches_a_call_within_its_window(tmp_path):
    # The two calls (and their baseline rows) are placed well over tol_s apart
    # so neither is a candidate for the other's baseline row -- otherwise, at
    # the interface's own tol_s=90.0 default, two real calls only 43 s apart
    # (as in an earlier version of this fixture) are BOTH within tolerance of
    # BOTH baseline rows, which is an ambiguous join, not a clean match (see
    # test_join_marks_ambiguous_when_two_baseline_rows_claim_one_call). This
    # fixture keeps tol_s at its real default and instead spaces the two
    # scenarios far enough apart in time to be genuinely unambiguous.
    p = tmp_path / "b.csv"
    p.write_text(
        "utc,sensor,call_type,x_local,y_local,z_local,x_utm,y_utm,z_utm,lat,lon,wrms,deltax\n"
        "2021-11-04 02:00:08,hf,pair,37000.55,22344.41,-30.0,68607.0,5007455.9,-30.0,"
        "45.0888,-128.4819,0.1259,8.796\n"
        "2021-11-04 02:05:02,hf,pair,37011.21,22382.38,-30.0,68596.3,5007493.9,-30.0,"
        "45.0891,-128.4821,0.1271,10.809\n", encoding="utf-8")
    baseline = parse_baseline_csv(p)
    picks = pd.DataFrame({
        "call": ["2021-11-04_02_00_02/1"] * 3 + ["2021-11-04_02_00_02/2"] * 3,
        "window": ["2021-11-04_02_00_02"] * 6,
        "time": [6.0, 6.4, 6.8, 300.0, 300.4, 300.8]})
    got = join_to_calls(baseline, picks, tol_s=90.0)
    assert got["call"].notna().sum() == 2
    assert got["call"].iloc[0] == "2021-11-04_02_00_02/1"
    assert (got["join_status"] == "matched").all()


def test_join_reports_unmatched_rows_rather_than_dropping_them(tmp_path):
    # A join that silently drops rows turns a reproduction failure into a
    # smaller, cleaner-looking sample.
    p = tmp_path / "b.csv"
    p.write_text(BASELINE_HEAD, encoding="utf-8")
    baseline = parse_baseline_csv(p)
    picks = pd.DataFrame({"call": ["2021-11-09_09_09_09/1"], "window": ["2021-11-09_09_09_09"],
                          "time": [1.0]})
    got = join_to_calls(baseline, picks, tol_s=90.0)
    assert len(got) == len(baseline)
    assert got["call"].isna().all()


def test_join_marks_ambiguous_when_two_baseline_rows_claim_one_call(tmp_path):
    # This is the failure review round 1 caught: fin whales call every
    # 10-15 s with several audible at once, so more than one baseline row can
    # be "nearest in time" to the same annotated call. A join that picks the
    # nearest anyway silently attaches a different whale's published position
    # to this call. Both rows here are within tol_s of the SAME single call,
    # so neither is a provably unique match and both must be excluded rather
    # than one being guessed as the "real" one.
    p = tmp_path / "b.csv"
    p.write_text(
        "utc,sensor,call_type,x_local,y_local,z_local,x_utm,y_utm,z_utm,lat,lon,wrms,deltax\n"
        "2021-11-04 02:00:08,hf,pair,37000.55,22344.41,-30.0,68607.0,5007455.9,-30.0,"
        "45.0888,-128.4819,0.1259,8.796\n"
        "2021-11-04 02:00:10,hf,pair,37011.21,22382.38,-30.0,68596.3,5007493.9,-30.0,"
        "45.0891,-128.4821,0.1271,10.809\n", encoding="utf-8")
    baseline = parse_baseline_csv(p)
    picks = pd.DataFrame({"call": ["2021-11-04_02_00_02/1"] * 3,
                          "window": ["2021-11-04_02_00_02"] * 3,
                          "time": [6.0, 6.4, 6.8]})
    got = join_to_calls(baseline, picks, tol_s=90.0)
    assert len(got) == len(baseline)
    assert got["call"].isna().all()
    assert (got["join_status"] == "ambiguous").all()


def test_join_marks_ambiguous_when_one_baseline_row_matches_two_calls(tmp_path):
    # The reverse direction of the same problem: two real calls close enough
    # in time that a single baseline row cannot be attributed to either one
    # uniquely.
    p = tmp_path / "b.csv"
    p.write_text(BASELINE_HEAD, encoding="utf-8")
    baseline = parse_baseline_csv(p).iloc[[0]].copy()
    picks = pd.DataFrame({
        "call": ["2021-11-04_02_00_02/1"] * 3 + ["2021-11-04_02_00_02/2"] * 3,
        "window": ["2021-11-04_02_00_02"] * 6,
        "time": [6.0, 6.4, 6.8, 8.0, 8.4, 8.8]})
    got = join_to_calls(baseline, picks, tol_s=90.0)
    assert len(got) == len(baseline)
    assert got["call"].isna().all()
    assert (got["join_status"] == "ambiguous").all()


def test_margin_mode_accepts_a_call_isolated_in_time(tmp_path):
    # Review round 2's rescue mode: accept the nearest candidate when it
    # clearly beats the second-nearest, even though the default (provably
    # unique) mode would already reject this as ambiguous at tol_s=90
    # (both calls fall within 90 s of the baseline row).
    p = tmp_path / "b.csv"
    p.write_text(BASELINE_HEAD.splitlines()[0] + "\n" +
                 "2021-11-04 02:00:08,hf,pair,37000.55,22344.41,-30.0,68607.0,5007455.9,-30.0,"
                 "45.0888,-128.4819,0.1259,8.796\n", encoding="utf-8")
    baseline = parse_baseline_csv(p)
    picks = pd.DataFrame({
        "call": ["2021-11-04_02_00_02/1"] * 3 + ["2021-11-04_02_00_02/2"] * 3,
        "window": ["2021-11-04_02_00_02"] * 6,
        # call 1 arrives right at the baseline utc (gap ~0s); call 2 is 12s
        # away -- well clear of a 1s margin, so call 1 should be accepted.
        "time": [6.0, 6.4, 6.8, 18.0, 18.4, 18.8]})
    got = join_to_calls(baseline, picks, margin_s=1.0)
    assert got["join_status"].iloc[0] == "matched"
    assert got["call"].iloc[0] == "2021-11-04_02_00_02/1"


def test_margin_mode_rejects_when_second_nearest_is_too_close(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text(BASELINE_HEAD.splitlines()[0] + "\n" +
                 "2021-11-04 02:00:08,hf,pair,37000.55,22344.41,-30.0,68607.0,5007455.9,-30.0,"
                 "45.0888,-128.4819,0.1259,8.796\n", encoding="utf-8")
    baseline = parse_baseline_csv(p)
    picks = pd.DataFrame({
        "call": ["2021-11-04_02_00_02/1"] * 3 + ["2021-11-04_02_00_02/2"] * 3,
        "window": ["2021-11-04_02_00_02"] * 6,
        # call 2 is only 0.5s further away than call 1 -- inside a 1s margin.
        "time": [6.0, 6.4, 6.8, 6.5, 6.9, 7.3]})
    got = join_to_calls(baseline, picks, margin_s=1.0)
    assert got["join_status"].iloc[0] == "ambiguous"
    assert got["call"].isna().all()


def test_margin_mode_marks_contested_when_two_rows_claim_the_same_call(tmp_path):
    # Each row individually clears its own margin test (there is no second
    # call nearby for either), but both independently pick the SAME call as
    # their nearest -- there is no principled way to prefer one baseline row
    # over the other without using position, so neither is kept.
    p = tmp_path / "b.csv"
    p.write_text(
        "utc,sensor,call_type,x_local,y_local,z_local,x_utm,y_utm,z_utm,lat,lon,wrms,deltax\n"
        "2021-11-04 02:00:08,hf,pair,37000.55,22344.41,-30.0,68607.0,5007455.9,-30.0,"
        "45.0888,-128.4819,0.1259,8.796\n"
        "2021-11-04 02:00:09,hf,pair,37011.21,22382.38,-30.0,68596.3,5007493.9,-30.0,"
        "45.0891,-128.4821,0.1271,10.809\n", encoding="utf-8")
    baseline = parse_baseline_csv(p)
    picks = pd.DataFrame({"call": ["2021-11-04_02_00_02/1"] * 3,
                          "window": ["2021-11-04_02_00_02"] * 3,
                          "time": [6.0, 6.4, 6.8]})
    got = join_to_calls(baseline, picks, margin_s=1.0)
    assert len(got) == len(baseline)
    assert got["call"].isna().all()
    assert (got["join_status"] == "contested").all()


def test_join_feasibility_sweep_returns_tidy_counts_for_both_sweeps(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text(BASELINE_HEAD, encoding="utf-8")
    baseline = parse_baseline_csv(p)
    picks = pd.DataFrame({
        "call": ["2021-11-04_02_00_02/1"] * 3 + ["2021-11-04_02_00_02/2"] * 3,
        "window": ["2021-11-04_02_00_02"] * 6,
        "time": [6.0, 6.4, 6.8, 49.0, 49.4, 49.8]})
    got = join_feasibility_sweep(baseline, picks, tol_s_values=(90.0, 10.0), margin_s_values=(1.0,))
    assert set(got["sweep_type"]) == {"tol_s", "margin_s"}
    assert set(got.loc[got["sweep_type"] == "tol_s", "param_value"]) == {90.0, 10.0}
    # Every (sweep_type, param_value) group's metric counts must sum to the
    # full baseline row count -- nothing dropped, nothing double-counted.
    for (_, value), group in got.groupby(["sweep_type", "param_value"]):
        assert group["value"].sum() == len(baseline)


def test_run_repro_does_not_mislabel_unmatched_rows_as_too_few_picks(tmp_path):
    # Regression: DataFrame.iterrows() builds each row as a single Series
    # spanning every column. Once the `call` column holds a real string
    # value on some rows and None on others (mixed dtypes across rows -- any
    # run with both a matched and an unmatched/ambiguous row), pandas
    # silently upcasts the None entries to float `nan`. `r["call"] is None`
    # then never matches for those rows, and they fall through into the
    # "too few picks" branch with an empty pick subset instead of being
    # labelled by their real join outcome. A run with only one outcome
    # present (e.g. the previous "0/176 matched" run) never exercises this.
    p = tmp_path / "b.csv"
    p.write_text(
        "utc,sensor,call_type,x_local,y_local,z_local,x_utm,y_utm,z_utm,lat,lon,wrms,deltax\n"
        "2021-11-04 02:00:08,hf,pair,37000.55,22344.41,-30.0,68607.0,5007455.9,-30.0,"
        "45.0888,-128.4819,0.1259,8.796\n"
        "2021-11-09 09:09:09,hf,pair,1.0,2.0,-30.0,0.0,0.0,-30.0,0.0,0.0,0.1,1.0\n",
        encoding="utf-8")
    baseline = parse_baseline_csv(p)
    picks = pd.DataFrame({
        "call": ["2021-11-04_02_00_02/1"] * 2,
        "window": ["2021-11-04_02_00_02"] * 2,
        "time": [6.0, 6.4],
        "px": [0.0, 100.0], "py": [0.0, 0.0], "pz": [-30.0, -30.0]})
    df = run_repro(picks, baseline, min_picks=8)
    by_call = dict(zip(df["call"].fillna("UNMATCHED"), df["excluded"]))
    assert by_call["2021-11-04_02_00_02/1"] == "too few picks"
    assert by_call["UNMATCHED"] == "unmatched"


def test_run_repro_reaches_the_matched_and_solved_branch(tmp_path):
    # Coverage gap (Task 14 review): every other run_repro test feeds too few
    # picks (min_picks=8) or an unmatched/ambiguous join, so repro.py:405-414
    # -- the branch that actually populates delta_m, delta_nearest_basin_m,
    # provenance and separation_m/separation_resolved -- was never reached by
    # any test. This synthesises enough picks (n=20) for a real solve, with a
    # baseline row timed to land well inside the join's default tol_s=90.0 of
    # that call's own absolute time, so the join succeeds unambiguously and
    # the estimator actually runs.
    from dclde_localization.physics.sound_speed_models import ConstantC

    n = 20
    px = np.linspace(10000.0, 60000.0, n)
    py = np.full(n, 20000.0)
    pz = np.full(n, -300.0)
    truth_x, truth_y = 35000.0, 26000.0
    t = ConstantC(1480.0).travel_time(truth_x, truth_y, -30.0, px, py, pz) + 5.0
    picks = pd.DataFrame({"call": ["2021-11-04_02_00_02/1"] * n,
                          "window": ["2021-11-04_02_00_02"] * n,
                          "time": t, "px": px, "py": py, "pz": pz})

    window_start = pd.Timestamp("2021-11-04 02:00:02", tz="UTC")
    call_abs_t = window_start + pd.Timedelta(seconds=float(t.min()))
    utc_str = call_abs_t.strftime("%Y-%m-%d %H:%M:%S")

    p = tmp_path / "b.csv"
    p.write_text(
        "utc,sensor,call_type,x_local,y_local,z_local,x_utm,y_utm,z_utm,lat,lon,wrms,deltax\n"
        f"{utc_str},hf,pair,{truth_x + 50.0},{truth_y - 20.0},-30.0,68607.0,5007455.9,-30.0,"
        "45.0888,-128.4819,0.1259,8.796\n", encoding="utf-8")
    baseline = parse_baseline_csv(p)

    df = run_repro(picks, baseline, min_picks=8)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["call"] == "2021-11-04_02_00_02/1"
    assert row["matched"] is True or bool(row["matched"]) is True
    assert row["excluded"] == ""
    assert row["n_basins"] >= 1
    assert row["provenance"] in ("local_min", "fallback", "unresolved")
    assert np.isfinite(row["delta_m"])
    assert np.isfinite(row["delta_nearest_basin_m"])
    # The baseline row was placed close to the true source, so the estimator
    # (given the true model and 20 clean picks) should land close to it too.
    assert row["delta_m"] < 500.0
    assert row["our_misfit"] < 1.0
