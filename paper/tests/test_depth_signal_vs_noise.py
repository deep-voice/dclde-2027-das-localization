import numpy as np
import pandas as pd

from paper.scripts.depth_signal_vs_noise import delta_chi2, picks_summary, summarise


def test_delta_chi2_sign_and_scale():
    assert delta_chi2(0.1, 0.1, 20, 0.1) == 0
    assert delta_chi2(0.1, 0.2, 10, 0.1) == 10 * (0.04 - 0.01) / 0.01   # worse fit -> positive
    assert delta_chi2(0.2, 0.1, 10, 0.1) < 0


def test_summarise_counts_sigma_thresholds_over_the_stated_denominator():
    t = pd.DataFrame(dict(delta_chi2=[0.5, -1.5, 5.0, -10.0], n_picks=[10, 20, 30, 50], n_cables=[1, 1, 2, 2],
                          resid_sigma_ms=[100.0] * 4, rms_equiv_ms=[5.0, 10.0, 20.0, 30.0]))
    s = summarise(t).set_index("group")
    assert s.loc["all stratum", "n_calls"] == 4
    assert (s.loc["all stratum", ["n_ge_1sigma", "n_ge_2sigma", "n_ge_3sigma"]] == [3, 2, 1]).all()
    assert s.loc["n_cables=2", "n_calls"] == 2


def test_picks_summary_counts_calls_and_two_cable_minimum():
    p = pd.DataFrame(dict(call=["a"] * 5 + ["b"] * 8, cable=["N"] * 5 + ["N"] * 3 + ["S"] * 5))
    s = picks_summary(p).iloc[0]
    assert s.calls == 2 and s.picks == 13 and s.calls_two_cable == 1 and s.calls_ge_4 == 2
    assert s.median_picks_on_sparser_cable_two_cable_calls == 3
