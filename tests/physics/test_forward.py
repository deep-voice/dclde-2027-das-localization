import numpy as np
import pytest

from dclde_localization.physics.forward import (
    DEFAULT_C,
    best_t0,
    misfit_field,
    misfit_value,
    mirror_penalty,
    signed_side,
    travel_time,
)
from dclde_localization.physics.geometry import load_cable

NORTH_CSV = "fixtures/north_DAS_multicoord.csv"


def test_travel_time_is_euclidean_over_c():
    t = travel_time(0.0, 0.0, 0.0, np.array([3000.0]), np.array([4000.0]), np.array([0.0]), 1000.0)
    assert t[0] == pytest.approx(5.0)


def test_travel_time_includes_depth():
    t = travel_time(0.0, 0.0, -30.0, np.array([0.0]), np.array([0.0]), np.array([-1030.0]), 1000.0)
    assert t[0] == pytest.approx(1.0)


def test_best_t0_l1_is_median_and_l2_is_mean():
    tt = np.array([1.0, 2.0, 3.0])
    obs = tt + np.array([10.0, 11.0, 30.0])
    assert best_t0(obs, tt, "l1") == pytest.approx(11.0)
    assert best_t0(obs, tt, "l2") == pytest.approx(17.0)


def test_misfit_value_is_zero_for_a_perfect_fit():
    tt = np.array([1.0, 2.0, 3.5])
    obs = tt + 7.0
    assert misfit_value(obs, tt, "l1") == pytest.approx(0.0)
    assert misfit_value(obs, tt, "l2") == pytest.approx(0.0)


def test_misfit_value_l1_is_the_abs_sum_not_the_signed_sum():
    # Zero residuals (the case above) cannot distinguish sum(r) from
    # sum(abs(r)) -- they agree trivially at 0. This fixture uses residuals
    # with mixed signs around the l1 t0 (the median), where the signed sum and
    # the L1 (abs) sum genuinely diverge, so a regression to `r.sum()` in
    # place of `np.abs(r).sum()` is caught here and not by the zero-residual
    # case above.
    tt = np.zeros(5)
    obs = np.array([-5.0, -1.0, 0.0, 1.0, 20.0])
    t0 = best_t0(obs, tt, "l1")
    assert t0 == pytest.approx(0.0)  # median of obs

    residuals = obs - tt - t0
    hand_l1 = float(np.abs(residuals).sum())  # 5 + 1 + 0 + 1 + 20 = 27
    hand_signed_sum = float(residuals.sum())  # -5 - 1 + 0 + 1 + 20 = 15
    assert hand_l1 != pytest.approx(hand_signed_sum), "fixture must actually distinguish the two"

    assert misfit_value(obs, tt, "l1") == pytest.approx(hand_l1)
    assert misfit_value(obs, tt, "l1") != pytest.approx(hand_signed_sum)


def test_misfit_field_recovers_a_synthetic_source():
    # Straight cable along x at depth 0; source 6 km off-axis.
    px = np.linspace(0.0, 50_000.0, 40)
    py = np.zeros_like(px)
    pz = np.zeros_like(px)
    true_x, true_y, sz = 25_000.0, 6_000.0, 0.0
    obs = 12.5 + travel_time(true_x, true_y, sz, px, py, pz, DEFAULT_C)

    xs = np.linspace(20_000.0, 30_000.0, 101)
    ys = np.linspace(0.0, 12_000.0, 121)
    field = misfit_field(xs, ys, sz, px, py, pz, obs, DEFAULT_C)

    assert field.shape == (len(ys), len(xs))
    j, i = np.unravel_index(field.argmin(), field.shape)
    assert xs[i] == pytest.approx(true_x, abs=150.0)
    assert abs(ys[j]) == pytest.approx(true_y, abs=150.0)
    assert field.min() < 1e-6


def test_misfit_field_is_bimodal_for_a_straight_cable():
    # A perfectly straight cable cannot distinguish the two sides: the mirror
    # solution must fit exactly as well.
    px = np.linspace(0.0, 50_000.0, 40)
    py = np.zeros_like(px)
    pz = np.zeros_like(px)
    obs = travel_time(25_000.0, 6_000.0, 0.0, px, py, pz, DEFAULT_C)

    xs = np.linspace(24_000.0, 26_000.0, 21)
    ys = np.array([-6_000.0, 6_000.0])
    field = misfit_field(xs, ys, 0.0, px, py, pz, obs, DEFAULT_C)
    assert field[0].min() == pytest.approx(field[1].min(), abs=1e-9)


def test_signed_side_splits_around_the_cable():
    cable = load_cable("North", NORTH_CSV)
    # Offsets perpendicular to a locally east-west cable are north/south of it.
    i = 15_000
    x0, y0 = cable.x[i], cable.y[i]
    s = signed_side(cable, np.array([x0, x0]), np.array([y0 + 5_000.0, y0 - 5_000.0]))
    assert s[0] == -s[1]
    assert set(np.unique(s)) <= {-1.0, 1.0}


def test_mirror_penalty_ratio_is_one_for_a_symmetric_synthetic_case():
    class _Straight:
        x = np.linspace(0.0, 50_000.0, 200)
        y = np.zeros(200)

    px = np.linspace(0.0, 50_000.0, 40)
    py = np.zeros_like(px)
    pz = np.zeros_like(px)
    obs = travel_time(25_000.0, 6_000.0, 0.0, px, py, pz, DEFAULT_C)

    xs = np.linspace(23_000.0, 27_000.0, 41)
    ys = np.linspace(-9_000.0, 9_000.0, 91)
    mp = mirror_penalty(_Straight(), xs, ys, 0.0, px, py, pz, obs, DEFAULT_C)
    assert mp.ratio == pytest.approx(1.0, abs=1e-6)
