import numpy as np
import pytest

from dclde_localization.physics.dense_picks import PickerConfig, _parabolic, pick_call, station_traces


def _chevron(seed_jitter_s=0.035, noise=1.0, fwhm_s=0.35, rng_seed=0):
    """A synthetic SNR image: Gaussian envelope along a hyperbolic moveout."""
    rng = np.random.default_rng(rng_seed)
    dist = np.arange(10_000.0, 30_000.0, 4.084)
    t = np.arange(0.0, 30.0, 0.005)
    true = 5.0 + np.sqrt(3000.0**2 + (dist - 18_000.0) ** 2) / 1480.0
    sig = fwhm_s / 2.3548
    img = 15.0 * np.exp(-0.5 * ((t[None, :] - true[:, None]) / sig) ** 2)
    img += noise * rng.standard_normal(img.shape)
    seed_d = np.arange(11_000.0, 29_000.0, 1100.0)
    seed_t = np.interp(seed_d, dist, true) + seed_jitter_s * rng.standard_normal(len(seed_d))
    return img, dist, t, true, seed_d, seed_t


def _rsig(x):
    x = np.asarray(x)
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def test_parabolic_recovers_subsample_peak():
    x = np.arange(7.0)
    cc = -((x - 3.3) ** 2)
    assert _parabolic(cc, 3) == pytest.approx(0.3, abs=1e-9)


def test_station_traces_average_the_gauge():
    block = np.arange(10.0)[:, None] * np.ones((1, 4))
    dist = np.arange(10.0) * 2.0
    out = station_traces(block, dist, np.array([4.0]), 4.0)  # channels at 2, 4, 6 m
    assert out[0, 0] == pytest.approx(2.0)


def test_dense_picks_beat_the_seed_jitter():
    img, dist, t, true, seed_d, seed_t = _chevron()
    picks = pick_call(img, dist, t, seed_d, seed_t)
    k = picks[picks.keep]
    assert len(k) > 0.9 * len(picks)
    err = k.time.to_numpy() - np.interp(k.dist.to_numpy(), dist, true)
    seed_err = seed_t - np.interp(seed_d, dist, true)
    assert _rsig(err) < 5e-3                 # a few ms, against 35 ms of seed jitter
    assert _rsig(err) < _rsig(seed_err) / 5
    # timing convention follows the seeds on average, within their own scatter
    assert abs(np.median(err)) < 0.035


def test_no_stations_outside_the_seed_span():
    img, dist, t, true, seed_d, seed_t = _chevron()
    picks = pick_call(img, dist, t, seed_d, seed_t, PickerConfig(station_m=100.0))
    assert picks.dist.min() >= seed_d.min() and picks.dist.max() <= seed_d.max()


def test_too_few_seeds_returns_empty():
    img, dist, t, *_ = _chevron()
    assert pick_call(img, dist, t, np.array([15_000.0]), np.array([8.0])).empty
