import numpy as np
import pytest

from backtest.bootstrap import (
    default_mean_block,
    reality_check,
    stationary_bootstrap_indices,
)


def noise(rows, cols, seed=0, scale=0.01):
    return np.random.default_rng(seed).standard_normal((rows, cols)) * scale




def test_indices_have_the_right_shape_and_stay_in_range():
    idx = stationary_bootstrap_indices(120, 40, 8.0, np.random.default_rng(0))
    assert idx.shape == (40, 120)
    assert idx.min() >= 0
    assert idx.max() < 120


def test_block_lengths_are_geometric_with_the_requested_mean():
    n, mean_block = 500, 10.0
    idx = stationary_bootstrap_indices(n, 400, mean_block, np.random.default_rng(1))
    continued = (idx[:, 1:] - idx[:, :-1]) % n == 1
    assert continued.mean() == pytest.approx(1 - 1 / mean_block, abs=0.02)


def test_a_mean_block_of_one_reduces_to_the_iid_bootstrap():
    idx = stationary_bootstrap_indices(200, 200, 1.0, np.random.default_rng(2))
    continued = (idx[:, 1:] - idx[:, :-1]) % 200 == 1
    assert continued.mean() < 0.02


def test_every_row_is_equally_likely_to_appear():
    idx = stationary_bootstrap_indices(50, 20000, 5.0, np.random.default_rng(3))
    frequency = np.bincount(idx.ravel(), minlength=50) / idx.size
    assert frequency == pytest.approx(np.full(50, 1 / 50), rel=0.05)


def test_default_mean_block_grows_with_the_sample_but_stays_at_least_one():
    assert default_mean_block(8000) == pytest.approx(20.0, rel=0.01)
    assert default_mean_block(1) == 1.0
    assert default_mean_block(27) < default_mean_block(1000)


def test_results_are_reproducible_from_a_seed():
    data = noise(400, 6, seed=4)
    assert reality_check(data, n_boot=200, rng=11)["p_value"] == (
        reality_check(data, n_boot=200, rng=11)["p_value"]
    )
    assert reality_check(data, n_boot=200, rng=11)["p_value"] != (
        reality_check(data, n_boot=200, rng=12)["p_value"]
    )


def test_a_genuine_edge_is_detected():
    data = noise(2000, 20, seed=5)
    data[:, 7] += 0.002
    result = reality_check(data, n_boot=1000, rng=1)
    assert result["p_value"] < 0.01


def test_pure_noise_is_not_flagged_as_an_edge():
    rejections = sum(
        reality_check(noise(300, 5, seed=seed), n_boot=200, rng=seed)["p_value"] < 0.05
        for seed in range(60)
    )
    assert rejections <= 9


def test_the_p_value_never_reaches_zero():
    data = noise(500, 3, seed=6) + 0.05  # an absurdly large edge
    assert reality_check(data, n_boot=100, rng=1)["p_value"] == pytest.approx(1 / 101)



def test_adding_independent_trials_makes_the_same_edge_harder_to_call_real():
    rng = np.random.default_rng(7)
    edge = rng.standard_normal((1500, 1)) * 0.01 + 0.0006

    alone = reality_check(edge, n_boot=500, rng=1)["p_value"]
    with_decoys = reality_check(
        np.hstack([edge, rng.standard_normal((1500, 25)) * 0.01]), n_boot=500, rng=1
    )["p_value"]

    assert with_decoys > alone


def test_perfectly_correlated_trials_cost_nothing():
    rng = np.random.default_rng(8)
    edge = rng.standard_normal((1500, 1)) * 0.01 + 0.0006

    alone = reality_check(edge, n_boot=500, rng=1)["p_value"]
    duplicated = reality_check(np.tile(edge, (1, 26)), n_boot=500, rng=1)["p_value"]

    assert duplicated == alone


def test_the_null_distribution_is_centred_near_zero_and_returned_intact():
    result = reality_check(noise(800, 10, seed=9) + 0.01, n_boot=300, rng=1)
    assert result["null"].shape == (300,)
    assert abs(np.median(result["null"])) < result["statistic"]



def test_metadata_records_the_choices_that_were_made():
    result = reality_check(noise(600, 4, seed=11), n_boot=250, mean_block=12.0, rng=1)
    assert result["n_boot"] == 250
    assert result["mean_block"] == 12.0


def test_a_one_dimensional_input_is_treated_as_a_single_trial():
    result = reality_check(noise(300, 1, seed=12).ravel(), n_boot=100, rng=1)
    assert 0 < result["p_value"] <= 1


def test_rejects_inputs_that_would_fail_silently():
    with pytest.raises(ValueError, match="at least 2 observations"):
        reality_check(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="NaN or infinite"):
        reality_check(np.array([[0.1, np.nan], [0.2, 0.3]]))
    with pytest.raises(ValueError, match="n_boot must be positive"):
        reality_check(noise(100, 2), n_boot=0)
    with pytest.raises(ValueError, match="mean_block must be positive"):
        stationary_bootstrap_indices(10, 5, 0.0, np.random.default_rng(0))
    with pytest.raises(ValueError, match="n must be positive"):
        stationary_bootstrap_indices(0, 5, 2.0, np.random.default_rng(0))
