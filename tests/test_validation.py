from itertools import pairwise

import pandas as pd
import pytest

from backtest.strategy import buy_and_hold, sma_crossover, vol_target, warmup_days
from backtest.validation import _windows, feasible_folds, walk_forward
from sweep import STRATEGY_GRIDS


def rising_prices(n=1200, step=0.4):
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"Close": [100 + i * step for i in range(n)]}, index=idx)


def constant_position(df, level):
    return pd.Series(float(level), index=df.index)


LEVEL_GRID = [{"level": 0.0}, {"level": 1.0}]




def test_folds_are_contiguous_and_cover_everything_after_the_first_block():
    windows = _windows(1000, 4)
    assert len(windows) == 4
    assert windows[0][1] == 200
    for (_, _, previous_end), (_, next_start, _) in pairwise(windows):
        assert previous_end == next_start
    assert windows[-1][2] == 1000


def test_folds_are_anchored_and_train_on_everything_before_the_test_window():
    assert [start for start, _, _ in _windows(1000, 4)] == [0, 0, 0, 0]


def test_too_many_folds_for_the_sample_is_an_error_not_a_silent_empty_window():
    with pytest.raises(ValueError, match="rows per block"):
        _windows(10, 8)
    with pytest.raises(ValueError, match="n_folds must be at least 1"):
        _windows(1000, 0)




def test_the_embargo_defaults_to_the_longest_warm_up_in_the_grid():
    fn, grid = STRATEGY_GRIDS["sma"]
    result = walk_forward(rising_prices(3000), fn, grid, n_folds=3)
    assert result["embargo"] == max(params["slow"] for params in grid)


def test_the_embargo_removes_exactly_the_days_it_claims_to():
    result = walk_forward(rising_prices(), constant_position, LEVEL_GRID,
                          n_folds=4, embargo=30)
    for fold in result["folds"]:
        assert fold["embargoed_days"] == 30
    assert result["pooled_days"] == sum(f["test_days"] for f in result["folds"])


def test_a_zero_embargo_scores_every_test_day():
    with_embargo = walk_forward(rising_prices(), constant_position, LEVEL_GRID,
                                n_folds=4, embargo=40)
    without = walk_forward(rising_prices(), constant_position, LEVEL_GRID,
                           n_folds=4, embargo=0)
    assert without["pooled_days"] == with_embargo["pooled_days"] + 4 * 40


def test_an_embargo_that_would_empty_a_fold_is_an_error():
    with pytest.raises(ValueError, match="the embargo removes"):
        walk_forward(rising_prices(500), constant_position, LEVEL_GRID,
                     n_folds=4, embargo=200)




def test_the_obviously_better_parameter_set_is_selected_in_every_fold():
    result = walk_forward(rising_prices(), constant_position, LEVEL_GRID,
                          n_folds=5, cost_bps=0.0, embargo=10)
    assert [fold["params"] for fold in result["folds"]] == [{"level": 1.0}] * 5
    assert result["folds_won"] == 0


def test_on_a_falling_market_it_selects_cash_and_beats_buy_and_hold():
    falling = rising_prices(step=-0.05)
    result = walk_forward(falling, constant_position, LEVEL_GRID,
                          n_folds=5, cost_bps=0.0, rf=0.02, embargo=10)
    assert [fold["params"] for fold in result["folds"]] == [{"level": 0.0}] * 5
    assert result["folds_won"] == 5
    assert result["pooled_sharpe"] > result["pooled_hold_sharpe"]


def test_each_fold_scores_the_strategy_and_the_benchmark_on_identical_days():
    result = walk_forward(rising_prices(2000), constant_position, LEVEL_GRID,
                          n_folds=4, embargo=25)
    for fold in result["folds"]:
        assert fold["test_days"] > 0
        assert fold["train_days"] > 0
    assert result["pooled"]["days"] == result["pooled_days"]




def test_it_reports_what_it_did():
    result = walk_forward(rising_prices(), constant_position, LEVEL_GRID,
                          n_folds=6, embargo=5)
    assert result["n_folds"] == 6
    assert result["embargo"] == 5
    assert len(result["folds"]) == 6
    assert [fold["fold"] for fold in result["folds"]] == [1, 2, 3, 4, 5, 6]


def test_it_runs_on_real_grids_for_both_strategies(spy):
    for name in ("sma", "vol"):
        fn, grid = STRATEGY_GRIDS[name]
        result = walk_forward(spy, fn, grid, n_folds=4, cost_bps=5.0, rf=0.04)
        assert 0 <= result["folds_won"] <= 4
        assert result["pooled_days"] > 0
        assert result["pooled"]["p_value"] <= 1.0


def test_an_empty_grid_is_an_error():
    with pytest.raises(ValueError, match="at least one parameter set"):
        walk_forward(rising_prices(), constant_position, [], n_folds=2)




def test_warm_up_is_the_longest_window_each_strategy_actually_reads():
    assert warmup_days(buy_and_hold, {}) == 0
    assert warmup_days(sma_crossover, {"fast": 50, "slow": 200}) == 200
    assert warmup_days(vol_target, {"target_vol": 0.1, "lookback": 20}) == 21


def test_warm_up_falls_back_to_each_strategy_s_own_defaults():
    assert warmup_days(sma_crossover, {}) == 200
    assert warmup_days(vol_target, {}) == 21


def test_an_unregistered_strategy_fails_loudly_rather_than_getting_no_embargo():
    with pytest.raises(ValueError, match="no warm-up rule registered"):
        warmup_days(constant_position, {"level": 1.0})




def test_feasible_folds_grants_the_preferred_count_on_a_long_sample():
    assert feasible_folds(8440, 250, preferred=8) == 8


def test_feasible_folds_shrinks_for_a_shorter_sample():
    assert feasible_folds(2000, 250, preferred=8) == 3


def test_feasible_folds_is_zero_when_even_one_fold_cannot_fit():
    assert feasible_folds(400, 250, preferred=8) == 0


def test_feasible_folds_never_exceeds_what_walk_forward_actually_accepts():
    for n in (400, 550, 2000, 8440):
        for embargo in (0, 20, 250):
            folds = feasible_folds(n, embargo, preferred=8)
            if folds == 0:
                continue
            series = pd.DataFrame({"Close": [100.0 + i * 0.1 for i in range(n)]},
                                  index=pd.bdate_range("2015-01-01", periods=n))
            walk_forward(series, constant_position, LEVEL_GRID, n_folds=folds,
                         embargo=embargo)
