from sweep import sweep_ticker


def test_sweep_ticker_returns_both_default_corrections(load):
    record = sweep_ticker("SPY", "sma", cost_bps=5.0, rf=0.04, load=load,
                          n_boot=100, n_folds=2)
    assert 0 < record["reality_check"]["p_value"] <= 1
    assert record["walk_forward"]["n_folds"] == 2
    assert "deflated" not in record
