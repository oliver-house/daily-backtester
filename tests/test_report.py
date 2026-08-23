import json
import re

import pytest

import report
from sweep import STRATEGY_GRIDS


@pytest.fixture(scope="module")
def evidence(request):
    spy = request.getfixturevalue("spy")
    return report.build_evidence(spy, "sma")


def test_every_grid_point_is_scored_on_both_halves(evidence):
    _, grid = STRATEGY_GRIDS["sma"]
    assert len(evidence["grid"]) == len(grid)
    assert {key for point in evidence["grid"] for key in point} == {"params", "is", "oos"}
    assert 0 <= evidence["best"] < len(grid)


def test_the_highlighted_point_really_is_the_in_sample_winner(evidence):
    best = evidence["grid"][evidence["best"]]
    assert best["is"] == max(point["is"] for point in evidence["grid"])


def test_the_null_histogram_accounts_for_every_bootstrap_draw(evidence):
    rc = evidence["reality_check"]
    assert sum(rc["null"]["counts"]) == rc["n_boot"]
    assert len(rc["null"]["edges"]) == len(rc["null"]["counts"]) + 1
    assert rc["null"]["edges"][0] <= rc["statistic"] <= rc["null"]["edges"][-1]
    assert 0 < rc["p_value"] <= 1


def test_it_is_reproducible(spy):
    first = report.build_evidence(spy, "sma")["reality_check"]["p_value"]
    second = report.build_evidence(spy, "sma")["reality_check"]["p_value"]
    assert first == second


def test_the_dashboard_uses_feasible_folds_at_the_evidence_fold_count():
    from backtest.validation import feasible_folds

    assert feasible_folds(8440, 250, report.EVIDENCE_FOLDS) == report.EVIDENCE_FOLDS
    assert feasible_folds(2000, 250, report.EVIDENCE_FOLDS) == 3


def test_a_history_too_short_for_any_fold_omits_the_chart_rather_than_crashing(spy):
    short = spy.iloc[:400]
    assert report.build_evidence(short, "sma")["walk_forward"] is None


def test_both_strategies_produce_a_panel(spy):
    for strategy in STRATEGY_GRIDS:
        built = report.build_evidence(spy, strategy)
        assert built["grid"] and built["reality_check"]["n_boot"] > 0


def test_the_deflated_sharpe_ratio_stays_out_of_the_panel(evidence):
    assert "deflated" not in evidence
    assert "moments" not in evidence


def test_the_payload_is_json_serialisable_and_small(spy):
    payload = json.dumps(report.build_evidence(spy, "sma"))
    assert len(payload) < 20_000
    assert "NaN" not in payload and "Infinity" not in payload


def test_the_built_page_is_self_contained(spy, tmp_path):
    data = {
        "generated": "2026-01-01",
        "defaults": report.DEFAULTS,
        "evidence_settings": {"cost_bps": 5.0, "rf": 0.04, "folds": 8, "n_boot": 1000},
        "tickers": ["SPY"],
        "series": {"SPY": {"dates": ["2020-01-01"], "prices": [100.0]}},
        "evidence": {"SPY": {"sma": report.build_evidence(spy, "sma")}},
    }
    page = report.render(data)

    assert not re.search(r'<(script|link)[^>]+(src|href)=["\']https?://', page)
    assert "/*DATA*/" not in page and "/*END*/" not in page
    assert json.loads(re.search(r"const DATA = (\{.*?\});\n", page, re.S).group(1))["tickers"] == ["SPY"]
