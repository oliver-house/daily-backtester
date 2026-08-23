import json
import math
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from backtest.engine import run
from tests.parity_cases import cases, python_positions

RUNNER = Path(__file__).parent / "parity_runner.js"
TOLERANCE = 1e-12

needs_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="Node is not installed; CI runs this test"
)


def enc(value):
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def compare(label, expected, actual):
    if isinstance(expected, str) or isinstance(actual, str):
        assert expected == actual, f"{label}: python {expected!r} vs js {actual!r}"
    else:
        assert actual == pytest.approx(expected, rel=TOLERANCE, abs=1e-15), (
            f"{label}: python {expected!r} vs js {actual!r}"
        )


@pytest.fixture(scope="module")
def js_results(tmp_path_factory):
    payload = tmp_path_factory.mktemp("parity") / "cases.json"
    payload.write_text(json.dumps(cases()), encoding="utf-8")
    completed = subprocess.run(
        ["node", str(RUNNER), str(payload)],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert completed.returncode == 0, f"node failed:\n{completed.stderr}"
    return {result["name"]: result for result in json.loads(completed.stdout)}


@needs_node
@pytest.mark.parametrize("case", cases(), ids=lambda c: c["name"])
def test_the_two_engines_agree(case, js_results):
    js = js_results[case["name"]]
    prices = pd.Series(case["prices"], index=pd.bdate_range("2020-01-01", periods=len(case["prices"])))
    positions = python_positions(case, prices)

    for day, (expected, actual) in enumerate(zip(positions, js["positions"], strict=True)):
        compare(f"positions[{day}]", enc(expected), actual)

    try:
        result = run(prices, positions, cost_bps=case["cost_bps"], rf_annual=case["rf"])
    except ValueError as exc:
        assert "wipe out" in str(exc), f"unexpected python error: {exc}"
        assert js.get("error") == "wipeout", f"python raised but js did not: {js.get('stats')}"
        assert prices.index[js["day_index"]].date() == pd.Timestamp(
            str(exc).split(" on ")[1].split(":")[0]
        ).date()
        return

    assert "error" not in js, f"js raised but python did not: {js['error']}"

    for day, (expected, actual) in enumerate(zip(result.daily_returns, js["daily"], strict=True)):
        compare(f"daily[{day}]", enc(expected), actual)
    for day, (expected, actual) in enumerate(zip(result.equity, js["equity"], strict=True)):
        compare(f"equity[{day}]", enc(expected), actual)
    for key, expected in result.stats.items():
        compare(f"stats.{key}", enc(expected), js["stats"][key])


@needs_node
@pytest.mark.parametrize("case", [c for c in cases() if c["strategy"] != "hold"],
                         ids=lambda c: c["name"])
def test_the_two_significance_tests_agree(case, js_results):
    js = js_results[case["name"]]
    if "error" in js:
        pytest.skip("case has no result to compare")

    from backtest import buy_and_hold, paired_test

    prices = pd.Series(case["prices"], index=pd.bdate_range("2020-01-01", periods=len(case["prices"])))
    result = run(prices, python_positions(case, prices), cost_bps=case["cost_bps"], rf_annual=case["rf"])
    benchmark = run(prices, buy_and_hold(pd.DataFrame({"Close": prices})),
                    cost_bps=case["cost_bps"], rf_annual=case["rf"])
    expected = paired_test(result.daily_returns, benchmark.daily_returns)

    for key in ("mean_diff", "se", "t_stat", "days", "lags"):
        compare(f"paired.{key}", enc(expected[key]), js["paired"][key])
    assert js["paired"]["p_value"] == pytest.approx(expected["p_value"], abs=1.5e-7)


def test_the_engine_file_is_what_the_dashboard_actually_ships():
    import report

    engine_source = report.ENGINE.read_text(encoding="utf-8")
    page = report.render({"generated": "2026-01-01", "defaults": report.DEFAULTS,
                          "tickers": [], "series": {}})
    assert engine_source in page
    assert "/*ENGINE*/" not in page


def test_deliberate_divergences_are_recorded_not_forgotten():
    from backtest import sma_crossover

    prices = pd.Series([100.0] * 10, index=pd.bdate_range("2020-01-01", periods=10))
    with pytest.raises(ValueError, match="fast window must be shorter"):
        sma_crossover(pd.DataFrame({"Close": prices}), fast=20, slow=5)

    assert not any(
        case["strategy"] == "sma" and case["params"]["fast"] >= case["params"]["slow"]
        for case in cases()
    )
