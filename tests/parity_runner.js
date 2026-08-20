// Runs the browser engine over the cases in tests/parity_cases.py and prints
// its results as JSON. tests/test_parity.py computes the same quantities with
// backtest/engine.py and compares. Invoked as:
//
//     node tests/parity_runner.js <cases.json>
//
// Nothing here reimplements anything: it only calls templates/engine.js, which
// is the exact text report.py inlines into the published dashboard.

const fs = require("fs");
const path = require("path");

const engine = require(path.join(__dirname, "..", "templates", "engine.js"));
const { Strategy, runBacktest, computeStats, pairedTest } = engine;

// JSON has no way to carry Infinity or NaN, and Sharpe is legitimately signed
// infinity when dispersion is zero. Encode those as tagged strings so the
// comparison is exact rather than quietly turning them into null.
function enc(v) {
  if (typeof v !== "number") return v;
  if (Number.isNaN(v)) return "nan";
  if (v === Infinity) return "inf";
  if (v === -Infinity) return "-inf";
  return v;
}

const encAll = (a) => a.map(enc);

function positionsFor(kase, prices) {
  if (kase.strategy === "sma") return Strategy.sma(prices, kase.params);
  if (kase.strategy === "vol") return Strategy.vol(prices, kase.params);
  return Strategy.hold(prices);
}

const cases = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));

const results = cases.map((kase) => {
  const prices = kase.prices;
  const positions = positionsFor(kase, prices);
  const out = { name: kase.name, positions: encAll(positions) };

  let result, benchmark;
  try {
    result = runBacktest(prices, positions, kase.cost_bps, kase.rf);
    benchmark = runBacktest(prices, Strategy.hold(prices), kase.cost_bps, kase.rf);
  } catch (err) {
    // The engine throws only on equity wipeout, and carries the offending day.
    return { ...out, error: "wipeout", day_index: err.dayIndex };
  }

  const stats = computeStats(result.equity, result.daily, result.rfDaily, result.trades);
  const benchStats = computeStats(
    benchmark.equity, benchmark.daily, benchmark.rfDaily, benchmark.trades);

  return {
    ...out,
    daily: encAll(result.daily),
    equity: encAll(result.equity),
    stats: Object.fromEntries(Object.entries(stats).map(([k, v]) => [k, enc(v)])),
    bench_sharpe: enc(benchStats.sharpe),
    paired: Object.fromEntries(
      Object.entries(pairedTest(result.daily, benchmark.daily)).map(([k, v]) => [k, enc(v)])),
  };
});

process.stdout.write(JSON.stringify(results));
