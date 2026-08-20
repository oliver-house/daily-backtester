const TRADING_DAYS = 252;
// Mirrors POSITION_EPSILON in backtest/engine.py; see the comment there for
// why exact equality miscounts trades. tests/test_parity.py checks the two
// agree on real vol-target series, which is where it bites.
const POSITION_EPSILON = 1e-12;

// ---------------------------------------------------------------------
// Engine: a direct port of backtest/engine.py, backtest/strategy.py, and
// backtest/inference.py. Every formula here has a corresponding line in
// the Python source, so the two stay in sync by inspection.
// ---------------------------------------------------------------------

function mean(a) { return a.reduce((s, v) => s + v, 0) / a.length; }

function sampleStd(a) {
  if (a.length < 2) return 0;
  const m = mean(a);
  const ss = a.reduce((s, v) => s + (v - m) ** 2, 0);
  return Math.sqrt(ss / (a.length - 1));
}

function rollingMean(a, window) {
  const out = new Array(a.length).fill(NaN);
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i];
    if (i >= window) sum -= a[i - window];
    if (i >= window - 1) out[i] = sum / window;
  }
  return out;
}

function rollingStd(a, window) {
  // Matches pandas rolling(window).std(): NaN unless the full window is
  // free of NaN, sample standard deviation (ddof=1).
  const out = new Array(a.length).fill(NaN);
  for (let i = window - 1; i < a.length; i++) {
    const win = a.slice(i - window + 1, i + 1);
    if (win.some(Number.isNaN)) continue;
    out[i] = sampleStd(win);
  }
  return out;
}

function pctChange(prices) {
  // pandas .pct_change(): NaN for the first element, undefined otherwise.
  const out = new Array(prices.length).fill(NaN);
  for (let i = 1; i < prices.length; i++) out[i] = (prices[i] - prices[i - 1]) / prices[i - 1];
  return out;
}

const Strategy = {
  hold: (prices) => prices.map(() => 1.0),

  sma: (prices, { fast, slow }) => {
    if (fast < 1 || fast >= slow) return prices.map(() => 0.0);
    const f = rollingMean(prices, fast), s = rollingMean(prices, slow);
    return prices.map((_, i) => (f[i] > s[i]) ? 1.0 : 0.0); // NaN > NaN is false, matches pandas
  },

  vol: (prices, { targetVol, lookback }) => {
    if (targetVol < 0 || lookback < 1) return prices.map(() => 0.0);
    const returns = pctChange(prices);
    const realised = rollingStd(returns, lookback).map(v => v * Math.sqrt(TRADING_DAYS));
    return realised.map(v => {
      if (!(v > 0)) return 0.0; // NaN (warm-up) or exactly 0 (unusable): flat
      // Capped at 1: unlevered, long-only. Matches backtest/strategy.py.
      return Math.min(targetVol / v, 1.0);
    });
  },
};

function riskAdjustedRatio(meanExcess, dispersion) {
  if (dispersion > 0) return (meanExcess / dispersion) * Math.sqrt(TRADING_DAYS);
  if (meanExcess > 0) return Infinity;
  if (meanExcess < 0) return -Infinity;
  return 0.0;
}

function runBacktest(prices, positions, costBps, rfAnnual) {
  const n = prices.length;
  const assetReturns = new Array(n).fill(0);
  for (let i = 1; i < n; i++) assetReturns[i] = (prices[i] - prices[i - 1]) / prices[i - 1];

  const held = new Array(n).fill(0);
  for (let i = 1; i < n; i++) held[i] = positions[i - 1];

  const costs = new Array(n).fill(0);
  costs[0] = costBps / 10000 * Math.abs(positions[0]);
  for (let i = 1; i < n; i++) costs[i] = costBps / 10000 * Math.abs(positions[i] - positions[i - 1]);

  const rfDaily = Math.pow(1 + rfAnnual, 1 / TRADING_DAYS) - 1;
  const daily = new Array(n);
  for (let i = 0; i < n; i++) {
    daily[i] = held[i] * assetReturns[i] + (1 - held[i]) * rfDaily - costs[i];
  }

  // 1 + d_t = h_t(1+r_t) + (1-h_t)(1+rho) - c_t must stay > 0, or equity hits
  // zero/negative -- only costs can cause this. Mirrors the raise in
  // backtest/engine.py rather than silently producing NaN/garbage equity.
  const equity = new Array(n);
  let cum = 1;
  for (let i = 0; i < n; i++) {
    const growth = 1 + daily[i];
    if (growth <= 0) {
      // Carry the raw day index rather than formatting a message here:
      // this function only has prices/positions, not the ticker's date
      // strings, so the caller (which does) builds the final wording.
      const err = new Error("equity wipeout");
      err.dayIndex = i;
      err.dailyReturn = daily[i];
      throw err;
    }
    cum *= growth;
    equity[i] = cum;
  }

  let trades = 0;
  if (Math.abs(positions[0]) > POSITION_EPSILON) trades++;
  for (let i = 1; i < n; i++) {
    if (Math.abs(positions[i] - positions[i - 1]) > POSITION_EPSILON) trades++;
  }

  return { daily, equity, rfDaily, trades };
}

function computeStats(equity, daily, rfDaily, trades) {
  const n = daily.length;
  const totalReturn = equity[n - 1] - 1;
  const years = n / TRADING_DAYS;
  const cagr = years > 0 ? Math.pow(equity[n - 1], 1 / years) - 1 : 0;
  const vol = sampleStd(daily) * Math.sqrt(TRADING_DAYS);

  const excess = daily.map(d => d - rfDaily);
  const sharpe = riskAdjustedRatio(mean(excess), sampleStd(excess));

  const downside = excess.map(e => Math.min(e, 0) ** 2);
  const downsideDev = Math.sqrt(mean(downside));
  const sortino = riskAdjustedRatio(mean(excess), downsideDev);

  let sharpeSe, ciLow, ciHigh;
  if (!isFinite(sharpe)) {
    sharpeSe = Infinity; ciLow = ciHigh = sharpe;
  } else {
    sharpeSe = Math.sqrt((TRADING_DAYS + 0.5 * sharpe ** 2) / n);
    ciLow = sharpe - 1.96 * sharpeSe; ciHigh = sharpe + 1.96 * sharpeSe;
  }

  let peak = 1, maxDrawdown = 0;
  for (let i = 0; i < n; i++) {
    peak = Math.max(peak, equity[i]);
    maxDrawdown = Math.min(maxDrawdown, equity[i] / peak - 1);
  }

  return { total_return: totalReturn, cagr, annual_vol: vol, sharpe,
           sharpe_se: sharpeSe, sharpe_ci_low: ciLow, sharpe_ci_high: ciHigh,
           sortino, max_drawdown: maxDrawdown, trades, days: n };
}

function erfc(x) {
  // Abramowitz-Stegun 7.1.26, |error| < 1.5e-7.
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const tau = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 +
    t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
    t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? tau : 2 - tau;
}

function pairedTest(strategyDaily, benchmarkDaily) {
  const diff = strategyDaily.map((d, i) => d - benchmarkDaily[i]);
  const n = diff.length;
  if (n === 0) return { mean_diff: 0, se: 0, t_stat: 0, p_value: 1, days: 0, lags: 0 };

  const meanDiff = mean(diff);
  const lags = Math.max(0, Math.min(Math.floor(4 * (n / 100) ** (2 / 9)), n - 1));
  const dev = diff.map(v => v - meanDiff);
  let variance = dev.reduce((s, v) => s + v * v, 0) / n;
  for (let j = 1; j <= lags; j++) {
    let cov = 0;
    for (let i = j; i < n; i++) cov += dev[i] * dev[i - j];
    cov /= n;
    variance += 2 * (1 - j / (lags + 1)) * cov;
  }
  const se = variance > 0 ? Math.sqrt(variance / n) : 0;
  const tStat = se > 0 ? meanDiff / se : 0;
  const pValue = se > 0 ? erfc(Math.abs(tStat) / Math.SQRT2) : 1;
  return { mean_diff: meanDiff, se, t_stat: tStat, p_value: pValue, days: n, lags };
}

// Exported for tests/parity_runner.js, which checks these functions against
// backtest/engine.py on shared fixtures. `module` is undefined in a browser,
// so this line does nothing once report.py inlines the file into the page.
if (typeof module !== "undefined") {
  module.exports = { TRADING_DAYS, mean, sampleStd, rollingMean, rollingStd, pctChange,
                     Strategy, riskAdjustedRatio, runBacktest, computeStats, erfc, pairedTest };
}
