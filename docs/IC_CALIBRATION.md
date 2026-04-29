# Information Coefficient (IC) Calibration Guide

## Overview

The IC calibration feature learns which technical indicators are most predictive for a specific stock by measuring their correlation with future returns. Instead of using a regression model, it uses **Information Coefficient (IC)**, a rank-based statistical measure that avoids overfitting and multicollinearity.

## What is Information Coefficient (IC)?

**IC** is the Spearman rank correlation between an indicator's value on day T and the stock's forward return 21 days later (day T+21).

```
IC = rank_correlation(indicator_value_T, return_T+21)
```

- **Range**: -1.0 to +1.0
- **Typical values**: ±0.02 to ±0.08 for single-stock technical indicators
- **Interpretation**:
  - IC > 0: indicator is **predictive** (high indicator → high future return)
  - IC ≈ 0: **no predictive power**
  - IC < 0: **anti-predictive** (high indicator → low future return)

### Why Rank Correlation?

Rank-based correlation is **nonparametric** — it doesn't assume linearity and is robust to outliers. This works better for technical indicators which often have nonlinear relationships with returns.

## Calibration Workflow

### Step 1: Backtest Period Selection
User selects an evaluation period in the UI:
- Example: **2 Years** (the recent 2-year backtest window)

### Step 2: IC Calibration Period Selection
User selects a separate, non-overlapping period for calibration:
- Example: **5 Years back** from today
- The calibration window is automatically cut at the start of the backtest period to prevent data leakage
- Actual calibration window: years 5–2 ago (data strictly before the 2-year backtest starts)

### Step 3: Fetch Historical Data
The calibration fetches `calib_period` worth of price history and computes all technical indicators (warmup period: 200 bars).

### Step 4: Compute IC for Each Indicator
For each technical indicator (RSI, MACD, BB%, MA20, MA50, MA200, moving average cross):

```python
# 1. Get indicator values (after 200-bar warmup)
indicator_values = df_calib['RSI']

# 2. Compute 21-day forward returns
price = df_calib['Close']
forward_return = price.shift(-21) / price - 1.0

# 3. Remove NaN values
valid = (indicator_values.notna()) & (forward_return.notna())

# 4. Rank both series
ind_rank = indicator_values[valid].rank()
ret_rank = forward_return[valid].rank()

# 5. Spearman correlation
ic_value = ind_rank.corr(ret_rank)
```

Example output:
```
RSI:      0.040
MACD:     0.062
BB%:     -0.018
MA20:    -0.005
MA50:     0.052
MA200:    0.038
MA Cross: 0.045
```

### Step 5: Convert IC to Weights
Indicators with positive IC get proportional weight; negative IC indicators are **suppressed to zero**.

```
Positive ICs: RSI(0.040) + MACD(0.062) + MA50(0.052) + MA200(0.038) + MA Cross(0.045)
Sum of positive ICs = 0.237
Number of positive indicators = 5

Normalized weights (so average positive weight = 1.0):
RSI:      0.040 / 0.237 × 5 = 0.844
MACD:     0.062 / 0.237 × 5 = 1.308
BB%:      0 (suppressed)
MA20:     0 (suppressed)
MA50:     0.052 / 0.237 × 5 = 1.097
MA200:    0.038 / 0.237 × 5 = 0.803
MA Cross: 0.045 / 0.237 × 5 = 0.950
```

These weights multiply the indicator's contribution to the final signal score. High-IC indicators get amplified; low/negative-IC indicators are zeroed or downweighted.

## Data Leakage Prevention

**Critical**: The calibration window must not overlap with the backtest evaluation window, otherwise the weights will be overfitted to the test period itself.

Separation guarantee:
```
Calibration period: 5 years of history
Backtest period:    2 years (most recent)
Backtest start:     today - 2 years = ~2025-04-08

Calibration window: everything before 2025-04-08
Actual data used:   years 5–2 ago (e.g., 2021-04-08 to 2025-04-08, minus last 2y)
```

This ensures:
- ✅ Weights are learned on historical patterns (2021–2025)
- ✅ Backtest evaluates on fresh, unseen data (2025–2026)
- ✅ No look-ahead bias or overfitting

## Usage in Backtest

Once IC weights are calibrated:

1. **Run Backtest** → baseline signal scores (equal weights)
2. **⚙ Calibrate IC Weights** → learns which indicators predict returns (non-overlapping periods)
3. **💾 Save Weights Config** → saves the calibrated weights to Step 4 list
4. **↻ Rerun with IC Weights** → reruns the same backtest period with IC-weighted indicator contributions

Example impact:
```
Baseline Signal Quality: 52.3%
Rerun with IC Weights:  58.7%  (+6.4% improvement)
```

## Why Not Linear Regression?

A naive approach would be OLS regression:
```
return = β₀ + β₁×RSI + β₂×MACD + β₃×BB% + ...
```

Problems:
1. **Multicollinearity**: MA20, MA50, MA200 are highly correlated (>0.90). OLS estimates become unstable.
2. **Overfitting**: Regression can fit noise in the training window and fail out-of-sample.
3. **Interpretation**: Coefficients are hard to interpret when features are correlated.
4. **No ranking of importance**: All features included equally.

IC approach:
- ✅ Rank-based (nonparametric, robust)
- ✅ One-indicator-at-a-time (avoids multicollinearity)
- ✅ Directly measures predictiveness (correlation with realized returns)
- ✅ Simple normalization (positive ICs → weights, negative ICs → zero)

## Advanced Metrics (In Code)

The system also computes:

### Walk-Forward Out-of-Sample (OOS) IC
Divides the historical period into non-overlapping 21-day test windows and computes IC on each. Measures consistency and detects regime shifts:
```
Window 1 IC: 0.045
Window 2 IC: 0.038
Window 3 IC: 0.052
...
Average IC: 0.042
Std Dev:    0.008
ICIR:       0.042 / 0.008 × √12 ≈ 1.82 (Information Ratio)
```

High ICIR = consistent predictive power across time.

### Factor Exposure Analysis
OLS regression on three broad factors:
- **Momentum** (12-month minus 1-month return)
- **Trend** (50-day moving average slope)
- **Low Volatility** (negative realized volatility)

Measures how much of forward return variance is explained by these factors.

### Signal Correlation Matrix
Spearman pairwise correlations between indicator time series. Identifies redundant signals:
- MA20 ↔ MA50: 0.98 (highly redundant)
- RSI ↔ MACD: 0.34 (independent signals)

High correlation suggests one indicator is redundant and could be suppressed further.

## Configuration Storage

Saved configurations (Step 4) include:

```json
{
  "config_id": "a1b2c3d4",
  "name": "NVDA IC weights (bt:2y cal:5y)",
  "ic_calibration_ticker": "NVDA",
  "ic_weights": {
    "rsi": 0.844,
    "macd": 1.308,
    "bb": 0,
    "ma20": 0,
    "ma50": 1.097,
    "ma200": 0.803,
    "cross": 0.950
  },
  "ic_calibration_meta": {
    "calib_period": "5y",
    "backtest_period": "2y",
    "backtest_start": "2025-04-08",
    "calib_end_date": "2025-04-08",
    "n_calib_bars": 987,
    "raw_ic": {
      "rsi": 0.040,
      "macd": 0.062,
      ...
    }
  }
}
```

## Typical Results

For a single stock with typical technical indicators:
- **5–10 indicators evaluated** per stock
- **2–3 positive-IC indicators** (the others suppressed)
- **Signal quality improvement**: 2–8% in backtest when moving from equal to IC-weighted
- **ICIR**: 0.8–2.0 (values > 1.0 indicate consistent predictive power)

## Limitations

1. **Single-stock calibration**: IC weights learned on one stock don't necessarily transfer to other stocks with different price behavior.
2. **Regime sensitivity**: IC can change over time (e.g., during bull/bear markets). Periodic recalibration recommended.
3. **Forward-looking bias in choice of 21 days**: The choice of T+21 for forward return is arbitrary; a 5-day or 63-day forward window would yield different weights.
4. **Small sample**: If the calibration window is short or the stock has low volatility, IC estimates become noisy.

## Further Reading

- **Information Ratio (ICIR)**: `mean(IC) / std(IC) × √12` — Sharpe ratio of the IC time series
- **Fundamental Law of Active Management**: `IR = IC × √BR`, where BR is breadth (number of independent bets)
- **Spearman vs Pearson**: Rank correlation is robust to outliers and nonlinear relationships
