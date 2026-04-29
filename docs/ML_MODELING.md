# Machine Learning Modeling — Complete Technical Reference

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Problem Formulation](#2-problem-formulation)
3. [Feature Engineering](#3-feature-engineering)
4. [Label Generation (Self-Supervised Targets)](#4-label-generation-self-supervised-targets)
5. [Model Architectures](#5-model-architectures)
6. [Training Pipeline](#6-training-pipeline)
7. [Walk-Forward Validation](#7-walk-forward-validation)
8. [Decision Policy](#8-decision-policy)
9. [Inference and Prediction](#9-inference-and-prediction)
10. [Model Persistence and Caching](#10-model-persistence-and-caching)
11. [Configuration Reference](#11-configuration-reference)
12. [Data Flow: End-to-End](#12-data-flow-end-to-end)

---

## 1. System Overview

The ML engine is a **multi-task supervised learning system** that simultaneously solves three problems from the same feature matrix:

| Task | Type | Output |
|---|---|---|
| Regime classification | Multi-class (5 classes) | Calibrated probability distribution over market regimes |
| Entry quality estimation | Regression | Scalar in [0, 1] — best realized entry quality across multiple horizons |
| Exit quality estimation | Regression | Scalar in [0, 1] — worst realized exit threat across multiple horizons |

The three outputs are produced by a **shared model architecture** (either a tree ensemble or a neural network) that learns a single internal representation of market state, then branches into three specialized heads — one per task.

Three model backends are available:

- **LightGBM** — gradient-boosted decision trees (default). One classifier + two regressors. GPU-accelerated when available.
- **Logistic Regression** — multinomial logistic + Ridge regression baseline. Fully interpretable, CPU-only. Used as a sanity check.
- **MLP (Multi-Layer Perceptron)** — small PyTorch neural network with a shared backbone and three output heads. GPU via CUDA when available.

Labels are **self-supervised**: they are derived entirely from future realized returns in the historical price series — no hand-labeling, no external signals. The model learns to predict future price behavior from current market state.

---

## 2. Problem Formulation

### 2.1 Inputs (Features)

A feature matrix `X` of shape `(n_samples, n_features)` where each row represents one trading day's market state. Up to **36 features** in full mode, 21 in minimal mode. See Section 3 for the complete description of every feature.

### 2.2 Outputs (Targets)

**Target 1 — Regime** (`y_regime`): a categorical label, one of:
- `TREND_UP` — asset is in uptrend and expected to continue rising
- `TREND_DOWN` — asset is in downtrend and expected to continue falling
- `REVERSAL_UP` — asset was falling but is reversing upward
- `REVERSAL_DOWN` — asset was rising but is reversing downward
- `RANGE` — no directional trend, consolidation

**Target 2 — Entry Quality** (`y_entry`): a continuous value in [0, 1] where 1.0 = perfect entry (price rose significantly with minimal drawdown) and 0.0 = terrible entry.

**Target 3 — Exit Quality** (`y_exit`): a continuous value in [0, 1] where 1.0 = perfect exit (price dropped significantly after this point, missing a large runup) and 0.0 = no need to exit.

### 2.3 Leakage Prevention

The system has an explicit, multi-layer defense against data leakage:

1. **Chronological sort enforcement** — the DataFrame index is checked for monotonic increase before any operation. If not sorted, an explicit sort is performed and a warning is logged.
2. **Duplicate timestamp removal** — duplicate index entries are dropped (`keep='last'`) before label generation.
3. **Explicit shift operations** — all future-looking calculations use `pct_change(N).shift(-N)`, which aligns future values to current rows. Any accidental look-ahead produces NaN.
4. **NaN removal after alignment** — all rows with any NaN in features or targets are dropped via `dropna()` before training. This removes the last `forward_horizon` rows where labels cannot be computed, and any early warmup rows where indicator calculations are incomplete.
5. **Regime features vs regime targets** — the regime columns used as *features* (`regime_adx`, `regime_mkt`, etc.) are computed from **past indicators only** by `compute_indicators()`. The regime *target* (`y_regime`) is computed from past trailing returns + future forward returns. These are deliberately separate paths.
6. **Purged walk-forward boundary** — the train/validation cutoff is purged by `max(wf_gap, forward_horizon)`. In practice the engine keeps a configurable gap of **63 bars (≈ 1 quarter)** by default, but it will never purge fewer bars than the label horizon. This prevents training labels from using returns that extend into the validation window.

---

## 3. Feature Engineering

All features are computed by `build_features()` from a pre-computed indicator DataFrame. The function accepts optional context dictionaries (`market_ctx`, `fundamentals`, `sentiment`) plus optional point-in-time aligned DataFrames (`pit_market_ctx`, `pit_fundamentals`, `pit_sentiment`).

### 3.1 Design Principles

1. **No redundancy** — each feature captures an orthogonal dimension of market state.
2. **Relative, not absolute** — ratios and percentages transfer across assets of different price levels.
3. **Dual encoding** — regime state is encoded both as a continuous raw value (for tree splits) and as a categorical bucket (to allow the model to condition indicator weights on market regime).
4. **Noise reduction** — momentum derivatives use EMA smoothing rather than raw first differences to avoid reacting to single-bar noise.

### 3.2 Group A — Oscillators & Momentum (4 features)

These capture where price currently sits relative to its statistical range.

| Feature | Formula / Source | What it captures |
|---|---|---|
| `RSI` | Wilder RSI(14) | Overbought/oversold — mean-reversion signal |
| `BB_Pct` | `(Close - Lower) / (Upper - Lower)` | Price position in Bollinger Band envelope |
| `MACD_Hist` | MACD(12,26,9) histogram | Short-term momentum direction and acceleration |
| `ret_3d` | `Close.pct_change(3) * 100` | 3-day return — captures sharp reversals and momentum bursts |

### 3.3 Group B — Trend Structure (3 features)

These capture the shape and maturity of the medium/long-term trend.

| Feature | Formula | What it captures |
|---|---|---|
| `pct_from_ma50` | `(Close - MA50) / MA50 * 100` | Displacement from 50-day trend — medium-term extension |
| `pct_from_ma200` | `(Close - MA200) / MA200 * 100` | Displacement from 200-day trend — long-term extension |
| `ma_spread` | `(MA50 - MA200) / MA200 * 100` | MA50/MA200 spread — trend maturity and golden/death cross proximity |

### 3.4 Group C — Volatility & Volume (4 features)

These capture the noise level, volume conviction, and where the asset is in its volatility cycle.

| Feature | Formula | What it captures |
|---|---|---|
| `ATR_Pct` | `ATR(14) / Close * 100` | Realized volatility as percentage of price |
| `Vol_Ratio` | `Volume / Volume.rolling(20).mean()` | Current volume vs 20-day average — conviction signal |
| `obv_slope` | Chaikin Money Flow (CMF, 14-bar) — see note | Bounded buying/selling pressure from price position × volume |
| `vol_cycle` | `ATR_Pct.diff().ewm(span=10).mean()` | EMA-smoothed ATR delta — is volatility expanding or contracting? |

**`obv_slope` — Chaikin Money Flow (CMF):** Despite the column name (kept for backward compatibility with cached models), this feature is computed as Chaikin Money Flow rather than OBV slope. OBV is non-stationary and accumulates unboundedly; CMF is bounded `[-1, +1]` and measures buying/selling pressure more directly:

```
money_flow_mult = ((Close - Low) - (High - Close)) / (High - Low)
CMF_14          = sum(money_flow_mult × Volume, 14) / sum(Volume, 14)
```

Positive CMF indicates net buying pressure; negative indicates distribution. Falls back to legacy OBV slope only when High/Low/Volume are unavailable.

**`vol_cycle`** is computed inside `build_features()` by differencing `ATR_Pct` and applying EWM with `span=10`. Positive = vol expanding; negative = contracting.

### 3.5 Group D — Regime Context: Dual Encoding (8 features)

This group encodes the current market regime in two parallel representations — continuous and categorical — so the model can use both fine-grained numeric splits and coarser category conditioning simultaneously.

| Feature | Type | Encoding | What it captures |
|---|---|---|---|
| `ADX` | Continuous | Raw ADX value (0–100) | Trend strength — tree can split at any precise level |
| `regime_adx` | Categorical | `MEAN_REVERSION`→0, `NEUTRAL`→1, `TREND`→2 | ADX-based regime bucket |
| `regime_mkt` | Categorical | `BEARISH`→0, `TRANSITION`→1, `BULLISH`→2 | SPY-relative market regime |
| `regime_vol` | Categorical | `LOW`→0, `NORMAL`→1, `HIGH`→2, `EXTREME`→3 | Volatility regime bucket |
| `Vol_Pctl` | Continuous | Rolling percentile of ATR (0–100) | Position in the volatility cycle |
| `trend_stage_enc` | Ordinal | `EARLY`→0, `HEALTHY`→1, `EXTENDED`→2, `OVEREXTENDED`→3, `PARABOLIC`→4 | How mature/extended the current trend is |
| `regime_chg_enc` | Signed | `BEAR_REV`→−2, `BEAR_CONF`→−1, `WEAK`→−0.5, `NONE`→0, `BOTTOM`→+0.5, `BULL_CONF`→+1, `BULL_REV`→+2 | Regime change signal with direction and strength |
| `Trend_Ext` | Continuous | Signed distance from MA50 | Continuous trend extension measure |

The categorical features are mapped to integers using dictionaries defined at the top of `build_features()`. Any unmapped value (e.g. NaN or an unknown regime string) defaults to the neutral bucket (1 for most; 0 for trend_stage). The result is cast to `float64` so all backends can process it uniformly.

The dual-encoding rationale: a tree model can split `ADX > 35` very precisely when it sees the raw float, but also can split on `regime_adx == 2` to condition entire subtrees on whether the market is trending at all. A neural network learns a continuous embedding of `regime_adx` but can also detect the exact ADX level.

### 3.6 Group E — Momentum Dynamics (3 features)

Raw first differences of oscillators are noisy. These features apply **EWM smoothing (span=3)** to the one-step delta of each indicator, producing a signal that says "is this indicator currently accelerating upward or downward?"

| Feature | Formula | What it captures |
|---|---|---|
| `rsi_accel` | `RSI.diff().ewm(span=3).mean()` | RSI momentum — is it still strengthening or starting to fade? |
| `adx_accel` | `ADX.diff().ewm(span=3).mean()` | Trend strength momentum — is the trend getting stronger or weaker? |
| `vol_ratio_accel` | `Vol_Ratio.diff().ewm(span=3).mean()` | Volume participation shift — is volume conviction building or declining? |

### 3.7 Group F — Cross-Asset Context (4 features)

These features place the asset in context relative to the broader market. During training the engine prefers point-in-time aligned market snapshots from `PointInTimeStore`; if unavailable it falls back to historical SPY/VIX series aligned row-by-row.

| Feature | Formula | What it captures |
|---|---|---|
| `rs_1m` | `Ret_21D - spy_ret_1m` | 1-month relative strength vs SPY |
| `rs_3m` | `Ret_63D - spy_ret_3m` | 3-month relative strength vs SPY |
| `spy_trend` | Per-row: `1.0 if SPY > SPY.MA200 else 0.0` | Is the market in a long-term uptrend? (historical, per-row during training) |
| `vix_norm` | Rolling 63-day z-score, clipped to `[-1, +1]` | Relative fear level — 0 = VIX at its own 63-day average; positive = elevated |

**`vix_norm` — z-scored VIX:** Previously computed as `VIX / 80`, which was nearly constant during normal markets and contributed little information. Now uses a rolling 63-day z-score: `(VIX − rolling_mean) / rolling_std`, clipped to `[-3, 3]` and scaled to `[-1, 1]`. This captures *relative* fear (spike above recent baseline) rather than absolute level.

**At inference**, callers can eliminate the training/inference distribution shift by passing `vix_ma63` (rolling 63-day VIX mean) and `vix_std63` (rolling 63-day VIX std) in `market_ctx`. If these keys are present, they are used as the reference instead of long-run constants (mean=20, σ=8). The long-run constants are only used as a fallback when recent VIX history is unavailable — in low-volatility regimes (VIX ≈ 13, rolling mean ≈ 14), using long-run mean=20 would misrepresent the current fear level. In normal markets, the difference is small; in regime transitions it can be material.

**Cross-asset leakage prevention:** In `training_mode=True`, `rs_1m`, `rs_3m`, and `spy_trend` are computed per-row from a historical SPY DataFrame aligned by date — each row sees only the SPY data available up to that date. `vix_norm` likewise uses the historical VIX series aligned per-row. This eliminates the lookahead that would occur if today's SPY/VIX values were broadcast backward across the full history.

When `market_ctx` is not provided and not in training mode, `spy_trend` defaults to `0.5` (neutral) and `vix_norm` to `0.0` (VIX at long-run average).

### 3.8 Group G — Fundamental Quality (4 composite features — ML-zeroed, overlay-applied)

These four features are **always set to 0.0** in the ML feature matrix — in both training and inference — because no point-in-time historical fundamental dataset is available. Broadcasting today's P/E ratio backward across 5 years of training rows is lookahead bias; and if we non-zero them only at inference, LightGBM never learned splits on them (constant-zero in training), creating a silent training/inference distribution shift.

**Instead, fundamentals are applied as a post-model overlay** via `compute_fund_quality_score()`, which produces a single composite score fed to `apply_decision_policy()` as the `fund_quality_score` parameter. This gives fundamentals real influence on position sizing and vetoes without corrupting the ML feature distribution.

**`fund_quality_score` overlay thresholds:**
- `< −0.5` → hard veto: entry is blocked (dangerous balance sheet — high debt + negative margins)
- `< −0.2` → position size capped at 60% of normal (weak fundamentals)
- `> +0.3` → position size boosted +10% (strong fundamentals)

The composite score is computed from the metrics below (safety is weighted most heavily since financial distress is the primary risk to avoid):

**`fund_value`** — composite valuation score, averaged from:
- PE ratio: <15 → +1.0, <25 → +0.5, <40 → −0.5, ≥40 → −1.0
- PB ratio: <2 → +1.0, <4 → +0.5, <8 → −0.5, ≥8 → −1.0
- PEG ratio: <1 → +1.0, <2 → +0.5, <3 → −0.5, ≥3 → −1.0

**`fund_quality`** — composite quality score, averaged from:
- ROE: >20% → +1.0, >10% → +0.5, >0% → −0.5, ≤0% → −1.0
- Net margin: >20% → +1.0, >10% → +0.5, >0% → −0.5, ≤0% → −1.0

**`fund_growth`** — composite growth score, averaged from:
- Revenue growth: >20% → +1.0, >5% → +0.5, >−5% → −0.5, ≤−5% → −1.0
- EPS growth: same thresholds

**`fund_safety`** — composite safety score, averaged from:
- Debt/equity: <30% → +1.0, <80% → +0.5, <150% → −0.5, ≥150% → −1.0
- Current ratio: >2 → +1.0, >1.2 → +0.5, >0.8 → −0.5, ≤0.8 → −1.0

The averaging across available signals means that missing components are simply excluded rather than treated as neutral. In point-in-time mode the features are aligned as-of each bar and forward-filled only after release; the engine does not broadcast a current fundamentals snapshot backward across the full training history.

### 3.9 Group H — Sentiment (3 features)

Sentiment contributes three features, but historical training only uses them when `use_sentiment_in_training=True` and point-in-time sentiment snapshots exist in `PointInTimeStore`. Without PIT history they are excluded from the training feature contract. When point-in-time sentiment is available, irregular updates are aligned as-of each bar and smoothed with a 5-bar rolling mean to prevent the model from reacting to single-point noise.

| Feature | What it captures |
|---|---|
| `sent_score` | Sentiment score [−1, +1] — negative to positive |
| `sent_momentum` | Trend in sentiment over time (rising or falling) |
| `sent_dispersion` | Spread of opinions — consensus vs. divided market |

When sentiment is unavailable at inference, all three features are set to 0.0.

### 3.10 Group I — Liquidity (3 features)

These features filter illiquid assets and capture shifts in market participation. All are computed from the asset's own OHLCV data — no external dependencies or lookahead risk.

| Feature | Formula | What it captures |
|---|---|---|
| `dollar_vol_rank` | `log(Close × Volume, rolling 20) / log(1e8)`, clipped to [0, 2] | Normalised average daily dollar volume — higher = more liquid |
| `spread_proxy` | `((High − Low) / Close).rolling(5).mean() × 100` | Intraday range as bid-ask proxy — lower = tighter spread = more liquid |
| `amihud` | `log(1 + mean(|ret| / dollar_vol, 20) × 1e6)`, clipped to [0, 15] | Amihud (2002) illiquidity ratio — higher = less liquid, larger price impact per dollar traded |

**Amihud illiquidity ratio** (Amihud 2002): the most empirically robust measure of equity illiquidity in the academic literature. It captures the price impact per unit of dollar volume — a large absolute return accompanied by low dollar volume indicates the stock is thin and moves easily. The log-scale compresses the fat tail. Higher values mean the asset is less liquid and any trade will move the price more.

### 3.11 Feature Selection

After construction, only features explicitly requested in the config are returned:

```python
available = [f for f in feature_names if f in feat.columns]
return feat[available]
```

This means that if a required indicator column (e.g. `MA200`) was not computed because the data was too short, the corresponding feature is silently dropped. The training pipeline then re-checks which features are actually available and uses only those.

---

## 4. Label Generation (Self-Supervised Targets)

All three targets are computed by `generate_labels()` using **only the price series** — no human annotation is required. The helper `_compute_path_stats(prices, start, horizon)` extracts `(max_drawdown, max_runup, realized_vol)` for a given forward window, used by the multi-horizon loop.

### 4.1 Regime Labels

Regime is classified using trailing momentum (past `forward_horizon` bars) vs. forward return (next `forward_horizon` bars):

| Condition | Label |
|---|---|
| `trail_ret > strong_thresh (6%)` AND `fwd_ret > weak_thresh (2%)` | `TREND_UP` |
| `trail_ret < −strong_thresh` AND `fwd_ret < −weak_thresh` | `TREND_DOWN` |
| `trail_ret < −strong_thresh` AND `fwd_ret > weak_thresh` | `REVERSAL_UP` |
| `trail_ret > strong_thresh` AND `fwd_ret < −weak_thresh` | `REVERSAL_DOWN` |
| Everything else | `RANGE` |

The `strong_threshold` (default **6%**, 21-day horizon) and `weak_threshold` (default **2%**) define what constitutes a significant directional move. `RANGE` is the residual class and dominates (~60–70% of bars), which is corrected by `class_weight='balanced'` in the classifier.

**Important**: regime is a *target*, not a feature. The regime *features* (`regime_adx`, `regime_mkt`) are computed from past indicators only by `compute_indicators()`.

### 4.2 Multi-Horizon Entry and Exit Quality

Entry and exit quality are computed across **three time horizons** to capture patterns at different holding speeds:

```
short_h  = forward_horizon // 3    # ≈ 7 days for 21d horizon
medium_h = forward_horizon * 2//3  # ≈ 14 days
long_h   = forward_horizon         # 21 days
```

For each horizon `h`, the `_compute_path_stats()` helper returns:
- `max_dd[i, h]`: worst intra-window drawdown from bar `i`
- `max_ru[i, h]`: best intra-window gain from bar `i`
- `fwd_vol[i, h]`: realized daily return volatility in the window (floored at 0.5%)

These are used to compute per-horizon raw scores:

```python
# Entry: reward-to-risk ratio, normalized by realized vol
entry_raw[h] = (max_ru[h].clip(0) - 0.5 * abs(max_dd[h])) / fwd_vol[h]

# Exit: threat-to-opportunity ratio, normalized by realized vol
exit_raw[h]  = (abs(max_dd[h]) - 0.5 * max_ru[h].clip(0)) / fwd_vol[h]
```

**Entry quality = `max` across all horizons.** Taking the maximum means: "at some point in the next 21 days, there was a good entry opportunity here." A stock that surges +8% within 7 days is rated as a good entry even if it later pulls back by day 21. This captures "quick win" patterns that a single coarse 21-day horizon would obscure.

**Exit quality = `max` across all horizons.** Similarly, if the stock crashes at *any* of the three horizons, the model learns that staying long was dangerous — even if it recovered later.

After computing across horizons, a sigmoid with scale=10 maps both scores to [0, 1]:

```
sigmoid(x, scale) = 1 / (1 + exp(-scale × x))
```

Scale=10 produces a smooth sigmoid, avoiding the overconfident sharp transitions of higher-scale values.

### 4.3 The 0.5× Risk/Opportunity Cost Weighting

Both formulas penalize the adverse component at 50% of the favourable component. This is a deliberate design choice:

- **Entries**: penalizing drawdown at only 50% avoids the model refusing to signal entries that have any drawdown risk. A trade with +10% upside and −3% drawdown still produces a strong positive raw score.
- **Exits**: penalizing opportunity cost at 50% prevents the model from being too reluctant to exit because it fears missing further upside.

---

## 5. Model Architectures

### 5.1 LightGBM (Default)

Three separate LightGBM models are trained: one classifier for regime, two regressors for entry and exit quality.

**Classifier parameters** (from `MLConfig`):
- `n_estimators`: number of boosting rounds (default 300)
- `max_depth`: maximum tree depth (default 5)
- `learning_rate`: shrinkage per round (default 0.03)
- `num_leaves`: max leaves per tree (default 31)
- `subsample`: 0.7 — row subsampling per round (reduces overfitting)
- `colsample_bytree`: 0.7 — feature subsampling per round
- `min_child_samples`: 30 — minimum samples per leaf
- `reg_alpha`: 0.1 — L1 regularisation
- `reg_lambda`: 1.0 — L2 regularisation
- `class_weight`: `"balanced"` — corrects RANGE class dominance by weighting minority classes (REVERSAL_UP/DOWN) inversely proportional to their frequency
- `n_jobs=-1`, `verbosity=-1`

The same parameters (without `class_weight`) are used for both regressors.

**Three-way train/eval/calibration split**: The data is split into three non-overlapping portions:

```
[0 .. 70%]   → Training data (LightGBM fitting)
[70% .. 85%] → Eval set (LightGBM early-stop logging)
[85% .. 100%]→ Calibration set (Platt/isotonic scaling)
```

```python
calib_split = int(len(X) * 0.85)
eval_split  = int(len(X) * 0.70)
```

**Probability Calibration (Platt / Isotonic Scaling)**: After fitting the base classifier, a `CalibratedClassifierCV(cv='prefit', method='isotonic')` is fitted on the held-out calibration set. This is a post-hoc calibration layer that maps the raw LightGBM class probabilities to better-calibrated output probabilities. Without calibration, LightGBM probabilities near class boundaries tend to be overconfident — a raw probability of 0.85 may not truly correspond to 85% empirical accuracy. After calibration, the `min_regime_confidence` threshold of 0.35 has a consistent probabilistic interpretation.

**Minimum calibration set size: 50 samples.** Isotonic regression has O(n) capacity and overfits aggressively on small datasets. If the 15% calibration slice contains fewer than 50 samples (common in early walk-forward folds with short training windows), calibration is skipped and the raw LightGBM probabilities are used directly. This is safer than a miscalibrated isotonic fit.

The calibrated classifier is stored as `self._calibrated_clf`. At prediction time, it is used in preference to the raw classifier (`getattr(self, '_calibrated_clf', None) or self.regime_clf`).

**Feature importance**: Gain-based importance from the raw classifier, normalized to sum to 1.0, top 15 returned.

**Prediction**: `predict_proba()` returns a `(n_samples, 5)` calibrated probability matrix. Entry and exit predictions are clipped to [0, 1].

### 5.2 Logistic Regression (Baseline)

A scikit-learn `LogisticRegression` with multinomial objective, L-BFGS solver, C=1.0 (inverse regularization strength), and max 500 iterations. Entry and exit regressors use scikit-learn `Ridge(alpha=1.0)`.

**Preprocessing**: A `StandardScaler` is fitted on the training data and stored. At prediction time, `transform()` is applied using the stored scaler. This is critical — logistic regression is sensitive to feature scale, unlike tree methods.

**Feature importance**: The classifier's coefficient matrix is `(n_classes, n_features)`. The mean absolute coefficient across classes is computed, normalized, sorted, and the top 15 are returned. This is an approximation since multinomial logistic regression coefficients are not strictly comparable across classes.

**Epoch log**: A single entry with the training log-loss is recorded. There is no iterative training to track.

### 5.3 MLP / PyTorch

A small feedforward neural network with a **shared backbone** and **three output heads**. The architecture is:

```
Input (n_features)
    ↓
[Linear → BatchNorm1d → GELU → Dropout] × len(hidden_dims)
    ↓ shared backbone
   /        |        \
Regime    Entry     Exit
Head      Head      Head
(5-way    (1-way    (1-way
softmax)  sigmoid)  sigmoid)
```

Default `hidden_dims = [128, 64, 32]`, giving a 4-layer backbone:
- `n_features → 128 → BatchNorm → GELU → Dropout(0.3)`
- `128 → 64 → BatchNorm → GELU → Dropout(0.3)`
- `64 → 32 → BatchNorm → GELU → Dropout(0.3)`

Output heads:
- **Regime head**: `Linear(32, 5)` — raw logits, softmax at inference time
- **Entry head**: `Linear(32, 1) → Sigmoid()` — naturally bounded to (0, 1)
- **Exit head**: `Linear(32, 1) → Sigmoid()` — naturally bounded to (0, 1)

**Device selection**: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`. All tensors and model parameters are moved to this device.

**Normalization**: Feature normalization is computed from the training data (`mean` and `std + 1e-8` for numerical stability) and stored as NumPy arrays on the model object. At inference time, `(X - mean) / std` is applied before passing to the network. This normalization is critical for BatchNorm and gradient stability.

**Class imbalance handling**: Class weights are computed as `1 / class_count` for each regime class, then normalized to sum to `n_classes`:

```python
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * n_classes
```

These weights are passed to `CrossEntropyLoss(weight=...)`. This prevents the model from ignoring rare regimes (e.g. REVERSAL_DOWN) in favor of the dominant RANGE class.

**Loss function**: A weighted sum of three losses:

```
total_loss = w_regime × CrossEntropy(regime_logits, y_regime)
           + w_entry  × MSE(entry_head, y_entry)
           + w_exit   × MSE(exit_head, y_exit)
```

Default weights are all 1.0. The loss is back-propagated through all three heads and the shared backbone simultaneously — the backbone receives gradients from all three tasks.

**Optimizer**: `AdamW` with `lr=1e-3` and `weight_decay=1e-4`. AdamW decouples the weight decay from the gradient update, providing stronger regularization than Adam.

**Scheduler**: `CosineAnnealingLR` from initial LR to `eta_min=1e-6` over `T_max=max(20, epochs//3)`. The cosine cycle is set to one-third of `config.epochs` rather than the full epoch count, so the LR reaches `eta_min` by the time early stopping typically fires. Using `T_max=epochs` caused the LR to still be near its initial value when early stopping triggered at epoch 20 of 100 — the last epochs received no fine-tuning benefit from a low LR.

**Gradient clipping**: `torch.nn.utils.clip_grad_norm_(all_params, 1.0)` is applied at every step. This prevents exploding gradients, which are common with BatchNorm + GELU in early training.

**DataLoader**:
- `drop_last=True` when `len(dataset) > batch_size` — prevents the last incomplete mini-batch from crashing `BatchNorm1d`, which requires at least 2 samples.
- `shuffle=True` — data is re-shuffled every epoch.
- Effective batch size: `min(config.batch_size, max(2, len(dataset)))`.

**Train/val split for early stopping**: The training data is split 85%/15%:

```python
val_split = int(len(X_norm) * 0.85)
```

The 85% partition is used for training; the 15% partition is used for two purposes: (1) early stopping patience monitoring and (2) temperature calibration (see below). It is NOT the walk-forward test set.

**Early stopping**:
- Patience: 10 epochs without improvement (configurable via `early_stopping_patience`)
- Best weights are snapshot at every validation improvement using `{k: v.clone() for k, v in state_dict.items()}`
- On triggering, best weights are restored before the function returns
- `self._early_stop_epoch` records the epoch where stopping occurred, for the dashboard display

**Batch training loop** (per epoch):
```
for each mini-batch:
    h = backbone(X_batch)
    loss = w_r × CE(regime_head(h), y_regime) + w_e × MSE(entry_head(h), y_entry) + w_x × MSE(exit_head(h), y_exit)
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(1.0)
    optimizer.step()

scheduler.step()  # once per epoch, not per batch

eval mode for validation:
    with no_grad:
        compute val_loss on held-out 15%
        check early stopping
```

**MC Dropout (inference-time uncertainty)**:

At inference time, multiple forward passes are run with dropout enabled to estimate prediction uncertainty:

```python
for _ in range(mc_passes):  # default 20
    # enable only Dropout modules (not BatchNorm)
    for m in backbone.modules():
        if isinstance(m, nn.Dropout): m.train()

    regime_probs[pass], entry[pass], exit[pass] = forward(X)

mean_probs = mean(all_probs, axis=0)
mean_entry = mean(all_entry)
mean_exit  = mean(all_exit)

uncertainty = {
    "regime_std": mean(std(all_probs, axis=0)),  # avg std across class probabilities
    "entry_std":  std(all_entry),
    "exit_std":   std(all_exit),
}
```

A critical implementation detail: `backbone.train()` would also put `BatchNorm1d` into per-batch mode, which crashes on single-sample inputs. Instead, only `Dropout` modules are switched to train mode individually. `BatchNorm1d` stays in eval mode (using stored running statistics).

The stored uncertainty is consumed by `predict_from_df()` and then reset to `None` to prevent stale values from being used on subsequent calls.

**Full-data retrain after early stopping**: After early stopping discovers the optimal epoch count (`optimal_epochs`), the model is retrained from its best weights on the **full dataset** (85% + 15% combined) for exactly `optimal_epochs` epochs with a fresh optimizer and `CosineAnnealingLR(T_max=optimal_epochs)`. This is equivalent to sklearn's `refit=True` in `GridSearchCV` — the hyperparameter search uses the held-out validation set, then the final model uses all available data. Without this step, the final model permanently wastes ~15% of training data.

**Probability calibration — Temperature Scaling**: After the full-data retrain, `_calibrate_temperature()` fits a single scalar `T` on the held-out 15% validation set. This is the MLP equivalent of LightGBM's `CalibratedClassifierCV`. Temperature scaling (Guo et al. 2017) divides the regime logits by `T` before softmax:

```
calibrated_probs = softmax(regime_logits / T)
```

`T > 1` softens overconfident distributions (typical for neural networks); `T = 1` is the raw softmax. `T` is optimised by minimising NLL on the calibration set using L-BFGS, parameterised in log-space to guarantee `T > 0`. Clipped to `[0.5, 5.0]`. Requires ≥ 50 calibration samples; falls back to `T = 1.0` otherwise. Applied to both deterministic and MC Dropout prediction paths.

**Feature importance for MLP**: Currently approximated as uniform distribution across all features (1/n_features each). Proper attribution would require gradient-based methods (e.g. integrated gradients) — listed as a future improvement.

---

## 6. Training Pipeline

### 6.1 Entry Points

The `MLEngine.train()` method is the sole entry point for training. It accepts:
- `ticker` — the stock symbol
- `df` — optional pre-computed indicator DataFrame (if None, fetches from yfinance)
- `market_ctx`, `fundamentals`, `sentiment` — optional context dicts

### 6.2 Cache Check

Before any computation, the engine checks for a cached model:

```python
cached = self._load_cached(ticker)
if cached is not None:
    self._models = cached
    return TrainResult(...)  # return immediately with cached model
```

Cache keys are derived from a deterministic hash of the config parameters:

```python
sig = f"{ticker}|{model_type}|{training_period}|{forward_horizon}|{strong_threshold}|{feature_set}|{n_trees}|{max_depth}"
key = hashlib.md5(sig.encode()).hexdigest()[:12]
```

A cached model is considered stale if its age exceeds `max_model_age_days` (default 7). The age is stored in a companion `.json` metadata file alongside the model file.

### 6.3 Data Preparation

The indicator DataFrame goes through:

1. `build_features()` → produces `X_df` (feature DataFrame)
2. `generate_labels()` → produces `regime`, `entry_q`, `exit_q` (label Series)
3. All three are joined into a combined DataFrame, then `dropna()` is called
4. The `X`, `y_regime`, `y_entry`, `y_exit` arrays are extracted and cast to the appropriate dtypes
5. `np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)` is applied as a final safety step to catch any remaining non-finite values

Auxiliary arrays are also extracted at this step:
- `prices_aligned` — Close prices reindexed to the same rows as the feature matrix
- `dates_aligned` — the DatetimeIndex of valid rows
- `vol_feat` — `Vol_Pctl / 100` for walk-forward volatility bucket assignment

### 6.4 Validation Mode: Walk-Forward vs Chronological Split

**Walk-forward mode** (default):

```
1. Run _walk_forward_evaluate() on full dataset
   → produces WalkForwardMetrics with per-fold results
2. Keep the last 15% as a final holdout window
   → purge the forward label horizon from the fit window
3. Train FINAL model on the purged first 85%
   → this is the model used for inference
4. Evaluate the final model on the held-out last 15%
```

The crucial distinction: walk-forward evaluation uses separate ephemeral models trained and evaluated per fold. The **final model** is then trained once on the purged pre-holdout segment. `WalkForwardMetrics` remain the primary out-of-sample trading report because they span multiple historical train/test boundaries, while the final 15% holdout gives a clean end-of-sample accuracy/F1/MAE check.

**Chronological split mode** (when `train_start/train_end/test_start/test_end` are provided):

```
1. Train model ONLY on train window
2. Evaluate on test window using _walk_forward_evaluate() with force_split=split
   → single "fold" — no expanding windows
3. This model is the final model (trained on train window only)
```

In chronological mode, the final model intentionally does NOT see the test data. This provides a harder, more realistic out-of-sample evaluation at the cost of using less data for training.

### 6.5 Final Accuracy Evaluation

After training, the final model is evaluated on the **last 15%** of rows:

```python
split = int(len(X) * 0.85)
probs, ent_pred, exit_pred, classes = models.predict(X[split:])
regime_preds = [classes[i] for i in probs.argmax(axis=1)]
acc = accuracy_score(y_regime[split:], regime_preds)
```

Per-class F1 scores are computed for each of the 5 regime classes:
```python
for cls in REGIME_CLASSES:
    y_bin = (y_regime[split:] == cls).astype(int)   # one-vs-rest
    p_bin = (regime_preds == cls).astype(int)
    f1s[cls] = f1_score(y_bin, p_bin, zero_division=0)
```

Entry and exit MAE are computed similarly:
```python
entry_mae = mean(|y_entry[split:] - ent_pred|)
exit_mae  = mean(|y_exit[split:] - exit_pred|)
```

**Important**: In walk-forward mode the final holdout is kept out of the fit entirely, and the fit window is additionally purged by `forward_horizon` before the split. These accuracy numbers are therefore out-of-sample for the final model. The broader trading-readiness judgement still comes from `WalkForwardMetrics` plus registry admission gates.

---

## 7. Walk-Forward Validation

### 7.1 Purpose

Walk-forward validation simulates how the trading strategy would have performed if you had trained on progressively more data and traded on each new unseen slice. It avoids look-ahead bias because each fold's model never touches future data during training.

### 7.2 Fold Construction

**TimeSeriesSplit** from scikit-learn is used. The actual number of splits is `min(n_splits, max(2, len(X) // 50))` — capped so each fold has at least 50 training samples.

For 1200 samples and 10 splits:
```
Fold 0:  Train [0..109]        Test [110..219]
Fold 1:  Train [0..219]        Test [220..329]
...
Fold 9:  Train [0..1089]       Test [1090..1199]
```

Each test slice is approximately `n / (n_splits + 1)` bars.

**Rolling window mode**: If `wf_window == "rolling"` and `wf_rolling_size` is set, `TimeSeriesSplit(max_train_size=wf_rolling_size)` caps the training window size. This means the train window moves forward rather than expanding:

```
Fold 3 (rolling, size=300):  Train [30..329]   Test [330..439]
Fold 4 (rolling, size=300):  Train [140..439]  Test [440..549]
```

**Gap enforcement**: After building each fold, the training indices are trimmed to exclude the last `max(wf_gap, forward_horizon)` bars before the test slice:

```python
cutoff = val_idx[0] - max(gap, forward_horizon)
train_idx = train_idx[train_idx < cutoff]
```

If after trimming fewer than 20 samples remain, the fold is skipped entirely. The default 63-bar gap (one full quarter) is conservative enough for slow-moving context features, while the `forward_horizon` floor guarantees that no training label can peek into validation prices even when the configured gap is smaller.

### 7.3 Per-Fold Process

For each fold:

**Step 1 — Train an ephemeral model** using `_create_model(config, n_features)`:
- A brand new model is instantiated with the same config
- Fitted only on `X[train_idx]`, `y_regime[train_idx]`, `y_entry[train_idx]`, `y_exit[train_idx]`
- Feature names are passed as dummy integers (not the real names) since this is evaluation only

**Step 2 — Predict on the test slice**:
```python
probs, entry_pred, exit_pred, classes = model.predict(X[val_idx])
regime_pred = [classes[i] for i in probs.argmax(axis=1)]
acc = accuracy_score(y_regime[val_idx], regime_pred)
```

**Step 3 — Simulate trades** (discrete long-only simulation):

The simulation iterates over each bar in the test window. Entry and exit thresholds are taken from `DecisionPolicy` defaults (not hardcoded) so the reported Sharpe reflects the same thresholds the live strategy uses:

```
if FLAT and entry_pred[i] > DecisionPolicy.entry_threshold and regime_pred[i] in (TREND_UP, REVERSAL_UP):
    → enter long at fold_prices[i]
    → deduct wf_trade_cost (0.1% per leg) from equity immediately
    → record entry_price, entry_idx

if LONG and (exit_pred[i] > DecisionPolicy.exit_threshold or regime_pred[i] in (TREND_DOWN, REVERSAL_DOWN)):
    → close position
    → deduct wf_trade_cost from equity immediately
    → trade.return = (exit_price - entry_price) / entry_price
    → trade.holding_days = current_bar - entry_bar
    → trade.regime = regime_labels[entry_idx]  (the regime at entry, not at exit)
    → trade.vol_bucket = LOW/MED/HIGH based on Vol_Pctl at entry
```

The per-leg cost (`wf_trade_cost`, default 0.001 = 10bps) is configured in `MLConfig`. This makes the walk-forward Sharpe a realistic estimate — without it, the Sharpe would be fictitiously high since no real strategy trades for free (20bps round-trip is conservative for liquid US equities). The cost is applied as an immediate equity deduction at the moment of entry or exit, not spread across holding days.

The regime at entry is captured from the ground-truth `y_regime` labels (the same labels used for training), not the model's prediction. This is intentional — it measures performance within each predicted regime independently.

**Step 4 — Build per-fold equity curve** (continuous signal-weighted):

Parallel to the discrete trade simulation, a continuous equity curve is computed:

```python
for i in range(1, len(fold_prices)):
    daily_ret = (fold_prices[i] - fold_prices[i-1]) / fold_prices[i-1]
    signal_weight = entry_pred[i] - exit_pred[i]
    strategy_ret = daily_ret * max(0, signal_weight)
    fold_equity *= (1 + strategy_ret)
```

This represents a continuous strategy where the position size each day is proportional to `max(0, entry_score - exit_score)`. It produces smoother equity curves than the discrete simulation and is used for Sharpe, MaxDD, and CAGR calculations.

### 7.4 Per-Fold Metrics

After each fold, the following are computed and stored in `fold_results`:

| Metric | Source | Formula |
|---|---|---|
| `accuracy` | Regime classification | `accuracy_score(y_true, y_pred)` on test slice |
| `n_trades` | Discrete simulation | Count of closed trades in this fold |
| `sharpe` | Continuous equity curve | `(mean(pnl) / std(pnl)) × √252` |
| `total_return` | Continuous equity curve | `equity[-1] / equity[0] - 1` |
| `max_drawdown` | Continuous equity curve | `min((equity - peak) / peak)` |
| `win_rate` | Discrete trades | `count(return > 0) / total_trades` |
| `volatility` | Continuous equity curve | `std(pnl) × √252` |
| `regime_dist` | Ground-truth labels | Count of each regime class in test slice |
| `by_regime_sharpe` | Discrete trades by entry regime | Annualized Sharpe per regime class across trades in this fold |

**Per-regime Sharpe** is computed by grouping the fold's discrete trades by the regime at entry, then computing annualized Sharpe for each group. This reveals whether the strategy extracts real alpha in RANGE markets (which often it does not), as opposed to only in trending conditions. The RANGE group's Sharpe feeds the automatic RANGE entry gate (see Section 8.4).

### 7.5 Aggregate Metrics — Two Views

The `WalkForwardMetrics` object reports two perspectives, plus cross-regime performance summaries. The `range_regime_sharpe` field holds the walk-forward Sharpe for RANGE-regime entries across all folds combined, and is the primary input to the automatic RANGE entry gate (see Section 8.4).

**All-folds aggregate**: Computed from the concatenation of all fold pnl/equity/trades. `equity_curve` starts at 1.0 and grows across all folds in sequence, simulating having run the strategy for the entire evaluated period.

**Last-fold-only**: Computed identically but using only the last fold's data. The last fold has the largest training set (most trained model) and therefore produces the most reliable performance estimate. Early fold models are undertrained and may produce misleading metrics.

Both views report the same set of metrics: Sharpe, MaxDD, CAGR, total return, hit rate, profit factor, n_trades, avg trade return, avg trades/month, avg holding period, performance by regime, and performance by volatility bucket.

### 7.6 Cross-Fold Consistency

After all folds complete, consistency metrics are computed across the fold-level Sharpe and return values:

- `worst_fold_idx`: index of the fold with the lowest Sharpe ratio
- `fold_sharpe_std`: standard deviation of per-fold Sharpe ratios — high std = inconsistent strategy
- `fold_return_std`: standard deviation of per-fold total returns
- `pct_folds_profitable`: fraction of folds where total return was positive

A strategy with high mean Sharpe but high `fold_sharpe_std` may be overfitting to specific market conditions.

---

## 8. Decision Policy

The `apply_decision_policy()` function maps the three raw model outputs (regime, entry_score, exit_score) into a formal `TradeDecision` struct with an action, position size, conviction level, and human-readable reasons.

### 8.1 Inputs

| Input | Source | Range |
|---|---|---|
| `regime` | argmax of regime probs | String enum (5 values) |
| `regime_confidence` | max regime probability | [0, 1] |
| `entry` | entry head output | [0, 1] |
| `exit_score` | exit head output | [0, 1] |
| `uncertainty` | MC Dropout std dict | Optional |
| `fund_quality_score` | `compute_fund_quality_score(fundamentals)` | [−1, +1] or None |

`fund_quality_score` is computed by the standalone `compute_fund_quality_score()` function from the raw fundamentals dict — it is **not** an ML feature. See Section 3.8 for why fundamentals are excluded from the model's feature matrix and applied as a post-model overlay instead.

### 8.2 Uncertainty Penalty

```python
uncertainty_penalty = min(1.0, (entry_std + regime_std) / 0.3)
```

Combines entry and regime uncertainty. If the total exceeds 0.3, the penalty saturates at 1.0 (maximum uncertainty). A penalty of 0.5 means the model is moderately uncertain; 1.0 means maximally uncertain.

### 8.3 Exit Rules (Priority: Higher = Earlier)

Rules are checked in priority order. The first matching rule returns immediately.

1. **Urgent exit** (`exit_score ≥ 0.80`): Full SELL at 100% position, HIGH conviction. No conditions — a very high exit score overrides everything.

2. **Regime-confirmed exit** (`exit_score ≥ 0.60` AND regime in TREND_DOWN/REVERSAL_DOWN): SELL at 75% position, HIGH conviction.

3. **Uncertainty-triggered reduction** (`exit_score ≥ 0.51` AND `uncertainty_penalty > 0.70`): REDUCE to 50% position, MEDIUM conviction. The model is unsure — de-risk.

4. **Moderate exit** (`exit_score ≥ 0.60`): REDUCE to 50% (RANGE regime) or 75% (other regimes), MEDIUM conviction.

### 8.4 Entry Rules

Two veto conditions are checked first:

- **Uncertainty veto**: `entry_std > 0.15` AND `uncertainty_penalty > 0.5` → entry vetoed
- **Regime confidence veto**: `regime_confidence < 0.35` → entry vetoed

Then entry rules (checked in order):

1. **Strong entry** (`entry ≥ 0.75` AND favorable regime AND not vetoed):
   ```
   raw_size = entry × (1 - 0.5 × uncertainty_penalty)
   size = clamp(raw_size, min_pos=0.10, max_pos=1.0)
   ```
   Action: BUY, conviction: HIGH.

2. **Standard entry** (`entry ≥ 0.60` AND favorable regime AND not vetoed):
   ```
   raw_size = entry × (1 - 0.6 × uncertainty_penalty)
   size = clamp(raw_size, min_pos=0.10, max_pos=0.70)
   ```
   Action: BUY, conviction: MEDIUM.

3. **Speculative entry** (`entry ≥ 0.60` AND RANGE regime AND not uncertainty-vetoed AND `disable_range_entries=False`):
   ```
   raw_size = entry × 0.4 × (1 - 0.7 × uncertainty_penalty)
   size = clamp(raw_size, min_pos=0.10, max_pos=0.40)
   ```
   Action: BUY, conviction: LOW. The smaller size cap and larger uncertainty multiplier reflect the lower quality of RANGE regime entries.

**RANGE entry gate** (`disable_range_entries` flag): The `DecisionPolicy` has a `disable_range_entries: bool = False` flag. When `True`, rule 3 (Speculative entry) is skipped entirely — the strategy never enters new positions in RANGE markets. This flag is automatically set by `MLEngine.predict_from_df()` when the walk-forward `range_regime_sharpe` is ≤ 0: if RANGE entries had negative Sharpe historically, the engine gates them out at inference time. This is a data-driven, automatic circuit breaker that prevents the model from repeatedly entering low-quality consolidation trades. The gate persists for the lifetime of the model object (it is set when `wf_metrics` is stored on the engine during training).

### 8.5 WATCH Rules

If no BUY or SELL is triggered, two WATCH conditions are checked:

- REVERSAL_UP detected with `regime_confidence > 0.45` → WATCH (potential bottom forming)
- `entry ≥ 0.51` in a favorable regime → WATCH (approaching threshold)

### 8.6 Signal String Mapping

The `TradeDecision` is mapped to a legacy signal string for backward compatibility:

| Action | Conviction | Signal |
|---|---|---|
| SELL | HIGH | EXIT |
| BUY | HIGH | STRONG ENTRY |
| BUY | MEDIUM | ENTRY |
| BUY | LOW | SPECULATIVE |
| REDUCE | any | REDUCE |
| WATCH | any | WATCH (REVERSAL) |
| HOLD | any | HOLD |

---

## 9. Inference and Prediction

### 9.1 Single-Bar Prediction (`predict_from_df`)

Takes a DataFrame with indicators, builds features from the **last row only**, runs prediction, and applies the decision policy.

```python
X = X_df.iloc[[-1]].values.astype(np.float32)
X = nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# For MLP: 20 MC Dropout passes
# For others: 1 deterministic pass
probs, entry, exit_, classes = model.predict(X, mc_passes=mc_passes)
```

Any feature columns present in the stored model's feature list but missing from the current DataFrame are filled with 0.0. This ensures inference works even when not all data sources are available.

The output is an `MLPrediction` containing: regime, regime confidence, all regime probabilities, entry score, exit score, signal string, full decision dict, feature importances, and uncertainty dict (if MC Dropout was run).

### 9.2 Full Timeseries Prediction (`predict_timeseries`)

Runs prediction on every valid row of a DataFrame for dashboard charting. Rows where all features are zero (warmup period) are excluded via a mask:

```python
valid_mask = ~(X == 0).all(axis=1)
```

Returns a dictionary with aligned lists of dates, prices, regime labels, entry scores, exit scores, and per-class probability time series. This is used to render the regime-colored price chart and the entry/exit score timeline in the ML Lab dashboard.

---

## 10. Model Persistence and Caching

### 10.1 File Format

| Backend | Model File | Metadata |
|---|---|---|
| LightGBM | `{ticker}_{key}.pkl` (pickle) | `{ticker}_{key}.json` |
| Logistic | `{ticker}_{key}.pkl` (pickle) | `{ticker}_{key}.json` |
| MLP | `{ticker}_{key}.pt` (torch.save) | `{ticker}_{key}.json` |

All files are stored in `app/ml_models/`.

The cache key is the first 12 hex characters of `MD5(config_signature)`. This ensures that changing any training parameter (period, model type, features, hyperparameters, thresholds) produces a different key and triggers retraining.

### 10.2 Metadata File

The `.json` metadata file contains:

```json
{
    "timestamp": 1711234567.89,
    "ticker": "AAPL",
    "model_type": "lightgbm",
    "n_features": 36
}
```

The `timestamp` is compared against `max_model_age_days` to determine staleness.

### 10.3 Cache Invalidation

When a new training is initiated via the API:
- Cache files matching `{ticker}_{hex8+}.*` are deleted before training begins
- Registry files matching `{ticker}_{modeltype}_v{N}.pkl` are preserved — the regex only matches hexadecimal hash filenames, not version-tagged registry files

### 10.4 Registry (Versioned Saves)

The `ModelRegistry` maintains a JSON index at `ml_models/registry.json`. Each saved version has a unique ID (`{ticker}_{model_type}_v{N}`), stores the full config dict, training metadata, and points to a pickle file.

When loading from the registry, a minimal `TrainResult` is reconstructed from the stored metadata to allow the engine to function without retraining.

### 10.5 Registry Admission Gates

The registry is no longer a passive archive. `train_pipeline.py --refresh-registry` applies two layers of admission control before a model is allowed back into `registry.json`:

1. **Single-name readiness gate** via `assess_live_readiness()`:
   - minimum walk-forward Sharpe
   - minimum CAGR
   - maximum drawdown
   - maximum fold Sharpe dispersion
   - minimum fraction of profitable folds
   - minimum trade count
2. **Portfolio-aware gate** via `portfolio_registry.py` across the surviving candidates:
   - average pairwise return correlation
   - average position overlap
   - average gross exposure
   - average net exposure
   - average daily turnover
   - median capacity in dollars

If no candidate set passes both gates, `registry.json` remains empty and the previous registry should be treated only as archival evidence, not as approved capital-ready inventory.

---

## 11. Configuration Reference

All parameters are in `MLConfig`:

| Parameter | Default | Description |
|---|---|---|
| `backend` | `"auto"` | `"auto"` (CPU), `"pytorch"` (force GPU) |
| `model_type` | `"lightgbm"` | `"lightgbm"`, `"mlp"`, `"logistic"` |
| `training_period` | `"5y"` | yfinance period string |
| `forward_horizon` | `21` | Days ahead for label generation (3 horizons: 7/14/21) |
| `strong_threshold` | `0.06` | 6% trailing move required for trend detection |
| `weak_threshold` | `0.02` | 2% forward continuation required for trend/reversal |
| `n_trees` | `300` | LightGBM: number of boosting estimators |
| `max_depth` | `5` | LightGBM: max tree depth |
| `learning_rate` | `0.03` | LightGBM: shrinkage rate per round |
| `num_leaves` | `31` | LightGBM: max leaves per tree |
| `feature_set` | `"full"` | `"full"` (36 features) or `"minimal"` (21 features) |
| `train_mode` | `"per_ticker"` | `"per_ticker"` by default; `universe` is research-only until portfolio execution/risk is mature |
| `cv_splits` | `5` | Walk-forward fold count |
| `wf_gap` | `63` | Bars gap between train end and test start (≈ 1 quarter) |
| `wf_window` | `"expanding"` | `"expanding"` (growing train) or `"rolling"` (fixed window) |
| `wf_rolling_size` | `None` | Max training bars in rolling mode |
| `use_fundamentals_in_training` | `False` | Include fundamentals only when PIT history exists |
| `use_sentiment_in_training` | `False` | Include sentiment only when PIT history exists |
| `target_annual_vol` | `0.15` | Vol-target used in walk-forward position sizing |
| `max_drawdown_trigger` | `0.15` | Exposure reduction trigger for deep drawdowns |
| `min_dollar_volume` | `5_000_000` | Liquidity floor for tradability-aware signals |
| `hidden_dims` | `[128, 64, 32]` | MLP: neurons per hidden layer |
| `epochs` | `100` | MLP: max training epochs |
| `batch_size` | `64` | MLP: mini-batch size |
| `pt_learning_rate` | `1e-3` | MLP: AdamW initial learning rate |
| `dropout` | `0.3` | MLP: dropout probability |
| `early_stopping_patience` | `10` | MLP: epochs without improvement before stopping |
| `loss_w_regime` | `1.0` | MLP: weight for regime classification loss |
| `loss_w_entry` | `1.0` | MLP: weight for entry regression loss |
| `loss_w_exit` | `1.0` | MLP: weight for exit regression loss |
| `mc_dropout_passes` | `20` | MLP: MC Dropout inference passes for uncertainty |
| `max_model_age_days` | `7` | Cache staleness threshold |

| `wf_trade_cost` | `0.001` | Per-leg cost in walk-forward simulation (10bps = 20bps round-trip) |

**LightGBM calibration**: after fitting the classifier, a `CalibratedClassifierCV(cv='prefit', method='isotonic')` is fitted on a held-out 15% calibration slice — provided it contains ≥50 samples. The raw LightGBM classifier is used directly when the calibration set is too small. This calibration step is not a config parameter — it always runs for the LightGBM backend when the data allows it.

---

## 12. Data Flow: End-to-End

```
yfinance (OHLCV)
    │
    ▼
compute_indicators()
    Adds: MA50, MA200, RSI, MACD, ATR, BB, CMF, ADX, Vol_Ratio, Vol_Pctl,
          Regime, Mkt_Regime, Vol_Regime, Trend_Stage, Trend_Ext, Regime_Chg,
          Ret_21D, Ret_63D (all from past data only)
    │
    ▼
build_features()
    Constructs X_df (36 or 21 columns) from indicator columns
    Applies EWM smoothing for momentum dynamics
    Maps categorical regimes to integers
    Computes CMF (bounded [-1,+1]) as `obv_slope` column
    Computes VIX rolling 63-day z-score as `vix_norm`
    Computes Amihud illiquidity ratio (log-scaled) as `amihud`
    Pulls PIT market/fundamental/sentiment snapshots when available
    Drops unsupported train-only features from the training feature contract
    Fills unavailable inference-time context with safe defaults
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
generate_labels()                  X_df (features)
    Computes fwd_ret, trail_ret        │
    _compute_path_stats() per row      │
    Three horizons: 7/14/21 days       │
    Takes max entry/exit across all    │
    horizons via nanmax                │
    Assigns regime class (5 classes)   │
    Applies sigmoid(x, scale=10)       │
    │                                  │
    └──────────────┬───────────────────┘
                   │
                   ▼
             dropna() alignment
             (drops last forward_horizon rows + warmup rows)
                   │
                   ▼
    ┌──────────────┴──────────────┐
    │                             │
    ▼                             ▼
Walk-Forward                 Final Training
(ephemeral models per fold)  (purged 85% fit → held-out 15% eval)
    │                              │
    ▼                              ▼
WalkForwardMetrics           CalibratedClassifierCV
(per-fold + aggregate        (isotonic, cv='prefit')
 by_regime_sharpe,           stored as _calibrated_clf
 range_regime_sharpe)             │
    │                             ▼
    │                        TrainResult
    │                        (accuracy, F1, MAE, importances)
    │                             │
    └──────────────┬──────────────┘
                   │
                   ▼
             TrainResult (combined)
             → assess_live_readiness()
             → portfolio_registry.py (batch refresh only)
             → range_regime_sharpe → disable_range_entries gate
             → API / registry / dashboard display
                   │
                   ▼
             predict_timeseries()
             → regime chart, entry/exit score timeline
                   │
                   ▼
             predict_from_df()  (latest bar)
             → apply_decision_policy(disable_range_entries=...)
             → TradeDecision (BUY/SELL/HOLD/REDUCE/WATCH)
             → MLPrediction
             → Live Signal panel
```

---

## Appendix: Known Limitations

1. **Entry/exit simulation vs. equity curve mismatch**: The walk-forward validation runs two parallel simulations — discrete trades (for hit rate, profit factor) and continuous signal-weighted equity (for Sharpe, MaxDD). These can diverge since they use different position sizing logic.

2. **Walk-forward remains the primary trading metric**: The final model now uses a purged 85/15 holdout split for accuracy/F1/MAE, but the more decision-relevant out-of-sample evidence is still the multi-fold `WalkForwardMetrics` object.

3. **MLP feature importance is uniform**: Proper attribution requires gradient-based methods; currently returns equal weights for all features.

4. **Fundamental features are excluded from the ML model**: Without a point-in-time historical fundamental dataset, fundamentals are zeroed in both training and inference ML features to prevent distribution shift. They are applied as a post-model overlay via `compute_fund_quality_score()` → `fund_quality_score` in `apply_decision_policy()`. A proper implementation would source quarterly snapshots aligned to SEC filing dates and include them as ML features with full temporal alignment.

5. **Walk-forward Sharpe includes a 10bps-per-leg cost** (`wf_trade_cost=0.001`): This is conservative for liquid large-cap US equities but may understate friction for small-cap or international names. Adjust `MLConfig.wf_trade_cost` to match expected execution costs for the asset class.

6. **Class imbalance in RANGE**: The RANGE class typically accounts for 50–70% of all labels. `class_weight='balanced'` corrects the training signal, but per-class F1 scores for rare classes (REVERSAL_DOWN, TREND_DOWN) will still be lower than RANGE/TREND_UP. The RANGE entry gate (Section 8.4) provides a runtime guard when RANGE entries underperform historically.

7. **Probability calibration requires sufficient calibration data**: The isotonic regression calibrator requires ≥50 samples in the 15% held-out calibration slice. Below that floor, the raw LightGBM probabilities are used directly. For very short training periods (<500 bars total), isotonic calibration will almost never fire — this is the correct behavior.

8. **Multi-horizon label max can mask trend persistence**: Taking the maximum entry score across 7/14/21 days captures the "best opportunity window" but does not reflect whether the opportunity was sustained. A stock that spikes +8% on day 7 then reverses −12% by day 21 will receive a high entry label. Buyers who held to day 21 would have lost money. The exit quality label partially compensates for this, but the entry/exit combination does not fully model holding-period risk.

9. **Execution is paper/shadow only by default**: The repo now includes a guarded execution interface, paper ledger, and soak-period tracking, but no live broker adapter ships in-tree. Any broker connection should be treated as a future integration and must route through `LiveRiskManager`.
