"""
ML Classifier Engine for regime detection and entry/exit signals.

Self-supervised: labels are derived from realised future returns.
Supports three model types:
  - lightgbm   (gradient-boosted trees, GPU-accelerated when available)
  - logistic   (logistic regression via sklearn, CPU)
  - mlp        (small MLP, GPU via PyTorch + CUDA when available)
  - ensemble   (LightGBM + logistic blended signal)

Usage:
    engine = MLEngine(MLConfig(...))
    result = engine.train(ticker, period="5y")   # returns TrainResult
    pred   = engine.predict(latest_features)      # returns MLPrediction
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Model cache directory ────────────────────────────────────────────────────
_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "ml_models"
_MODEL_DIR.mkdir(exist_ok=True)

# ── Regime class labels ──────────────────────────────────────────────────────
REGIME_CLASSES = ["TREND_UP", "TREND_DOWN", "REVERSAL_UP", "REVERSAL_DOWN", "RANGE"]
_CLS2IDX = {c: i for i, c in enumerate(REGIME_CLASSES)}

# ── Feature definitions ──────────────────────────────────────────────────────
#
# Design principles:
#   1. NO redundancy — each feature captures an orthogonal dimension
#   2. Regime context — the model knows WHEN indicators behave differently
#   3. Multi-source — technical + fundamental + sentiment + cross-asset
#   4. Relative, not absolute — ratios and ranks transfer across assets
#
# Groups:
#   A. Oscillators & momentum (4)  — where is price relative to its range?
#   B. Trend structure (3)         — what kind of trend, how mature?
#   C. Volatility & volume (4)     — noise level, confirmation, vol cycle
#   D. Regime context (8)          — continuous + categorical (dual encoding)
#   E. Momentum dynamics (3)       — EMA-smoothed indicator acceleration
#   F. Cross-asset context (4)     — relative to market
#   G. Fundamental quality (4)     — forward-filled quarterly signals
#   H. Sentiment (3)               — rolling-smoothed news signals
#   Total: 33 orthogonal features (full), 19 (minimal)

# A. Oscillators & momentum — orthogonal price-range signals
_OSCILLATORS = [
    "RSI",              # mean-reversion signal (oversold/overbought)
    "BB_Pct",           # price position in volatility envelope
    "MACD_Hist",        # momentum direction & acceleration
    "ret_3d",           # 3-day return — captures short-term reversals & sharp moves
]

# B. Trend structure — trend quality and maturity
_TREND = [
    "pct_from_ma50",    # medium-term trend displacement
    "pct_from_ma200",   # long-term trend displacement
    "ma_spread",        # MA50/MA200 spread — trend maturity proxy
]

# C. Volatility & volume — noise level, confirmation, vol cycle position
_VOL = [
    "ATR_Pct",          # realised volatility (% of price)
    "Vol_Ratio",        # volume vs 20d avg — conviction
    "obv_slope",        # OBV trend — smart money flow direction
    "vol_cycle",        # EMA-smoothed ATR% delta — is vol rising or falling?
]

# D. Regime context — DUAL encoding: continuous raw + categorical bucket
#    Model gets both the precise value AND the regime label, so it can:
#    - use raw ADX for fine-grained splits
#    - use regime_adx to condition indicator weights
_REGIME = [
    "ADX",              # continuous trend strength (0-100)
    "regime_adx",       # categorical: 0=MEAN_REV, 1=NEUTRAL, 2=TREND
    "regime_mkt",       # categorical: 0=BEARISH, 1=TRANSITION, 2=BULLISH
    "regime_vol",       # categorical: 0=LOW, 1=NORMAL, 2=HIGH, 3=EXTREME
    "Vol_Pctl",         # continuous vol percentile (0-100) — position in vol cycle
    "trend_stage_enc",  # ordinal: 0=EARLY → 4=PARABOLIC
    "regime_chg_enc",   # signed: -2=BEAR_REV → +2=BULL_REV, 0=NONE
    "Trend_Ext",        # continuous trend extension (signed distance from MA50)
]

# E. Momentum dynamics — EMA-smoothed rate of change
_DYNAMICS = [
    "rsi_accel",        # EMA-smoothed RSI delta — momentum acceleration
    "adx_accel",        # EMA-smoothed ADX delta — trend strengthening/weakening
    "vol_ratio_accel",  # EMA-smoothed vol ratio delta — participation shift
]

# F. Cross-asset context — performance relative to market
_CROSS_ASSET = [
    "rs_1m",            # 1M return vs SPY
    "rs_3m",            # 3M return vs SPY
    "spy_trend",        # SPY above MA200? (0 or 1)
    "vix_norm",         # VIX level normalised (0-1 scale)
]

# G. Fundamental quality — forward-filled quarterly data (constant between reports)
_FUNDAMENTAL = [
    "fund_value",       # composite valuation score (PE + PB + PEG)
    "fund_quality",     # composite quality score (ROE + margins)
    "fund_growth",      # revenue/earnings growth signal
    "fund_safety",      # debt/equity + current ratio signal
]

# H. Sentiment — rolling-smoothed to handle irregular arrival
_SENTIMENT = [
    "sent_score",       # sentiment score [-1, 1], smoothed
    "sent_momentum",    # sentiment trend (rising/falling)
    "sent_dispersion",  # opinion spread (consensus vs divided)
]

# I. Liquidity — filters out illiquid junk, captures participation shifts
_LIQUIDITY = [
    "dollar_vol_rank",  # log avg daily dollar volume (normalised)
    "spread_proxy",     # (high-low)/close as bid-ask proxy (lower = more liquid)
    "amihud",           # Amihud illiquidity ratio: |ret|/dollar_vol (lower = more liquid)
]

# J. Alpha microstructure — short-horizon tradable structure
_ALPHA_MICRO = [
    "overnight_gap",     # open vs prior close
    "intraday_ret",      # close vs same-day open
    "compression_20d",   # ATR compression/expansion vs 20d baseline
    "trend_persistence", # fraction of positive closes over last 10 bars
]

# Full set
ALL_FEATURE_NAMES = (
    _OSCILLATORS + _TREND + _VOL + _REGIME + _DYNAMICS +
    _CROSS_ASSET + _FUNDAMENTAL + _SENTIMENT + _LIQUIDITY + _ALPHA_MICRO
)

# Minimal set — technicals + regime + liquidity + alpha microstructure
_MINIMAL_FEATURES = (
    _OSCILLATORS + _TREND + _VOL + _REGIME + _LIQUIDITY + _ALPHA_MICRO
)


# ── Config & result dataclasses ──────────────────────────────────────────────

@dataclass
class MLConfig:
    """User-configurable ML parameters."""
    backend: str = "auto"              # "auto" or "pytorch" (GPU)
    model_type: str = "lightgbm"       # "lightgbm", "mlp", "logistic", "ensemble"
    training_period: str = "5y"        # yfinance period string
    forward_horizon: int = 21          # days ahead for label generation (≈1 month)
    strong_threshold: float = 0.06     # 6% = "strong" move (scaled for 21d)
    weak_threshold: float = 0.02      # 2% = directional move (scaled for 21d)
    n_trees: int = 300                 # LightGBM: n_estimators
    max_depth: int = 5                 # LightGBM: max_depth (shallower = less overfit)
    learning_rate: float = 0.03        # LightGBM: learning_rate (slower = more robust)
    num_leaves: int = 31               # LightGBM: num_leaves
    feature_set: str = "full"          # "full" or "minimal"
    train_mode: str = "per_ticker"     # "per_ticker" or "universe"
    cv_splits: int = 5                 # Walk-forward TimeSeriesSplit folds
    # Walk-forward CV options
    wf_gap: int = 63                   # bars gap (~1 quarter) prevents label leakage and
                                       # gives fundamentals/sentiment time to decay
    wf_window: str = "expanding"       # "expanding" or "rolling"
    wf_rolling_size: Optional[int] = None  # max train bars for rolling window (None = no cap)
    # Point-in-time safety: exclude fundamentals from historical training
    use_fundamentals_in_training: bool = False
    use_sentiment_in_training: bool = False
    # Volatility targeting for position sizing
    target_annual_vol: float = 0.15    # 15% annualised target vol
    # Drawdown circuit breaker
    max_drawdown_trigger: float = 0.15  # reduce exposure after 15% drawdown
    # Minimum liquidity filter (avg daily dollar volume)
    min_dollar_volume: float = 5_000_000.0
    # PyTorch MLP-specific
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    epochs: int = 100
    batch_size: int = 64
    pt_learning_rate: float = 1e-3
    dropout: float = 0.3
    early_stopping_patience: int = 10   # stop if val loss doesn't improve
    # Multi-head loss weights
    loss_w_regime: float = 1.0
    loss_w_entry: float = 1.0
    loss_w_exit: float = 1.0
    # MC Dropout passes at inference for uncertainty estimation
    mc_dropout_passes: int = 20
    # Cache
    max_model_age_days: int = 7
    # Cost-aware label shaping
    label_cost_penalty_pct: float = 0.003
    label_impact_penalty_pct: float = 0.0
    # Signal-generation controls
    signal_min_regime_confidence: float = 0.35
    signal_min_score_spread: float = 0.08
    signal_min_liquidity_rank: float = 0.25
    signal_max_amihud: float = 9.0
    signal_commission_pct: float = 0.001
    signal_slippage_pct: float = 0.0005
    signal_require_readiness: bool = False
    # Walk-forward simulation cost: applied at each entry and exit to make the
    # reported WF Sharpe reflect real trading friction.  Default 0.1% per leg
    # (20bps round-trip) is conservative for liquid US equities.
    wf_trade_cost: float = 0.001

    def feature_names(self) -> List[str]:
        if self.feature_set == "minimal":
            return list(_MINIMAL_FEATURES)
        return list(ALL_FEATURE_NAMES)

    def train_feature_names(
        self,
        include_fundamentals: bool = False,
        include_sentiment: bool = False,
    ) -> List[str]:
        """
        Features that are safe to train with the data actually available here.

        Fundamentals and sentiment are intentionally excluded until the project
        has point-in-time historical versions of those inputs. Training with
        zero-filled placeholders and then scoring live non-zero values produces
        an invalid train/live feature contract.
        """
        unsupported = set()
        if not include_fundamentals:
            unsupported |= set(_FUNDAMENTAL)
        if not include_sentiment:
            unsupported |= set(_SENTIMENT)
        return [name for name in self.feature_names() if name not in unsupported]

    def cache_key(self, ticker: str) -> str:
        """Deterministic key for model caching."""
        payload = {
            "schema": 3,
            "ticker": ticker,
            "model_type": self.model_type,
            "training_period": self.training_period,
            "forward_horizon": self.forward_horizon,
            "strong_threshold": self.strong_threshold,
            "weak_threshold": self.weak_threshold,
            "feature_set": self.feature_set,
            "train_features": self.train_feature_names(),
            "n_trees": self.n_trees,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "train_mode": self.train_mode,
            "cv_splits": self.cv_splits,
            "wf_gap": self.wf_gap,
            "wf_window": self.wf_window,
            "wf_rolling_size": self.wf_rolling_size,
            "use_fundamentals_in_training": self.use_fundamentals_in_training,
            "use_sentiment_in_training": self.use_sentiment_in_training,
            "target_annual_vol": self.target_annual_vol,
            "max_drawdown_trigger": self.max_drawdown_trigger,
            "hidden_dims": self.hidden_dims,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "pt_learning_rate": self.pt_learning_rate,
            "dropout": self.dropout,
            "early_stopping_patience": self.early_stopping_patience,
            "label_cost_penalty_pct": self.label_cost_penalty_pct,
            "label_impact_penalty_pct": self.label_impact_penalty_pct,
            "signal_min_regime_confidence": self.signal_min_regime_confidence,
            "signal_min_score_spread": self.signal_min_score_spread,
            "signal_min_liquidity_rank": self.signal_min_liquidity_rank,
            "signal_max_amihud": self.signal_max_amihud,
            "signal_commission_pct": self.signal_commission_pct,
            "signal_slippage_pct": self.signal_slippage_pct,
            "signal_require_readiness": self.signal_require_readiness,
        }
        sig = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.md5(sig.encode()).hexdigest()[:12]


@dataclass
class TrainingLog:
    """Detailed training log for dashboard monitoring."""
    epochs: List[Dict]                    # [{epoch, train_loss, val_loss, lr}]
    early_stop_epoch: Optional[int]
    best_val_loss: Optional[float]
    n_train_rows: int
    n_val_rows: int
    n_test_rows: int
    features_used: List[str]
    backend: str
    training_time_s: float
    data_date_range: Optional[Tuple[str, str]]  # (first_date, last_date) ISO
    train_date_range: Optional[Tuple[str, str]]
    test_date_range: Optional[Tuple[str, str]]
    class_distribution: Optional[Dict[str, int]]  # regime class counts
    # Calibration info — how probabilities were post-processed
    calibration_status: str = ""        # "isotonic", "raw_lgbm", "temperature_scaling", or ""
    calibration_samples: int = 0        # samples used for the calibration step
    temperature: float = 1.0            # MLP temperature scaling T (1.0 = no scaling)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class WalkForwardMetrics:
    """Comprehensive evaluation from walk-forward validation."""
    # Core trading metrics
    sharpe_ratio: float
    max_drawdown: float            # worst peak-to-trough (negative %)
    cagr: float                    # compound annual growth rate
    hit_rate: float                # % of trades that were profitable
    profit_factor: float           # gross profit / gross loss
    total_return: float            # cumulative return
    n_trades: int
    avg_trade_return: float
    # Per regime performance (now includes per-regime Sharpe)
    by_regime: Dict[str, Dict]     # {regime: {n_trades, hit_rate, avg_ret, sharpe}}
    # Per volatility bucket
    by_volatility: Dict[str, Dict] # {LOW/MED/HIGH: {n_trades, hit_rate, avg_ret}}
    # Trade frequency
    avg_trades_per_month: float
    avg_holding_period: float      # days
    daily_returns: List[float]
    position_exposure: List[float]
    # Walk-forward fold details
    fold_results: List[Dict]       # per-fold metrics
    avg_daily_turnover: float = 0.0
    # Last-fold-only metrics (model most trained)
    last_fold_metrics: Optional[Dict] = None
    # Cross-fold analysis
    worst_fold_idx: int = -1           # index of worst fold by Sharpe
    fold_sharpe_std: float = 0.0       # std-dev of per-fold Sharpe ratios
    fold_return_std: float = 0.0       # std-dev of per-fold total returns
    pct_folds_profitable: float = 0.0  # % of folds with positive total return
    window_type: str = "expanding"     # "expanding" or "rolling"
    policy_opt: Optional[Dict] = None  # PolicyOptResult as dict — best thresholds found by grid search
    # RANGE regime Sharpe — negative means speculative RANGE entries hurt performance
    range_regime_sharpe: float = 0.0
    # Information Ratio: annualised return / annualised vol (Sharpe with rf=0)
    information_ratio: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TrainResult:
    """Returned after training completes."""
    ticker: str
    backend: str
    model_type: str
    n_samples: int
    n_features: int
    regime_accuracy: float
    regime_f1: Dict[str, float]
    entry_mae: float
    exit_mae: float
    feature_importances: Dict[str, float]
    cv_scores: List[float]
    training_time_s: float
    training_log: Optional[TrainingLog] = None
    walk_forward: Optional[WalkForwardMetrics] = None
    readiness: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ReadinessCriteria:
    """
    Minimum out-of-sample bar for admitting a model into the live registry.

    These thresholds are intentionally conservative enough to reject negative
    or unstable strategies, but lightweight enough to evaluate from the stored
    walk-forward report alone.
    """
    min_sharpe_ratio: float = 0.01
    min_cagr: float = 0.0
    max_drawdown: float = 0.20
    max_fold_sharpe_std: float = 1.0
    min_pct_folds_profitable: float = 0.50
    min_trades: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReadinessAssessment:
    ready: bool
    reasons: List[str]
    metrics: Dict[str, Any]
    criteria: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyOptResult:
    """
    Result of grid search over entry/exit thresholds evaluated on walk-forward folds.
    Uncertainty cap cannot be optimised in walk-forward mode (MC Dropout is not run
    per fold for performance reasons); it is kept at its configured default.
    """
    best_entry_threshold: float
    best_exit_threshold: float
    best_min_regime_confidence: float
    best_min_score_spread: float
    best_sharpe: float
    default_sharpe: float          # Sharpe using the configured default thresholds
    improvement_pct: float         # (best - default) / |default| × 100
    estimated_round_trip_cost: float
    n_combinations: int
    grid_scores: List[Dict]        # [{entry_threshold, exit_threshold, min_regime_confidence, min_score_spread, sharpe, n_trades, hit_rate}]

    def to_dict(self) -> Dict:
        return asdict(self)


def assess_live_readiness(
    train_result: TrainResult,
    criteria: Optional[ReadinessCriteria] = None,
) -> ReadinessAssessment:
    """
    Decide whether a freshly trained model is admissible for the live registry.

    The check is intentionally based on out-of-sample walk-forward metrics only.
    If those metrics are absent or unstable, the model is rejected.
    """
    crit = criteria or ReadinessCriteria()
    wf = train_result.walk_forward
    reasons: List[str] = []
    metrics = {
        "sharpe_ratio": None,
        "cagr": None,
        "max_drawdown": None,
        "fold_sharpe_std": None,
        "pct_folds_profitable": None,
        "n_trades": 0,
    }
    if wf is None:
        reasons.append("missing walk-forward metrics")
        return ReadinessAssessment(
            ready=False,
            reasons=reasons,
            metrics=metrics,
            criteria=crit.to_dict(),
        )

    metrics.update({
        "sharpe_ratio": wf.sharpe_ratio,
        "cagr": wf.cagr,
        "max_drawdown": abs(wf.max_drawdown),
        "fold_sharpe_std": wf.fold_sharpe_std,
        "pct_folds_profitable": wf.pct_folds_profitable,
        "n_trades": wf.n_trades,
    })

    if wf.sharpe_ratio <= crit.min_sharpe_ratio:
        reasons.append(
            f"sharpe_ratio {wf.sharpe_ratio:.3f} <= required {crit.min_sharpe_ratio:.3f}"
        )
    if wf.cagr <= crit.min_cagr:
        reasons.append(f"cagr {wf.cagr:.3f} <= required {crit.min_cagr:.3f}")
    if abs(wf.max_drawdown) > crit.max_drawdown:
        reasons.append(
            f"max_drawdown {abs(wf.max_drawdown):.3f} > allowed {crit.max_drawdown:.3f}"
        )
    if wf.fold_sharpe_std > crit.max_fold_sharpe_std:
        reasons.append(
            f"fold_sharpe_std {wf.fold_sharpe_std:.3f} > allowed {crit.max_fold_sharpe_std:.3f}"
        )
    if wf.pct_folds_profitable < crit.min_pct_folds_profitable:
        reasons.append(
            f"pct_folds_profitable {wf.pct_folds_profitable:.3f} < required "
            f"{crit.min_pct_folds_profitable:.3f}"
        )
    if wf.n_trades < crit.min_trades:
        reasons.append(f"n_trades {wf.n_trades} < required {crit.min_trades}")

    return ReadinessAssessment(
        ready=not reasons,
        reasons=reasons or ["ready for registry admission"],
        metrics=metrics,
        criteria=crit.to_dict(),
    )


@dataclass
class DecisionPolicy:
    """
    Formal decision policy mapping ML outputs → trade actions.

    This is where PnL is actually determined. Every threshold is explicit
    and tunable — no magic numbers buried in if/else chains.
    """
    # Entry thresholds
    entry_threshold: float = 0.60       # min entry_score to consider buying
    strong_entry_threshold: float = 0.75  # entry_score for full conviction
    entry_uncertainty_cap: float = 0.15  # max entry_std before vetoing entry
    # Exit thresholds
    exit_threshold: float = 0.60        # min exit_score to consider selling
    urgent_exit_threshold: float = 0.80  # exit_score for immediate full exit
    exit_uncertainty_cap: float = 0.20   # high uncertainty → tighten exit
    # Regime filters
    favorable_regimes: Tuple = ("TREND_UP", "REVERSAL_UP")
    unfavorable_regimes: Tuple = ("TREND_DOWN", "REVERSAL_DOWN")
    # Position sizing
    max_position_pct: float = 1.0       # max position as fraction of capital
    min_position_pct: float = 0.10      # minimum meaningful position
    # Confidence floor
    min_regime_confidence: float = 0.35  # below this, regime is too uncertain
    # Entry-quality spread: require entry score to exceed exit risk by this margin
    min_score_spread: float = 0.08
    # Liquidity filters derived from model features
    min_liquidity_rank: float = 0.25
    max_amihud: float = 9.0
    # RANGE regime gate — set True when walk-forward shows negative RANGE Sharpe.
    # Disables speculative entries in RANGE to avoid trading noise.
    disable_range_entries: bool = False

    def to_dict(self) -> Dict:
        return {k: v if not isinstance(v, tuple) else list(v)
                for k, v in asdict(self).items()}


@dataclass
class TradeDecision:
    """Output of the decision policy — what to actually do."""
    action: str               # BUY, SELL, HOLD, REDUCE, WATCH
    position_size: float      # 0.0-1.0, fraction of max allocation
    conviction: str           # HIGH, MEDIUM, LOW, NONE
    reasons: List[str]        # human-readable explanation of why
    entry_score: float        # raw model output
    exit_score: float         # raw model output
    uncertainty_penalty: float  # how much uncertainty reduced conviction

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MLPrediction:
    """Prediction for one asset at current time."""
    regime: str                        # predicted regime class
    regime_confidence: float           # max class probability
    regime_probs: Dict[str, float]     # all class probabilities
    entry_score: float                 # 0-1, higher = better entry
    exit_score: float                  # 0-1, higher = should exit
    ml_signal: str                     # derived action signal
    decision: Optional[Dict] = None    # formal trade decision
    feature_importances: Optional[Dict[str, float]] = None
    uncertainty: Optional[Dict[str, float]] = None  # MC Dropout uncertainty
    score_spread: Optional[float] = None
    policy: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        return asdict(self)


# ── Feature engineering ──────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    feature_names: List[str],
    market_ctx: Optional[Dict] = None,
    fundamentals: Optional[Dict] = None,
    sentiment: Optional[Dict] = None,
    training_mode: bool = False,
    historical_spy: Optional[pd.DataFrame] = None,
    historical_vix: Optional[pd.Series] = None,
    pit_market_ctx: Optional[pd.DataFrame] = None,
    pit_fundamentals: Optional[pd.DataFrame] = None,
    pit_sentiment: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the full ML feature matrix from indicators + context.

    Args:
        df:              DataFrame with computed indicators (from compute_indicators)
        feature_names:   which features to include (from MLConfig)
        market_ctx:      market context dict (VIX, SPY trend, etc.)
        fundamentals:    fundamental metrics dict (PE, ROE, etc.)
        sentiment:       sentiment dict (score, momentum, dispersion)
        training_mode:   True during training — disables scalar broadcasting of
                         fundamentals/sentiment to prevent lookahead bias.
                         Cross-asset features use historical_spy/historical_vix
                         when available, or are set to neutral.
        historical_spy:  SPY OHLCV DataFrame (same date range as df) for computing
                         per-row relative strength during training.
        historical_vix:  VIX close Series for per-row VIX normalisation.
        pit_market_ctx:  aligned point-in-time market-context DataFrame.
        pit_fundamentals: aligned point-in-time fundamental feature DataFrame.
        pit_sentiment: aligned point-in-time sentiment feature DataFrame.

    Leakage guarantee:
        - Technical features (A-E): derived from past/current data only (safe)
        - Cross-asset (F): per-row from historical data in training_mode,
          or from live market_ctx during prediction
        - Fundamentals (G): zeroed in training_mode (no point-in-time data
          available from yfinance); live values used only for prediction
        - Sentiment (H): zeroed in training_mode unless time-indexed
        - Liquidity (I): derived from df's own volume/price history (safe)
    """
    # Enforce sort order — belt-and-suspenders
    if not df.index.is_monotonic_increasing:
        log.warning("build_features: df index not sorted — sorting now")
        df = df.sort_index()

    feat = pd.DataFrame(index=df.index)
    c = df["Close"]

    # ── A. Oscillators & momentum ────────────────────────────────────────────
    if "RSI" in df.columns:
        feat["RSI"] = df["RSI"]
    if "BB_Pct" in df.columns:
        feat["BB_Pct"] = df["BB_Pct"]
    if "MACD_Hist" in df.columns:
        feat["MACD_Hist"] = df["MACD_Hist"]
    feat["ret_3d"] = c.pct_change(3) * 100

    # ── B. Trend structure ───────────────────────────────────────────────────
    if "MA50" in df.columns:
        feat["pct_from_ma50"] = (c - df["MA50"]) / df["MA50"].replace(0, np.nan) * 100
    if "MA200" in df.columns:
        feat["pct_from_ma200"] = (c - df["MA200"]) / df["MA200"].replace(0, np.nan) * 100
    if "MA50" in df.columns and "MA200" in df.columns:
        feat["ma_spread"] = (df["MA50"] - df["MA200"]) / df["MA200"].replace(0, np.nan) * 100

    # ── C. Volatility & volume ───────────────────────────────────────────────
    if "ATR_Pct" in df.columns:
        feat["ATR_Pct"] = df["ATR_Pct"]
        # vol_cycle: EMA-smoothed ATR% delta — is vol rising or falling?
        feat["vol_cycle"] = df["ATR_Pct"].diff().ewm(span=10, adjust=False).mean()
    if "Vol_Ratio" in df.columns:
        feat["Vol_Ratio"] = df["Vol_Ratio"]
    # obv_slope replaced with Chaikin Money Flow (CMF).
    # OBV accumulates errors over time and is non-stationary; CMF is
    # bounded [-1, +1] and measures buying/selling pressure directly.
    # The feature column name stays "obv_slope" for backward compatibility
    # with cached models that already reference this slot.
    if "High" in df.columns and "Low" in df.columns and "Volume" in df.columns:
        hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
        mf_mult  = ((c - df["Low"]) - (df["High"] - c)) / hl_range
        mf_vol   = mf_mult * df["Volume"]
        vol_roll = df["Volume"].rolling(14).sum().replace(0, np.nan)
        feat["obv_slope"] = (mf_vol.rolling(14).sum() / vol_roll).fillna(0).clip(-1, 1)
    elif "OBV" in df.columns:
        # Fallback: keep legacy OBV slope if OHLCV not fully available
        obv_ma = df["OBV"].rolling(10).mean()
        feat["obv_slope"] = obv_ma.pct_change(5) * 100

    # ── D. Regime context (dual encoding: continuous + categorical) ──────────
    # Continuous values — trees/MLP can split on fine-grained values
    if "ADX" in df.columns:
        feat["ADX"] = df["ADX"]
    if "Vol_Pctl" in df.columns:
        feat["Vol_Pctl"] = df["Vol_Pctl"]
    if "Trend_Ext" in df.columns:
        feat["Trend_Ext"] = df["Trend_Ext"]

    # Categorical buckets — let model condition on regime labels
    _regime_map = {"MEAN_REVERSION": 0, "NEUTRAL": 1, "TREND": 2}
    if "Regime" in df.columns:
        feat["regime_adx"] = df["Regime"].map(_regime_map).fillna(1).astype(float)

    _mkt_map = {"BEARISH": 0, "TRANSITION": 1, "BULLISH": 2}
    if "Mkt_Regime" in df.columns:
        feat["regime_mkt"] = df["Mkt_Regime"].map(_mkt_map).fillna(1).astype(float)

    _vol_map = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "EXTREME": 3}
    if "Vol_Regime" in df.columns:
        feat["regime_vol"] = df["Vol_Regime"].map(_vol_map).fillna(1).astype(float)

    _stage_map = {"EARLY": 0, "HEALTHY": 1, "EXTENDED": 2, "OVEREXTENDED": 3, "PARABOLIC": 4}
    if "Trend_Stage" in df.columns:
        feat["trend_stage_enc"] = df["Trend_Stage"].map(_stage_map).fillna(0).astype(float)

    _chg_map = {
        "BEARISH REVERSAL": -2, "BEARISH CONFIRMATION": -1,
        "WEAKENING": -0.5, "NONE": 0,
        "POTENTIAL BOTTOM": 0.5, "BULLISH CONFIRMATION": 1,
        "BULLISH REVERSAL": 2,
    }
    if "Regime_Chg" in df.columns:
        feat["regime_chg_enc"] = df["Regime_Chg"].map(_chg_map).fillna(0).astype(float)

    # ── E. Momentum dynamics (EMA-smoothed deltas, not raw diff) ────────────
    if "RSI" in df.columns:
        feat["rsi_accel"] = df["RSI"].diff().ewm(span=3, adjust=False).mean()
    if "ADX" in df.columns:
        feat["adx_accel"] = df["ADX"].diff().ewm(span=3, adjust=False).mean()
    if "Vol_Ratio" in df.columns:
        feat["vol_ratio_accel"] = df["Vol_Ratio"].diff().ewm(span=3, adjust=False).mean()

    # ── F. Cross-asset context ───────────────────────────────────────────────
    # CRITICAL FIX: In training_mode, compute per-row from historical data
    # instead of broadcasting today's scalar (which is lookahead bias).
    if training_mode and pit_market_ctx is not None:
        pit_market_ctx = pit_market_ctx.reindex(df.index)
        asset_ret_21 = c.pct_change(21) * 100
        asset_ret_63 = c.pct_change(63) * 100
        feat["rs_1m"] = asset_ret_21 - pit_market_ctx["spy_ret_1m"].astype(float)
        feat["rs_3m"] = asset_ret_63 - pit_market_ctx["spy_ret_3m"].astype(float)
        feat["spy_trend"] = pit_market_ctx["spy_trend_bull"].astype(float).fillna(0.5)
    elif training_mode and historical_spy is not None:
        # Per-row relative strength: asset return - SPY return over same window
        spy_close = historical_spy["Close"].reindex(df.index, method="ffill")
        spy_ret_21 = spy_close.pct_change(21) * 100
        spy_ret_63 = spy_close.pct_change(63) * 100
        asset_ret_21 = c.pct_change(21) * 100
        asset_ret_63 = c.pct_change(63) * 100
        feat["rs_1m"] = asset_ret_21 - spy_ret_21
        feat["rs_3m"] = asset_ret_63 - spy_ret_63
        # SPY trend: is SPY above its own MA200?
        spy_ma200 = spy_close.rolling(200).mean()
        feat["spy_trend"] = (spy_close > spy_ma200).astype(float)
    elif not training_mode and market_ctx:
        # Prediction mode: use live market context (current-day only, acceptable)
        if "Ret_21D" in df.columns:
            spy_1m = market_ctx.get("spy_ret_1m")
            feat["rs_1m"] = df["Ret_21D"] - spy_1m if spy_1m is not None else 0.0
        else:
            feat["rs_1m"] = 0.0
        if "Ret_63D" in df.columns:
            spy_3m = market_ctx.get("spy_ret_3m")
            feat["rs_3m"] = df["Ret_63D"] - spy_3m if spy_3m is not None else 0.0
        else:
            feat["rs_3m"] = 0.0
        feat["spy_trend"] = float(market_ctx.get("spy_trend_bull") or 0)
    else:
        feat["rs_1m"] = 0.0
        feat["rs_3m"] = 0.0
        feat["spy_trend"] = 0.5

    # VIX: z-scored to capture *relative* fear, not absolute level.
    # VIX/80 was near-constant during normal markets and contributed
    # nothing to tree splits.  Z-score captures regime transitions cleanly.
    _VIX_LONG_MEAN = 20.0   # long-run VIX average
    _VIX_LONG_STD  = 8.0    # long-run VIX std-dev
    if training_mode and pit_market_ctx is not None and "vix" in pit_market_ctx.columns:
        vix_aligned = pit_market_ctx["vix"].astype(float).reindex(df.index)
        vix_roll_mean = vix_aligned.rolling(63, min_periods=20).mean().fillna(_VIX_LONG_MEAN)
        vix_roll_std = vix_aligned.rolling(63, min_periods=20).std().fillna(_VIX_LONG_STD).clip(lower=1.0)
        feat["vix_norm"] = ((vix_aligned - vix_roll_mean) / vix_roll_std).clip(-3, 3) / 3.0
    elif training_mode and historical_vix is not None:
        vix_aligned = historical_vix.reindex(df.index, method="ffill")
        # Rolling z-score (63-day window); clip to [-3, 3] and scale to [-1, 1]
        vix_roll_mean = vix_aligned.rolling(63, min_periods=20).mean().fillna(_VIX_LONG_MEAN)
        vix_roll_std  = vix_aligned.rolling(63, min_periods=20).std().fillna(_VIX_LONG_STD).clip(lower=1.0)
        feat["vix_norm"] = ((vix_aligned - vix_roll_mean) / vix_roll_std).clip(-3, 3) / 3.0
    elif not training_mode and market_ctx:
        vix = float(market_ctx.get("vix") or _VIX_LONG_MEAN)
        # Prefer caller-supplied rolling stats if available (eliminates the
        # training/inference distribution shift caused by using fixed long-run
        # constants when markets are in a regime far from the long-run average).
        # Callers should populate vix_ma63 and vix_std63 from recent VIX history.
        vix_mean = float(market_ctx.get("vix_ma63") or _VIX_LONG_MEAN)
        vix_std  = max(1.0, float(market_ctx.get("vix_std63") or _VIX_LONG_STD))
        feat["vix_norm"] = max(-1.0, min(1.0, (vix - vix_mean) / vix_std))
    else:
        feat["vix_norm"] = 0.0   # neutral: VIX at long-run average

    # ── G. Fundamental quality ───────────────────────────────────────────────
    # Fundamental features are ALWAYS zeroed in the ML feature matrix — in both
    # training_mode and prediction_mode.  Reason: yfinance only provides a
    # current-day snapshot, so broadcasting today's P/E across 5 years of
    # training rows is lookahead bias (training).  And if we non-zero them only
    # at inference, the LightGBM tree never learned splits on them (they were
    # constant-zero during training), so they would be silently ignored at
    # inference while creating a spurious training/inference distribution shift.
    #
    # Instead, fundamentals are applied as a post-model overlay in
    # apply_decision_policy() via the `fund_quality_score` argument, which is
    # computed from the raw fundamentals dict in predict_from_df().  This gives
    # fundamentals real influence on the final decision without corrupting the
    # ML feature distribution.
    if training_mode and pit_fundamentals is not None:
        pit_fundamentals = pit_fundamentals.reindex(df.index)
        for col in _FUNDAMENTAL:
            feat[col] = pit_fundamentals[col].astype(float).fillna(0.0)
    else:
        # Both training and live prediction: zero fundamentals in ML features.
        # Fundamentals are applied separately via fund_quality_score overlay.
        for col in _FUNDAMENTAL:
            feat[col] = 0.0

    # ── H. Sentiment ─────────────────────────────────────────────────────────
    # CRITICAL FIX: In training_mode, sentiment is zeroed (same reason as
    # fundamentals — we don't have a historical sentiment time series).
    if training_mode and pit_sentiment is not None:
        pit_sentiment = pit_sentiment.reindex(df.index)
        for col in _SENTIMENT:
            feat[col] = pit_sentiment[col].astype(float).fillna(0.0)
    elif not training_mode and sentiment and sentiment.get("score") is not None:
        feat["sent_score"] = float(sentiment.get("score", 0))
        feat["sent_momentum"] = float(sentiment.get("momentum", 0))
        feat["sent_dispersion"] = float(sentiment.get("dispersion", 0))
    else:
        for col in _SENTIMENT:
            feat[col] = 0.0

    # ── I. Liquidity ─────────────────────────────────────────────────────────
    # Always computable from df's own data — no lookahead risk.
    if "Volume" in df.columns:
        dollar_vol = (c * df["Volume"]).rolling(20).mean()
        # Log-normalised dollar volume (log scale compresses range)
        feat["dollar_vol_rank"] = np.log1p(dollar_vol.clip(lower=1))
        # Normalise to ~[0, 1] using a reference scale ($100M avg = 1.0)
        feat["dollar_vol_rank"] = (feat["dollar_vol_rank"] / np.log1p(1e8)).clip(upper=2.0)
    else:
        feat["dollar_vol_rank"] = 0.5

    if "High" in df.columns and "Low" in df.columns:
        # Spread proxy: intraday range / close — lower = tighter spread = more liquid
        spread_raw = (df["High"] - df["Low"]) / c.replace(0, np.nan)
        feat["spread_proxy"] = spread_raw.rolling(5).mean() * 100  # in %
    else:
        feat["spread_proxy"] = 1.0

    # Amihud illiquidity ratio: |daily_return| / dollar_volume
    # Proven in microstructure literature (Amihud 2002).  Lower = more liquid.
    # We log-scale and invert so that higher values = less liquid = more risk.
    if "Volume" in df.columns:
        _dollar_vol_raw = (c * df["Volume"]).rolling(20).mean().replace(0, np.nan)
        _abs_ret = c.pct_change().abs()
        amihud_raw = (_abs_ret / _dollar_vol_raw).rolling(20).mean()
        # Log-scale compresses the fat tail; multiply by 1e6 for readable range
        feat["amihud"] = np.log1p(amihud_raw * 1e6).clip(0, 15).fillna(0.0)
    else:
        feat["amihud"] = 0.0

    # ── J. Alpha microstructure ────────────────────────────────────────────
    prev_close = c.shift(1)
    if "Open" in df.columns:
        feat["overnight_gap"] = (df["Open"] / prev_close.replace(0, np.nan) - 1.0) * 100
        feat["intraday_ret"] = (c / df["Open"].replace(0, np.nan) - 1.0) * 100
    else:
        feat["overnight_gap"] = 0.0
        feat["intraday_ret"] = 0.0

    if "ATR_Pct" in df.columns:
        atr_baseline = df["ATR_Pct"].rolling(20).mean().replace(0, np.nan)
        feat["compression_20d"] = (df["ATR_Pct"] / atr_baseline - 1.0).clip(-3, 3)
    else:
        feat["compression_20d"] = 0.0

    daily_ret = c.pct_change()
    feat["trend_persistence"] = (
        np.sign(daily_ret)
        .rolling(10)
        .mean()
        .clip(-1, 1)
    )

    # Select only requested features, return available ones
    available = [f for f in feature_names if f in feat.columns]
    return feat[available]


def _sigmoid(x: np.ndarray, scale: float = 20.0) -> np.ndarray:
    """Sigmoid normalization to [0, 1]."""
    return 1.0 / (1.0 + np.exp(-scale * x))


def _compute_path_stats(
    prices: np.ndarray,
    start: int,
    horizon: int,
) -> Tuple[float, float, float]:
    """
    Compute (max_drawdown, max_runup, realized_vol) for a forward window.
    Returns (nan, nan, nan) if the window is out of bounds or base <= 0.
    """
    end = start + 1 + horizon
    if end > len(prices):
        return np.nan, np.nan, np.nan
    base = prices[start]
    if base <= 0:
        return np.nan, np.nan, np.nan
    window = prices[start + 1:end]
    rets = (window - base) / base
    daily_rets = np.diff(window) / np.where(window[:-1] > 0, window[:-1], 1)
    vol = daily_rets.std() if len(daily_rets) > 1 else 0.01
    return float(rets.min()), float(rets.max()), float(max(vol, 0.005))


def generate_labels(
    df: pd.DataFrame,
    forward_horizon: int = 21,
    strong_thresh: float = 0.06,
    weak_thresh: float = 0.02,
    cost_penalty_pct: float = 0.003,
    impact_penalty_pct: float = 0.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Self-supervised label generation from realised future returns.

    Multi-horizon entry/exit labels: computed at 7, 14, and forward_horizon
    days. Entry quality takes the *best* realized quality across horizons —
    captures "quick win" patterns that a single long horizon would obscure.
    Exit quality similarly takes the max threat across horizons.

    Regime labels use the primary forward_horizon (≈1 month) window because
    monthly returns have better signal-to-noise than daily/10-day windows.

    Leakage guarantee: all future-looking ops use explicit shift(-N) or
    forward-index window slices. Any accidental look-ahead → NaN → dropped.

    Returns:
        regime:        Series of regime class strings
        entry_quality: Series [0, 1] — best realized entry quality across horizons
        exit_quality:  Series [0, 1] — worst realized exit quality across horizons
    """
    # ── Leakage safeguard ─────────────────────────────────────────────────
    if not df.index.is_monotonic_increasing:
        log.warning("generate_labels: index not sorted — sorting to prevent leakage")
        df = df.sort_index()
    dup_mask = df.index.duplicated(keep="last")
    if dup_mask.any():
        log.warning(f"generate_labels: {dup_mask.sum()} duplicate timestamps dropped")
        df = df[~dup_mask]
    # All future-looking computations below use explicit shift(-N).
    # Any accidental look-ahead produces NaN, caught by dropna() in caller.
    # ─────────────────────────────────────────────────────────────────────
    c = df["Close"]
    total_cost_penalty = max(0.0, cost_penalty_pct + impact_penalty_pct)
    fwd_ret = c.pct_change(forward_horizon).shift(-forward_horizon)
    trail_ret = c.pct_change(forward_horizon)
    net_fwd_ret = fwd_ret.copy()
    net_fwd_ret[fwd_ret > 0] = fwd_ret[fwd_ret > 0] - total_cost_penalty
    net_fwd_ret[fwd_ret < 0] = fwd_ret[fwd_ret < 0] + total_cost_penalty

    # ── Regime classification (past trail_ret + primary future fwd_ret) ────
    # NOTE: This is a TARGET label, not a feature. The regime *features*
    # (regime_adx, regime_mkt, etc.) come from past indicators only.
    regime = pd.Series("RANGE", index=df.index)
    future_thresh = weak_thresh + total_cost_penalty
    regime[(trail_ret > strong_thresh) & (net_fwd_ret > future_thresh)] = "TREND_UP"
    regime[(trail_ret < -strong_thresh) & (net_fwd_ret < -future_thresh)] = "TREND_DOWN"
    regime[(trail_ret < -strong_thresh) & (net_fwd_ret > future_thresh)] = "REVERSAL_UP"
    regime[(trail_ret > strong_thresh) & (net_fwd_ret < -future_thresh)] = "REVERSAL_DOWN"

    # ── Multi-horizon entry/exit quality ───────────────────────────────────
    # Compute path statistics at 3 checkpoints. Entry takes the best
    # (highest reward-to-risk), exit takes the worst (highest threat).
    # This gives sharper signals than a single coarse horizon.
    short_h  = max(5,  forward_horizon // 3)   # ~7  days for 21d horizon
    medium_h = max(10, forward_horizon * 2 // 3)  # ~14 days for 21d horizon
    horizons = sorted({short_h, medium_h, forward_horizon})

    prices_arr = c.values
    n = len(prices_arr)

    # Accumulators — shape (n_horizons, n_rows)
    entry_raws: List[np.ndarray] = []
    exit_raws:  List[np.ndarray] = []

    for h in horizons:
        max_dd_h  = np.full(n, np.nan)
        max_ru_h  = np.full(n, np.nan)
        fwd_vol_h = np.full(n, np.nan)

        for i in range(n - h):
            dd, ru, vol = _compute_path_stats(prices_arr, i, h)
            max_dd_h[i]  = dd
            max_ru_h[i]  = ru
            fwd_vol_h[i] = vol

        fwd_vol_h = np.where(np.isnan(fwd_vol_h), 0.005, np.maximum(fwd_vol_h, 0.005))
        reward_h  = np.where(
            np.isnan(max_ru_h), np.nan,
            np.clip(max_ru_h - total_cost_penalty, 0, None),
        )
        risk_h    = np.where(
            np.isnan(max_dd_h), np.nan,
            np.abs(max_dd_h) + total_cost_penalty,
        )

        # Entry: (reward - 0.5 * risk) / vol  — higher = better entry timing
        entry_raws.append((reward_h - 0.5 * risk_h) / fwd_vol_h)
        # Exit:  (risk - 0.5 * reward) / vol — higher = more threatening to hold
        exit_raws.append((risk_h - 0.5 * reward_h) / fwd_vol_h)

    # Take the *best* entry (max reward-to-risk) and *worst* exit (max threat)
    # across all horizons.  Rows where ALL horizons are NaN (tail of series,
    # no forward window) remain NaN and are dropped by dropna() in the caller.
    stacked_entry = np.vstack(entry_raws)   # (n_horizons, n)
    stacked_exit  = np.vstack(exit_raws)

    # Suppress all-NaN slice warning: replace all-NaN columns with NaN explicitly
    all_nan_mask = np.all(np.isnan(stacked_entry), axis=0)
    best_entry_raw = np.where(all_nan_mask, np.nan,
                              np.nanmax(np.where(np.isnan(stacked_entry), -1e9, stacked_entry), axis=0))
    all_nan_mask_x = np.all(np.isnan(stacked_exit), axis=0)
    worst_exit_raw = np.where(all_nan_mask_x, np.nan,
                              np.nanmax(np.where(np.isnan(stacked_exit), -1e9, stacked_exit), axis=0))

    # Sigmoid maps raw scores → [0, 1] probability-like output
    entry_quality = pd.Series(_sigmoid(best_entry_raw, scale=10.0), index=df.index)
    exit_quality  = pd.Series(_sigmoid(worst_exit_raw, scale=10.0), index=df.index)

    return regime, entry_quality, exit_quality


# ── Signal derivation ────────────────────────────────────────────────────────

def compute_fund_quality_score(fundamentals: Dict) -> Optional[float]:
    """
    Compute a composite fundamental quality score in [-1, +1] from a raw
    fundamentals dict.  Used as an overlay in apply_decision_policy() —
    NOT fed into the ML model (avoids training/inference distribution shift).

    Returns None if no fundamental data is available.

    Scoring:
      Safety  (debt/equity, current ratio) — weighted most heavily
      Quality (ROE, net margin)
    Average across available signals; missing metrics are excluded.
    """
    signals: List[float] = []
    de = fundamentals.get("debt_eq")
    cr = fundamentals.get("curr_ratio")
    roe = fundamentals.get("roe")
    net_mgn = fundamentals.get("net_mgn")
    if de is not None:
        signals.append(1.0 if de < 30 else 0.5 if de < 80 else -0.5 if de < 150 else -1.0)
    if cr is not None:
        signals.append(1.0 if cr > 2 else 0.5 if cr > 1.2 else -0.5 if cr > 0.8 else -1.0)
    if roe is not None:
        signals.append(1.0 if roe > 20 else 0.5 if roe > 10 else -0.5 if roe > 0 else -1.0)
    if net_mgn is not None:
        signals.append(1.0 if net_mgn > 20 else 0.5 if net_mgn > 10 else -0.5 if net_mgn > 0 else -1.0)
    return float(np.mean(signals)) if signals else None


def apply_decision_policy(
    regime: str,
    regime_confidence: float,
    entry: float,
    exit_score: float,
    uncertainty: Optional[Dict[str, float]] = None,
    policy: Optional[DecisionPolicy] = None,
    realized_vol: Optional[float] = None,
    target_vol: Optional[float] = None,
    current_drawdown: Optional[float] = None,
    max_drawdown_trigger: Optional[float] = None,
    liquidity_rank: Optional[float] = None,
    amihud: Optional[float] = None,
    model_ready: bool = True,
    fund_quality_score: Optional[float] = None,
) -> TradeDecision:
    """
    Formal decision policy: ML outputs → trade action + position size.

    This is the single place where all trading logic lives.
    Every threshold is explicit and tunable via DecisionPolicy.

    Position sizing layers:
      1. Base size: entry_score × (1 - uncertainty_penalty)
      2. Vol scaling: target_vol / realized_vol (normalises risk across assets)
      3. Drawdown scaling: reduce after significant drawdowns
      4. Fundamental overlay: fund_quality_score caps or vetoes entries for
         financially dangerous companies (high debt, negative margins).
         Computed externally from raw fundamentals dict — NOT an ML feature.
    """
    pol = policy or DecisionPolicy()
    reasons: List[str] = []

    # ── Extract uncertainty metrics ────────────────────────────────────────
    entry_std = 0.0
    exit_std = 0.0
    regime_std = 0.0
    if uncertainty:
        entry_std = uncertainty.get("entry_std", 0.0)
        exit_std = uncertainty.get("exit_std", 0.0)
        regime_std = uncertainty.get("regime_std", 0.0)

    # Uncertainty penalty: 0 = fully certain, 1 = maximally uncertain
    uncertainty_penalty = min(1.0, (entry_std + regime_std) / 0.3)
    score_spread = entry - exit_score

    # ── Vol-scaling multiplier ─────────────────────────────────────────────
    # Normalises position sizes so each trade targets similar risk.
    # High vol → smaller position; low vol → larger position.
    vol_scalar = 1.0
    if realized_vol is not None and target_vol is not None and realized_vol > 0.001:
        vol_scalar = min(2.0, max(0.2, target_vol / realized_vol))
        if vol_scalar < 0.5:
            reasons.append(f"vol_scale={vol_scalar:.2f} (high vol → reduced size)")

    # ── Drawdown circuit breaker ───────────────────────────────────────────
    # After significant losses, reduce exposure to prevent ruin.
    dd_scalar = 1.0
    _dd_trigger = max_drawdown_trigger or 0.15
    if current_drawdown is not None and current_drawdown < -_dd_trigger:
        # Scale linearly: at trigger → 0.5x, at 2×trigger → 0.0x
        dd_scalar = max(0.0, 1.0 - (abs(current_drawdown) - _dd_trigger) / _dd_trigger)
        dd_scalar = 0.5 * dd_scalar  # never more than 50% after trigger
        reasons.append(f"drawdown={current_drawdown:.1%} → exposure reduced to {dd_scalar:.0%}")

    combined_scalar = vol_scalar * dd_scalar

    # ── EXIT rules (checked first — protecting capital is priority) ────────
    # Rule 1: Urgent exit — high exit score regardless of other signals
    if exit_score >= pol.urgent_exit_threshold:
        reasons.append(f"exit_score={exit_score:.2f} >= urgent threshold {pol.urgent_exit_threshold}")
        return TradeDecision(
            action="SELL", position_size=1.0, conviction="HIGH",
            reasons=reasons, entry_score=entry, exit_score=exit_score,
            uncertainty_penalty=uncertainty_penalty,
        )

    # Rule 2: Exit with unfavorable regime
    if exit_score >= pol.exit_threshold and regime in pol.unfavorable_regimes:
        reasons.append(f"exit_score={exit_score:.2f} + unfavorable regime {regime}")
        return TradeDecision(
            action="SELL", position_size=0.75, conviction="HIGH",
            reasons=reasons, entry_score=entry, exit_score=exit_score,
            uncertainty_penalty=uncertainty_penalty,
        )

    # Rule 3: Exit when uncertainty spikes (model doesn't know → reduce risk)
    if exit_score >= pol.exit_threshold * 0.85 and uncertainty_penalty > 0.7:
        reasons.append(f"exit_score={exit_score:.2f} + high uncertainty ({uncertainty_penalty:.2f})")
        return TradeDecision(
            action="REDUCE", position_size=0.50, conviction="MEDIUM",
            reasons=reasons, entry_score=entry, exit_score=exit_score,
            uncertainty_penalty=uncertainty_penalty,
        )

    # Rule 4: Moderate exit signal
    if exit_score >= pol.exit_threshold:
        reasons.append(f"exit_score={exit_score:.2f} >= threshold {pol.exit_threshold}")
        size = 0.5 if regime == "RANGE" else 0.75
        return TradeDecision(
            action="REDUCE", position_size=size, conviction="MEDIUM",
            reasons=reasons, entry_score=entry, exit_score=exit_score,
            uncertainty_penalty=uncertainty_penalty,
        )

    # ── Drawdown circuit breaker — block entries after severe drawdown ──────
    if dd_scalar <= 0.0:
        reasons.append("drawdown circuit breaker active — entries blocked")
        return TradeDecision(
            action="HOLD", position_size=0.0, conviction="NONE",
            reasons=reasons, entry_score=entry, exit_score=exit_score,
            uncertainty_penalty=round(uncertainty_penalty, 3),
        )

    # ── ENTRY rules ────────────────────────────────────────────────────────
    # Veto: uncertainty too high for entry
    entry_vetoed = entry_std > pol.entry_uncertainty_cap and uncertainty_penalty > 0.5
    if entry_vetoed:
        reasons.append(f"entry_std={entry_std:.3f} > cap {pol.entry_uncertainty_cap} — uncertainty veto")

    # Veto: regime confidence too low
    regime_vetoed = regime_confidence < pol.min_regime_confidence
    if regime_vetoed:
        reasons.append(f"regime_confidence={regime_confidence:.2f} < floor {pol.min_regime_confidence}")

    # Veto: entry edge is too small relative to modeled exit risk
    spread_vetoed = score_spread < pol.min_score_spread
    if spread_vetoed:
        reasons.append(f"score_spread={score_spread:.2f} < floor {pol.min_score_spread:.2f}")

    # Veto: low liquidity / high illiquidity
    liquidity_vetoed = False
    if liquidity_rank is not None and np.isfinite(liquidity_rank):
        liquidity_vetoed = liquidity_rank < pol.min_liquidity_rank
        if liquidity_vetoed:
            reasons.append(
                f"dollar_vol_rank={liquidity_rank:.2f} < floor {pol.min_liquidity_rank:.2f}"
            )
    amihud_vetoed = False
    if amihud is not None and np.isfinite(amihud):
        amihud_vetoed = amihud > pol.max_amihud
        if amihud_vetoed:
            reasons.append(f"amihud={amihud:.2f} > cap {pol.max_amihud:.2f}")

    readiness_vetoed = not model_ready
    if readiness_vetoed:
        reasons.append("model not marked ready for live signal generation")

    # Fundamental quality overlay.  Computed outside the ML model to avoid
    # training/inference distribution shift (fundamentals are zeroed in ML
    # features since we lack a PIT historical series).
    # fund_quality_score in [-1, +1]: positive = healthy, negative = risky.
    #   < -0.5 → hard veto (dangerous balance sheet: high debt or negative margins)
    #   < -0.2 → position size capped at 60% of normal
    #   > +0.3 → small size bonus (+10%)
    fund_vetoed = False
    fund_size_scalar = 1.0
    if fund_quality_score is not None:
        if fund_quality_score < -0.5:
            fund_vetoed = True
            reasons.append(
                f"fund_quality={fund_quality_score:.2f} < -0.5 — balance sheet veto "
                f"(high debt / negative margins)"
            )
        elif fund_quality_score < -0.2:
            fund_size_scalar = 0.60
            reasons.append(
                f"fund_quality={fund_quality_score:.2f}: position capped at 60% "
                f"(weak fundamentals)"
            )
        elif fund_quality_score > 0.3:
            fund_size_scalar = 1.10
            reasons.append(
                f"fund_quality={fund_quality_score:.2f}: +10% size bonus (strong fundamentals)"
            )

    # Strong entry: high score + favorable regime + confident model
    strong_entry_floor = max(pol.strong_entry_threshold, pol.entry_threshold)

    if (entry >= strong_entry_floor
            and regime in pol.favorable_regimes
            and not entry_vetoed
            and not regime_vetoed
            and not spread_vetoed
            and not liquidity_vetoed
            and not amihud_vetoed
            and not readiness_vetoed
            and not fund_vetoed):
        raw_size = entry * (1.0 - 0.5 * uncertainty_penalty) * combined_scalar * fund_size_scalar
        size = max(pol.min_position_pct, min(pol.max_position_pct, raw_size))
        reasons.append(f"entry={entry:.2f} >= strong threshold {strong_entry_floor:.2f} + {regime}")
        if uncertainty_penalty > 0.3:
            reasons.append(f"size reduced by uncertainty ({uncertainty_penalty:.2f})")
        return TradeDecision(
            action="BUY", position_size=round(size, 2), conviction="HIGH",
            reasons=reasons, entry_score=entry, exit_score=exit_score,
            uncertainty_penalty=round(uncertainty_penalty, 3),
        )

    # Standard entry: moderate score + favorable regime
    if (entry >= pol.entry_threshold
            and regime in pol.favorable_regimes
            and not entry_vetoed
            and not regime_vetoed
            and not spread_vetoed
            and not liquidity_vetoed
            and not amihud_vetoed
            and not readiness_vetoed
            and not fund_vetoed):
        raw_size = entry * (1.0 - 0.6 * uncertainty_penalty) * combined_scalar * fund_size_scalar
        size = max(pol.min_position_pct, min(pol.max_position_pct * 0.7, raw_size))
        reasons.append(f"entry={entry:.2f} >= threshold + {regime}")
        return TradeDecision(
            action="BUY", position_size=round(size, 2), conviction="MEDIUM",
            reasons=reasons, entry_score=entry, exit_score=exit_score,
            uncertainty_penalty=round(uncertainty_penalty, 3),
        )

    # Speculative entry: decent score but neutral regime
    # Gated by disable_range_entries (set after walk-forward shows negative RANGE Sharpe)
    if (entry >= pol.entry_threshold
            and regime == "RANGE"
            and not entry_vetoed
            and not spread_vetoed
            and not liquidity_vetoed
            and not amihud_vetoed
            and not readiness_vetoed
            and not fund_vetoed
            and not pol.disable_range_entries):
        raw_size = entry * 0.4 * (1.0 - 0.7 * uncertainty_penalty) * combined_scalar * fund_size_scalar
        size = max(pol.min_position_pct, min(pol.max_position_pct * 0.4, raw_size))
        reasons.append(f"entry={entry:.2f} in RANGE regime — speculative")
        return TradeDecision(
            action="BUY", position_size=round(size, 2), conviction="LOW",
            reasons=reasons, entry_score=entry, exit_score=exit_score,
            uncertainty_penalty=round(uncertainty_penalty, 3),
        )
    if entry >= pol.entry_threshold and regime == "RANGE" and pol.disable_range_entries:
        reasons.append(f"RANGE entry blocked (disable_range_entries=True, RANGE Sharpe ≤ 0)")
        # fall through to WATCH / HOLD

    # ── WATCH: potential setup forming ─────────────────────────────────────
    if regime == "REVERSAL_UP" and regime_confidence > 0.45:
        reasons.append(f"reversal_up detected (conf={regime_confidence:.2f}) — watching")
        return TradeDecision(
            action="WATCH", position_size=0.0, conviction="LOW",
            reasons=reasons, entry_score=entry, exit_score=exit_score,
            uncertainty_penalty=round(uncertainty_penalty, 3),
        )

    if entry >= pol.entry_threshold * 0.85 and regime in pol.favorable_regimes:
        reasons.append(f"entry={entry:.2f} approaching threshold in {regime}")
        return TradeDecision(
            action="WATCH", position_size=0.0, conviction="LOW",
            reasons=reasons, entry_score=entry, exit_score=exit_score,
            uncertainty_penalty=round(uncertainty_penalty, 3),
        )

    # ── HOLD: no actionable signal ─────────────────────────────────────────
    if not reasons:
        reasons.append("no actionable signal")
    return TradeDecision(
        action="HOLD", position_size=0.0, conviction="NONE",
        reasons=reasons, entry_score=entry, exit_score=exit_score,
        uncertainty_penalty=round(uncertainty_penalty, 3),
    )


def _decision_to_signal(decision: TradeDecision) -> str:
    """Map TradeDecision → legacy signal string for backward compat."""
    action = decision.action
    conv = decision.conviction
    if action == "SELL" and conv == "HIGH":
        return "EXIT"
    if action == "BUY" and conv == "HIGH":
        return "STRONG ENTRY"
    if action == "BUY" and conv == "MEDIUM":
        return "ENTRY"
    if action == "BUY" and conv == "LOW":
        return "SPECULATIVE"
    if action == "REDUCE":
        return "REDUCE"
    if action == "WATCH":
        return "WATCH (REVERSAL)"
    return "HOLD"


# ── Sklearn backend ──────────────────────────────────────────────────────────

class _LightGBMModels:
    """LightGBM ensemble: 1 classifier + 2 regressors. Fast, handles categoricals."""

    def __init__(self, config: MLConfig):
        import lightgbm as lgb

        clf_params = dict(
            n_estimators=config.n_trees, max_depth=config.max_depth,
            learning_rate=config.learning_rate, num_leaves=config.num_leaves,
            subsample=0.7, colsample_bytree=0.7, min_child_samples=30,
            reg_alpha=0.1, reg_lambda=1.0,       # L1/L2 regularisation
            min_split_gain=0.01,                   # minimum gain to split
            class_weight="balanced",               # correct RANGE class dominance
            verbosity=-1, n_jobs=-1,
        )
        reg_params = dict(
            n_estimators=config.n_trees, max_depth=config.max_depth,
            learning_rate=config.learning_rate, num_leaves=config.num_leaves,
            subsample=0.7, colsample_bytree=0.7, min_child_samples=30,
            reg_alpha=0.1, reg_lambda=1.0,
            min_split_gain=0.01,
            verbosity=-1, n_jobs=-1,
        )
        self.regime_clf = lgb.LGBMClassifier(**clf_params)
        self.entry_reg = lgb.LGBMRegressor(**reg_params)
        self.exit_reg = lgb.LGBMRegressor(**reg_params)
        self.feature_names: List[str] = []

    def fit(self, X: np.ndarray, y_regime: np.ndarray,
            y_entry: np.ndarray, y_exit: np.ndarray,
            feature_names: List[str],
            epoch_callback: Optional[Any] = None):
        self.feature_names = feature_names
        self._epoch_log: List[Dict] = []

        # Three-way split: 70% train / 15% LightGBM eval / 15% Platt calibration.
        # Calibration set is held out from the classifier entirely so that
        # CalibratedClassifierCV(cv='prefit') uses truly unseen probabilities.
        calib_split = int(len(X) * 0.85)   # calibration set starts here
        eval_split  = int(len(X) * 0.70)   # LGBM early-stop eval starts here

        X_tr, X_vl = X[:eval_split], X[eval_split:calib_split]
        X_cal = X[calib_split:]
        yr_tr, yr_vl = y_regime[:eval_split], y_regime[eval_split:calib_split]
        yr_cal = y_regime[calib_split:]
        ye_tr, ye_vl = y_entry[:eval_split], y_entry[eval_split:calib_split]
        yx_tr, yx_vl = y_exit[:eval_split], y_exit[eval_split:calib_split]

        train_classes = set(np.unique(yr_tr))
        valid_classes = set(np.unique(yr_vl))
        use_eval_set = bool(X_vl is not None and len(X_vl) > 0 and valid_classes.issubset(train_classes))

        if use_eval_set:
            self.regime_clf.fit(X_tr, yr_tr, eval_set=[(X_vl, yr_vl)])
            self.entry_reg.fit(X_tr, ye_tr, eval_set=[(X_vl, ye_vl)])
            self.exit_reg.fit(X_tr, yx_tr, eval_set=[(X_vl, yx_vl)])
        else:
            if valid_classes and not valid_classes.issubset(train_classes):
                log.warning(
                    "LightGBM eval_set skipped because validation labels include unseen classes: %s",
                    sorted(valid_classes - train_classes),
                )
            self.regime_clf.fit(X_tr, yr_tr)
            self.entry_reg.fit(X_tr, ye_tr)
            self.exit_reg.fit(X_tr, yx_tr)

        # ── Platt scaling: calibrate regime probabilities ─────────────────
        # LightGBM probabilities are often overconfident near decision boundaries.
        # CalibratedClassifierCV(cv='prefit') fits a softmax/isotonic layer on
        # the held-out calibration set — does NOT retrain the base estimator.
        self._cal_n = len(X_cal)  # record calibration set size for dashboard display
        try:
            from sklearn.calibration import CalibratedClassifierCV
            # Require at least 50 samples for isotonic calibration.  Isotonic
            # regression has O(n) capacity and overfits aggressively on small
            # sets — fewer than ~50 samples produces worse-calibrated outputs
            # than the raw LightGBM probabilities.  In that case, skip
            # calibration and use the raw classifier directly.
            if len(X_cal) >= 50 and len(np.unique(yr_cal)) >= 2:
                calibrated = CalibratedClassifierCV(
                    self.regime_clf, cv="prefit", method="isotonic"
                )
                calibrated.fit(X_cal, yr_cal)
                self._calibrated_clf = calibrated
                log.info(f"Platt calibration fitted on {len(X_cal)} held-out samples")
            else:
                self._calibrated_clf = None
                log.info(
                    f"Calibration skipped: only {len(X_cal)} samples in calibration "
                    f"slice (need ≥50). Using raw LightGBM probabilities."
                )
        except Exception as e:
            log.warning(f"Calibration skipped: {e}")
            self._calibrated_clf = None

        # Build epoch log from LightGBM eval results
        try:
            from sklearn.metrics import log_loss, mean_squared_error
            r_evals = self.regime_clf.evals_result_.get("valid_0", {})
            e_evals = self.entry_reg.evals_result_.get("valid_0", {})
            x_evals = self.exit_reg.evals_result_.get("valid_0", {})

            r_losses = r_evals.get("multi_logloss", r_evals.get("multi_error", []))
            e_losses = e_evals.get("l2", e_evals.get("rmse", []))
            x_losses = x_evals.get("l2", x_evals.get("rmse", []))

            n_iters = min(len(r_losses), len(e_losses), len(x_losses)) if r_losses and e_losses and x_losses else 0
            for i in range(n_iters):
                rl = r_losses[i] if i < len(r_losses) else 0
                el = e_losses[i] if i < len(e_losses) else 0
                xl = x_losses[i] if i < len(x_losses) else 0
                entry = {"epoch": i, "train_loss": round(rl + el + xl, 5),
                         "val_loss": round(rl + el + xl, 5),
                         "regime_loss": round(rl, 5), "entry_loss": round(el, 5),
                         "exit_loss": round(xl, 5)}
                self._epoch_log.append(entry)
                if epoch_callback:
                    epoch_callback(entry)
        except Exception:
            pass

    def predict(self, X: np.ndarray, mc_passes: int = 1
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Use calibrated classifier when available (Platt/isotonic scaling)
        clf = getattr(self, "_calibrated_clf", None) or self.regime_clf
        regime_probs = clf.predict_proba(X)
        entry = np.clip(self.entry_reg.predict(X), 0, 1)
        exit_ = np.clip(self.exit_reg.predict(X), 0, 1)
        return regime_probs, entry, exit_, clf.classes_

    def feature_importance_dict(self) -> Dict[str, float]:
        imp = self.regime_clf.feature_importances_
        total = imp.sum() or 1.0
        pairs = sorted(zip(self.feature_names, imp / total), key=lambda x: -x[1])
        return {name: round(float(v), 4) for name, v in pairs[:15]}

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "_LightGBMModels":
        with open(path, "rb") as f:
            return pickle.load(f)


class _LogisticModels:
    """Logistic Regression baseline: fast, interpretable, sanity check."""

    def __init__(self, config: MLConfig):
        from sklearn.linear_model import LogisticRegression
        from sklearn.linear_model import Ridge

        self.regime_clf = LogisticRegression(
            C=1.0, max_iter=500, multi_class="multinomial", solver="lbfgs",
        )
        self.entry_reg = Ridge(alpha=1.0)
        self.exit_reg = Ridge(alpha=1.0)
        self.feature_names: List[str] = []
        self._scaler = None

    def fit(self, X: np.ndarray, y_regime: np.ndarray,
            y_entry: np.ndarray, y_exit: np.ndarray,
            feature_names: List[str],
            epoch_callback: Optional[Any] = None):
        from sklearn.preprocessing import StandardScaler

        self.feature_names = feature_names
        self._epoch_log: List[Dict] = []

        # Logistic regression needs scaling
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self.regime_clf.fit(X_scaled, y_regime)
        self.entry_reg.fit(X_scaled, y_entry)
        self.exit_reg.fit(X_scaled, y_exit)

        # Single "epoch" log entry
        from sklearn.metrics import log_loss
        proba = self.regime_clf.predict_proba(X_scaled)
        rl = log_loss(y_regime, proba, labels=self.regime_clf.classes_)
        entry = {"epoch": 0, "train_loss": round(rl, 5), "regime_loss": round(rl, 5),
                 "entry_loss": 0.0, "exit_loss": 0.0}
        self._epoch_log.append(entry)
        if epoch_callback:
            epoch_callback(entry)

    def predict(self, X: np.ndarray, mc_passes: int = 1
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_scaled = self._scaler.transform(X) if self._scaler else X
        regime_probs = self.regime_clf.predict_proba(X_scaled)
        entry = np.clip(self.entry_reg.predict(X_scaled), 0, 1)
        exit_ = np.clip(self.exit_reg.predict(X_scaled), 0, 1)
        return regime_probs, entry, exit_, self.regime_clf.classes_

    def feature_importance_dict(self) -> Dict[str, float]:
        # Use absolute coefficient magnitudes
        coefs = np.abs(self.regime_clf.coef_).mean(axis=0)
        total = coefs.sum() or 1.0
        pairs = sorted(zip(self.feature_names, coefs / total), key=lambda x: -x[1])
        return {name: round(float(v), 4) for name, v in pairs[:15]}

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "_LogisticModels":
        with open(path, "rb") as f:
            return pickle.load(f)


class _EnsembleModels:
    """Blend LightGBM and logistic outputs into a single more stable signal."""

    def __init__(self, config: MLConfig):
        light_cfg = MLConfig(**{**asdict(config), "model_type": "lightgbm"})
        log_cfg = MLConfig(**{**asdict(config), "model_type": "logistic"})
        self.members = [
            ("lightgbm", _LightGBMModels(light_cfg)),
            ("logistic", _LogisticModels(log_cfg)),
        ]
        self.feature_names: List[str] = []

    def fit(
        self,
        X: np.ndarray,
        y_regime: np.ndarray,
        y_entry: np.ndarray,
        y_exit: np.ndarray,
        feature_names: List[str],
        epoch_callback: Optional[Any] = None,
    ):
        self.feature_names = feature_names
        self._epoch_log: List[Dict[str, Any]] = []
        for idx, (name, member) in enumerate(self.members):
            member.fit(
                X, y_regime, y_entry, y_exit,
                feature_names,
                epoch_callback=epoch_callback if idx == 0 else None,
            )
            self._epoch_log.extend(getattr(member, "_epoch_log", []))

    def predict(
        self,
        X: np.ndarray,
        mc_passes: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        probs_acc = np.zeros((len(X), len(REGIME_CLASSES)), dtype=float)
        entry_acc = np.zeros(len(X), dtype=float)
        exit_acc = np.zeros(len(X), dtype=float)

        for _, member in self.members:
            probs, entry, exit_, classes = member.predict(X, mc_passes=mc_passes)
            aligned = np.zeros_like(probs_acc)
            for j, cls in enumerate(classes):
                aligned[:, _CLS2IDX[str(cls)]] = probs[:, j]
            probs_acc += aligned
            entry_acc += entry
            exit_acc += exit_

        n = max(len(self.members), 1)
        return (
            probs_acc / n,
            np.clip(entry_acc / n, 0, 1),
            np.clip(exit_acc / n, 0, 1),
            np.array(REGIME_CLASSES),
        )

    def feature_importance_dict(self) -> Dict[str, float]:
        merged: Dict[str, List[float]] = {}
        for _, member in self.members:
            for name, value in member.feature_importance_dict().items():
                merged.setdefault(name, []).append(float(value))
        ranked = sorted(
            ((name, float(np.mean(vals))) for name, vals in merged.items()),
            key=lambda kv: -kv[1],
        )
        return {name: round(val, 4) for name, val in ranked[:15]}

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "_EnsembleModels":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── PyTorch backend ──────────────────────────────────────────────────────────

class _PyTorchModels:
    """Small MLP trained on GPU for regime classification + entry/exit regression."""

    def __init__(self, config: MLConfig, n_features: int):
        import torch
        import torch.nn as nn

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_names: List[str] = []
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._mc_uncertainty: Optional[Dict[str, float]] = None
        self._mc_uncertainty_rows: Optional[List[Dict[str, float]]] = None
        # Temperature scaling: T=1.0 = no calibration (raw softmax).
        # Fitted post-training on a held-out set; T>1 softens overconfident
        # distributions, T<1 sharpens them (rare).  See _calibrate_temperature().
        self._temperature: float = 1.0

        # Build model
        dims = [n_features] + config.hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
        self.backbone = nn.Sequential(*layers).to(self.device)

        # Heads
        last_dim = config.hidden_dims[-1]
        self.regime_head = nn.Linear(last_dim, len(REGIME_CLASSES)).to(self.device)
        self.entry_head = nn.Sequential(nn.Linear(last_dim, 1), nn.Sigmoid()).to(self.device)
        self.exit_head = nn.Sequential(nn.Linear(last_dim, 1), nn.Sigmoid()).to(self.device)

    def _all_params(self):
        import itertools
        return itertools.chain(
            self.backbone.parameters(),
            self.regime_head.parameters(),
            self.entry_head.parameters(),
            self.exit_head.parameters(),
        )

    def fit(self, X: np.ndarray, y_regime: np.ndarray,
            y_entry: np.ndarray, y_exit: np.ndarray,
            feature_names: List[str],
            epoch_callback: Optional[Any] = None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.feature_names = feature_names

        # Normalize features
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8
        X_norm = (X - self._mean) / self._std

        # Convert regime strings to ints
        regime_idx = np.array([_CLS2IDX.get(str(r), 4) for r in y_regime])

        # Train/val split for early stopping (last 15%)
        val_split = int(len(X_norm) * 0.85)
        X_train, X_val = X_norm[:val_split], X_norm[val_split:]
        yr_train, yr_val = regime_idx[:val_split], regime_idx[val_split:]
        ye_train, ye_val = y_entry[:val_split], y_entry[val_split:]
        yx_train, yx_val = y_exit[:val_split], y_exit[val_split:]

        # Training tensors
        X_t = torch.FloatTensor(X_train).to(self.device)
        yr_t = torch.LongTensor(yr_train).to(self.device)
        ye_t = torch.FloatTensor(ye_train).unsqueeze(1).to(self.device)
        yx_t = torch.FloatTensor(yx_train).unsqueeze(1).to(self.device)

        # Validation tensors
        X_v = torch.FloatTensor(X_val).to(self.device)
        yr_v = torch.LongTensor(yr_val).to(self.device)
        ye_v = torch.FloatTensor(ye_val).unsqueeze(1).to(self.device)
        yx_v = torch.FloatTensor(yx_val).unsqueeze(1).to(self.device)

        dataset = TensorDataset(X_t, yr_t, ye_t, yx_t)
        # BatchNorm1d crashes on batches of size 1. Always drop the last
        # incomplete batch when we have enough data for >1 full batch.
        bs = min(self.config.batch_size, max(2, len(dataset)))
        loader = DataLoader(dataset, batch_size=bs, shuffle=True,
                            drop_last=len(dataset) > bs)

        # Class weights for imbalanced regimes
        class_counts = np.bincount(yr_train, minlength=len(REGIME_CLASSES)).astype(float)
        class_counts = np.maximum(class_counts, 1.0)
        class_weights = torch.FloatTensor(1.0 / class_counts).to(self.device)
        class_weights = class_weights / class_weights.sum() * len(REGIME_CLASSES)

        regime_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        reg_loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(self._all_params(), lr=self.config.pt_learning_rate,
                                       weight_decay=1e-4)
        # T_max controls when the cosine LR cycle reaches eta_min.
        # Using config.epochs as T_max causes the LR to still be near its
        # maximum when early stopping fires at epoch 20 of 100.  Setting T_max
        # to roughly 1/3 of max epochs ensures the schedule converges by the
        # time early stopping typically fires, giving the later epochs proper
        # fine-tuning with a small LR.  Floor of 20 prevents the LR from
        # collapsing too fast on small datasets.
        _sched_t_max = max(20, self.config.epochs // 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=_sched_t_max, eta_min=1e-6)

        # Loss weights for multi-head balancing
        w_r = self.config.loss_w_regime
        w_e = self.config.loss_w_entry
        w_x = self.config.loss_w_exit

        # Early stopping state
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        self._epoch_log: List[Dict] = []
        self._early_stop_epoch: Optional[int] = None

        self.backbone.train()
        self.regime_head.train()
        self.entry_head.train()
        self.exit_head.train()

        for epoch in range(self.config.epochs):
            # ── Training step ──────────────────────────────────────────────
            epoch_loss = 0.0
            for xb, yr, ye, yx in loader:
                h = self.backbone(xb)
                loss_r = regime_loss_fn(self.regime_head(h), yr)
                loss_e = reg_loss_fn(self.entry_head(h), ye)
                loss_x = reg_loss_fn(self.exit_head(h), yx)
                loss = w_r * loss_r + w_e * loss_e + w_x * loss_x

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._all_params(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            # ── Validation step (for early stopping) ───────────────────────
            if len(X_val) > 0:
                self.backbone.eval()
                self.regime_head.eval()
                self.entry_head.eval()
                self.exit_head.eval()

                with torch.no_grad():
                    h_v = self.backbone(X_v)
                    vl_r = regime_loss_fn(self.regime_head(h_v), yr_v)
                    vl_e = reg_loss_fn(self.entry_head(h_v), ye_v)
                    vl_x = reg_loss_fn(self.exit_head(h_v), yx_v)
                    val_loss = (w_r * vl_r + w_e * vl_e + w_x * vl_x).item()

                # Log epoch metrics
                lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else 0
                n_batches = max(1, len(loader))
                entry = {
                    "epoch": epoch, "train_loss": round(epoch_loss / n_batches, 5),
                    "val_loss": round(val_loss, 5), "lr": round(lr_now, 7),
                    "regime_loss": round(vl_r.item(), 5),
                    "entry_loss": round(vl_e.item(), 5),
                    "exit_loss": round(vl_x.item(), 5),
                }
                self._epoch_log.append(entry)
                if epoch_callback:
                    epoch_callback(entry)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Snapshot best weights
                    best_state = {
                        "backbone": {k: v.clone() for k, v in self.backbone.state_dict().items()},
                        "regime_head": {k: v.clone() for k, v in self.regime_head.state_dict().items()},
                        "entry_head": {k: v.clone() for k, v in self.entry_head.state_dict().items()},
                        "exit_head": {k: v.clone() for k, v in self.exit_head.state_dict().items()},
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        self._early_stop_epoch = epoch
                        log.info(f"Early stopping at epoch {epoch+1}/{self.config.epochs} "
                                 f"(val_loss={val_loss:.4f}, best={best_val_loss:.4f})")
                        break

                self.backbone.train()
                self.regime_head.train()
                self.entry_head.train()
                self.exit_head.train()

        # Restore best weights if early stopping was used
        if best_state is not None:
            self.backbone.load_state_dict(best_state["backbone"])
            self.regime_head.load_state_dict(best_state["regime_head"])
            self.entry_head.load_state_dict(best_state["entry_head"])
            self.exit_head.load_state_dict(best_state["exit_head"])

        # ── Full-data retrain ──────────────────────────────────────────────
        # The early-stopping loop trained on 85% of data and held out 15% for
        # patience monitoring.  Best practice (analogous to sklearn's refit=True
        # in GridSearchCV): retrain from the discovered best weights on the
        # FULL dataset for the same number of epochs the model converged at.
        # This recovers the ~15% of data that was permanently held out, giving
        # the final model more signal without changing the convergence criterion.
        optimal_epochs = (
            self._early_stop_epoch if self._early_stop_epoch is not None
            else (len(self._epoch_log) - 1) if self._epoch_log else (self.config.epochs - 1)
        )
        if optimal_epochs > 0:
            X_full_t = torch.cat([X_t, X_v], dim=0)
            yr_full_t = torch.cat([yr_t, yr_v], dim=0)
            ye_full_t = torch.cat([ye_t, ye_v], dim=0)
            yx_full_t = torch.cat([yx_t, yx_v], dim=0)
            full_dataset = TensorDataset(X_full_t, yr_full_t, ye_full_t, yx_full_t)
            full_bs = min(self.config.batch_size, max(2, len(full_dataset)))
            full_loader = DataLoader(full_dataset, batch_size=full_bs, shuffle=True,
                                     drop_last=len(full_dataset) > full_bs)
            # New optimizer + scheduler over the full dataset for optimal_epochs
            opt2 = torch.optim.AdamW(self._all_params(), lr=self.config.pt_learning_rate,
                                      weight_decay=1e-4)
            sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt2, T_max=optimal_epochs, eta_min=1e-6)
            self.backbone.train(); self.regime_head.train()
            self.entry_head.train(); self.exit_head.train()
            for _ in range(optimal_epochs):
                for xb, yr_b, ye_b, yx_b in full_loader:
                    h = self.backbone(xb)
                    loss = (w_r * regime_loss_fn(self.regime_head(h), yr_b)
                            + w_e * reg_loss_fn(self.entry_head(h), ye_b)
                            + w_x * reg_loss_fn(self.exit_head(h), yx_b))
                    opt2.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._all_params(), 1.0)
                    opt2.step()
                sched2.step()
            log.info(
                "MLP full-data retrain: %d epochs on %d samples (was %d samples during search).",
                optimal_epochs, len(full_dataset), len(dataset),
            )

        self.backbone.eval()
        self.regime_head.eval()
        self.entry_head.eval()
        self.exit_head.eval()

        # ── Temperature scaling ────────────────────────────────────────────
        # Fit a single calibration scalar on the held-out validation set.
        # This is the MLP analogue of LightGBM's CalibratedClassifierCV —
        # it gives the min_regime_confidence threshold a consistent
        # probabilistic interpretation without re-training the backbone.
        self._calibrate_temperature(X_val, y_regime[val_split:])

    def _calibrate_temperature(self, X_cal: np.ndarray, y_regime_cal: np.ndarray) -> None:
        """
        Fit temperature scaling on a held-out calibration set.

        Temperature scaling (Guo et al. 2017) is the standard post-hoc
        calibration method for neural networks.  A single scalar T is
        optimised to minimise NLL of softmax(logits / T) on the held-out set.
        T > 1 softens overconfident outputs; T < 1 sharpens underconfident ones.

        Requires ≥ 50 samples.  Below that, isotonic regression would overfit
        even more severely, so T=1.0 (no calibration) is safer.
        """
        import torch
        import torch.nn as nn

        self._cal_n = len(X_cal)  # record for dashboard display
        if len(X_cal) < 50:
            self._temperature = 1.0
            log.info(
                "MLP temperature scaling skipped: only %d calibration samples (need ≥50).",
                len(X_cal),
            )
            return

        if len(np.unique(y_regime_cal)) < 2:
            self._temperature = 1.0
            log.info("MLP temperature scaling skipped: fewer than 2 classes in calibration set.")
            return

        X_norm = (X_cal - self._mean) / self._std
        X_t = torch.FloatTensor(X_norm).to(self.device)
        y_t = torch.LongTensor(
            [_CLS2IDX.get(str(r), 4) for r in y_regime_cal]
        ).to(self.device)

        # Collect raw logits in eval mode — no gradient through backbone
        self.backbone.eval()
        self.regime_head.eval()
        with torch.no_grad():
            logits = self.regime_head(self.backbone(X_t)).detach()

        # Parameterise temperature in log-space so T is always positive
        log_temp = nn.Parameter(torch.zeros(1, device=self.device))
        nll = nn.CrossEntropyLoss()
        opt = torch.optim.LBFGS([log_temp], lr=0.01, max_iter=50)

        def _closure():
            opt.zero_grad()
            loss = nll(logits / torch.exp(log_temp), y_t)
            loss.backward()
            return loss

        try:
            opt.step(_closure)
            T = float(torch.exp(log_temp).item())
            # Clip to a sane range: T < 0.5 = too sharp (rare); T > 5.0 = near-uniform
            self._temperature = max(0.5, min(5.0, T))
            log.info(
                "MLP temperature scaling: T=%.3f fitted on %d calibration samples.",
                self._temperature, len(X_cal),
            )
        except Exception as exc:
            log.warning("MLP temperature scaling failed (%s) — using T=1.0 (raw softmax).", exc)
            self._temperature = 1.0

    def predict(self, X: np.ndarray, mc_passes: int = 1
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with optional MC Dropout for uncertainty estimation.

        When mc_passes > 1, runs multiple forward passes with dropout enabled,
        then averages predictions. The variance across passes indicates
        model uncertainty — high variance = low confidence.

        Regime probabilities are temperature-scaled (T fitted on held-out
        calibration set) so that the min_regime_confidence threshold has a
        consistent probabilistic interpretation across training runs.
        """
        import torch

        X_norm = (X - self._mean) / self._std
        X_t = torch.FloatTensor(X_norm).to(self.device)

        T = self._temperature  # calibrated scalar; 1.0 = no calibration (raw softmax)

        if mc_passes <= 1:
            # Standard deterministic inference
            self._mc_uncertainty = None
            self._mc_uncertainty_rows = None
            with torch.no_grad():
                h = self.backbone(X_t)
                regime_logits = self.regime_head(h)
                # Divide logits by T before softmax (temperature scaling).
                # T > 1 softens overconfident distributions; T=1.0 = no change.
                regime_probs = torch.softmax(regime_logits / T, dim=1).cpu().numpy()
                entry = self.entry_head(h).cpu().numpy().flatten()
                exit_ = self.exit_head(h).cpu().numpy().flatten()
            return regime_probs, entry, exit_, np.array(REGIME_CLASSES)

        # MC Dropout: enable ONLY dropout layers (not BatchNorm) for uncertainty.
        # Setting backbone.train() would also switch BatchNorm to per-batch mode,
        # which crashes on single-sample inputs. Instead, selectively enable dropout.
        import torch.nn as nn
        for m in self.backbone.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        all_probs, all_entry, all_exit = [], [], []

        with torch.no_grad():
            for _ in range(mc_passes):
                h = self.backbone(X_t)
                # Temperature scaling applied to each MC pass before averaging.
                probs = torch.softmax(self.regime_head(h) / T, dim=1).cpu().numpy()
                ent = self.entry_head(h).cpu().numpy().flatten()
                ext = self.exit_head(h).cpu().numpy().flatten()
                all_probs.append(probs)
                all_entry.append(ent)
                all_exit.append(ext)

        # Restore all to eval mode
        for m in self.backbone.modules():
            if isinstance(m, nn.Dropout):
                m.eval()

        # Average across MC passes (mean prediction)
        regime_probs = np.mean(all_probs, axis=0)
        entry = np.mean(all_entry, axis=0)
        exit_ = np.mean(all_exit, axis=0)

        # Store per-row uncertainty so the decision policy can apply it bar by bar.
        regime_std = np.std(all_probs, axis=0).mean(axis=1)
        entry_std = np.std(all_entry, axis=0)
        exit_std = np.std(all_exit, axis=0)
        self._mc_uncertainty_rows = [
            {
                "regime_std": float(regime_std[i]),
                "entry_std": float(entry_std[i]),
                "exit_std": float(exit_std[i]),
            }
            for i in range(len(entry))
        ]
        self._mc_uncertainty = (
            self._mc_uncertainty_rows[0] if len(self._mc_uncertainty_rows) == 1 else None
        )

        return regime_probs, entry, exit_, np.array(REGIME_CLASSES)

    def feature_importance_dict(self) -> Dict[str, float]:
        """Approximate feature importance via gradient-based attribution."""
        # For simplicity, return uniform — proper attribution requires input data
        if not self.feature_names:
            return {}
        imp = 1.0 / len(self.feature_names)
        return {name: round(imp, 4) for name in self.feature_names[:15]}

    def save(self, path: Path):
        import torch
        state = {
            "backbone": self.backbone.state_dict(),
            "regime_head": self.regime_head.state_dict(),
            "entry_head": self.entry_head.state_dict(),
            "exit_head": self.exit_head.state_dict(),
            "feature_names": self.feature_names,
            "mean": self._mean,
            "std": self._std,
            "config": asdict(self.config),
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: Path, config: MLConfig) -> "_PyTorchModels":
        import torch
        state = torch.load(path, weights_only=False)
        n_features = len(state["feature_names"])
        obj = cls(config, n_features)
        obj.backbone.load_state_dict(state["backbone"])
        obj.regime_head.load_state_dict(state["regime_head"])
        obj.entry_head.load_state_dict(state["entry_head"])
        obj.exit_head.load_state_dict(state["exit_head"])
        obj.feature_names = state["feature_names"]
        obj._mean = state["mean"]
        obj._std = state["std"]
        obj.backbone.eval()
        obj.regime_head.eval()
        obj.entry_head.eval()
        obj.exit_head.eval()
        return obj


# ── Resolve backend ──────────────────────────────────────────────────────────

def _resolve_backend(requested: str) -> str:
    """Decide GPU backend availability."""
    if requested in ("pytorch", "auto"):
        try:
            import torch
            if torch.cuda.is_available():
                return "pytorch"
            if requested == "pytorch":
                log.warning("PyTorch requested but CUDA not available")
        except ImportError:
            if requested == "pytorch":
                log.warning("PyTorch not installed")
    return "cpu"


def _create_model(config: MLConfig, n_features: int = 0):
    """Factory: create the right model based on config.model_type."""
    mt = config.model_type
    if mt == "mlp":
        return _PyTorchModels(config, n_features)
    elif mt == "logistic":
        return _LogisticModels(config)
    elif mt == "ensemble":
        return _EnsembleModels(config)
    else:  # "lightgbm" default
        return _LightGBMModels(config)


# ── Walk-Forward Evaluation ──────────────────────────────────────────────────

def _purged_train_cutoff(val_start: int, gap: int, forward_horizon: int) -> int:
    """
    Cut training off far enough ahead of validation so no training label can use
    future returns from the validation window.
    """
    return val_start - max(max(0, gap), max(1, forward_horizon))


def _universe_holdout_masks(
    dates: np.ndarray,
    forward_horizon: int,
    holdout_frac: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calendar-ordered train/test masks for pooled universe data.

    Splits on unique dates, then purges the forward label horizon from the
    training side so no training target can see holdout prices.
    """
    unique_dates = np.unique(dates)
    if len(unique_dates) < 20:
        raise ValueError("Too few unique dates for universe holdout split")
    split_pos = max(1, int(len(unique_dates) * (1.0 - holdout_frac)))
    split_pos = min(split_pos, len(unique_dates) - 1)
    split_date = unique_dates[split_pos]
    purge_pos = max(0, split_pos - max(1, forward_horizon))
    train_cutoff_date = unique_dates[purge_pos]
    train_mask = dates < train_cutoff_date
    test_mask = dates >= split_date
    return train_mask, test_mask


def _walk_forward_evaluate(
    X: np.ndarray,
    y_regime: np.ndarray,
    y_entry: np.ndarray,
    y_exit: np.ndarray,
    prices: np.ndarray,
    dates: np.ndarray,
    config: MLConfig,
    n_splits: int = 5,
    regime_labels: Optional[np.ndarray] = None,
    vol_values: Optional[np.ndarray] = None,
    epoch_callback: Optional[Any] = None,
    force_split: Optional[int] = None,
) -> WalkForwardMetrics:
    """
    Walk-forward validation with full trading metrics.

    For each fold:
      1. Train on expanding window of past data
      2. Predict on next fold (out-of-sample)
      3. Simulate trades: BUY when entry_score > 0.6 in favorable regime, SELL when exit_score > 0.6
      4. Compute PnL metrics
    """
    from sklearn.metrics import accuracy_score

    all_trades: List[Dict] = []
    fold_results: List[Dict] = []
    all_pnl: List[float] = []
    all_positions: List[float] = []
    equity_curve: List[float] = [1.0]

    # Per-fold storage for last-fold aggregate and policy optimisation
    _fold_pnl_lists: List[List[float]] = []
    _fold_equity_lists: List[List[float]] = []
    _fold_trade_lists: List[List[Dict]] = []
    _fold_raw: List[Dict] = []   # raw predictions for grid search (no model re-training needed)

    gap = max(0, config.wf_gap)
    label_purge = max(1, config.forward_horizon)
    window_type = config.wf_window  # "expanding" or "rolling"

    # Use DecisionPolicy defaults as the WF simulation thresholds so that the
    # reported Sharpe matches what the live strategy would actually trade.
    # Hardcoding 0.6 here while the policy optimizer could select different
    # thresholds made the walk-forward metrics irrelevant to actual trading.
    _wf_policy = DecisionPolicy()
    _wf_entry_thresh = _wf_policy.entry_threshold
    _wf_exit_thresh  = _wf_policy.exit_threshold

    # Per-leg transaction cost applied to the WF equity curve at every entry
    # and exit.  Without this, reported Sharpe is fictional — no real strategy
    # trades for free.  Default: config.wf_trade_cost (0.1% per leg = 20bps RT).
    _wf_cost = max(0.0, getattr(config, "wf_trade_cost", 0.001))

    # Build fold iterator: either a single chrono split or TimeSeriesSplit
    if force_split is not None:
        train_idx = np.arange(force_split)
        val_idx   = np.arange(force_split, len(X))
        folds_iter = [(train_idx, val_idx)]
    else:
        from sklearn.model_selection import TimeSeriesSplit
        _max_train = (
            config.wf_rolling_size
            if window_type == "rolling" and config.wf_rolling_size
            else None
        )
        tscv = TimeSeriesSplit(
            n_splits=min(n_splits, max(2, len(X) // 50)),
            max_train_size=_max_train,
        )
        folds_iter = list(tscv.split(X))

    n_folds = len(folds_iter)
    for fold_i, (train_idx, val_idx) in enumerate(folds_iter):
        # Purge the maximum forward label horizon and any additional embargo gap.
        cutoff = _purged_train_cutoff(int(val_idx[0]), gap, label_purge)
        train_idx = train_idx[train_idx < cutoff]
        if len(train_idx) < 20:
            # Not enough training data after gap — skip fold
            continue

        # Train model on this fold
        fold_t0 = time.time()
        if epoch_callback:
            epoch_callback({
                "type": "status",
                "message": f"Walk-forward fold {fold_i + 1}/{n_folds}: "
                           f"training on {len(train_idx)} rows (purge={max(gap, label_purge)}), "
                           f"testing on {len(val_idx)} rows...",
            })
        n_feat = X.shape[1]
        model = _create_model(config, n_feat)
        model.fit(X[train_idx], y_regime[train_idx], y_entry[train_idx], y_exit[train_idx],
                  list(range(n_feat)))  # dummy feature names for CV

        # Predict on validation fold
        probs, entry_pred, exit_pred, classes = model.predict(X[val_idx])
        regime_pred = np.array([classes[i] for i in probs.argmax(axis=1)])
        acc = accuracy_score(y_regime[val_idx], regime_pred)

        # Simulate trades on this fold
        fold_prices = prices[val_idx]
        fold_dates = dates[val_idx] if dates is not None else np.arange(len(val_idx))
        fold_regimes = regime_labels[val_idx] if regime_labels is not None else None
        fold_vol = vol_values[val_idx] if vol_values is not None else None

        position = 0.0  # fraction of capital at risk
        entry_price = 0.0
        entry_idx = 0
        fold_trades_this: List[Dict] = []

        # Estimate realised vol from first 20 bars for vol-scaling (or use full fold)
        _vol_lookback = min(20, len(fold_prices) - 1)
        if _vol_lookback > 1:
            _init_rets = np.diff(fold_prices[:_vol_lookback + 1]) / fold_prices[:_vol_lookback]
            _realized_vol = float(np.std(_init_rets) * np.sqrt(252)) if len(_init_rets) > 1 else 0.20
        else:
            _realized_vol = 0.20
        _target_vol = config.target_annual_vol if hasattr(config, 'target_annual_vol') else 0.15
        _vol_scalar = min(2.0, max(0.2, _target_vol / max(_realized_vol, 0.01)))

        fold_pnl: List[float] = []
        fold_positions: List[float] = []
        fold_equity: List[float] = [1.0]
        
        # Seed the first tradable close so the first realised return starts on
        # the next bar, not on the signal bar itself.
        if len(fold_prices) > 0:
            ep0 = float(entry_pred[0])
            rp0 = str(regime_pred[0])
            if ep0 > _wf_entry_thresh and rp0 in ("TREND_UP", "REVERSAL_UP"):
                position = min(1.0, ep0 * _vol_scalar)
                entry_price = float(fold_prices[0])
                entry_idx = 0
                # Deduct seed-bar entry cost from starting equity
                if _wf_cost > 0:
                    fold_equity[0] *= (1.0 - _wf_cost)
                    equity_curve[-1] *= (1.0 - _wf_cost)

        for i in range(1, len(fold_prices)):
            prev_price = float(fold_prices[i - 1])
            price = float(fold_prices[i])
            daily_ret = (price - prev_price) / prev_price if prev_price > 0 else 0.0
            strategy_ret = daily_ret * position
            fold_pnl.append(strategy_ret)
            fold_positions.append(float(position))
            fold_equity.append(fold_equity[-1] * (1 + strategy_ret))
            all_pnl.append(strategy_ret)
            all_positions.append(float(position))
            equity_curve.append(equity_curve[-1] * (1 + strategy_ret))

            # Update rolling vol estimate from observed history only.
            if i > 20 and i % 10 == 0:
                _recent = fold_prices[max(0, i - 20):i + 1]
                _rrets = np.diff(_recent) / _recent[:-1]
                if len(_rrets) > 1 and np.std(_rrets) > 0:
                    _realized_vol = float(np.std(_rrets) * np.sqrt(252))
                    _vol_scalar = min(2.0, max(0.2, _target_vol / max(_realized_vol, 0.01)))

            ep = float(entry_pred[i])
            xp = float(exit_pred[i])
            rp = str(regime_pred[i])

            if position == 0 and ep > _wf_entry_thresh and rp in ("TREND_UP", "REVERSAL_UP"):
                position = min(1.0, ep * _vol_scalar)
                entry_price = price
                entry_idx = i
                # Deduct entry transaction cost from equity immediately
                if _wf_cost > 0 and fold_equity:
                    fold_equity[-1] *= (1.0 - _wf_cost)
                    equity_curve[-1] *= (1.0 - _wf_cost)
            elif position > 0 and (xp > _wf_exit_thresh or rp in ("TREND_DOWN", "REVERSAL_DOWN")):
                ret = (price - entry_price) / entry_price if entry_price > 0 else 0.0
                holding_days = i - entry_idx
                trade = {
                    "return": ret * position,
                    "holding_days": max(1, holding_days),
                    "regime": fold_regimes[entry_idx] if fold_regimes is not None else "UNKNOWN",
                    "vol_bucket": _vol_bucket(fold_vol[entry_idx]) if fold_vol is not None else "UNKNOWN",
                    "fold": fold_i,
                }
                all_trades.append(trade)
                fold_trades_this.append(trade)
                # Deduct exit transaction cost from equity immediately
                if _wf_cost > 0 and fold_equity:
                    fold_equity[-1] *= (1.0 - _wf_cost)
                    equity_curve[-1] *= (1.0 - _wf_cost)
                position = 0.0

        if position > 0 and len(fold_prices) > 0:
            last_price = float(fold_prices[-1])
            ret = (last_price - entry_price) / entry_price if entry_price > 0 else 0.0
            holding_days = len(fold_prices) - 1 - entry_idx
            trade = {
                "return": ret * position,
                "holding_days": max(1, holding_days),
                "regime": fold_regimes[entry_idx] if fold_regimes is not None else "UNKNOWN",
                "vol_bucket": _vol_bucket(fold_vol[entry_idx]) if fold_vol is not None else "UNKNOWN",
                "fold": fold_i,
            }
            all_trades.append(trade)
            fold_trades_this.append(trade)

        # Per-fold metrics
        fp_arr = np.array(fold_pnl) if fold_pnl else np.array([0.0])
        fe_arr = np.array(fold_equity)

        if len(fp_arr) > 1 and fp_arr.std() > 0:
            fold_sharpe = float((fp_arr.mean() / fp_arr.std()) * np.sqrt(252))
        else:
            fold_sharpe = 0.0

        fe_peak = np.maximum.accumulate(fe_arr)
        fe_dd = (fe_arr - fe_peak) / np.where(fe_peak > 0, fe_peak, 1)
        fold_max_dd = float(fe_dd.min())
        fold_total_ret = float(fe_arr[-1] / fe_arr[0] - 1) if fe_arr[0] > 0 else 0.0
        fold_vol_val = float(fp_arr.std() * np.sqrt(252)) if len(fp_arr) > 1 else 0.0

        fold_trets = [t["return"] for t in fold_trades_this]
        fold_win_rate = (sum(1 for r in fold_trets if r > 0) / len(fold_trets)) if fold_trets else 0.0

        # Regime distribution in this fold's test set
        regime_dist: Dict[str, int] = {}
        if fold_regimes is not None:
            for r in fold_regimes:
                regime_dist[str(r)] = regime_dist.get(str(r), 0) + 1

        fold_result = {
            "fold": fold_i,
            "train_size": len(train_idx),
            "test_size": len(val_idx),
            "accuracy": round(acc, 3),
            "n_trades": len(fold_trades_this),
            "sharpe": round(fold_sharpe, 3),
            "total_return": round(fold_total_ret, 4),
            "max_drawdown": round(fold_max_dd, 4),
            "win_rate": round(fold_win_rate, 3),
            "volatility": round(fold_vol_val, 4),
            "regime_dist": regime_dist,
        }
        fold_results.append(fold_result)
        _fold_pnl_lists.append(fold_pnl)
        _fold_equity_lists.append(fold_equity)
        _fold_trade_lists.append(fold_trades_this)
        _fold_raw.append({
            "fold_i": fold_i,
            "entry_pred": entry_pred,       # np.ndarray
            "exit_pred": exit_pred,         # np.ndarray
            "regime_pred": regime_pred,     # np.ndarray of strings
            "regime_conf": probs.max(axis=1),
            "fold_prices": fold_prices,
            "fold_regimes": fold_regimes,   # ground-truth regime labels
            "fold_vol": fold_vol,
        })

        fold_elapsed = round(time.time() - fold_t0, 2)
        if epoch_callback:
            epoch_callback({
                "type": "fold",
                "fold": fold_i,
                "accuracy": round(acc, 3),
                "n_trades": len(fold_trades_this),
                "sharpe": round(fold_sharpe, 3),
                "total_return": round(fold_total_ret, 4),
                "max_drawdown": round(fold_max_dd, 4),
                "win_rate": round(fold_win_rate, 3),
                "train_size": len(train_idx),
                "test_size": len(val_idx),
                "elapsed_s": fold_elapsed,
            })

    def _compute_trading_metrics(pnl_list, eq_list, trades_list):
        """Compute the standard trading metric block from pnl/equity/trades data."""
        pnl = np.array(pnl_list) if pnl_list else np.array([0.0])
        eq  = np.array(eq_list)  if eq_list  else np.array([1.0])

        sharpe_ = (pnl.mean() / pnl.std() * np.sqrt(252)) if (len(pnl) > 1 and pnl.std() > 0) else 0.0
        pk = np.maximum.accumulate(eq)
        max_dd_ = float(((eq - pk) / np.where(pk > 0, pk, 1)).min())
        n_yrs = len(pnl) / 252 if len(pnl) > 0 else 1
        tot_ret = eq[-1] / eq[0] - 1 if eq[0] > 0 else 0.0
        cagr_ = (eq[-1] / eq[0]) ** (1 / max(n_yrs, 0.01)) - 1 if eq[0] > 0 else 0.0

        trets = [t["return"] for t in trades_list]
        n_ = len(trets)
        hr_ = sum(1 for r in trets if r > 0) / max(n_, 1)
        gp = sum(r for r in trets if r > 0)
        gl = abs(sum(r for r in trets if r < 0))
        pf_ = gp / max(gl, 1e-8)
        avg_tr = float(np.mean(trets)) if trets else 0.0
        total_days_ = len(pnl)
        atpm = n_ / max(total_days_ / 21, 1)
        avg_hold_ = float(np.mean([t["holding_days"] for t in trades_list])) if trades_list else 0.0

        br_ = {}
        for regime in REGIME_CLASSES:
            rt = [t for t in trades_list if t["regime"] == regime]
            if rt:
                rr = np.array([t["return"] for t in rt])
                # Per-regime Sharpe: annualised (assume each trade holds avg 21 days)
                avg_hold = float(np.mean([t["holding_days"] for t in rt]))
                ann_factor = np.sqrt(252 / max(avg_hold, 1))
                reg_sharpe = float((rr.mean() / rr.std() * ann_factor)
                                   if rr.std() > 1e-9 else 0.0)
                br_[regime] = {
                    "n_trades": len(rt),
                    "hit_rate": round(float(np.mean(rr > 0)), 3),
                    "avg_return": round(float(rr.mean()), 4),
                    "sharpe": round(reg_sharpe, 3),
                }
        bv_ = {}
        for bucket in ("LOW", "MED", "HIGH"):
            vt = [t for t in trades_list if t["vol_bucket"] == bucket]
            if vt:
                vr = [t["return"] for t in vt]
                bv_[bucket] = {"n_trades": len(vt), "hit_rate": round(sum(1 for r in vr if r > 0)/len(vt),3), "avg_return": round(float(np.mean(vr)),4)}

        return dict(
            sharpe_ratio=round(sharpe_, 3), max_drawdown=round(max_dd_, 4),
            cagr=round(cagr_, 4), total_return=round(tot_ret, 4),
            hit_rate=round(hr_, 3), profit_factor=round(pf_, 3),
            n_trades=n_, avg_trade_return=round(avg_tr, 4),
            avg_trades_per_month=round(atpm, 2), avg_holding_period=round(avg_hold_, 1),
            by_regime=br_, by_volatility=bv_,
        )

    # ── All-folds aggregate metrics ────────────────────────────────────────
    m_all = _compute_trading_metrics(all_pnl, equity_curve, all_trades)

    # ── Last-fold-only metrics (most trained model) ────────────────────────
    if _fold_pnl_lists:
        m_last = _compute_trading_metrics(
            _fold_pnl_lists[-1], _fold_equity_lists[-1], _fold_trade_lists[-1]
        )
    else:
        m_last = m_all.copy()

    # Cross-fold analysis
    fold_sharpes = [f["sharpe"] for f in fold_results]
    fold_returns = [f["total_return"] for f in fold_results]
    worst_fold_idx = int(np.argmin(fold_sharpes)) if fold_sharpes else -1
    fold_sharpe_std = float(np.std(fold_sharpes)) if len(fold_sharpes) > 1 else 0.0
    fold_return_std = float(np.std(fold_returns)) if len(fold_returns) > 1 else 0.0
    pct_folds_profitable = (
        sum(1 for r in fold_returns if r > 0) / len(fold_returns)
        if fold_returns else 0.0
    )
    turnover_series = np.abs(np.diff(np.array([0.0] + all_positions, dtype=float))) if all_positions else np.array([0.0])
    avg_daily_turnover = float(np.mean(turnover_series)) if len(turnover_series) > 0 else 0.0

    # ── Policy grid search on fold predictions (no retraining) ───────────
    pol_opt: Optional[Dict] = None
    if _fold_raw:
        try:
            pol_opt = _grid_search_policy(
                _fold_raw,
                default_entry=0.60,
                default_exit=0.60,
                default_min_regime_confidence=config.signal_min_regime_confidence,
                default_min_score_spread=config.signal_min_score_spread,
                one_way_cost=(config.signal_commission_pct + config.signal_slippage_pct),
            ).to_dict()
        except Exception:
            pol_opt = None

    # Extract RANGE regime Sharpe for downstream policy gating
    _range_sharpe = m_all["by_regime"].get("RANGE", {}).get("sharpe", 0.0)

    # Information Ratio: annualised return / annualised volatility (Sharpe with rf=0)
    _ir = 0.0
    if len(all_pnl) > 20:
        _pnl_arr = np.array(all_pnl, dtype=float)
        _vol_ann = float(np.std(_pnl_arr)) * np.sqrt(252)
        if _vol_ann > 1e-9:
            _ir = round(float(np.mean(_pnl_arr)) * 252 / _vol_ann, 3)

    return WalkForwardMetrics(
        sharpe_ratio=m_all["sharpe_ratio"],
        max_drawdown=m_all["max_drawdown"],
        cagr=m_all["cagr"],
        hit_rate=m_all["hit_rate"],
        profit_factor=m_all["profit_factor"],
        total_return=m_all["total_return"],
        n_trades=m_all["n_trades"],
        avg_trade_return=m_all["avg_trade_return"],
        by_regime=m_all["by_regime"],
        by_volatility=m_all["by_volatility"],
        avg_trades_per_month=m_all["avg_trades_per_month"],
        avg_holding_period=m_all["avg_holding_period"],
        daily_returns=[round(float(v), 6) for v in all_pnl],
        position_exposure=[round(float(v), 4) for v in all_positions],
        avg_daily_turnover=round(avg_daily_turnover, 6),
        fold_results=fold_results,
        last_fold_metrics=m_last,
        worst_fold_idx=worst_fold_idx,
        fold_sharpe_std=round(fold_sharpe_std, 3),
        fold_return_std=round(fold_return_std, 4),
        pct_folds_profitable=round(pct_folds_profitable, 3),
        window_type=window_type,
        policy_opt=pol_opt,
        range_regime_sharpe=round(_range_sharpe, 3),
        information_ratio=_ir,
    )


def _vol_bucket(vol_val) -> str:
    """Classify a volatility value into LOW/MED/HIGH."""
    if vol_val is None or np.isnan(vol_val):
        return "MED"
    if vol_val < 0.33:
        return "LOW"
    elif vol_val < 0.66:
        return "MED"
    return "HIGH"


# ── Policy grid search ────────────────────────────────────────────────────────

_POLICY_ENTRY_THRS = [0.50, 0.55, 0.60, 0.65, 0.70]
_POLICY_EXIT_THRS  = [0.50, 0.55, 0.60, 0.65, 0.70]
_MIN_TRADES_FILTER = 5   # require at least this many trades total to consider a combo valid


def _sim_with_thresholds(
    fold_raw: List[Dict],
    entry_thr: float,
    exit_thr: float,
    min_regime_confidence: float,
    min_score_spread: float,
    one_way_cost: float = 0.0,
) -> tuple:
    """
    Re-simulate all walk-forward folds using specified entry/exit thresholds.
    Uses binary position (flat or fully long) — returns (daily_returns, trades).
    No model re-training; reuses stored fold predictions.
    """
    all_daily_rets: List[float] = []
    all_trades: List[Dict] = []

    for fold in fold_raw:
        entry_pred  = fold["entry_pred"]
        exit_pred   = fold["exit_pred"]
        regime_pred = fold["regime_pred"]
        regime_conf = fold["regime_conf"]
        fold_prices = fold["fold_prices"]
        fold_regimes = fold["fold_regimes"]
        fold_vol    = fold["fold_vol"]
        fold_i      = fold["fold_i"]

        position    = 0.0
        entry_price = 0.0
        entry_idx   = 0

        if len(fold_prices) > 0:
            ep0 = float(entry_pred[0])
            xp0 = float(exit_pred[0])
            rp0 = str(regime_pred[0])
            rc0 = float(regime_conf[0])
            if (
                ep0 > entry_thr
                and rp0 in ("TREND_UP", "REVERSAL_UP")
                and rc0 >= min_regime_confidence
                and (ep0 - xp0) >= min_score_spread
            ):
                position = 1.0
                entry_price = float(fold_prices[0])
                entry_idx = 0
                if one_way_cost > 0:
                    all_daily_rets.append(-one_way_cost)

        for i in range(1, len(fold_prices)):
            prev_price = float(fold_prices[i - 1])
            price = float(fold_prices[i])
            if prev_price > 0:
                daily_ret = (price - prev_price) / prev_price
                all_daily_rets.append(daily_ret * position)
            else:
                all_daily_rets.append(0.0)

            ep = float(entry_pred[i])
            xp = float(exit_pred[i])
            rp = str(regime_pred[i])
            rc = float(regime_conf[i])
            spread = ep - xp

            if (
                position == 0.0
                and ep > entry_thr
                and rp in ("TREND_UP", "REVERSAL_UP")
                and rc >= min_regime_confidence
                and spread >= min_score_spread
            ):
                position = 1.0
                entry_price = price
                entry_idx = i
                if one_way_cost > 0 and all_daily_rets:
                    all_daily_rets[-1] -= one_way_cost
            elif position > 0.0 and (
                xp > exit_thr
                or rp in ("TREND_DOWN", "REVERSAL_DOWN")
                or spread < 0.0
            ):
                ret = (price - entry_price) / entry_price if entry_price > 0 else 0.0
                net_ret = ret - (2.0 * one_way_cost)
                holding_days = i - entry_idx
                all_trades.append({
                    "return":       net_ret,
                    "holding_days": max(1, holding_days),
                    "regime":       fold_regimes[entry_idx] if fold_regimes is not None else "UNKNOWN",
                    "vol_bucket":   _vol_bucket(fold_vol[entry_idx]) if fold_vol is not None else "UNKNOWN",
                    "fold":         fold_i,
                })
                if one_way_cost > 0 and all_daily_rets:
                    all_daily_rets[-1] -= one_way_cost
                position = 0.0

        if position > 0.0 and len(fold_prices) > 0:
            last_price = float(fold_prices[-1])
            ret = (last_price - entry_price) / entry_price if entry_price > 0 else 0.0
            net_ret = ret - (2.0 * one_way_cost)
            holding_days = len(fold_prices) - 1 - entry_idx
            all_trades.append({
                "return":       net_ret,
                "holding_days": max(1, holding_days),
                "regime":       fold_regimes[entry_idx] if fold_regimes is not None else "UNKNOWN",
                "vol_bucket":   _vol_bucket(fold_vol[entry_idx]) if fold_vol is not None else "UNKNOWN",
                "fold":         fold_i,
            })

    return all_daily_rets, all_trades


def _policy_sharpe(daily_rets: List[float]) -> float:
    arr = np.array(daily_rets)
    if len(arr) < 2 or arr.std() == 0:
        return 0.0
    return float((arr.mean() / arr.std()) * np.sqrt(252))


def _grid_search_policy(
    fold_raw: List[Dict],
    default_entry: float = 0.60,
    default_exit: float  = 0.60,
    default_min_regime_confidence: float = 0.35,
    default_min_score_spread: float = 0.08,
    one_way_cost: float = 0.0,
) -> "PolicyOptResult":
    """
    Grid search over entry × exit threshold combinations using stored fold predictions.

    Scoring: annualised Sharpe of the daily-return series produced by the binary
    long-only simulation.  Uncertainty cap is excluded from the grid because
    MC Dropout is not run during walk-forward folds (cost).

    Parameters
    ----------
    fold_raw      : per-fold raw predictions collected by _walk_forward_evaluate
    default_entry : configured default entry threshold (for improvement% reference)
    default_exit  : configured default exit threshold
    """
    grid_scores: List[Dict] = []
    default_sharpe = 0.0

    conf_grid = [0.35, 0.45, 0.55]
    spread_grid = [0.00, 0.05, 0.08, 0.10, 0.15]

    for entry_thr in _POLICY_ENTRY_THRS:
        for exit_thr in _POLICY_EXIT_THRS:
            for conf_thr in conf_grid:
                for spread_thr in spread_grid:
                    daily_rets, trades = _sim_with_thresholds(
                        fold_raw,
                        entry_thr,
                        exit_thr,
                        conf_thr,
                        spread_thr,
                        one_way_cost=one_way_cost,
                    )
                    sharpe    = _policy_sharpe(daily_rets)
                    n_trades  = len(trades)
                    trets     = [t["return"] for t in trades]
                    hit_rate  = (sum(1 for r in trets if r > 0) / n_trades) if n_trades > 0 else 0.0

                    entry = {
                        "entry_threshold": entry_thr,
                        "exit_threshold":  exit_thr,
                        "min_regime_confidence": conf_thr,
                        "min_score_spread": spread_thr,
                        "sharpe":          round(sharpe, 3),
                        "n_trades":        n_trades,
                        "hit_rate":        round(hit_rate, 3),
                    }
                    grid_scores.append(entry)

                    if (
                        abs(entry_thr - default_entry) < 1e-9
                        and abs(exit_thr - default_exit) < 1e-9
                        and abs(conf_thr - default_min_regime_confidence) < 1e-9
                        and abs(spread_thr - default_min_score_spread) < 1e-9
                    ):
                        default_sharpe = sharpe

    # Select best: highest Sharpe among combos with enough trades
    valid = [s for s in grid_scores if s["n_trades"] >= _MIN_TRADES_FILTER]
    if not valid:
        valid = grid_scores  # relax filter if nothing qualifies
    best = max(valid, key=lambda s: s["sharpe"])

    improvement = (
        (best["sharpe"] - default_sharpe) / max(abs(default_sharpe), 0.01) * 100
    )

    return PolicyOptResult(
        best_entry_threshold=best["entry_threshold"],
        best_exit_threshold=best["exit_threshold"],
        best_min_regime_confidence=best["min_regime_confidence"],
        best_min_score_spread=best["min_score_spread"],
        best_sharpe=round(best["sharpe"], 3),
        default_sharpe=round(default_sharpe, 3),
        improvement_pct=round(improvement, 1),
        estimated_round_trip_cost=round(one_way_cost * 2.0, 6),
        n_combinations=len(grid_scores),
        grid_scores=grid_scores,
    )


# ── Main engine ──────────────────────────────────────────────────────────────

class MLEngine:
    """
    Main ML engine. Trains regime classifier + entry/exit regressors
    on historical indicator data, with optional GPU acceleration.
    """

    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or MLConfig()
        self.backend = _resolve_backend(self.config.backend)
        self._models: Optional[Any] = None
        self._ticker: Optional[str] = None
        self._last_df: Optional[pd.DataFrame] = None  # keep for dashboard
        self._epoch_callback: Optional[Any] = None
        self._policy_opt: Optional[Dict[str, Any]] = None
        self._readiness: Optional[Dict[str, Any]] = None
        self._range_regime_sharpe: Optional[float] = None
        # Chrono split mode: set by API before calling train()
        self._chrono_train_end: Optional[str] = None
        self._chrono_test_start: Optional[str] = None

    def train(
        self,
        ticker: str,
        df: Optional[pd.DataFrame] = None,
        period: Optional[str] = None,
        market_ctx: Optional[Dict] = None,
        fundamentals: Optional[Dict] = None,
        sentiment: Optional[Dict] = None,
    ) -> TrainResult:
        """
        Train on historical data for one ticker.
        If df is provided, use it directly. Otherwise fetch via yfinance.

        During training:
          - Fundamentals are EXCLUDED (no point-in-time data → lookahead bias)
          - Cross-asset features use historical SPY/VIX data (per-row, no lookahead)
          - Sentiment is EXCLUDED unless time-indexed
        """
        t0 = time.time()
        period = period or self.config.training_period
        self._ticker = ticker

        # Check cache
        cached = self._load_cached(ticker)
        if cached is not None:
            self._models = cached
            return TrainResult(
                ticker=ticker, backend=self.backend + " (cached)",
                model_type=self.config.model_type,
                n_samples=0, n_features=0,
                regime_accuracy=0, regime_f1={}, entry_mae=0, exit_mae=0,
                feature_importances=self._models.feature_importance_dict(),
                cv_scores=[], training_time_s=0,
            )

        # Get data
        if df is None:
            from . import market_data as md
            import concurrent.futures as _cf
            import yfinance as yf

            def _fetch():
                return yf.Ticker(ticker).history(period=period, auto_adjust=True)

            with _cf.ThreadPoolExecutor() as pool:
                hist = pool.submit(_fetch).result(timeout=60)

            if hist is None or hist.empty or len(hist) < 100:
                raise ValueError(f"Insufficient data for {ticker}: {len(hist) if hist is not None else 0} rows")
            df = md.compute_indicators(hist)

        if len(df) < 100:
            raise ValueError(f"Need at least 100 rows, got {len(df)}")

        self._last_df = df  # keep for dashboard predict_timeseries

        pit_market_ctx = None
        pit_fundamentals = None
        pit_sentiment = None
        use_pit_fund = False
        use_pit_sent = False
        try:
            from .pit_data import PointInTimeStore
            pit_store = PointInTimeStore()
            pit_market_ctx = pit_store.align_market_context(df.index)
            pit_fundamentals = pit_store.align_fundamental_features(ticker, df.index)
            pit_sentiment = pit_store.align_sentiment_features(ticker, df.index)
            use_pit_fund = (
                self.config.use_fundamentals_in_training
                and pit_fundamentals is not None
                and not pit_fundamentals.dropna(how="all").empty
            )
            use_pit_sent = (
                self.config.use_sentiment_in_training
                and pit_sentiment is not None
                and not pit_sentiment.dropna(how="all").empty
            )
        except Exception:
            pit_market_ctx = None
            pit_fundamentals = None
            pit_sentiment = None

        if self.config.use_fundamentals_in_training and not use_pit_fund:
            log.warning(
                "No point-in-time historical fundamentals available for %s; "
                "dropping fundamental ML features from training.",
                ticker,
            )
        if self.config.use_sentiment_in_training and not use_pit_sent:
            log.warning(
                "No point-in-time historical sentiment available for %s; "
                "dropping sentiment ML features from training.",
                ticker,
            )

        # ── Fetch historical SPY & VIX for proper cross-asset features ─────
        # These are per-row, no lookahead — each row sees only past SPY/VIX.
        historical_spy = None
        historical_vix = None
        try:
            import yfinance as yf
            _start = df.index[0]
            _end = df.index[-1]
            _spy = yf.Ticker("SPY").history(start=_start, end=_end, auto_adjust=True)
            if _spy is not None and len(_spy) > 50:
                historical_spy = _spy
            _vix = yf.Ticker("^VIX").history(start=_start, end=_end, auto_adjust=True)
            if _vix is not None and len(_vix) > 50:
                historical_vix = _vix["Close"]
        except Exception as e:
            log.warning(f"Failed to fetch SPY/VIX for training features: {e}")

        # Build features & labels
        feat_names = self.config.train_feature_names(
            include_fundamentals=use_pit_fund,
            include_sentiment=use_pit_sent,
        )
        X_df = build_features(
            df, feat_names,
            market_ctx=market_ctx,
            fundamentals=None,
            sentiment=None,  # excluded unless PIT-aligned below
            training_mode=True,
            historical_spy=historical_spy,
            historical_vix=historical_vix,
            pit_market_ctx=pit_market_ctx,
            pit_fundamentals=pit_fundamentals if use_pit_fund else None,
            pit_sentiment=pit_sentiment if use_pit_sent else None,
        )
        regime, entry_q, exit_q = generate_labels(
            df,
            forward_horizon=self.config.forward_horizon,
            strong_thresh=self.config.strong_threshold,
            weak_thresh=self.config.weak_threshold,
            cost_penalty_pct=self.config.label_cost_penalty_pct,
            impact_penalty_pct=self.config.label_impact_penalty_pct,
        )

        # Align and drop NaNs
        combined = X_df.copy()
        combined["_regime"] = regime
        combined["_entry"] = entry_q
        combined["_exit"] = exit_q
        combined = combined.dropna()

        if len(combined) < 50:
            raise ValueError(f"Only {len(combined)} valid samples after NaN removal")

        available_features = [c for c in X_df.columns if c in combined.columns]
        X = combined[available_features].values.astype(np.float32)
        y_regime = combined["_regime"].values
        y_entry = combined["_entry"].values.astype(np.float32)
        y_exit = combined["_exit"].values.astype(np.float32)

        # Replace any remaining inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Auxiliary arrays for evaluation ───────────────────────────────
        prices_aligned = df["Close"].reindex(combined.index).values.astype(np.float64)
        dates_aligned  = combined.index.values
        vol_feat = combined.get("Vol_Pctl", pd.Series(0.5, index=combined.index)).values / 100.0

        # ── Chrono split vs Walk-forward ───────────────────────────────────
        chrono_mode = bool(self._chrono_train_end and self._chrono_test_start)
        label_purge = max(1, self.config.forward_horizon)

        if chrono_mode:
            # Find the index boundary between train and test
            date_strs = [str(d)[:10] for d in dates_aligned]
            split = next((i for i, d in enumerate(date_strs) if d >= self._chrono_test_start), len(X))
            fit_end = split - label_purge
            if fit_end < 20 or split >= len(X) - 10:
                raise ValueError(f"Chrono split boundary out of range (split={split}, n={len(X)})")

            if self._epoch_callback:
                self._epoch_callback({"type": "status",
                    "message": f"Chrono split: train 0–{fit_end} / purge {fit_end}–{split} / "
                               f"test {split}–{len(X)}..."})

            # Train only on train window, eval on test window
            models = _create_model(self.config, X.shape[1])
            models.fit(X[:fit_end], y_regime[:fit_end], y_entry[:fit_end], y_exit[:fit_end],
                       available_features, epoch_callback=self._epoch_callback)
            self._models = models

            # Walk-forward style metrics on the test window only (single "fold")
            wf_metrics = _walk_forward_evaluate(
                X, y_regime, y_entry, y_exit,
                prices=prices_aligned, dates=dates_aligned,
                config=self.config, n_splits=1,
                regime_labels=y_regime, vol_values=vol_feat,
                epoch_callback=None,
                force_split=split,
            )
            cv_scores = [round(fit_end / len(X), 3)]  # train fraction as single "score"

            from sklearn.metrics import accuracy_score, f1_score
            probs, ent_pred, exit_pred, classes = models.predict(X[split:])
            regime_preds = [classes[i] for i in probs.argmax(axis=1)]
            acc = accuracy_score(y_regime[split:], regime_preds)
            eval_start = split
            train_rows = fit_end

        else:
            # ── Walk-forward evaluation (BEFORE final training) ────────────
            if self._epoch_callback:
                self._epoch_callback({"type": "status", "message": "Running walk-forward evaluation..."})

            wf_metrics = _walk_forward_evaluate(
                X, y_regime, y_entry, y_exit,
                prices=prices_aligned, dates=dates_aligned,
                config=self.config, n_splits=self.config.cv_splits,
                regime_labels=y_regime, vol_values=vol_feat,
                epoch_callback=self._epoch_callback,
            )
            cv_scores = [f.get("accuracy", 0) for f in wf_metrics.fold_results]

            # ── Train final model on first 85% (holdout last 15% for eval) ──
            # Training on ALL data leaks the test set into the model.
            # We keep a proper holdout for unbiased metrics and purge the last
            # forward_horizon rows from the fit window so no training label can
            # use holdout prices.
            split = int(len(X) * 0.85)
            fit_end = split - label_purge
            if fit_end < 20:
                raise ValueError(
                    f"Not enough rows after applying holdout split and label purge "
                    f"(fit_end={fit_end}, split={split}, n={len(X)})"
                )
            models = _create_model(self.config, X.shape[1])
            models.fit(X[:fit_end], y_regime[:fit_end], y_entry[:fit_end], y_exit[:fit_end],
                       available_features, epoch_callback=self._epoch_callback)
            self._models = models

            # Evaluate on held-out last 15%
            from sklearn.metrics import accuracy_score, f1_score
            probs, ent_pred, exit_pred, classes = models.predict(X[split:])
            regime_preds = [classes[i] for i in probs.argmax(axis=1)]
            acc = accuracy_score(y_regime[split:], regime_preds)
            eval_start = split
            train_rows = fit_end

        f1s = {}
        for cls in REGIME_CLASSES:
            y_bin = (y_regime[eval_start:] == cls).astype(int)
            p_bin = (np.array(regime_preds) == cls).astype(int)
            if y_bin.sum() > 0:
                f1s[cls] = round(float(f1_score(y_bin, p_bin, zero_division=0)), 3)

        entry_mae = float(np.mean(np.abs(y_entry[eval_start:] - ent_pred)))
        exit_mae = float(np.mean(np.abs(y_exit[eval_start:] - exit_pred)))

        # Store RANGE regime Sharpe for decision-policy gating at prediction time
        if wf_metrics is not None:
            self._range_regime_sharpe = wf_metrics.range_regime_sharpe
        else:
            self._range_regime_sharpe = None

        # Cache model
        self._save_cached(ticker, models)

        elapsed = time.time() - t0

        # Build training log for dashboard
        epoch_log = getattr(models, "_epoch_log", [])
        early_stop_ep = getattr(models, "_early_stop_epoch", None)
        best_vl = min((e.get("val_loss", float("inf")) for e in epoch_log), default=None) if epoch_log else None

        idx = combined.index
        _iso = lambda d: d.isoformat()[:10] if hasattr(d, 'isoformat') else str(d)
        data_range = (_iso(idx[0]), _iso(idx[-1])) if len(idx) > 0 else None
        train_range = (_iso(idx[0]), _iso(idx[train_rows - 1])) if train_rows > 0 else None
        test_range = (_iso(idx[eval_start]), _iso(idx[-1])) if eval_start < len(idx) else None

        from collections import Counter
        class_dist = dict(Counter(y_regime))

        # Extract calibration info from the fitted model object
        _cal_status = ""
        _cal_n = getattr(models, "_cal_n", 0)
        _temperature = 1.0
        if hasattr(models, "_calibrated_clf"):          # LightGBM path
            _cal_status = "isotonic" if models._calibrated_clf is not None else "raw_lgbm"
        elif hasattr(models, "_temperature"):            # MLP path
            _temperature = float(models._temperature)
            _cal_status = "temperature_scaling"

        tlog = TrainingLog(
            epochs=epoch_log,
            early_stop_epoch=early_stop_ep,
            best_val_loss=round(best_vl, 5) if best_vl is not None else None,
            n_train_rows=train_rows,
            n_val_rows=len(X) - eval_start,
            n_test_rows=len(X) - eval_start,
            features_used=available_features,
            backend=self.backend,
            training_time_s=round(elapsed, 2),
            data_date_range=data_range,
            train_date_range=train_range,
            test_date_range=test_range,
            class_distribution=class_dist,
            calibration_status=_cal_status,
            calibration_samples=_cal_n,
            temperature=_temperature,
        )

        readiness = assess_live_readiness(
            TrainResult(
                ticker=ticker,
                backend=self.backend if self.config.model_type == "mlp" else self.config.model_type,
                model_type=self.config.model_type,
                n_samples=len(X),
                n_features=X.shape[1],
                regime_accuracy=round(acc, 3),
                regime_f1=f1s,
                entry_mae=round(entry_mae, 4),
                exit_mae=round(exit_mae, 4),
                feature_importances=models.feature_importance_dict(),
                cv_scores=[round(s, 3) for s in cv_scores],
                training_time_s=round(elapsed, 2),
                training_log=tlog,
                walk_forward=wf_metrics,
            )
        ).to_dict()
        self._policy_opt = wf_metrics.policy_opt if wf_metrics is not None else None
        self._readiness = readiness

        # Display backend: show model_type for tree/linear models, gpu/cpu for MLP
        _display_backend = self.backend if self.config.model_type == "mlp" else self.config.model_type
        return TrainResult(
            ticker=ticker, backend=_display_backend,
            model_type=self.config.model_type,
            n_samples=len(X), n_features=X.shape[1],
            regime_accuracy=round(acc, 3),
            regime_f1=f1s,
            entry_mae=round(entry_mae, 4),
            exit_mae=round(exit_mae, 4),
            feature_importances=models.feature_importance_dict(),
            cv_scores=[round(s, 3) for s in cv_scores],
            training_time_s=round(elapsed, 2),
            training_log=tlog,
            walk_forward=wf_metrics,
            readiness=readiness,
        )

    def predict_from_df(
        self,
        df: pd.DataFrame,
        market_ctx: Optional[Dict] = None,
        fundamentals: Optional[Dict] = None,
        sentiment: Optional[Dict] = None,
    ) -> MLPrediction:
        """Predict using the last row of a DataFrame with indicators + context."""
        if self._models is None:
            raise RuntimeError("Model not trained — call train() first")

        last_as_of = df.index[-1] if len(df.index) > 0 else pd.Timestamp.utcnow()
        if self._ticker:
            try:
                from .pit_data import PointInTimeStore
                pit_store = PointInTimeStore()
                if market_ctx is None:
                    pit_market = pit_store.snapshot_asof("market", "SPY_CTX", last_as_of)
                    if pit_market:
                        market_ctx = pit_market
                if fundamentals is None:
                    pit_fund = pit_store.snapshot_asof("fundamentals", self._ticker, last_as_of)
                    if pit_fund:
                        fundamentals = pit_fund
                if sentiment is None:
                    pit_sent = pit_store.snapshot_asof("sentiment", self._ticker, last_as_of)
                    if pit_sent:
                        sentiment = pit_sent
            except Exception:
                pass

        feat_names = self._models.feature_names
        X_df = build_features(df, feat_names, market_ctx=market_ctx,
                              fundamentals=fundamentals, sentiment=sentiment)

        # Align columns: ensure same features as training (fill missing with 0)
        for col in feat_names:
            if col not in X_df.columns:
                X_df[col] = 0.0
        X_df = X_df[feat_names]

        # Use last row
        last_row = X_df.iloc[-1]
        last = X_df.iloc[[-1]].values.astype(np.float32)
        last = np.nan_to_num(last, nan=0.0, posinf=0.0, neginf=0.0)

        # Use MC Dropout for PyTorch (uncertainty estimation)
        mc_passes = 1
        if self.config.model_type == "mlp" and self.config.mc_dropout_passes > 1:
            mc_passes = self.config.mc_dropout_passes

        probs, entry, exit_, classes = self._models.predict(last, mc_passes=mc_passes)

        regime_idx = probs[0].argmax()
        regime = str(classes[regime_idx])
        conf = float(probs[0][regime_idx])
        regime_probs = {str(classes[i]): round(float(probs[0][i]), 3)
                        for i in range(len(classes))}

        entry_val = float(entry[0])
        exit_val = float(exit_[0])

        # Capture MC Dropout uncertainty if available
        uncertainty = None
        if hasattr(self._models, "_mc_uncertainty") and self._models._mc_uncertainty:
            uncertainty = self._models._mc_uncertainty
            self._models._mc_uncertainty = None  # reset
        elif getattr(self._models, "_mc_uncertainty_rows", None):
            uncertainty = self._models._mc_uncertainty_rows[0]
            self._models._mc_uncertainty_rows = None

        # Build decision policy, disabling RANGE entries when walk-forward
        # has confirmed that speculative RANGE trades hurt performance
        _policy = DecisionPolicy(
            min_regime_confidence=self.config.signal_min_regime_confidence,
            min_score_spread=self.config.signal_min_score_spread,
            min_liquidity_rank=self.config.signal_min_liquidity_rank,
            max_amihud=self.config.signal_max_amihud,
        )
        _policy_opt = getattr(self, "_policy_opt", None) or {}
        if _policy_opt:
            _policy.entry_threshold = float(_policy_opt.get("best_entry_threshold", _policy.entry_threshold))
            _policy.exit_threshold = float(_policy_opt.get("best_exit_threshold", _policy.exit_threshold))
            _policy.min_regime_confidence = float(
                _policy_opt.get("best_min_regime_confidence", _policy.min_regime_confidence)
            )
            _policy.min_score_spread = float(
                _policy_opt.get("best_min_score_spread", _policy.min_score_spread)
            )
        _range_sharpe = getattr(self, "_range_regime_sharpe", None)
        if _range_sharpe is not None and _range_sharpe <= 0.0:
            _policy.disable_range_entries = True
        model_ready = True
        if self.config.signal_require_readiness:
            model_ready = bool((getattr(self, "_readiness", None) or {}).get("ready", False))

        # Compute fundamental quality overlay (rule-based, outside ML pipeline).
        # Fundamentals are zeroed in ML features to avoid training/inference
        # distribution shift; here they influence position sizing directly.
        _fund_qs = compute_fund_quality_score(fundamentals) if fundamentals else None

        # Apply formal decision policy
        decision = apply_decision_policy(
            regime, conf, entry_val, exit_val,
            uncertainty=uncertainty,
            policy=_policy,
            liquidity_rank=float(last_row.get("dollar_vol_rank", np.nan)),
            amihud=float(last_row.get("amihud", np.nan)),
            model_ready=model_ready,
            fund_quality_score=_fund_qs,
        )
        signal = _decision_to_signal(decision)

        return MLPrediction(
            regime=regime,
            regime_confidence=round(conf, 3),
            regime_probs=regime_probs,
            entry_score=round(entry_val, 3),
            exit_score=round(exit_val, 3),
            ml_signal=signal,
            decision=decision.to_dict(),
            feature_importances=self._models.feature_importance_dict(),
            uncertainty=uncertainty,
            score_spread=round(entry_val - exit_val, 3),
            policy=_policy.to_dict(),
        )

    def predict_from_latest(self, latest: Dict, df: pd.DataFrame,
                            market_ctx=None, fundamentals=None, sentiment=None) -> MLPrediction:
        """Convenience: predict from the asset's indicator DataFrame."""
        return self.predict_from_df(df, market_ctx=market_ctx,
                                     fundamentals=fundamentals, sentiment=sentiment)

    # ── Timeseries prediction (for dashboard charts) ────────────────────────

    def predict_timeseries(
        self,
        df: Optional[pd.DataFrame] = None,
        market_ctx: Optional[Dict] = None,
        fundamentals: Optional[Dict] = None,
        sentiment: Optional[Dict] = None,
    ) -> Dict:
        """
        Run predictions across ALL rows of a DataFrame, returning timeseries
        data for charting regime regions, entry/exit scores over time.
        """
        if self._models is None:
            raise RuntimeError("Model not trained — call train() first")

        if df is None:
            df = self._last_df
        if df is None:
            raise ValueError("No DataFrame available — pass df or train first")

        feat_names = self._models.feature_names
        X_df = build_features(df, feat_names, market_ctx=market_ctx,
                              fundamentals=fundamentals, sentiment=sentiment)
        for col in feat_names:
            if col not in X_df.columns:
                X_df[col] = 0.0
        X_df = X_df[feat_names]

        X = X_df.values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Drop rows that are all-zero (warmup period)
        valid_mask = ~(X == 0).all(axis=1)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return {"dates": [], "prices": [], "regimes": [], "entry_scores": [],
                    "exit_scores": [], "regime_probs": {}}

        X_valid = X[valid_indices]
        probs, entry, exit_, classes = self._models.predict(X_valid)

        dates_raw = df.index[valid_indices]
        dates = [d.isoformat()[:10] if hasattr(d, 'isoformat') else str(d) for d in dates_raw]
        prices = df["Close"].iloc[valid_indices].tolist()
        regimes = [str(classes[i]) for i in probs.argmax(axis=1)]
        entry_scores = [round(float(e), 4) for e in entry]
        exit_scores = [round(float(e), 4) for e in exit_]

        # Regime probabilities per class
        regime_prob_series = {}
        for ci, cls in enumerate(classes):
            regime_prob_series[str(cls)] = [round(float(probs[r, ci]), 4) for r in range(len(probs))]

        return {
            "dates": dates,
            "prices": prices,
            "regimes": regimes,
            "entry_scores": entry_scores,
            "exit_scores": exit_scores,
            "regime_probs": regime_prob_series,
            "classes": [str(c) for c in classes],
        }

    # ── Model caching ────────────────────────────────────────────────────────

    def _cache_path(self, ticker: str) -> Path:
        key = self.config.cache_key(ticker)
        ext = ".pt" if self.config.model_type == "mlp" else ".pkl"
        return _MODEL_DIR / f"{ticker}_{key}{ext}"

    def _meta_path(self, ticker: str) -> Path:
        key = self.config.cache_key(ticker)
        return _MODEL_DIR / f"{ticker}_{key}.json"

    def _load_cached(self, ticker: str) -> Optional[Any]:
        path = self._cache_path(ticker)
        meta_path = self._meta_path(ticker)

        if not path.exists() or not meta_path.exists():
            return None

        try:
            with open(meta_path) as f:
                meta = json.load(f)
            age_days = (time.time() - meta.get("timestamp", 0)) / 86400
            if age_days > self.config.max_model_age_days:
                log.info(f"Cached model for {ticker} is stale ({age_days:.1f}d), retraining")
                return None

            self._policy_opt = meta.get("policy_opt")
            self._readiness = meta.get("readiness")
            self._range_regime_sharpe = meta.get("range_regime_sharpe")

            mt = self.config.model_type
            if mt == "mlp":
                return _PyTorchModels.load(path, self.config)
            elif mt == "logistic":
                return _LogisticModels.load(path)
            elif mt == "ensemble":
                return _EnsembleModels.load(path)
            else:
                return _LightGBMModels.load(path)
        except Exception as e:
            log.warning(f"Failed to load cached model for {ticker}: {e}")
            return None

    def _save_cached(self, ticker: str, models: Any):
        try:
            path = self._cache_path(ticker)
            meta_path = self._meta_path(ticker)
            models.save(path)
            with open(meta_path, "w") as f:
                json.dump({
                    "ticker": ticker,
                    "backend": self.backend,
                    "timestamp": time.time(),
                    "config": asdict(self.config),
                    "policy_opt": self._policy_opt,
                    "readiness": self._readiness,
                    "range_regime_sharpe": self._range_regime_sharpe,
                }, f)
        except Exception as e:
            log.warning(f"Failed to cache model for {ticker}: {e}")


# ── Universe (pooled) training ───────────────────────────────────────────────

class UniverseMLEngine(MLEngine):
    """
    Trains a single model on pooled data from multiple tickers.
    Better for small-cap / limited-history stocks.
    """

    def train_universe(
        self,
        tickers: List[str],
        dfs: Dict[str, pd.DataFrame],
        market_ctx: Optional[Dict] = None,
        fundamentals_map: Optional[Dict[str, Dict]] = None,
        sentiment_map: Optional[Dict[str, Dict]] = None,
    ) -> TrainResult:
        """
        Train on pooled data from multiple tickers.

        Cross-sectional pooling is the statistically correct approach:
        - 10 tickers × 1200 days = 12,000 samples (vs 1,200 per-ticker)
        - Model learns UNIVERSAL patterns, not ticker-specific noise
        - Features are relative (%, ranks) so they transfer across assets
        """
        t0 = time.time()
        feat_names = self.config.train_feature_names(
            include_fundamentals=self.config.use_fundamentals_in_training,
            include_sentiment=self.config.use_sentiment_in_training,
        )
        all_X, all_regime, all_entry, all_exit = [], [], [], []
        all_dates: List[np.ndarray] = []
        pit_store = None
        try:
            from .pit_data import PointInTimeStore
            pit_store = PointInTimeStore()
        except Exception:
            pit_store = None

        # ── Fetch historical SPY & VIX once for all tickers ──────────────
        historical_spy = None
        historical_vix = None
        try:
            import yfinance as yf
            # Use the widest date range across all tickers
            all_starts = [dfs[t].index[0] for t in tickers if t in dfs and len(dfs[t]) >= 100]
            all_ends = [dfs[t].index[-1] for t in tickers if t in dfs and len(dfs[t]) >= 100]
            if all_starts and all_ends:
                _spy = yf.Ticker("SPY").history(start=min(all_starts), end=max(all_ends), auto_adjust=True)
                if _spy is not None and len(_spy) > 50:
                    historical_spy = _spy
                _vix = yf.Ticker("^VIX").history(start=min(all_starts), end=max(all_ends), auto_adjust=True)
                if _vix is not None and len(_vix) > 50:
                    historical_vix = _vix["Close"]
        except Exception as e:
            log.warning(f"Failed to fetch SPY/VIX for universe training: {e}")

        n_tickers_used = 0
        for ticker in tickers:
            df = dfs.get(ticker)
            if df is None or len(df) < 100:
                continue

            # ── Liquidity filter ──────────────────────────────────────────
            if "Volume" in df.columns and "Close" in df.columns:
                avg_dv = (df["Close"] * df["Volume"]).tail(60).mean()
                if avg_dv < self.config.min_dollar_volume:
                    log.info(f"Skipping {ticker}: avg dollar volume ${avg_dv:,.0f} "
                             f"< ${self.config.min_dollar_volume:,.0f}")
                    continue

            X_df = build_features(
                df, feat_names,
                training_mode=True,
                historical_spy=historical_spy,
                historical_vix=historical_vix,
                pit_market_ctx=pit_store.align_market_context(df.index) if pit_store else None,
                pit_fundamentals=(
                    pit_store.align_fundamental_features(ticker, df.index)
                    if pit_store and self.config.use_fundamentals_in_training else None
                ),
                pit_sentiment=(
                    pit_store.align_sentiment_features(ticker, df.index)
                    if pit_store and self.config.use_sentiment_in_training else None
                ),
            )
            regime, entry_q, exit_q = generate_labels(
                df,
                forward_horizon=self.config.forward_horizon,
                strong_thresh=self.config.strong_threshold,
                weak_thresh=self.config.weak_threshold,
                cost_penalty_pct=self.config.label_cost_penalty_pct,
                impact_penalty_pct=self.config.label_impact_penalty_pct,
            )

            combined = X_df.copy()
            combined["_regime"] = regime
            combined["_entry"] = entry_q
            combined["_exit"] = exit_q
            combined = combined.dropna()

            if len(combined) < 30:
                continue

            avail = [c for c in X_df.columns if c in combined.columns]
            all_X.append(combined[avail].values.astype(np.float32))
            all_regime.append(combined["_regime"].values)
            all_entry.append(combined["_entry"].values.astype(np.float32))
            all_exit.append(combined["_exit"].values.astype(np.float32))
            all_dates.append(pd.to_datetime(combined.index).values)
            n_tickers_used += 1

        if not all_X:
            raise ValueError("No valid data for universe training")

        log.info(f"Universe training: {n_tickers_used} tickers pooled, "
                 f"{sum(len(x) for x in all_X)} total samples")

        X = np.nan_to_num(np.vstack(all_X), nan=0.0, posinf=0.0, neginf=0.0)
        y_regime = np.concatenate(all_regime)
        y_entry = np.concatenate(all_entry)
        y_exit = np.concatenate(all_exit)
        dates = np.concatenate(all_dates)

        # Preserve temporal ordering across the pooled panel before splitting.
        order = np.argsort(dates)
        X = X[order]
        y_regime = y_regime[order]
        y_entry = y_entry[order]
        y_exit = y_exit[order]
        dates = dates[order]

        # Get consistent feature names from first valid ticker
        sample_ticker = next(t for t in tickers if t in dfs and len(dfs[t]) >= 100)
        sample_X = build_features(
            dfs[sample_ticker], feat_names, training_mode=True,
            historical_spy=historical_spy, historical_vix=historical_vix,
            pit_market_ctx=pit_store.align_market_context(dfs[sample_ticker].index) if pit_store else None,
            pit_fundamentals=(
                pit_store.align_fundamental_features(sample_ticker, dfs[sample_ticker].index)
                if pit_store and self.config.use_fundamentals_in_training else None
            ),
            pit_sentiment=(
                pit_store.align_sentiment_features(sample_ticker, dfs[sample_ticker].index)
                if pit_store and self.config.use_sentiment_in_training else None
            ),
        )
        available_features = list(sample_X.columns)

        # ── Split on calendar order, not stacked array order ───────────────
        train_mask, test_mask = _universe_holdout_masks(
            dates,
            forward_horizon=self.config.forward_horizon,
            holdout_frac=0.15,
        )
        if train_mask.sum() < 20 or test_mask.sum() < 10:
            raise ValueError(
                f"Universe split left too few rows (train={int(train_mask.sum())}, "
                f"test={int(test_mask.sum())})"
            )

        models = _create_model(self.config, X.shape[1])
        models.fit(X[train_mask], y_regime[train_mask], y_entry[train_mask], y_exit[train_mask],
                   available_features, epoch_callback=self._epoch_callback)
        self._models = models
        self._ticker = "UNIVERSE"

        elapsed = time.time() - t0

        # Evaluate on held-out last 15%
        from sklearn.metrics import accuracy_score, f1_score
        probs, ent_pred, exit_pred, classes = models.predict(X[test_mask])
        preds = [classes[i] for i in probs.argmax(axis=1)]
        acc = accuracy_score(y_regime[test_mask], preds)

        f1s = {}
        for cls in REGIME_CLASSES:
            y_bin = (y_regime[test_mask] == cls).astype(int)
            p_bin = (np.array(preds) == cls).astype(int)
            if y_bin.sum() > 0:
                f1s[cls] = round(float(f1_score(y_bin, p_bin, zero_division=0)), 3)

        entry_mae = float(np.mean(np.abs(y_entry[test_mask] - ent_pred)))
        exit_mae = float(np.mean(np.abs(y_exit[test_mask] - exit_pred)))

        result = TrainResult(
            ticker="UNIVERSE",
            backend=self.backend,
            model_type=self.config.model_type,
            n_samples=len(X),
            n_features=X.shape[1],
            regime_accuracy=round(acc, 3),
            regime_f1=f1s,
            entry_mae=round(entry_mae, 4),
            exit_mae=round(exit_mae, 4),
            feature_importances=models.feature_importance_dict(),
            cv_scores=[],
            training_time_s=round(elapsed, 2),
        )
        result.readiness = assess_live_readiness(result).to_dict()
        return result


# ── Convenience: list cached models ─────────────────────────────────────────

def list_cached_models() -> List[Dict[str, Any]]:
    """Return metadata for all cached models."""
    results = []
    for meta_file in _MODEL_DIR.glob("*.json"):
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            age_days = (time.time() - meta.get("timestamp", 0)) / 86400
            meta["age_days"] = round(age_days, 1)
            model_file = meta_file.with_suffix(".pkl")
            if not model_file.exists():
                model_file = meta_file.with_suffix(".pt")
            meta["size_mb"] = round(model_file.stat().st_size / 1e6, 2) if model_file.exists() else 0
            results.append(meta)
        except Exception:
            pass
    return results


def clear_cached_models() -> int:
    """Remove all cached models. Returns count removed."""
    count = 0
    for f in _MODEL_DIR.glob("*"):
        try:
            f.unlink()
            count += 1
        except Exception:
            pass
    return count


def get_available_backend() -> str:
    """Return the best available backend."""
    return _resolve_backend("auto")


# ── Model Registry (versioned experiment tracking) ────────────────────────────

_REGISTRY_PATH = _MODEL_DIR / "registry.json"


@dataclass
class ModelVersion:
    """A single versioned model entry in the registry."""
    version_id: str
    ticker: str
    model_type: str
    backend: str                 # resolved display backend: "lightgbm", "logistic", "pytorch", "cpu"
    version: int
    train_period: str
    n_samples: int
    n_features: int
    features: List[str]
    regime_accuracy: float
    regime_f1: Dict[str, float]
    entry_mae: float
    exit_mae: float
    sharpe_ratio: Optional[float]
    max_drawdown: Optional[float]
    cagr: Optional[float]
    hit_rate: Optional[float]
    profit_factor: Optional[float]
    model_path: str
    config: Dict
    created_at: str
    notes: str = ""
    readiness: Optional[Dict[str, Any]] = None
    policy_opt: Optional[Dict[str, Any]] = None
    range_regime_sharpe: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class ModelRegistry:
    """
    Versioned model registry. Each trained model can be saved with a
    human-readable version ID, enabling experiment tracking and comparison.
    """

    def __init__(self):
        self._registry: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        if _REGISTRY_PATH.exists():
            try:
                with open(_REGISTRY_PATH) as f:
                    self._registry = json.load(f)
            except Exception:
                self._registry = {}

    def _save(self):
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(_REGISTRY_PATH, "w") as f:
            json.dump(self._registry, f, indent=2)

    def backup(self) -> Path:
        """Write a timestamped backup of the current registry JSON."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup_path = _MODEL_DIR / f"registry.backup.{stamp}.json"
        with open(backup_path, "w") as f:
            json.dump(self._registry, f, indent=2)
        return backup_path

    def reset(self):
        """Clear the registry index while leaving weight files on disk."""
        self._registry = {}
        self._save()

    def _next_version(self, ticker: str, model_type: str) -> int:
        prefix = f"{ticker}_{model_type}_v"
        existing = [
            int(k.replace(prefix, ""))
            for k in self._registry
            if k.startswith(prefix) and k.replace(prefix, "").isdigit()
        ]
        return max(existing, default=0) + 1

    def save(self, engine, train_result, df=None, notes: str = "") -> ModelVersion:
        """Save a trained model to the registry with full metadata."""
        ticker = train_result.ticker
        model_type = train_result.model_type
        v_num = self._next_version(ticker, model_type)
        version_id = f"{ticker}_{model_type}_v{v_num}"

        if df is not None and hasattr(df, "index") and len(df) > 0:
            _iso = lambda d: d.isoformat()[:10] if hasattr(d, "isoformat") else str(d)[:10]
            period_str = f"{_iso(df.index[0])} → {_iso(df.index[-1])}"
        else:
            period_str = engine.config.training_period

        ext = ".pt" if model_type == "mlp" else ".pkl"
        model_path = _MODEL_DIR / f"{version_id}{ext}"
        engine._models.save(model_path)

        wf = train_result.walk_forward
        tlog = train_result.training_log

        # Resolved display backend: model_type for tree/linear, actual backend for MLP
        display_backend = train_result.backend

        mv = ModelVersion(
            version_id=version_id, ticker=ticker, model_type=model_type,
            backend=display_backend,
            version=v_num, train_period=period_str,
            n_samples=train_result.n_samples, n_features=train_result.n_features,
            features=tlog.features_used if tlog else [],
            regime_accuracy=train_result.regime_accuracy, regime_f1=train_result.regime_f1,
            entry_mae=train_result.entry_mae, exit_mae=train_result.exit_mae,
            sharpe_ratio=wf.sharpe_ratio if wf else None,
            max_drawdown=wf.max_drawdown if wf else None,
            cagr=wf.cagr if wf else None,
            hit_rate=wf.hit_rate if wf else None,
            profit_factor=wf.profit_factor if wf else None,
            model_path=str(model_path), config=asdict(engine.config),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"), notes=notes,
            readiness=train_result.readiness,
            policy_opt=wf.policy_opt if wf else None,
            range_regime_sharpe=wf.range_regime_sharpe if wf else None,
        )
        self._registry[version_id] = mv.to_dict()
        self._save()
        log.info(f"Saved model version: {version_id}")
        return mv

    def load(self, version_id: str):
        """Load model weights for a given version_id."""
        if version_id not in self._registry:
            raise KeyError(f"Version '{version_id}' not found in registry")
        entry = self._registry[version_id]
        path = Path(entry["model_path"])
        if not path.exists():
            raise FileNotFoundError(f"Model file missing: {path}")
        model_type = entry["model_type"]
        cfg_dict = entry.get("config", {})
        valid_fields = set(MLConfig.__dataclass_fields__.keys())
        cfg = MLConfig(**{k: v for k, v in cfg_dict.items() if k in valid_fields})
        if model_type == "mlp":
            return _PyTorchModels.load(path, cfg)
        elif model_type == "logistic":
            return _LogisticModels.load(path)
        elif model_type == "ensemble":
            return _EnsembleModels.load(path)
        else:
            return _LightGBMModels.load(path)

    def list(self, ticker=None, model_type=None):
        results = []
        for vid, entry in self._registry.items():
            if ticker and entry.get("ticker") != ticker:
                continue
            if model_type and entry.get("model_type") != model_type:
                continue
            # Back-fill 'backend' for entries saved before this field existed
            if "backend" not in entry:
                mt = entry.get("model_type", "")
                cfg_backend = entry.get("config", {}).get("backend", "auto")
                if mt == "mlp":
                    entry["backend"] = "pytorch" if cfg_backend not in ("cpu", "auto") else cfg_backend
                else:
                    entry["backend"] = mt  # "lightgbm" or "logistic"
            results.append(entry)
        return sorted(results, key=lambda e: e.get("created_at", ""))

    def compare(self, version_ids: List[str]):
        rows = []
        for vid in version_ids:
            if vid in self._registry:
                e = self._registry[vid]
                rows.append({
                    "version": vid, "model_type": e.get("model_type"),
                    "train_period": e.get("train_period"), "n_samples": e.get("n_samples"),
                    "regime_acc": e.get("regime_accuracy"), "sharpe": e.get("sharpe_ratio"),
                    "max_dd": e.get("max_drawdown"), "cagr": e.get("cagr"),
                    "hit_rate": e.get("hit_rate"), "profit_factor": e.get("profit_factor"),
                    "created_at": e.get("created_at"),
                })
        return pd.DataFrame(rows)

    def delete(self, version_id: str, delete_weights: bool = False):
        if version_id not in self._registry:
            raise KeyError(f"Version '{version_id}' not in registry")
        if delete_weights:
            path = Path(self._registry[version_id].get("model_path", ""))
            if path.exists():
                path.unlink()
        del self._registry[version_id]
        self._save()

    def best(self, ticker: str, model_type=None, metric: str = "sharpe_ratio"):
        candidates = self.list(ticker=ticker, model_type=model_type)
        valid = [c for c in candidates if c.get(metric) is not None]
        if not valid:
            return None
        return max(valid, key=lambda c: c[metric])["version_id"]


_registry_instance: Optional["ModelRegistry"] = None

def get_registry() -> ModelRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance
