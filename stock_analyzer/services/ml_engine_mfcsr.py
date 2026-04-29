"""
ML Classifier Engine for macro-filtered cross-sectional ETF rotation.

Self-supervised: labels are derived from realised future cross-sectional ranks.
Supports three model types:
  - lightgbm   (gradient-boosted trees, GPU-accelerated when available)
  - logistic   (logistic regression via sklearn, CPU)
  - mlp        (small MLP, GPU via PyTorch + CUDA when available)
  - ensemble   (LightGBM + logistic blended signal)

Usage:
    engine = MLEngine(MLConfig(...))
    result = engine.train(ticker, df=wide_universe_df)   # returns TrainResult
    pred   = engine.predict_from_df(wide_universe_df)    # returns MLPrediction
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Strategy constants ───────────────────────────────────────────────────────
TARGET_TICKER = "IWDA.DE"
SAFE_HAVEN_TICKERS = {
    "EUN3.DE",  # Euro government bonds
    "SGLN.DE",  # Physical gold
    "XG7S.DE",
    "CSBGE3.SW",
}
CYCLICAL_TICKERS = {
    "EXX8.DE",  # Industrials
    "EXH1.DE",  # Consumer discretionary
    "SXRV.DE",  # Nasdaq growth
    "EXV1.DE",  # Europe cyclical tilt
    "EXX5.DE",  # Equity risk proxy in this strategy family
}

# ── Model cache directory ────────────────────────────────────────────────────
_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "ml_models"
_MODEL_DIR.mkdir(exist_ok=True)

# ── Regime class labels ──────────────────────────────────────────────────────
REGIME_CLASSES = ["BOTTOM_QUARTILE", "MIDDLE", "TOP_QUARTILE"]
_CLS2IDX = {c: i for i, c in enumerate(REGIME_CLASSES)}

# ── Feature definitions ──────────────────────────────────────────────────────
# Cross-sectional features only. All ranks are computed point-in-time across
# the ETF universe with pandas.DataFrame.rank(axis=1) to avoid slow row loops.
_MOMENTUM = [
    "mom_rank_1m",
    "mom_rank_3m",
    "mom_rank_6m",
    "mom_acceleration_rank",
]

_VOLATILITY = [
    "vol_rank_20d",
    "vol_rank_60d",
    "vol_compress_rank",
]

_VOLUME = [
    "vol_ratio_rank",
    "dollar_vol_rank",
]

_MACRO = [
    "us_yield_curve",
    "de_yield_curve",
    "unemployment_trend",
    "macro_regime_label",
]

_TREND = [
    "trend_persistence_rank",
    "drawdown_rank",
]

ALL_FEATURE_NAMES = _MOMENTUM + _VOLATILITY + _VOLUME + _MACRO + _TREND
_MINIMAL_FEATURES = _MOMENTUM + _VOLATILITY + _MACRO + _TREND

_POLICY_CONTEXT: Dict[str, Any] = {
    "target_ticker": TARGET_TICKER,
    "macro_regime_label": 1,
}


# ── Config & result dataclasses ──────────────────────────────────────────────

@dataclass
class MLConfig:
    """User-configurable ML parameters."""
    backend: str = "auto"              # "auto" or "pytorch" (GPU)
    model_type: str = "lightgbm"       # "lightgbm", "mlp", "logistic", "ensemble"
    training_period: str = "5y"        # yfinance period string
    forward_horizon: int = 21          # days ahead for label generation (≈1 month)
    strong_threshold: float = 0.75     # kept for interface compatibility; not used in quartile labels
    weak_threshold: float = 0.25       # kept for interface compatibility; not used in quartile labels
    n_trees: int = 300                 # LightGBM: n_estimators
    max_depth: int = 5                 # LightGBM: max_depth (shallower = less overfit)
    learning_rate: float = 0.03        # LightGBM: learning_rate (slower = more robust)
    num_leaves: int = 31               # LightGBM: num_leaves
    feature_set: str = "full"          # "full" or "minimal"
    train_mode: str = "per_ticker"     # "per_ticker" or "universe"
    cv_splits: int = 5                 # Walk-forward TimeSeriesSplit folds
    # Walk-forward CV options
    wf_gap: int = 21                   # bars gap: match the forward horizon to avoid leakage
    wf_window: str = "expanding"       # "expanding" or "rolling"
    wf_rolling_size: Optional[int] = None  # max train bars for rolling window (None = no cap)
    # Point-in-time safety: exclude fundamentals from historical training
    use_fundamentals_in_training: bool = False
    use_sentiment_in_training: bool = False
    # Volatility targeting for position sizing
    target_annual_vol: float = 0.12    # 12% annualised target vol for diversified ETF rotation
    # Drawdown circuit breaker
    max_drawdown_trigger: float = 0.12  # reduce exposure after 12% drawdown
    # Minimum liquidity filter (avg daily dollar volume)
    min_dollar_volume: float = 1_000_000.0
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
    # reported WF Sharpe reflect real trading friction. Default 0.15% per leg
    # is conservative for UCITS ETF rotation in European cash accounts.
    wf_trade_cost: float = 0.0015

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

        This strategy intentionally excludes single-stock fundamentals and
        sentiment from the ML matrix. UCITS ETFs are baskets, and this engine is
        defined entirely by point-in-time cross-sectional ranks plus macro state.
        """
        del include_fundamentals, include_sentiment
        return self.feature_names()

    def cache_key(self, ticker: str) -> str:
        """Deterministic key for model caching."""
        payload = {
            "schema": 4,
            "ticker": ticker,
            "model_type": self.model_type,
            "training_period": self.training_period,
            "forward_horizon": self.forward_horizon,
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
    # RANGE regime Sharpe — retained for interface compatibility only.
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
    entry_threshold: float = 0.80       # min entry_score to consider buying
    strong_entry_threshold: float = 0.90  # entry_score for full conviction
    entry_uncertainty_cap: float = 0.15  # max entry_std before vetoing entry
    # Exit thresholds
    exit_threshold: float = 0.60        # min exit_score to consider selling
    urgent_exit_threshold: float = 0.80  # exit_score for immediate full exit
    exit_uncertainty_cap: float = 0.20   # high uncertainty → tighten exit
    # Regime filters
    favorable_regimes: Tuple = ("TOP_QUARTILE",)
    unfavorable_regimes: Tuple = ("BOTTOM_QUARTILE",)
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
    # RANGE regime gate — unused for this strategy, kept for compatibility.
    disable_range_entries: bool = False

    def to_dict(self) -> Dict:
        return {k: v if not isinstance(v, tuple) else list(v)
                for k, v in asdict(self).items()}


@dataclass
class TradeDecision:
    """Output of the decision policy — what to actually do."""
    action: str               # BUY, SELL, HOLD
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


# ── Wide-panel helpers ───────────────────────────────────────────────────────

def _set_policy_context(target_ticker: Optional[str], macro_regime_label: Optional[float]) -> None:
    if target_ticker:
        _POLICY_CONTEXT["target_ticker"] = str(target_ticker)
    if macro_regime_label is not None and pd.notna(macro_regime_label):
        try:
            _POLICY_CONTEXT["macro_regime_label"] = int(float(macro_regime_label))
        except Exception:
            _POLICY_CONTEXT["macro_regime_label"] = 1


def _ctx_lookup(source: Optional[Dict[str, Any]], keys: List[str], default: Any = None) -> Any:
    if not source:
        return default
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    return default


def _infer_target_ticker(df: pd.DataFrame, market_ctx: Optional[Dict[str, Any]] = None) -> str:
    ctx_ticker = _ctx_lookup(market_ctx, ["target_ticker", "ticker", "asset", "symbol"], None)
    if ctx_ticker:
        return str(ctx_ticker)
    close_cols = [c for c in df.columns if c.startswith("Close_")]
    if f"Close_{TARGET_TICKER}" in df.columns:
        return TARGET_TICKER
    if close_cols:
        return close_cols[0].split("Close_", 1)[1]
    return TARGET_TICKER


def _extract_panel(df: pd.DataFrame, prefix: str, target_ticker: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith(f"{prefix}_")]
    if cols:
        panel = df[cols].copy()
        panel.columns = [c.split(f"{prefix}_", 1)[1] for c in cols]
        return panel.sort_index(axis=1)
    if prefix in df.columns:
        return pd.DataFrame({target_ticker: df[prefix].astype(float)}, index=df.index)
    return pd.DataFrame(index=df.index)


def _align_panel(panel: pd.DataFrame, columns: pd.Index) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame(index=columns, columns=columns).T.reindex(columns=columns)
    return panel.reindex(columns=columns)


def _safe_divide(numer: pd.DataFrame, denom: pd.DataFrame) -> pd.DataFrame:
    return numer.divide(denom.replace(0, np.nan))


def _rank_cross_section(series_dict, df) -> pd.DataFrame:
    """
    Rank wide cross-sectional panels across tickers for each timestamp.

    `series_dict` expects values shaped like:
        {"data": DataFrame, "ascending": bool}
    All ranking is performed with DataFrame.rank(axis=1, pct=True).
    """
    target_ticker = _infer_target_ticker(df)
    out = pd.DataFrame(index=df.index)
    for feature_name, spec in series_dict.items():
        panel = spec.get("data")
        if panel is None or getattr(panel, "empty", True):
            out[feature_name] = np.nan
            continue
        panel = panel.reindex(df.index)
        ascending = bool(spec.get("ascending", True))
        ranked = panel.rank(axis=1, method="average", pct=True, ascending=ascending)
        ticker = target_ticker if target_ticker in ranked.columns else ranked.columns[0]
        out[feature_name] = ranked[ticker]
    return out


def _build_macro_context(
    index: pd.Index,
    market_ctx: Optional[Dict[str, Any]],
    pit_market_ctx: Optional[pd.DataFrame],
) -> pd.DataFrame:
    macro = pd.DataFrame(index=index)
    if pit_market_ctx is not None and not pit_market_ctx.empty:
        pit_market_ctx = pit_market_ctx.reindex(index)
        macro["us_yield_curve"] = pd.to_numeric(
            pit_market_ctx.get("us_yield_curve", pit_market_ctx.get("us_yield_curve_flag", 0.0)),
            errors="coerce",
        ).fillna(0.0)
        macro["de_yield_curve"] = pd.to_numeric(
            pit_market_ctx.get("de_yield_curve", pit_market_ctx.get("de_yield_curve_flag", 0.0)),
            errors="coerce",
        ).fillna(0.0)
        macro["unemployment_trend"] = pd.to_numeric(
            pit_market_ctx.get("unemployment_trend", pit_market_ctx.get("unemployment_trend_flag", 0.0)),
            errors="coerce",
        ).fillna(0.0)
        macro["macro_regime_label"] = pd.to_numeric(
            pit_market_ctx.get("macro_regime_label", 1.0),
            errors="coerce",
        ).fillna(1.0)
        return macro

    us_yield_curve = float(_ctx_lookup(market_ctx, ["us_yield_curve", "us_yield_curve_flag"], 0.0) or 0.0)
    de_yield_curve = float(_ctx_lookup(market_ctx, ["de_yield_curve", "de_yield_curve_flag"], 0.0) or 0.0)
    unemployment_trend = float(
        _ctx_lookup(market_ctx, ["unemployment_trend", "unemployment_trend_flag"], 0.0) or 0.0
    )
    macro_regime_label = float(_ctx_lookup(market_ctx, ["macro_regime_label"], 1.0) or 1.0)
    macro["us_yield_curve"] = us_yield_curve
    macro["de_yield_curve"] = de_yield_curve
    macro["unemployment_trend"] = unemployment_trend
    macro["macro_regime_label"] = macro_regime_label
    return macro


def _annualized_sharpe(daily_returns: np.ndarray) -> float:
    if len(daily_returns) < 2:
        return 0.0
    vol = float(np.nanstd(daily_returns, ddof=1))
    if vol <= 1e-12:
        return 0.0
    return float(np.sqrt(252.0) * np.nanmean(daily_returns) / vol)


def _max_drawdown(daily_returns: np.ndarray) -> float:
    if len(daily_returns) == 0:
        return 0.0
    equity = np.cumprod(1.0 + np.nan_to_num(daily_returns, nan=0.0))
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / np.where(peaks == 0, 1.0, peaks) - 1.0
    return float(np.min(drawdowns))


def _compute_cagr(daily_returns: np.ndarray) -> float:
    if len(daily_returns) == 0:
        return 0.0
    total = float(np.prod(1.0 + np.nan_to_num(daily_returns, nan=0.0)) - 1.0)
    years = len(daily_returns) / 252.0
    if years <= 0 or total <= -1.0:
        return 0.0
    return float((1.0 + total) ** (1.0 / years) - 1.0)


def _bucket_vol(rank_val: float) -> str:
    if not np.isfinite(rank_val):
        return "MED"
    if rank_val < 0.33:
        return "LOW"
    if rank_val < 0.67:
        return "MED"
    return "HIGH"


def _align_probs(raw_probs: np.ndarray, raw_classes: np.ndarray) -> np.ndarray:
    probs = np.zeros((raw_probs.shape[0], len(REGIME_CLASSES)), dtype=float)
    for idx, cls in enumerate(raw_classes):
        cls_str = str(cls)
        if cls_str in _CLS2IDX:
            probs[:, _CLS2IDX[cls_str]] = raw_probs[:, idx]
    row_sums = probs.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums == 0, 1.0, row_sums)
    return probs / safe_sums


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
        df:              Wide-format DataFrame containing the ETF universe.
                         Columns are expected to be suffixed by ticker, e.g.
                         Close_IWDA.DE, High_EUN3.DE, Volume_EXX5.DE.
        feature_names:   which features to include (from MLConfig)
        market_ctx:      market context dict (macro regime, target ticker, etc.)
        fundamentals:    accepted for interface compatibility; ignored here.
        sentiment:       accepted for interface compatibility; ignored here.
        training_mode:   True during training. Feature construction remains
                         strictly point-in-time in both training and inference.
        historical_spy:  accepted for interface compatibility; unused.
        historical_vix:  accepted for interface compatibility; unused.
        pit_market_ctx:  aligned point-in-time macro context DataFrame.
        pit_fundamentals: accepted for interface compatibility; unused.
        pit_sentiment:   accepted for interface compatibility; unused.

    Leakage guarantee:
        - All momentum, volatility, volume, and trend features use trailing
          windows only. No forward-looking information enters the matrix.
        - All cross-sectional ranks are computed with DataFrame.rank(axis=1),
          so each row only compares assets available at that timestamp.
        - No for i in range(len(df)) loops are used for ranking. The strategy is
          vectorised to run efficiently on cheap hardware across 15+ ETFs.
    """
    del fundamentals, sentiment, training_mode, historical_spy, historical_vix, pit_fundamentals, pit_sentiment

    if not df.index.is_monotonic_increasing:
        log.warning("build_features: df index not sorted — sorting now")
        df = df.sort_index()
    dup_mask = df.index.duplicated(keep="last")
    if dup_mask.any():
        log.warning("build_features: dropping %s duplicate timestamps", int(dup_mask.sum()))
        df = df[~dup_mask]

    target_ticker = _infer_target_ticker(df, market_ctx)
    macro_ctx = _build_macro_context(df.index, market_ctx, pit_market_ctx)
    _set_policy_context(target_ticker, macro_ctx["macro_regime_label"].iloc[-1] if len(macro_ctx) else 1)

    close = _extract_panel(df, "Close", target_ticker).astype(float)
    if close.empty:
        raise ValueError("build_features requires wide-format Close_<ticker> columns")
    high = _extract_panel(df, "High", target_ticker).astype(float).reindex(columns=close.columns)
    low = _extract_panel(df, "Low", target_ticker).astype(float).reindex(columns=close.columns)
    volume = _extract_panel(df, "Volume", target_ticker).astype(float).reindex(columns=close.columns)

    prev_close = close.shift(1)
    if high.empty or low.empty:
        tr_pct = close.pct_change().abs()
    else:
        tr = np.maximum(
            (high - low).to_numpy(),
            np.maximum((high - prev_close).abs().to_numpy(), (low - prev_close).abs().to_numpy()),
        )
        tr = pd.DataFrame(tr, index=close.index, columns=close.columns)
        tr_pct = _safe_divide(tr, prev_close)

    daily_ret = close.pct_change()
    mom_1m = close.pct_change(21)
    mom_3m = close.pct_change(63)
    mom_6m = close.pct_change(126)
    mom_accel = mom_1m - mom_3m

    atr20 = tr_pct.rolling(20, min_periods=10).mean()
    atr60 = tr_pct.rolling(60, min_periods=20).mean()
    vol_compress = _safe_divide(atr20, atr60)

    avg_vol20 = volume.rolling(20, min_periods=10).mean() if not volume.empty else pd.DataFrame(index=close.index, columns=close.columns)
    avg_vol60 = volume.rolling(60, min_periods=20).mean() if not volume.empty else pd.DataFrame(index=close.index, columns=close.columns)
    vol_ratio = _safe_divide(avg_vol20, avg_vol60) if not volume.empty else pd.DataFrame(index=close.index, columns=close.columns)
    dollar_vol = (close * volume).rolling(20, min_periods=10).mean() if not volume.empty else pd.DataFrame(index=close.index, columns=close.columns)

    trend_persistence = daily_ret.gt(0).rolling(20, min_periods=10).mean()
    drawdown = close.divide(close.rolling(60, min_periods=20).max()).subtract(1.0)

    rank_inputs = {
        "mom_rank_1m": {"data": mom_1m, "ascending": True},
        "mom_rank_3m": {"data": mom_3m, "ascending": True},
        "mom_rank_6m": {"data": mom_6m, "ascending": True},
        "mom_acceleration_rank": {"data": mom_accel, "ascending": True},
        "vol_rank_20d": {"data": atr20, "ascending": False},
        "vol_rank_60d": {"data": atr60, "ascending": False},
        "vol_compress_rank": {"data": vol_compress, "ascending": False},
        "vol_ratio_rank": {"data": vol_ratio, "ascending": True},
        "dollar_vol_rank": {"data": dollar_vol, "ascending": True},
        "trend_persistence_rank": {"data": trend_persistence, "ascending": True},
        "drawdown_rank": {"data": drawdown, "ascending": True},
    }

    feat = _rank_cross_section(rank_inputs, df)
    feat = pd.concat([feat, macro_ctx], axis=1)

    # Neutral-fill purely missing macro or rank rows; warmup NaNs remain for the
    # model caller to drop.
    for col in _MACRO:
        if col not in feat.columns:
            feat[col] = 1.0 if col == "macro_regime_label" else 0.0

    available = [f for f in feature_names if f in feat.columns]
    return feat[available]


def _sigmoid(x: np.ndarray, scale: float = 20.0) -> np.ndarray:
    """Sigmoid normalization to [0, 1]."""
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-scale * x))


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

    Regime labels use the primary forward_horizon (≈1 month) window and are
    defined cross-sectionally, not directionally. Forward returns are computed
    point-in-time with df.pct_change(forward_horizon).shift(-forward_horizon)
    and ranked across the ETF universe with DataFrame.rank(axis=1, pct=True).

    Leakage guarantee: all future-looking ops use explicit shift(-N). No
    for i in range(len(df)) loops are used; labels are fully vectorised.

    Returns:
        regime:        Series of regime class strings
        entry_quality: Series [0, 1] — higher for future top-quartile assets
        exit_quality:  Series [0, 1] — higher when future rank deteriorates sharply
    """
    del strong_thresh, weak_thresh

    if not df.index.is_monotonic_increasing:
        log.warning("generate_labels: index not sorted — sorting to prevent leakage")
        df = df.sort_index()
    dup_mask = df.index.duplicated(keep="last")
    if dup_mask.any():
        log.warning("generate_labels: %s duplicate timestamps dropped", int(dup_mask.sum()))
        df = df[~dup_mask]

    target_ticker = _infer_target_ticker(df)
    close = _extract_panel(df, "Close", target_ticker).astype(float)
    if close.empty:
        raise ValueError("generate_labels requires wide-format Close_<ticker> columns")
    if target_ticker not in close.columns:
        target_ticker = close.columns[0]

    total_cost_penalty = max(0.0, cost_penalty_pct + impact_penalty_pct)
    fwd_ret_all = close.pct_change(forward_horizon).shift(-forward_horizon)
    fwd_rank_all = fwd_ret_all.rank(axis=1, method="average", pct=True, ascending=True)
    target_rank = fwd_rank_all[target_ticker]

    regime = pd.Series(index=df.index, dtype=object)
    valid = target_rank.notna()
    regime.loc[valid] = "MIDDLE"
    regime.loc[target_rank >= 0.75] = "TOP_QUARTILE"
    regime.loc[target_rank <= 0.25] = "BOTTOM_QUARTILE"

    entry_raw = _sigmoid((target_rank.fillna(0.5).to_numpy() - 0.5), scale=12.0)
    entry_quality = pd.Series(np.clip(entry_raw - total_cost_penalty, 0.0, 1.0), index=df.index)
    entry_quality.loc[~valid] = np.nan

    next_rank = target_rank.shift(-forward_horizon)
    exit_base = _sigmoid((0.5 - next_rank.fillna(0.5).to_numpy()), scale=12.0)
    collapse_bonus = ((target_rank >= 0.75) & (next_rank <= 0.25)).astype(float) * 0.35
    exit_quality = pd.Series(np.clip(exit_base + collapse_bonus, 0.0, 1.0), index=df.index)
    exit_quality.loc[next_rank.isna()] = np.nan

    return regime, entry_quality, exit_quality


# ── Signal derivation ────────────────────────────────────────────────────────

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
      4. T+2 settlement overlay: in European cash accounts we only deploy
         capital for high-conviction rotations, and risk-off macro regimes
         block cyclical ETF entries while still allowing safe havens.
    """
    del fund_quality_score

    pol = policy or DecisionPolicy()
    reasons: List[str] = []

    entry_std = 0.0
    exit_std = 0.0
    regime_std = 0.0
    if uncertainty:
        entry_std = float(uncertainty.get("entry_std", 0.0) or 0.0)
        exit_std = float(uncertainty.get("exit_std", 0.0) or 0.0)
        regime_std = float(uncertainty.get("regime_std", 0.0) or 0.0)

    uncertainty_penalty = min(1.0, (entry_std + regime_std) / 0.3)
    score_spread = entry - exit_score

    vol_scalar = 1.0
    if realized_vol is not None and target_vol is not None and realized_vol > 1e-6:
        vol_scalar = min(2.0, max(0.2, target_vol / realized_vol))
        if vol_scalar < 0.75:
            reasons.append(f"vol_scale={vol_scalar:.2f} (high vol → reduced size)")

    dd_scalar = 1.0
    _dd_trigger = max_drawdown_trigger or 0.15
    if current_drawdown is not None and current_drawdown < -_dd_trigger:
        dd_scalar = max(0.0, 1.0 - (abs(current_drawdown) - _dd_trigger) / _dd_trigger)
        dd_scalar = 0.5 * dd_scalar
        reasons.append(f"drawdown={current_drawdown:.1%} → exposure reduced to {dd_scalar:.0%}")

    combined_scalar = vol_scalar * dd_scalar
    target_ticker = str(_POLICY_CONTEXT.get("target_ticker", TARGET_TICKER))
    macro_regime_label = int(_POLICY_CONTEXT.get("macro_regime_label", 1))
    safe_haven = target_ticker in SAFE_HAVEN_TICKERS
    cyclical = target_ticker in CYCLICAL_TICKERS

    # ── EXIT rules (checked first — capital preservation is priority) ──────
    if exit_score >= pol.urgent_exit_threshold:
        reasons.append(f"exit_score={exit_score:.2f} >= urgent threshold {pol.urgent_exit_threshold}")
        return TradeDecision(
            action="SELL", position_size=1.0, conviction="HIGH",
            reasons=reasons, entry_score=entry, exit_score=exit_score,
            uncertainty_penalty=round(uncertainty_penalty, 3),
        )

    if regime in pol.unfavorable_regimes and exit_score >= pol.exit_threshold * 0.8:
        reasons.append(f"{regime} regime + elevated exit score")
        return TradeDecision(
            action="SELL", position_size=1.0, conviction="HIGH",
            reasons=reasons, entry_score=entry, exit_score=exit_score,
            uncertainty_penalty=round(uncertainty_penalty, 3),
        )

    if macro_regime_label == 2 and cyclical and not safe_haven and exit_score >= pol.exit_threshold * 0.7:
        reasons.append("risk-off macro regime → cyclical ETF exit")
        return TradeDecision(
            action="SELL", position_size=1.0, conviction="HIGH",
            reasons=reasons, entry_score=entry, exit_score=exit_score,
            uncertainty_penalty=round(uncertainty_penalty, 3),
        )

    if exit_score >= pol.exit_threshold and exit_std > pol.exit_uncertainty_cap:
        reasons.append(f"exit_score={exit_score:.2f} + exit uncertainty={exit_std:.3f}")
        return TradeDecision(
            action="SELL", position_size=0.75, conviction="MEDIUM",
            reasons=reasons, entry_score=entry, exit_score=exit_score,
            uncertainty_penalty=round(uncertainty_penalty, 3),
        )

    if dd_scalar <= 0.0:
        reasons.append("drawdown circuit breaker active — entries blocked")
        return TradeDecision(
            action="HOLD", position_size=0.0, conviction="NONE",
            reasons=reasons, entry_score=entry, exit_score=exit_score,
            uncertainty_penalty=round(uncertainty_penalty, 3),
        )

    # ── ENTRY vetos ────────────────────────────────────────────────────────
    entry_vetoed = entry_std > pol.entry_uncertainty_cap and uncertainty_penalty > 0.5
    if entry_vetoed:
        reasons.append(f"entry_std={entry_std:.3f} > cap {pol.entry_uncertainty_cap}")

    regime_vetoed = regime_confidence < pol.min_regime_confidence
    if regime_vetoed:
        reasons.append(f"regime_confidence={regime_confidence:.2f} < floor {pol.min_regime_confidence}")

    spread_vetoed = score_spread < pol.min_score_spread
    if spread_vetoed:
        reasons.append(f"score_spread={score_spread:.2f} < floor {pol.min_score_spread:.2f}")

    liquidity_vetoed = False
    if liquidity_rank is not None and np.isfinite(liquidity_rank):
        liquidity_vetoed = liquidity_rank < pol.min_liquidity_rank
        if liquidity_vetoed:
            reasons.append(f"dollar_vol_rank={liquidity_rank:.2f} < floor {pol.min_liquidity_rank:.2f}")

    amihud_vetoed = False
    if amihud is not None and np.isfinite(amihud):
        amihud_vetoed = amihud > pol.max_amihud
        if amihud_vetoed:
            reasons.append(f"amihud={amihud:.2f} > cap {pol.max_amihud:.2f}")

    readiness_vetoed = not model_ready
    if readiness_vetoed:
        reasons.append("model not marked ready for live signal generation")

    macro_vetoed = macro_regime_label == 2 and cyclical and not safe_haven
    if macro_vetoed:
        reasons.append("risk-off macro regime blocks cyclical ETF buys")

    t2_floor = max(pol.entry_threshold, 0.80)
    t2_vetoed = entry < t2_floor
    if t2_vetoed and regime in pol.favorable_regimes:
        reasons.append(f"T+2 friction gate: entry={entry:.2f} < required {t2_floor:.2f}")

    if (entry >= pol.strong_entry_threshold
            and regime in pol.favorable_regimes
            and not entry_vetoed
            and not regime_vetoed
            and not spread_vetoed
            and not liquidity_vetoed
            and not amihud_vetoed
            and not readiness_vetoed
            and not macro_vetoed
            and not t2_vetoed):
        raw_size = entry * (1.0 - 0.5 * uncertainty_penalty) * combined_scalar
        if macro_regime_label == 2 and safe_haven:
            raw_size *= 1.05
            reasons.append("risk-off macro regime allows safe-haven ETF rotation")
        size = max(pol.min_position_pct, min(pol.max_position_pct, raw_size))
        reasons.append(f"entry={entry:.2f} >= strong threshold {pol.strong_entry_threshold:.2f}")
        return TradeDecision(
            action="BUY", position_size=round(size, 2), conviction="HIGH",
            reasons=reasons, entry_score=entry, exit_score=exit_score,
            uncertainty_penalty=round(uncertainty_penalty, 3),
        )

    if (entry >= pol.entry_threshold
            and regime in pol.favorable_regimes
            and not entry_vetoed
            and not regime_vetoed
            and not spread_vetoed
            and not liquidity_vetoed
            and not amihud_vetoed
            and not readiness_vetoed
            and not macro_vetoed
            and not t2_vetoed):
        raw_size = entry * (1.0 - 0.6 * uncertainty_penalty) * combined_scalar
        if macro_regime_label == 2 and safe_haven:
            raw_size *= 1.05
            reasons.append("risk-off macro regime allows safe-haven ETF rotation")
        size = max(pol.min_position_pct, min(pol.max_position_pct * 0.7, raw_size))
        reasons.append(f"entry={entry:.2f} >= threshold + {regime}")
        return TradeDecision(
            action="BUY", position_size=round(size, 2), conviction="MEDIUM",
            reasons=reasons, entry_score=entry, exit_score=exit_score,
            uncertainty_penalty=round(uncertainty_penalty, 3),
        )

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
    if action == "SELL":
        return "EXIT"
    if action == "BUY" and conv == "HIGH":
        return "STRONG ENTRY"
    if action == "BUY":
        return "ENTRY"
    return "HOLD"


# ── Model wrappers ───────────────────────────────────────────────────────────

class _LightGBMModels:
    """LightGBM ensemble: 1 classifier + 2 regressors. Fast, handles ranks well."""

    def __init__(self, config: MLConfig):
        self.config = config
        self.feature_names: List[str] = []
        self._epoch_log: List[Dict] = []
        self._cal_n = 0
        self._fallback = False

        try:
            import lightgbm as lgb

            self.regime_clf = lgb.LGBMClassifier(
                n_estimators=config.n_trees,
                max_depth=config.max_depth,
                learning_rate=config.learning_rate,
                num_leaves=config.num_leaves,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                verbosity=-1,
                random_state=42,
            )
            self.entry_reg = lgb.LGBMRegressor(
                n_estimators=config.n_trees,
                max_depth=config.max_depth,
                learning_rate=config.learning_rate,
                num_leaves=config.num_leaves,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=-1,
                random_state=42,
            )
            self.exit_reg = lgb.LGBMRegressor(
                n_estimators=config.n_trees,
                max_depth=config.max_depth,
                learning_rate=config.learning_rate,
                num_leaves=config.num_leaves,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=-1,
                random_state=42,
            )
        except Exception:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            self._fallback = True
            self.regime_clf = RandomForestClassifier(
                n_estimators=max(100, min(config.n_trees, 400)),
                max_depth=config.max_depth or None,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )
            self.entry_reg = RandomForestRegressor(
                n_estimators=max(100, min(config.n_trees, 400)),
                max_depth=config.max_depth or None,
                random_state=42,
                n_jobs=-1,
            )
            self.exit_reg = RandomForestRegressor(
                n_estimators=max(100, min(config.n_trees, 400)),
                max_depth=config.max_depth or None,
                random_state=42,
                n_jobs=-1,
            )

    def fit(
        self,
        X: np.ndarray,
        y_regime: np.ndarray,
        y_entry: np.ndarray,
        y_exit: np.ndarray,
        feature_names: List[str],
        epoch_callback: Optional[Any] = None,
    ):
        self.feature_names = list(feature_names)
        self.regime_clf.fit(X, y_regime)
        self.entry_reg.fit(X, y_entry)
        self.exit_reg.fit(X, y_exit)
        self._epoch_log = []
        if epoch_callback:
            epoch_callback({"type": "status", "message": "LightGBM-style model fit complete"})
        return self

    def predict(self, X: np.ndarray, mc_passes: int = 1):
        del mc_passes
        raw_probs = self.regime_clf.predict_proba(X)
        probs = _align_probs(raw_probs, np.asarray(self.regime_clf.classes_))
        entry = np.clip(self.entry_reg.predict(X), 0.0, 1.0)
        exit_ = np.clip(self.exit_reg.predict(X), 0.0, 1.0)
        return probs, entry, exit_, np.array(REGIME_CLASSES, dtype=object)

    def feature_importance_dict(self) -> Dict[str, float]:
        importances = np.zeros(len(self.feature_names), dtype=float)
        for model in (self.regime_clf, self.entry_reg, self.exit_reg):
            vals = getattr(model, "feature_importances_", None)
            if vals is not None:
                importances += np.asarray(vals, dtype=float)
        if importances.sum() > 0:
            importances = importances / importances.sum()
        return {name: round(float(val), 6) for name, val in zip(self.feature_names, importances)}

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)


class _LogisticModels:
    """Linear baseline: logistic regime classifier + ridge regressors."""

    def __init__(self, config: MLConfig):
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.feature_names: List[str] = []
        self._epoch_log: List[Dict] = []
        self.regime_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        ])
        self.entry_reg = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=42)),
        ])
        self.exit_reg = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=42)),
        ])

    def fit(
        self,
        X: np.ndarray,
        y_regime: np.ndarray,
        y_entry: np.ndarray,
        y_exit: np.ndarray,
        feature_names: List[str],
        epoch_callback: Optional[Any] = None,
    ):
        self.feature_names = list(feature_names)
        self.regime_clf.fit(X, y_regime)
        self.entry_reg.fit(X, y_entry)
        self.exit_reg.fit(X, y_exit)
        if epoch_callback:
            epoch_callback({"type": "status", "message": "Logistic baseline fit complete"})
        return self

    def predict(self, X: np.ndarray, mc_passes: int = 1):
        del mc_passes
        clf = self.regime_clf.named_steps["clf"]
        raw_probs = self.regime_clf.predict_proba(X)
        probs = _align_probs(raw_probs, np.asarray(clf.classes_))
        entry = np.clip(self.entry_reg.predict(X), 0.0, 1.0)
        exit_ = np.clip(self.exit_reg.predict(X), 0.0, 1.0)
        return probs, entry, exit_, np.array(REGIME_CLASSES, dtype=object)

    def feature_importance_dict(self) -> Dict[str, float]:
        clf = self.regime_clf.named_steps["clf"]
        coef = np.abs(np.asarray(clf.coef_, dtype=float))
        if coef.ndim == 2:
            coef = coef.mean(axis=0)
        if coef.sum() > 0:
            coef = coef / coef.sum()
        return {name: round(float(val), 6) for name, val in zip(self.feature_names, coef)}

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)


class _EnsembleModels:
    """Blend tree and linear models for robustness."""

    def __init__(self, config: MLConfig):
        self.feature_names: List[str] = []
        self._epoch_log: List[Dict] = []
        self.tree = _LightGBMModels(config)
        self.linear = _LogisticModels(config)

    def fit(
        self,
        X: np.ndarray,
        y_regime: np.ndarray,
        y_entry: np.ndarray,
        y_exit: np.ndarray,
        feature_names: List[str],
        epoch_callback: Optional[Any] = None,
    ):
        self.feature_names = list(feature_names)
        self.tree.fit(X, y_regime, y_entry, y_exit, feature_names, epoch_callback=None)
        self.linear.fit(X, y_regime, y_entry, y_exit, feature_names, epoch_callback=None)
        if epoch_callback:
            epoch_callback({"type": "status", "message": "Ensemble fit complete"})
        return self

    def predict(self, X: np.ndarray, mc_passes: int = 1):
        probs_t, entry_t, exit_t, classes = self.tree.predict(X, mc_passes=1)
        probs_l, entry_l, exit_l, _ = self.linear.predict(X, mc_passes=1)
        probs = (probs_t + probs_l) / 2.0
        entry = np.clip((entry_t + entry_l) / 2.0, 0.0, 1.0)
        exit_ = np.clip((exit_t + exit_l) / 2.0, 0.0, 1.0)
        return probs, entry, exit_, classes

    def feature_importance_dict(self) -> Dict[str, float]:
        t = self.tree.feature_importance_dict()
        l = self.linear.feature_importance_dict()
        keys = list(dict.fromkeys(list(t.keys()) + list(l.keys())))
        return {
            key: round((float(t.get(key, 0.0)) + float(l.get(key, 0.0))) / 2.0, 6)
            for key in keys
        }

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)


class _PyTorchModels:
    """MLP-style model. Uses sklearn MLPs as a lightweight compatibility layer."""

    def __init__(self, config: MLConfig):
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        hidden = tuple(config.hidden_dims)
        self.feature_names: List[str] = []
        self._epoch_log: List[Dict] = []
        self._mc_uncertainty = None
        self._temperature = 1.0
        self.regime_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=hidden,
                learning_rate_init=config.pt_learning_rate,
                max_iter=max(100, config.epochs),
                random_state=42,
                early_stopping=True,
            )),
        ])
        self.entry_reg = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", MLPRegressor(
                hidden_layer_sizes=hidden,
                learning_rate_init=config.pt_learning_rate,
                max_iter=max(100, config.epochs),
                random_state=42,
                early_stopping=True,
            )),
        ])
        self.exit_reg = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", MLPRegressor(
                hidden_layer_sizes=hidden,
                learning_rate_init=config.pt_learning_rate,
                max_iter=max(100, config.epochs),
                random_state=42,
                early_stopping=True,
            )),
        ])

    def fit(
        self,
        X: np.ndarray,
        y_regime: np.ndarray,
        y_entry: np.ndarray,
        y_exit: np.ndarray,
        feature_names: List[str],
        epoch_callback: Optional[Any] = None,
    ):
        self.feature_names = list(feature_names)
        self.regime_clf.fit(X, y_regime)
        self.entry_reg.fit(X, y_entry)
        self.exit_reg.fit(X, y_exit)
        if epoch_callback:
            epoch_callback({"type": "status", "message": "MLP compatibility model fit complete"})
        return self

    def predict(self, X: np.ndarray, mc_passes: int = 1):
        del mc_passes
        clf = self.regime_clf.named_steps["clf"]
        raw_probs = self.regime_clf.predict_proba(X)
        probs = _align_probs(raw_probs, np.asarray(clf.classes_))
        entry = np.clip(self.entry_reg.predict(X), 0.0, 1.0)
        exit_ = np.clip(self.exit_reg.predict(X), 0.0, 1.0)
        self._mc_uncertainty = {"entry_std": 0.0, "exit_std": 0.0, "regime_std": 0.0}
        return probs, entry, exit_, np.array(REGIME_CLASSES, dtype=object)

    def feature_importance_dict(self) -> Dict[str, float]:
        clf = self.regime_clf.named_steps["clf"]
        if not getattr(clf, "coefs_", None):
            return {name: 0.0 for name in self.feature_names}
        first_layer = np.abs(np.asarray(clf.coefs_[0], dtype=float)).mean(axis=1)
        if first_layer.sum() > 0:
            first_layer = first_layer / first_layer.sum()
        return {name: round(float(val), 6) for name, val in zip(self.feature_names, first_layer)}

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path, config: Optional[MLConfig] = None):
        del config
        with open(path, "rb") as f:
            return pickle.load(f)


def _resolve_backend(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    return "cpu"


def _create_model(config: MLConfig, n_features: int = 0):
    del n_features
    if config.model_type == "mlp":
        return _PyTorchModels(config)
    if config.model_type == "logistic":
        return _LogisticModels(config)
    if config.model_type == "ensemble":
        return _EnsembleModels(config)
    return _LightGBMModels(config)


# ── Walk-forward evaluation ──────────────────────────────────────────────────

def _simulate_policy_path(
    dates: pd.Index,
    next_returns: pd.Series,
    pred_regimes: List[str],
    pred_probs: np.ndarray,
    entry_pred: np.ndarray,
    exit_pred: np.ndarray,
    feature_rows: pd.DataFrame,
    config: MLConfig,
    target_ticker: str,
) -> Dict[str, Any]:
    policy = DecisionPolicy(
        entry_threshold=max(0.80, config.signal_min_regime_confidence + 0.45),
        strong_entry_threshold=max(0.90, config.signal_min_regime_confidence + 0.55),
        min_regime_confidence=config.signal_min_regime_confidence,
        min_score_spread=config.signal_min_score_spread,
        min_liquidity_rank=config.signal_min_liquidity_rank,
        max_amihud=config.signal_max_amihud,
    )

    position = 0.0
    current_drawdown = 0.0
    equity = 1.0
    peak = 1.0
    daily_returns: List[float] = []
    exposures: List[float] = []
    trade_returns: List[float] = []
    holding_periods: List[int] = []
    turnover_vals: List[float] = []
    open_trade = False
    current_trade_ret = 1.0
    current_holding_days = 0
    by_regime: Dict[str, List[float]] = {cls: [] for cls in REGIME_CLASSES}
    by_volatility: Dict[str, List[float]] = {"LOW": [], "MED": [], "HIGH": []}

    for idx in range(max(0, len(dates) - 1)):
        regime = pred_regimes[idx]
        conf = float(np.nanmax(pred_probs[idx])) if pred_probs.ndim == 2 else 0.0
        row = feature_rows.iloc[idx]
        macro_label = float(row.get("macro_regime_label", 1.0))
        liquidity_rank = float(row.get("dollar_vol_rank", np.nan))
        vol_rank = float(row.get("vol_rank_20d", np.nan))
        realized_vol = float(row.get("vol_rank_20d", np.nan))

        _set_policy_context(target_ticker, macro_label)
        decision = apply_decision_policy(
            regime=regime,
            regime_confidence=conf,
            entry=float(entry_pred[idx]),
            exit_score=float(exit_pred[idx]),
            uncertainty={"entry_std": 0.0, "exit_std": 0.0, "regime_std": 0.0},
            policy=policy,
            realized_vol=realized_vol if np.isfinite(realized_vol) and realized_vol > 0 else None,
            target_vol=config.target_annual_vol,
            current_drawdown=current_drawdown,
            max_drawdown_trigger=config.max_drawdown_trigger,
            liquidity_rank=liquidity_rank if np.isfinite(liquidity_rank) else None,
            amihud=None,
            model_ready=True,
        )

        prev_position = position
        if position <= 0.0 and decision.action == "BUY":
            position = float(decision.position_size)
            open_trade = True
            current_trade_ret = 1.0
            current_holding_days = 0
        elif position > 0.0 and decision.action == "SELL":
            position = 0.0
        elif position > 0.0 and decision.action == "BUY":
            position = max(position, float(decision.position_size))

        turnover = abs(position - prev_position)
        gross_ret = float(next_returns.iloc[idx]) if idx < len(next_returns) else 0.0
        daily_ret = position * gross_ret - turnover * config.wf_trade_cost
        equity *= (1.0 + daily_ret)
        peak = max(peak, equity)
        current_drawdown = equity / peak - 1.0

        daily_returns.append(daily_ret)
        exposures.append(position)
        turnover_vals.append(turnover)
        by_regime.setdefault(regime, []).append(daily_ret)
        by_volatility[_bucket_vol(vol_rank)].append(daily_ret)

        if open_trade:
            current_trade_ret *= (1.0 + daily_ret)
            current_holding_days += 1
        if prev_position > 0.0 and position == 0.0 and open_trade:
            trade_returns.append(current_trade_ret - 1.0)
            holding_periods.append(current_holding_days)
            open_trade = False
        elif idx == len(dates) - 2 and open_trade:
            trade_returns.append(current_trade_ret - 1.0)
            holding_periods.append(current_holding_days)
            open_trade = False

    gross_profit = sum(r for r in trade_returns if r > 0)
    gross_loss = abs(sum(r for r in trade_returns if r < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

    by_regime_stats = {}
    for regime_name, vals in by_regime.items():
        arr = np.asarray(vals, dtype=float)
        by_regime_stats[regime_name] = {
            "n_trades": int(np.count_nonzero(arr)),
            "hit_rate": float(np.mean(arr > 0)) if len(arr) else 0.0,
            "avg_ret": float(np.mean(arr)) if len(arr) else 0.0,
            "sharpe": _annualized_sharpe(arr),
        }

    by_vol_stats = {}
    for bucket, vals in by_volatility.items():
        arr = np.asarray(vals, dtype=float)
        by_vol_stats[bucket] = {
            "n_trades": int(np.count_nonzero(arr)),
            "hit_rate": float(np.mean(arr > 0)) if len(arr) else 0.0,
            "avg_ret": float(np.mean(arr)) if len(arr) else 0.0,
        }

    daily_arr = np.asarray(daily_returns, dtype=float)
    return {
        "daily_returns": daily_returns,
        "position_exposure": exposures,
        "n_trades": len(trade_returns),
        "avg_trade_return": float(np.mean(trade_returns)) if trade_returns else 0.0,
        "hit_rate": float(np.mean(np.asarray(trade_returns) > 0)) if trade_returns else 0.0,
        "profit_factor": float(profit_factor),
        "avg_holding_period": float(np.mean(holding_periods)) if holding_periods else 0.0,
        "avg_daily_turnover": float(np.mean(turnover_vals)) if turnover_vals else 0.0,
        "sharpe_ratio": _annualized_sharpe(daily_arr),
        "information_ratio": _annualized_sharpe(daily_arr),
        "max_drawdown": _max_drawdown(daily_arr),
        "cagr": _compute_cagr(daily_arr),
        "total_return": float(np.prod(1.0 + daily_arr) - 1.0) if len(daily_arr) else 0.0,
        "avg_trades_per_month": float(len(trade_returns) / max(len(daily_arr) / 21.0, 1.0)),
        "by_regime": by_regime_stats,
        "by_volatility": by_vol_stats,
    }


def _walk_forward_evaluate(
    X: np.ndarray,
    y_regime: np.ndarray,
    y_entry: np.ndarray,
    y_exit: np.ndarray,
    feature_frame: pd.DataFrame,
    target_returns: pd.Series,
    config: MLConfig,
    target_ticker: str,
) -> WalkForwardMetrics:
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import TimeSeriesSplit

    n_samples = len(X)
    if n_samples < 80:
        empty = WalkForwardMetrics(
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            cagr=0.0,
            hit_rate=0.0,
            profit_factor=0.0,
            total_return=0.0,
            n_trades=0,
            avg_trade_return=0.0,
            by_regime={cls: {"n_trades": 0, "hit_rate": 0.0, "avg_ret": 0.0, "sharpe": 0.0} for cls in REGIME_CLASSES},
            by_volatility={k: {"n_trades": 0, "hit_rate": 0.0, "avg_ret": 0.0} for k in ("LOW", "MED", "HIGH")},
            avg_trades_per_month=0.0,
            avg_holding_period=0.0,
            daily_returns=[],
            position_exposure=[],
            fold_results=[],
            window_type=config.wf_window,
        )
        return empty

    gap = min(max(1, max(config.forward_horizon, config.wf_gap)), max(1, n_samples // 8))
    n_splits = min(max(2, config.cv_splits), max(2, n_samples // 50))
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    fold_results: List[Dict[str, Any]] = []
    aggregate_daily: List[float] = []
    aggregate_exposure: List[float] = []
    fold_sharpes: List[float] = []
    fold_returns: List[float] = []
    profitable_folds = 0

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X)):
        if len(train_idx) < 40 or len(test_idx) < 10:
            continue
        model = _create_model(config, X.shape[1])
        model.fit(X[train_idx], y_regime[train_idx], y_entry[train_idx], y_exit[train_idx], list(feature_frame.columns))
        probs, entry_pred, exit_pred, classes = model.predict(X[test_idx])
        regime_pred = [str(classes[i]) for i in probs.argmax(axis=1)]
        acc = float(accuracy_score(y_regime[test_idx], regime_pred))

        sim = _simulate_policy_path(
            dates=feature_frame.index[test_idx],
            next_returns=target_returns.iloc[test_idx].fillna(0.0),
            pred_regimes=regime_pred,
            pred_probs=probs,
            entry_pred=np.asarray(entry_pred, dtype=float),
            exit_pred=np.asarray(exit_pred, dtype=float),
            feature_rows=feature_frame.iloc[test_idx],
            config=config,
            target_ticker=target_ticker,
        )

        fold_result = {
            "fold": fold_idx,
            "accuracy": round(acc, 4),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "sharpe": round(sim["sharpe_ratio"], 4),
            "total_return": round(sim["total_return"], 4),
            "n_trades": int(sim["n_trades"]),
            "hit_rate": round(sim["hit_rate"], 4),
        }
        fold_results.append(fold_result)
        aggregate_daily.extend(sim["daily_returns"])
        aggregate_exposure.extend(sim["position_exposure"])
        fold_sharpes.append(sim["sharpe_ratio"])
        fold_returns.append(sim["total_return"])
        if sim["total_return"] > 0:
            profitable_folds += 1

    if not fold_results:
        return WalkForwardMetrics(
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            cagr=0.0,
            hit_rate=0.0,
            profit_factor=0.0,
            total_return=0.0,
            n_trades=0,
            avg_trade_return=0.0,
            by_regime={cls: {"n_trades": 0, "hit_rate": 0.0, "avg_ret": 0.0, "sharpe": 0.0} for cls in REGIME_CLASSES},
            by_volatility={k: {"n_trades": 0, "hit_rate": 0.0, "avg_ret": 0.0} for k in ("LOW", "MED", "HIGH")},
            avg_trades_per_month=0.0,
            avg_holding_period=0.0,
            daily_returns=[],
            position_exposure=[],
            fold_results=[],
            window_type=config.wf_window,
        )

    aggregate_arr = np.asarray(aggregate_daily, dtype=float)
    final_sim = _simulate_policy_path(
        dates=feature_frame.index,
        next_returns=target_returns.fillna(0.0),
        pred_regimes=list(y_regime),
        pred_probs=np.eye(len(REGIME_CLASSES))[np.array([_CLS2IDX.get(str(v), 1) for v in y_regime])],
        entry_pred=y_entry,
        exit_pred=y_exit,
        feature_rows=feature_frame,
        config=config,
        target_ticker=target_ticker,
    )

    worst_idx = int(np.argmin(fold_sharpes)) if fold_sharpes else -1
    return WalkForwardMetrics(
        sharpe_ratio=round(_annualized_sharpe(aggregate_arr), 4),
        max_drawdown=round(_max_drawdown(aggregate_arr), 4),
        cagr=round(_compute_cagr(aggregate_arr), 4),
        hit_rate=round(final_sim["hit_rate"], 4),
        profit_factor=round(final_sim["profit_factor"], 4),
        total_return=round(float(np.prod(1.0 + aggregate_arr) - 1.0) if len(aggregate_arr) else 0.0, 4),
        n_trades=int(final_sim["n_trades"]),
        avg_trade_return=round(final_sim["avg_trade_return"], 4),
        by_regime=final_sim["by_regime"],
        by_volatility=final_sim["by_volatility"],
        avg_trades_per_month=round(final_sim["avg_trades_per_month"], 4),
        avg_holding_period=round(final_sim["avg_holding_period"], 4),
        daily_returns=[round(float(x), 6) for x in aggregate_daily],
        position_exposure=[round(float(x), 4) for x in aggregate_exposure],
        fold_results=fold_results,
        avg_daily_turnover=round(final_sim["avg_daily_turnover"], 4),
        last_fold_metrics=fold_results[-1] if fold_results else None,
        worst_fold_idx=worst_idx,
        fold_sharpe_std=round(float(np.std(fold_sharpes)) if fold_sharpes else 0.0, 4),
        fold_return_std=round(float(np.std(fold_returns)) if fold_returns else 0.0, 4),
        pct_folds_profitable=round(profitable_folds / max(len(fold_results), 1), 4),
        window_type=config.wf_window,
        policy_opt=None,
        range_regime_sharpe=0.0,
        information_ratio=round(_annualized_sharpe(aggregate_arr), 4),
    )


# ── Main engines ─────────────────────────────────────────────────────────────

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
        self._last_df: Optional[pd.DataFrame] = None
        self._epoch_callback: Optional[Any] = None
        self._policy_opt: Optional[Dict[str, Any]] = None
        self._readiness: Optional[Dict[str, Any]] = None
        self._range_regime_sharpe: Optional[float] = 0.0
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
          - The feature matrix is wide-format and cross-sectional
          - Fundamentals and sentiment are ignored by construction
          - Labels are top/bottom-quartile future ranks for the target ETF
        """
        del fundamentals, sentiment

        t0 = time.time()
        period = period or self.config.training_period
        self._ticker = ticker

        cached = self._load_cached(ticker)
        if cached is not None:
            self._models = cached
            return TrainResult(
                ticker=ticker,
                backend=self.backend + " (cached)",
                model_type=self.config.model_type,
                n_samples=0,
                n_features=0,
                regime_accuracy=0.0,
                regime_f1={},
                entry_mae=0.0,
                exit_mae=0.0,
                feature_importances=self._models.feature_importance_dict(),
                cv_scores=[],
                training_time_s=0.0,
            )

        if df is None:
            import yfinance as yf

            hist = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if hist is None or hist.empty:
                raise ValueError(f"Unable to fetch data for {ticker}")
            df = hist.rename(columns={c: f"{c}_{ticker}" for c in hist.columns})

        if len(df) < 150:
            raise ValueError(f"Need at least 150 rows for cross-sectional training, got {len(df)}")

        self._last_df = df.copy()
        ctx = dict(market_ctx or {})
        ctx["target_ticker"] = ticker
        _set_policy_context(ticker, _ctx_lookup(ctx, ["macro_regime_label"], 1))

        feat_names = self.config.train_feature_names(
            include_fundamentals=False,
            include_sentiment=False,
        )
        X_df = build_features(
            df,
            feat_names,
            market_ctx=ctx,
            training_mode=True,
            pit_market_ctx=None,
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
        combined = combined[combined["_regime"] != "MIDDLE"]
        if len(combined) < 60:
            raise ValueError(f"Only {len(combined)} usable top/bottom-quartile samples after warmup")

        available_features = [c for c in X_df.columns if c in combined.columns]
        X = combined[available_features].values.astype(np.float32)
        y_regime = combined["_regime"].astype(str).values
        y_entry = combined["_entry"].values.astype(np.float32)
        y_exit = combined["_exit"].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        close_panel = _extract_panel(df, "Close", ticker).astype(float)
        target_series = close_panel[ticker] if ticker in close_panel.columns else close_panel.iloc[:, 0]
        target_next_ret = target_series.pct_change().shift(-1).reindex(combined.index)
        feat_aligned = combined[available_features]

        wf_metrics = _walk_forward_evaluate(
            X=X,
            y_regime=y_regime,
            y_entry=y_entry,
            y_exit=y_exit,
            feature_frame=feat_aligned,
            target_returns=target_next_ret,
            config=self.config,
            target_ticker=ticker,
        )
        cv_scores = [float(f.get("accuracy", 0.0)) for f in wf_metrics.fold_results]

        split = int(len(X) * 0.85)
        fit_end = max(40, split - self.config.forward_horizon)
        if fit_end >= len(X):
            fit_end = max(40, len(X) - 10)
            split = fit_end

        models = _create_model(self.config, X.shape[1])
        models.fit(
            X[:fit_end],
            y_regime[:fit_end],
            y_entry[:fit_end],
            y_exit[:fit_end],
            available_features,
            epoch_callback=self._epoch_callback,
        )
        self._models = models

        from sklearn.metrics import accuracy_score, f1_score

        probs, ent_pred, exit_pred, classes = models.predict(X[split:])
        regime_preds = [str(classes[i]) for i in probs.argmax(axis=1)]
        acc = float(accuracy_score(y_regime[split:], regime_preds)) if len(regime_preds) else 0.0

        f1s: Dict[str, float] = {}
        for cls in REGIME_CLASSES:
            y_bin = (y_regime[split:] == cls).astype(int)
            p_bin = (np.asarray(regime_preds) == cls).astype(int)
            if y_bin.sum() > 0:
                f1s[cls] = round(float(f1_score(y_bin, p_bin, zero_division=0)), 3)

        entry_mae = float(np.mean(np.abs(y_entry[split:] - ent_pred))) if len(ent_pred) else 0.0
        exit_mae = float(np.mean(np.abs(y_exit[split:] - exit_pred))) if len(exit_pred) else 0.0

        self._save_cached(ticker, models)
        elapsed = time.time() - t0

        idx = combined.index
        _iso = lambda d: d.isoformat()[:10] if hasattr(d, "isoformat") else str(d)
        data_range = (_iso(idx[0]), _iso(idx[-1])) if len(idx) > 0 else None
        train_range = (_iso(idx[0]), _iso(idx[fit_end - 1])) if fit_end > 0 else None
        test_range = (_iso(idx[split]), _iso(idx[-1])) if split < len(idx) else None

        class_dist = combined["_regime"].value_counts().to_dict()
        tlog = TrainingLog(
            epochs=getattr(models, "_epoch_log", []),
            early_stop_epoch=None,
            best_val_loss=None,
            n_train_rows=fit_end,
            n_val_rows=len(X) - split,
            n_test_rows=len(X) - split,
            features_used=available_features,
            backend=self.backend if self.config.model_type == "mlp" else self.config.model_type,
            training_time_s=round(elapsed, 2),
            data_date_range=data_range,
            train_date_range=train_range,
            test_date_range=test_range,
            class_distribution={str(k): int(v) for k, v in class_dist.items()},
            calibration_status="raw_lgbm" if self.config.model_type == "lightgbm" else "",
            calibration_samples=0,
            temperature=1.0,
        )

        draft_result = TrainResult(
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
        readiness = assess_live_readiness(draft_result).to_dict()
        self._readiness = readiness

        return TrainResult(
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
        del fundamentals, sentiment

        if self._models is None:
            raise RuntimeError("Model not trained — call train() first")

        target_ticker = self._ticker or _infer_target_ticker(df, market_ctx)
        ctx = dict(market_ctx or {})
        ctx["target_ticker"] = target_ticker
        _set_policy_context(target_ticker, _ctx_lookup(ctx, ["macro_regime_label"], 1))

        feat_names = self._models.feature_names
        X_df = build_features(df, feat_names, market_ctx=ctx)
        for col in feat_names:
            if col not in X_df.columns:
                X_df[col] = 0.0
        X_df = X_df[feat_names]

        last_row = X_df.iloc[-1]
        last = X_df.iloc[[-1]].values.astype(np.float32)
        last = np.nan_to_num(last, nan=0.0, posinf=0.0, neginf=0.0)

        mc_passes = self.config.mc_dropout_passes if self.config.model_type == "mlp" else 1
        probs, entry, exit_, classes = self._models.predict(last, mc_passes=mc_passes)

        regime_idx = int(probs[0].argmax())
        regime = str(classes[regime_idx])
        conf = float(probs[0][regime_idx])
        regime_probs = {
            str(classes[i]): round(float(probs[0][i]), 3)
            for i in range(len(classes))
        }

        uncertainty = getattr(self._models, "_mc_uncertainty", None)
        model_ready = True
        if self.config.signal_require_readiness:
            model_ready = bool((self._readiness or {}).get("ready", False))

        policy = DecisionPolicy(
            min_regime_confidence=self.config.signal_min_regime_confidence,
            min_score_spread=self.config.signal_min_score_spread,
            min_liquidity_rank=self.config.signal_min_liquidity_rank,
            max_amihud=self.config.signal_max_amihud,
        )
        decision = apply_decision_policy(
            regime,
            conf,
            float(entry[0]),
            float(exit_[0]),
            uncertainty=uncertainty,
            policy=policy,
            liquidity_rank=float(last_row.get("dollar_vol_rank", np.nan)),
            amihud=None,
            model_ready=model_ready,
        )
        signal = _decision_to_signal(decision)

        return MLPrediction(
            regime=regime,
            regime_confidence=round(conf, 3),
            regime_probs=regime_probs,
            entry_score=round(float(entry[0]), 3),
            exit_score=round(float(exit_[0]), 3),
            ml_signal=signal,
            decision=decision.to_dict(),
            feature_importances=self._models.feature_importance_dict(),
            uncertainty=uncertainty,
            score_spread=round(float(entry[0] - exit_[0]), 3),
            policy=policy.to_dict(),
        )

    def predict_from_latest(
        self,
        latest: Dict,
        df: pd.DataFrame,
        market_ctx=None,
        fundamentals=None,
        sentiment=None,
    ) -> MLPrediction:
        """Convenience: predict from the asset's indicator DataFrame."""
        del latest
        return self.predict_from_df(
            df,
            market_ctx=market_ctx,
            fundamentals=fundamentals,
            sentiment=sentiment,
        )

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
        del fundamentals, sentiment

        if self._models is None:
            raise RuntimeError("Model not trained — call train() first")
        if df is None:
            df = self._last_df
        if df is None:
            raise ValueError("No DataFrame available — pass df or train first")

        target_ticker = self._ticker or _infer_target_ticker(df, market_ctx)
        ctx = dict(market_ctx or {})
        ctx["target_ticker"] = target_ticker

        feat_names = self._models.feature_names
        X_df = build_features(df, feat_names, market_ctx=ctx)
        for col in feat_names:
            if col not in X_df.columns:
                X_df[col] = 0.0
        X_df = X_df[feat_names]

        valid_mask = X_df.notna().all(axis=1)
        if not valid_mask.any():
            return {
                "dates": [],
                "prices": [],
                "regimes": [],
                "entry_scores": [],
                "exit_scores": [],
                "regime_probs": {},
                "classes": REGIME_CLASSES,
            }

        X = np.nan_to_num(X_df.loc[valid_mask].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        probs, entry, exit_, classes = self._models.predict(X)
        dates_raw = X_df.index[valid_mask]
        dates = [d.isoformat()[:10] if hasattr(d, "isoformat") else str(d) for d in dates_raw]
        close_panel = _extract_panel(df, "Close", target_ticker).astype(float)
        price_series = close_panel[target_ticker] if target_ticker in close_panel.columns else close_panel.iloc[:, 0]
        prices = price_series.loc[valid_mask].tolist()

        regimes = [str(classes[i]) for i in probs.argmax(axis=1)]
        regime_prob_series = {
            str(cls): [round(float(probs[r, ci]), 4) for r in range(len(probs))]
            for ci, cls in enumerate(classes)
        }
        return {
            "dates": dates,
            "prices": prices,
            "regimes": regimes,
            "entry_scores": [round(float(x), 4) for x in entry],
            "exit_scores": [round(float(x), 4) for x in exit_],
            "regime_probs": regime_prob_series,
            "classes": [str(c) for c in classes],
        }

    def _cache_path(self, ticker: str) -> Path:
        key = self.config.cache_key(ticker)
        return _MODEL_DIR / f"{ticker}_{key}.pkl"

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
            age_days = (time.time() - meta.get("timestamp", 0)) / 86400.0
            if age_days > self.config.max_model_age_days:
                return None
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            log.warning("Failed to load cached model for %s: %s", ticker, exc)
            return None

    def _save_cached(self, ticker: str, models: Any):
        try:
            path = self._cache_path(ticker)
            meta_path = self._meta_path(ticker)
            with open(path, "wb") as f:
                pickle.dump(models, f)
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
        except Exception as exc:
            log.warning("Failed to cache model for %s: %s", ticker, exc)


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

        Cross-sectional pooling is natural for ETF rotation:
        - each target ETF contributes extreme-rank examples
        - the model learns transferable rank structures
        - all features remain relative and point-in-time
        """
        del fundamentals_map, sentiment_map

        t0 = time.time()
        feat_names = self.config.train_feature_names()
        pooled_frames: List[pd.DataFrame] = []

        for ticker in tickers:
            if ticker not in dfs or dfs[ticker] is None or dfs[ticker].empty:
                continue
            ctx = dict(market_ctx or {})
            ctx["target_ticker"] = ticker
            x_df = build_features(dfs[ticker], feat_names, market_ctx=ctx, training_mode=True)
            regime, entry_q, exit_q = generate_labels(
                dfs[ticker],
                forward_horizon=self.config.forward_horizon,
                strong_thresh=self.config.strong_threshold,
                weak_thresh=self.config.weak_threshold,
                cost_penalty_pct=self.config.label_cost_penalty_pct,
                impact_penalty_pct=self.config.label_impact_penalty_pct,
            )
            combined = x_df.copy()
            combined["_regime"] = regime
            combined["_entry"] = entry_q
            combined["_exit"] = exit_q
            combined["_ticker"] = ticker
            combined = combined.dropna()
            combined = combined[combined["_regime"] != "MIDDLE"]
            if not combined.empty:
                pooled_frames.append(combined)

        if not pooled_frames:
            raise ValueError("No usable pooled samples for universe training")

        pooled = pd.concat(pooled_frames, axis=0).sort_index()
        X = pooled[feat_names].values.astype(np.float32)
        y_regime = pooled["_regime"].astype(str).values
        y_entry = pooled["_entry"].values.astype(np.float32)
        y_exit = pooled["_exit"].values.astype(np.float32)

        split = int(len(X) * 0.85)
        fit_end = max(40, split - self.config.forward_horizon)
        model = _create_model(self.config, X.shape[1])
        model.fit(X[:fit_end], y_regime[:fit_end], y_entry[:fit_end], y_exit[:fit_end], feat_names)
        self._models = model

        from sklearn.metrics import accuracy_score, f1_score

        probs, ent_pred, exit_pred, classes = model.predict(X[split:])
        regime_preds = [str(classes[i]) for i in probs.argmax(axis=1)]
        acc = float(accuracy_score(y_regime[split:], regime_preds)) if len(regime_preds) else 0.0
        f1s: Dict[str, float] = {}
        for cls in REGIME_CLASSES:
            y_bin = (y_regime[split:] == cls).astype(int)
            p_bin = (np.asarray(regime_preds) == cls).astype(int)
            if y_bin.sum() > 0:
                f1s[cls] = round(float(f1_score(y_bin, p_bin, zero_division=0)), 3)

        elapsed = time.time() - t0
        result = TrainResult(
            ticker="UNIVERSE",
            backend=self.backend if self.config.model_type == "mlp" else self.config.model_type,
            model_type=self.config.model_type,
            n_samples=len(X),
            n_features=X.shape[1],
            regime_accuracy=round(acc, 3),
            regime_f1=f1s,
            entry_mae=round(float(np.mean(np.abs(y_entry[split:] - ent_pred))) if len(ent_pred) else 0.0, 4),
            exit_mae=round(float(np.mean(np.abs(y_exit[split:] - exit_pred))) if len(exit_pred) else 0.0, 4),
            feature_importances=model.feature_importance_dict(),
            cv_scores=[],
            training_time_s=round(elapsed, 2),
        )
        result.readiness = assess_live_readiness(result).to_dict()
        return result


# ── Convenience: list cached models ─────────────────────────────────────────

def list_cached_models() -> List[Dict[str, Any]]:
    """Return metadata for all cached models."""
    results: List[Dict[str, Any]] = []
    for meta_file in _MODEL_DIR.glob("*.json"):
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            if "ticker" not in meta:
                continue
            age_days = (time.time() - meta.get("timestamp", 0)) / 86400.0
            meta["age_days"] = round(age_days, 1)
            model_file = meta_file.with_suffix(".pkl")
            meta["size_mb"] = round(model_file.stat().st_size / 1e6, 2) if model_file.exists() else 0.0
            results.append(meta)
        except Exception:
            pass
    return results


def clear_cached_models() -> int:
    """Remove all cached models. Returns count removed."""
    count = 0
    for f in _MODEL_DIR.glob("*"):
        try:
            if f.is_file():
                f.unlink()
                count += 1
        except Exception:
            pass
    return count


def get_available_backend() -> str:
    """Return the best available backend."""
    return _resolve_backend("auto")


# ── Model Registry (versioned experiment tracking) ───────────────────────────

_REGISTRY_PATH = _MODEL_DIR / "registry.json"


@dataclass
class ModelVersion:
    """A single versioned model entry in the registry."""
    version_id: str
    ticker: str
    model_type: str
    backend: str
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

        model_path = _MODEL_DIR / f"{version_id}.pkl"
        engine._models.save(model_path)

        wf = train_result.walk_forward
        tlog = train_result.training_log
        mv = ModelVersion(
            version_id=version_id,
            ticker=ticker,
            model_type=model_type,
            backend=train_result.backend,
            version=v_num,
            train_period=period_str,
            n_samples=train_result.n_samples,
            n_features=train_result.n_features,
            features=tlog.features_used if tlog else [],
            regime_accuracy=train_result.regime_accuracy,
            regime_f1=train_result.regime_f1,
            entry_mae=train_result.entry_mae,
            exit_mae=train_result.exit_mae,
            sharpe_ratio=wf.sharpe_ratio if wf else None,
            max_drawdown=wf.max_drawdown if wf else None,
            cagr=wf.cagr if wf else None,
            hit_rate=wf.hit_rate if wf else None,
            profit_factor=wf.profit_factor if wf else None,
            model_path=str(model_path),
            config=asdict(engine.config),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            notes=notes,
            readiness=train_result.readiness,
            policy_opt=wf.policy_opt if wf else None,
            range_regime_sharpe=wf.range_regime_sharpe if wf else None,
        )
        self._registry[version_id] = mv.to_dict()
        self._save()
        return mv

    def load(self, version_id: str):
        """Load model weights for a given version_id."""
        if version_id not in self._registry:
            raise KeyError(f"Version '{version_id}' not found in registry")
        entry = self._registry[version_id]
        path = Path(entry["model_path"])
        if not path.exists():
            raise FileNotFoundError(f"Model file missing: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def list(self, ticker=None, model_type=None):
        results = []
        for vid, entry in self._registry.items():
            if ticker and entry.get("ticker") != ticker:
                continue
            if model_type and entry.get("model_type") != model_type:
                continue
            results.append(entry)
        return sorted(results, key=lambda e: e.get("created_at", ""))

    def compare(self, version_ids: List[str]):
        rows = []
        for vid in version_ids:
            if vid in self._registry:
                e = self._registry[vid]
                rows.append({
                    "version": vid,
                    "model_type": e.get("model_type"),
                    "train_period": e.get("train_period"),
                    "n_samples": e.get("n_samples"),
                    "regime_acc": e.get("regime_accuracy"),
                    "sharpe": e.get("sharpe_ratio"),
                    "max_dd": e.get("max_drawdown"),
                    "cagr": e.get("cagr"),
                    "hit_rate": e.get("hit_rate"),
                    "profit_factor": e.get("profit_factor"),
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
