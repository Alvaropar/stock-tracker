"""
Formal backtesting engine for the ML trading system.

Improvements over v1:
  - Proper benchmark comparison (fetches SPY, computes real alpha/beta/IR)
  - Volatility-targeted position sizing (target_vol / realized_vol)
  - Drawdown circuit breaker (reduces exposure after significant losses)
  - Rolling realized vol estimation for dynamic position scaling
  - Realistic transaction costs (commission + slippage)

Usage:
    from app.backend.services.backtest import Backtester, BacktestConfig
    bt = Backtester(BacktestConfig(commission_pct=0.001))
    result = bt.run(engine, df)
    print(result.summary())
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001       # 0.1% per side (entry + exit)
    slippage_pct: float = 0.0005        # 0.05% adverse fill per side
    max_position_pct: float = 1.0       # max fraction of capital in one trade
    # Override decision policy thresholds if desired (None = use model defaults)
    entry_threshold: Optional[float] = None
    exit_threshold: Optional[float] = None
    # Stop-loss / take-profit guardrails (None = off)
    stop_loss_pct: Optional[float] = 0.08     # 8% stop-loss
    take_profit_pct: Optional[float] = None   # optional
    # Benchmark ticker for comparison (None = skip)
    benchmark: str = "SPY"
    # Volatility targeting
    target_annual_vol: float = 0.15     # 15% annualised target volatility
    vol_lookback: int = 21              # days for rolling vol estimation
    # Drawdown circuit breaker
    max_drawdown_trigger: float = 0.15  # reduce exposure after 15% drawdown
    # Minimum dollar volume to trade (liquidity filter)
    min_dollar_volume: float = 5_000_000.0
    # Market impact model: sqrt(order_value / avg_daily_volume) * coeff
    # Almgren-Chriss style linear-sqrt impact.  0 = disabled (flat slippage only).
    # Realistic range: 0.05–0.10 for US equities.
    market_impact_coeff: float = 0.05


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_date: str
    exit_date: str
    ticker: str
    entry_price: float
    exit_price: float
    shares: float
    position_size: float        # fraction of capital allocated
    gross_pnl: float            # before costs
    net_pnl: float              # after commission + slippage
    pnl_pct: float              # net return as fraction
    holding_days: int
    regime: str
    entry_score: float
    exit_score: float
    conviction: str
    exit_reason: str            # "signal", "stop_loss", "take_profit", "end_of_data"
    commission_paid: float
    slippage_cost: float


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    ticker: str
    config: BacktestConfig

    # Equity & returns
    equity_curve: List[float]           # daily equity values
    dates: List[str]                    # aligned with equity_curve
    daily_returns: List[float]          # daily % changes
    benchmark_equity: List[float]       # buy-and-hold benchmark
    close_prices: List[float]           # raw close price aligned to dates

    # Trades
    trades: List[Trade]

    # Core metrics
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    hit_rate: float
    profit_factor: float
    avg_trade_return: float
    avg_winning_trade: float
    avg_losing_trade: float
    best_trade: float
    worst_trade: float
    avg_holding_days: float
    n_trades: int
    n_winning: int
    n_losing: int
    total_commission: float
    total_slippage: float

    # Regime/vol breakdown
    by_regime: Dict[str, Dict]
    by_volatility: Dict[str, Dict]

    # Streak stats
    max_win_streak: int
    max_loss_streak: int

    # Benchmark comparison
    benchmark_return: float
    alpha: float                        # strategy CAGR - benchmark CAGR
    beta: float                         # regression beta vs benchmark
    information_ratio: float            # annualised alpha / tracking error

    # Metadata
    backtest_date: str
    elapsed_s: float

    def summary(self) -> str:
        lines = [
            f"{'='*55}",
            f"  Backtest: {self.ticker}  |  {self.dates[0]} -> {self.dates[-1]}",
            f"{'='*55}",
            f"  Total Return    : {self.total_return*100:+.2f}%",
            f"  CAGR            : {self.cagr*100:+.2f}%",
            f"  Sharpe Ratio    : {self.sharpe_ratio:.3f}",
            f"  Sortino Ratio   : {self.sortino_ratio:.3f}",
            f"  Calmar Ratio    : {self.calmar_ratio:.3f}",
            f"  Max Drawdown    : {self.max_drawdown*100:.2f}%  ({self.max_drawdown_duration_days}d)",
            f"{'~'*55}",
            f"  Benchmark ({self.config.benchmark:>5}): {self.benchmark_return*100:+.2f}%",
            f"  Alpha           : {self.alpha*100:+.2f}%",
            f"  Beta            : {self.beta:.3f}",
            f"  Info Ratio      : {self.information_ratio:.3f}",
            f"{'~'*55}",
            f"  Trades          : {self.n_trades}  (W:{self.n_winning} / L:{self.n_losing})",
            f"  Hit Rate        : {self.hit_rate*100:.1f}%",
            f"  Profit Factor   : {self.profit_factor:.3f}",
            f"  Avg Trade       : {self.avg_trade_return*100:+.2f}%",
            f"  Avg Win         : {self.avg_winning_trade*100:+.2f}%",
            f"  Avg Loss        : {self.avg_losing_trade*100:+.2f}%",
            f"  Best / Worst    : {self.best_trade*100:+.2f}% / {self.worst_trade*100:+.2f}%",
            f"  Avg Hold        : {self.avg_holding_days:.1f} days",
            f"  Win/Loss Streak : {self.max_win_streak} / {self.max_loss_streak}",
            f"{'~'*55}",
            f"  Commission      : ${self.total_commission:,.0f}",
            f"  Slippage        : ${self.total_slippage:,.0f}",
            f"{'='*55}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["trades"] = [asdict(t) for t in self.trades]
        d.pop("config", None)
        return d


# ── Open position tracker ─────────────────────────────────────────────────────

@dataclass
class _Position:
    entry_date: str
    entry_price: float
    shares: float
    initial_shares: float
    position_size: float
    entry_score: float
    regime: str
    conviction: str
    entry_commission_remaining: float
    entry_slippage_remaining: float


@dataclass
class _PendingOrder:
    action: str
    target_size: float
    reason: str
    conviction: str
    entry_score: float = 0.0
    exit_score: float = 0.0
    regime: str = ""


# ── Benchmark fetcher ────────────────────────────────────────────────────────

def _fetch_benchmark(ticker: str, start, end) -> Optional[pd.Series]:
    """Fetch benchmark close prices aligned to the backtest date range."""
    try:
        import yfinance as yf
        bench = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if bench is not None and len(bench) > 10:
            return bench["Close"]
    except Exception as e:
        log.warning(f"Failed to fetch benchmark {ticker}: {e}")
    return None


def _effective_slippage(
    base_slippage: float,
    market_impact_coeff: float,
    order_value: float,
    adv: float,
) -> float:
    impact = (
        market_impact_coeff * np.sqrt(order_value / adv)
        if adv > 0 and np.isfinite(adv) and market_impact_coeff > 0
        else 0.0
    )
    return max(0.0, base_slippage + impact)


# ── Engine ────────────────────────────────────────────────────────────────────

class Backtester:
    """
    Applies a trained MLEngine to a full price history, simulates trades,
    and computes comprehensive performance metrics.

    Key improvements:
      - Vol-targeting: positions sized to target constant risk
      - Drawdown circuit breaker: halts entries after severe losses
      - Real benchmark comparison with proper alpha/beta/IR
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        engine,               # trained MLEngine instance
        df: pd.DataFrame,
        market_ctx: Optional[Dict] = None,
        fundamentals: Optional[Dict] = None,
        sentiment: Optional[Dict] = None,
        progress_callback=None,
    ) -> BacktestResult:
        """
        Run a full backtest on historical data.

        Args:
            engine: a trained MLEngine (engine._models must be set)
            df: OHLCV + indicator DataFrame (same format as training)
            market_ctx / fundamentals / sentiment: optional context
        """
        from .ml_engine import (
            build_features, apply_decision_policy, DecisionPolicy,
            REGIME_CLASSES
        )

        t0 = time.time()
        cfg = self.config
        ticker = getattr(engine, "_ticker", "UNKNOWN")

        if engine._models is None:
            raise RuntimeError("Engine has no trained model. Call engine.train() first.")

        # ── Build features for every row ─────────────────────────────────
        feature_names = (
            getattr(engine._models, "feature_names", None)
            or engine.config.feature_names()
        )
        # Build backtest features in point-in-time mode so historical SPY/VIX are
        # aligned per row and no live scalar is broadcast backwards.
        historical_spy = None
        historical_vix = None
        try:
            import yfinance as yf
            _spy = yf.Ticker("SPY").history(
                start=df.index[0], end=df.index[-1], auto_adjust=True)
            if _spy is not None and len(_spy) > 50:
                historical_spy = _spy
            _vix = yf.Ticker("^VIX").history(
                start=df.index[0], end=df.index[-1], auto_adjust=True)
            if _vix is not None and len(_vix) > 50:
                historical_vix = _vix["Close"]
        except Exception:
            pass

        feat_df = build_features(
            df, feature_names,
            market_ctx=market_ctx,
            fundamentals=fundamentals,
            sentiment=sentiment,
            training_mode=True,  # use per-row historical data, not broadcast
            historical_spy=historical_spy,
            historical_vix=historical_vix,
        )
        feat_df = feat_df.dropna()

        # Align close prices to feature rows
        close = df["Close"].reindex(feat_df.index)
        open_ = df["Open"].reindex(feat_df.index) if "Open" in df.columns else close
        dates_idx = feat_df.index

        # Use only model's actual feature list; fill missing with 0
        for fn in feature_names:
            if fn not in feat_df.columns:
                feat_df[fn] = 0.0

        X = feat_df[feature_names].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Run predictions ───────────────────────────────────────────────
        mc_passes = 1
        if engine.config.model_type == "mlp" and engine.config.mc_dropout_passes > 1:
            mc_passes = engine.config.mc_dropout_passes
        probs, entry_scores, exit_scores, classes = engine._models.predict(X, mc_passes=mc_passes)

        # Get MC uncertainty if available
        unc_list = getattr(engine._models, "_mc_uncertainty_rows", None)
        if not unc_list or len(unc_list) != len(X):
            unc_list = [None] * len(X)
        if hasattr(engine._models, "_mc_uncertainty_rows"):
            engine._models._mc_uncertainty_rows = None
        if hasattr(engine._models, "_mc_uncertainty"):
            engine._models._mc_uncertainty = None

        # ── Pre-compute rolling ADV for market impact model ───────────────
        # avg_daily_dollar_volume[i] = rolling-20 average dollar volume at bar i
        if "Volume" in df.columns and cfg.market_impact_coeff > 0:
            _dv_series = (df["Close"] * df["Volume"]).reindex(feat_df.index).rolling(20).mean()
            adv_arr = _dv_series.fillna(cfg.min_dollar_volume).values
        else:
            adv_arr = np.full(len(X), np.inf)   # infinity → no market impact

        # ── Override decision thresholds if specified ─────────────────────
        policy = DecisionPolicy()
        _policy_opt = getattr(engine, "_policy_opt", None) or {}
        _range_sharpe = getattr(engine, "_range_regime_sharpe", None)
        if _range_sharpe is not None and _range_sharpe <= 0.0:
            policy.disable_range_entries = True
        if _policy_opt:
            policy.entry_threshold = float(_policy_opt.get("best_entry_threshold", policy.entry_threshold))
            policy.exit_threshold = float(_policy_opt.get("best_exit_threshold", policy.exit_threshold))
            policy.min_regime_confidence = float(
                _policy_opt.get("best_min_regime_confidence", policy.min_regime_confidence)
            )
            policy.min_score_spread = float(
                _policy_opt.get("best_min_score_spread", policy.min_score_spread)
            )
        min_conf_floor = float(getattr(engine.config, "signal_min_regime_confidence", policy.min_regime_confidence))
        min_spread_floor = float(getattr(engine.config, "signal_min_score_spread", policy.min_score_spread))
        policy.min_regime_confidence = max(policy.min_regime_confidence, min_conf_floor)
        policy.min_score_spread = max(policy.min_score_spread, min_spread_floor)
        policy.min_liquidity_rank = float(getattr(engine.config, "signal_min_liquidity_rank", policy.min_liquidity_rank))
        policy.max_amihud = float(getattr(engine.config, "signal_max_amihud", policy.max_amihud))
        if cfg.entry_threshold is not None:
            policy.entry_threshold = cfg.entry_threshold
        if cfg.exit_threshold is not None:
            policy.exit_threshold = cfg.exit_threshold
        model_ready = True
        if bool(getattr(engine.config, "signal_require_readiness", False)):
            model_ready = bool((getattr(engine, "_readiness", None) or {}).get("ready", False))

        # ── Simulate trades ───────────────────────────────────────────────
        capital = cfg.initial_capital
        position: Optional[_Position] = None
        pending_order: Optional[_PendingOrder] = None
        trades: List[Trade] = []

        equity_curve = [capital]
        equity_dates = [str(dates_idx[0])[:10]]
        close_prices_list = [float(close.iloc[0]) if len(close) > 0 else 0.0]

        regime_pred = [classes[i] for i in probs.argmax(axis=1)]
        regime_conf = probs.max(axis=1).tolist()

        # Rolling vol estimation buffer
        _recent_returns: List[float] = []
        _realized_vol = cfg.target_annual_vol  # initialise at target
        _peak_equity = capital

        def _execute_entry(order: _PendingOrder, fill_price_raw: float, adv_i: float, date_str: str):
            nonlocal capital, position
            if position is not None:
                return
            alloc = min(order.target_size, cfg.max_position_pct)
            if alloc <= 0 or fill_price_raw <= 0:
                return
            order_value = capital * alloc
            eff_slippage = _effective_slippage(
                cfg.slippage_pct, cfg.market_impact_coeff, order_value, adv_i
            )
            fill_price = fill_price_raw * (1 + eff_slippage)
            commission = order_value * cfg.commission_pct
            shares = (order_value - commission) / fill_price if fill_price > 0 else 0.0
            if shares <= 0:
                return
            entry_slippage = max(0.0, shares * max(fill_price - fill_price_raw, 0.0))
            position = _Position(
                entry_date=date_str,
                entry_price=fill_price,
                shares=shares,
                initial_shares=shares,
                position_size=alloc,
                entry_score=order.entry_score,
                regime=order.regime,
                conviction=order.conviction,
                entry_commission_remaining=commission,
                entry_slippage_remaining=entry_slippage,
            )
            capital -= (shares * fill_price + commission)

        def _execute_exit(
            order: _PendingOrder,
            fill_price_raw: float,
            adv_i: float,
            date_str: str,
            force_full: bool = False,
        ):
            nonlocal capital, position
            if position is None or fill_price_raw <= 0:
                return

            pre_shares = position.shares
            reduce_fraction = 1.0 if force_full or order.action == "SELL" else max(0.0, min(1.0, order.target_size))
            shares_to_sell = pre_shares * reduce_fraction
            if shares_to_sell <= 0:
                return

            exit_value = shares_to_sell * fill_price_raw
            eff_slippage = _effective_slippage(
                cfg.slippage_pct, cfg.market_impact_coeff, exit_value, adv_i
            )
            fill_price = fill_price_raw * (1 - eff_slippage)
            commission_exit = shares_to_sell * fill_price * cfg.commission_pct
            proceeds = shares_to_sell * fill_price - commission_exit

            share_frac = shares_to_sell / pre_shares if pre_shares > 0 else 1.0
            entry_commission = position.entry_commission_remaining * share_frac
            entry_slippage = position.entry_slippage_remaining * share_frac
            cost_basis = shares_to_sell * position.entry_price
            gross_pnl = shares_to_sell * (fill_price - position.entry_price)
            net_pnl = proceeds - cost_basis - entry_commission
            total_cost_basis = cost_basis + entry_commission
            pnl_pct = net_pnl / total_cost_basis if total_cost_basis > 0 else 0.0
            holding_days = max(1, (pd.Timestamp(date_str) - pd.Timestamp(position.entry_date)).days)
            exit_slippage = max(0.0, shares_to_sell * max(fill_price_raw - fill_price, 0.0))

            trades.append(Trade(
                entry_date=position.entry_date,
                exit_date=date_str,
                ticker=ticker,
                entry_price=position.entry_price,
                exit_price=fill_price,
                shares=shares_to_sell,
                position_size=position.position_size * share_frac,
                gross_pnl=round(gross_pnl, 4),
                net_pnl=round(net_pnl, 4),
                pnl_pct=round(pnl_pct, 6),
                holding_days=holding_days,
                regime=position.regime,
                entry_score=round(position.entry_score, 4),
                exit_score=round(order.exit_score, 4),
                conviction=position.conviction,
                exit_reason=order.reason,
                commission_paid=round(entry_commission + commission_exit, 4),
                slippage_cost=round(entry_slippage + exit_slippage, 4),
            ))

            capital += proceeds
            position.entry_commission_remaining -= entry_commission
            position.entry_slippage_remaining -= entry_slippage
            position.shares -= shares_to_sell
            position.position_size *= max(0.0, 1.0 - share_frac)
            if position.shares <= 1e-9:
                position = None

        for i in range(len(X)):
            price = float(close.iloc[i])
            open_price = float(open_.iloc[i]) if pd.notna(open_.iloc[i]) else price
            date_str = str(dates_idx[i])[:10]
            adv_i = float(adv_arr[i]) if i < len(adv_arr) else np.inf

            if price <= 0:
                equity_curve.append(equity_curve[-1])
                equity_dates.append(date_str)
                close_prices_list.append(close_prices_list[-1])
                continue

            if open_price <= 0:
                open_price = price

            # Execute the prior bar's signal at today's open.
            if pending_order is not None:
                if pending_order.action == "BUY":
                    _execute_entry(pending_order, open_price, adv_i, date_str)
                else:
                    _execute_exit(pending_order, open_price, adv_i, date_str)
                pending_order = None

            # Mark-to-market equity at today's close after any open execution.
            mtm = capital + (position.shares * price if position else 0.0)
            equity_curve.append(mtm)
            equity_dates.append(date_str)
            close_prices_list.append(price)

            # ── Update rolling vol & drawdown ────────────────────────────
            if len(equity_curve) > 1:
                daily_ret = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
                _recent_returns.append(daily_ret)
                if len(_recent_returns) > cfg.vol_lookback:
                    _recent_returns.pop(0)
                if len(_recent_returns) >= 5:
                    _realized_vol = float(np.std(_recent_returns) * np.sqrt(252))
                    _realized_vol = max(_realized_vol, 0.01)

            _peak_equity = max(_peak_equity, equity_curve[-1])
            _current_dd = (equity_curve[-1] - _peak_equity) / _peak_equity

            ep = float(entry_scores[i])
            xp = float(exit_scores[i])
            reg = regime_pred[i]
            conf = float(regime_conf[i])

            decision = apply_decision_policy(
                regime=reg,
                regime_confidence=conf,
                entry=ep,
                exit_score=xp,
                uncertainty=unc_list[i],
                policy=policy,
                realized_vol=_realized_vol,
                target_vol=cfg.target_annual_vol,
                current_drawdown=_current_dd,
                max_drawdown_trigger=cfg.max_drawdown_trigger,
                liquidity_rank=float(feat_df.iloc[i].get("dollar_vol_rank", np.nan)),
                amihud=float(feat_df.iloc[i].get("amihud", np.nan)),
                model_ready=model_ready,
            )

            next_order: Optional[_PendingOrder] = None

            # ── Check stop-loss / take-profit on open position ────────────
            if position is not None and cfg.stop_loss_pct is not None:
                loss_pct = (price - position.entry_price) / position.entry_price
                if loss_pct <= -cfg.stop_loss_pct:
                    next_order = _PendingOrder(
                        action="SELL",
                        target_size=1.0,
                        reason="stop_loss",
                        conviction="HIGH",
                        exit_score=xp,
                        regime=reg,
                    )

            if next_order is None and position is not None and cfg.take_profit_pct is not None:
                gain_pct = (price - position.entry_price) / position.entry_price
                if gain_pct >= cfg.take_profit_pct:
                    next_order = _PendingOrder(
                        action="SELL",
                        target_size=1.0,
                        reason="take_profit",
                        conviction="HIGH",
                        exit_score=xp,
                        regime=reg,
                    )

            if next_order is None:
                if position is None and decision.action == "BUY":
                    next_order = _PendingOrder(
                        action="BUY",
                        target_size=min(decision.position_size, cfg.max_position_pct),
                        reason="signal",
                        conviction=decision.conviction,
                        entry_score=ep,
                        regime=reg,
                    )
                elif position is not None and decision.action in ("SELL", "REDUCE"):
                    next_order = _PendingOrder(
                        action=decision.action,
                        target_size=decision.position_size,
                        reason="signal",
                        conviction=decision.conviction,
                        exit_score=xp,
                        regime=reg,
                    )

            if next_order is not None and i < len(X) - 1:
                pending_order = next_order

        # Close any open position at last price
        if position is not None:
            last_price = float(close.iloc[-1])
            _execute_exit(
                _PendingOrder(
                    action="SELL",
                    target_size=1.0,
                    reason="end_of_data",
                    conviction=position.conviction,
                    exit_score=0.0,
                    regime=position.regime,
                ),
                last_price,
                np.inf,
                equity_dates[-1],
                force_full=True,
            )

        equity_curve[-1] = capital  # finalize

        # ── Fetch benchmark ────────────────────────────────────────────────
        bench_close = None
        if cfg.benchmark:
            bench_close = _fetch_benchmark(
                cfg.benchmark,
                start=dates_idx[0],
                end=dates_idx[-1],
            )

        # ── Compute metrics ───────────────────────────────────────────────
        result = _compute_metrics(
            ticker=ticker,
            equity_curve=equity_curve,
            dates=equity_dates,
            close_prices=close_prices_list,
            trades=trades,
            config=cfg,
            initial_capital=cfg.initial_capital,
            elapsed_s=round(time.time() - t0, 3),
            benchmark_close=bench_close,
        )

        return result


# ── Metrics computation ───────────────────────────────────────────────────────

def _compute_metrics(
    ticker: str,
    equity_curve: List[float],
    dates: List[str],
    close_prices: List[float],
    trades: List[Trade],
    config: BacktestConfig,
    initial_capital: float,
    elapsed_s: float,
    benchmark_close: Optional[pd.Series] = None,
) -> BacktestResult:

    eq = np.array(equity_curve, dtype=np.float64)
    daily_rets = np.diff(eq) / np.where(eq[:-1] != 0, eq[:-1], 1)
    daily_rets = np.nan_to_num(daily_rets, nan=0.0)

    total_return = (eq[-1] - initial_capital) / initial_capital

    # CAGR
    n_days = max(1, len(eq) - 1)
    years = n_days / 252.0
    cagr = (eq[-1] / initial_capital) ** (1.0 / max(years, 0.01)) - 1.0

    # Sharpe (annualised, risk-free = 0)
    std = daily_rets.std()
    sharpe = (daily_rets.mean() / std * np.sqrt(252)) if std > 1e-9 else 0.0

    # Sortino (downside only)
    neg = daily_rets[daily_rets < 0]
    down_std = neg.std() if len(neg) > 1 else 1e-9
    sortino = (daily_rets.mean() / down_std * np.sqrt(252)) if down_std > 1e-9 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak > 0, peak, 1)
    max_dd = float(dd.min())

    # Max drawdown duration
    in_dd = dd < 0
    max_dd_dur = 0
    cur = 0
    for v in in_dd:
        cur = cur + 1 if v else 0
        max_dd_dur = max(max_dd_dur, cur)

    # Calmar
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-9 else 0.0

    # Trade stats
    n = len(trades)
    pnl_pcts = [t.pnl_pct for t in trades]
    wins = [p for p in pnl_pcts if p > 0]
    losses = [p for p in pnl_pcts if p <= 0]

    hit_rate = len(wins) / n if n > 0 else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else (999.0 if gross_profit > 0 else 0.0)

    avg_trade = float(np.mean(pnl_pcts)) if pnl_pcts else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    best = max(pnl_pcts) if pnl_pcts else 0.0
    worst = min(pnl_pcts) if pnl_pcts else 0.0
    avg_hold = float(np.mean([t.holding_days for t in trades])) if trades else 0.0
    total_comm = sum(t.commission_paid for t in trades)
    total_slip = sum(t.slippage_cost for t in trades)

    # Win/loss streaks
    max_win_streak = max_loss_streak = cur_w = cur_l = 0
    for p in pnl_pcts:
        if p > 0:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_win_streak = max(max_win_streak, cur_w)
        max_loss_streak = max(max_loss_streak, cur_l)

    # By-regime breakdown
    by_regime: Dict[str, Dict] = {}
    for t in trades:
        reg = t.regime
        if reg not in by_regime:
            by_regime[reg] = {"n_trades": 0, "returns": []}
        by_regime[reg]["n_trades"] += 1
        by_regime[reg]["returns"].append(t.pnl_pct)
    for reg, v in by_regime.items():
        rets = v.pop("returns")
        v["hit_rate"] = round(sum(1 for r in rets if r > 0) / len(rets), 3)
        v["avg_return"] = round(float(np.mean(rets)), 4)
        v["total_return"] = round(float(sum(rets)), 4)

    # By-volatility breakdown
    by_volatility: Dict[str, Dict] = {}
    if len(daily_rets) > 20:
        vol_window = pd.Series(daily_rets).rolling(20).std().fillna(0).values
        _vw_pos = vol_window[vol_window > 0]
        vol_pctile = np.percentile(_vw_pos, [33, 66]) if len(_vw_pos) > 0 else [0, 0]
        for t in trades:
            try:
                di = dates.index(t.entry_date)
            except ValueError:
                di = 0
            v = vol_window[min(di, len(vol_window) - 1)]
            if v < vol_pctile[0]:
                bucket = "LOW"
            elif v < vol_pctile[1]:
                bucket = "MED"
            else:
                bucket = "HIGH"
            if bucket not in by_volatility:
                by_volatility[bucket] = {"n_trades": 0, "returns": []}
            by_volatility[bucket]["n_trades"] += 1
            by_volatility[bucket]["returns"].append(t.pnl_pct)
        for bk, bv in by_volatility.items():
            rets = bv.pop("returns")
            bv["hit_rate"] = round(sum(1 for r in rets if r > 0) / len(rets), 3)
            bv["avg_return"] = round(float(np.mean(rets)), 4)

    # ── Benchmark comparison (REAL, not placeholder) ────────────────────────
    benchmark_eq = [initial_capital] * len(equity_curve)
    benchmark_ret = 0.0
    alpha = total_return
    beta = 0.0
    information_ratio = 0.0

    if benchmark_close is not None and len(benchmark_close) > 10:
        # Build benchmark equity curve: buy-and-hold from day 1
        bench_aligned = benchmark_close.reindex(
            pd.to_datetime([d for d in dates]), method="ffill"
        )
        bench_vals = bench_aligned.values
        valid_mask = ~np.isnan(bench_vals) & (bench_vals > 0)

        if valid_mask.sum() > 10:
            # Find first valid benchmark price
            first_valid = np.argmax(valid_mask)
            bench_base = bench_vals[first_valid]
            benchmark_eq = []
            for i, bv in enumerate(bench_vals):
                if np.isnan(bv) or bv <= 0:
                    benchmark_eq.append(benchmark_eq[-1] if benchmark_eq else initial_capital)
                else:
                    benchmark_eq.append(initial_capital * bv / bench_base)

            # Pad to match equity_curve length
            while len(benchmark_eq) < len(equity_curve):
                benchmark_eq.append(benchmark_eq[-1])
            benchmark_eq = benchmark_eq[:len(equity_curve)]

            bench_eq_arr = np.array(benchmark_eq, dtype=np.float64)
            bench_daily_rets = np.diff(bench_eq_arr) / np.where(bench_eq_arr[:-1] != 0, bench_eq_arr[:-1], 1)
            bench_daily_rets = np.nan_to_num(bench_daily_rets, nan=0.0)

            # Benchmark total return
            benchmark_ret = (bench_eq_arr[-1] - initial_capital) / initial_capital

            # Alpha = strategy CAGR - benchmark CAGR
            bench_cagr = (bench_eq_arr[-1] / initial_capital) ** (1.0 / max(years, 0.01)) - 1.0
            alpha = cagr - bench_cagr

            # Beta = Cov(strategy, benchmark) / Var(benchmark)
            min_len = min(len(daily_rets), len(bench_daily_rets))
            if min_len > 10:
                strat_r = daily_rets[:min_len]
                bench_r = bench_daily_rets[:min_len]
                bench_var = np.var(bench_r)
                if bench_var > 1e-12:
                    beta = float(np.cov(strat_r, bench_r)[0, 1] / bench_var)

                # Information Ratio = annualised alpha / tracking error
                excess = strat_r - bench_r
                tracking_error = float(np.std(excess) * np.sqrt(252))
                if tracking_error > 1e-9:
                    information_ratio = float(np.mean(excess) * 252 / (tracking_error * np.sqrt(252)))
                    # Simplify: IR = mean(excess) * sqrt(252) / std(excess)
                    information_ratio = float((np.mean(excess) / np.std(excess)) * np.sqrt(252))
    else:
        # No benchmark data — use close prices as simple buy-and-hold benchmark
        if len(close_prices) > 1 and close_prices[0] > 0:
            base = close_prices[0]
            benchmark_eq = [initial_capital * p / base for p in close_prices]
            while len(benchmark_eq) < len(equity_curve):
                benchmark_eq.append(benchmark_eq[-1])
            benchmark_eq = benchmark_eq[:len(equity_curve)]
            benchmark_ret = (benchmark_eq[-1] - initial_capital) / initial_capital
            alpha = total_return - benchmark_ret

    return BacktestResult(
        ticker=ticker,
        config=config,
        equity_curve=[round(v, 2) for v in equity_curve],
        dates=dates,
        daily_returns=[round(float(r), 6) for r in daily_rets],
        benchmark_equity=[round(v, 2) for v in benchmark_eq],
        close_prices=[round(float(p), 4) for p in close_prices],
        trades=trades,
        total_return=round(total_return, 6),
        cagr=round(cagr, 6),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        calmar_ratio=round(calmar, 4),
        max_drawdown=round(max_dd, 6),
        max_drawdown_duration_days=max_dd_dur,
        hit_rate=round(hit_rate, 4),
        profit_factor=round(profit_factor, 4),
        avg_trade_return=round(avg_trade, 6),
        avg_winning_trade=round(avg_win, 6),
        avg_losing_trade=round(avg_loss, 6),
        best_trade=round(best, 6),
        worst_trade=round(worst, 6),
        avg_holding_days=round(avg_hold, 2),
        n_trades=n,
        n_winning=len(wins),
        n_losing=len(losses),
        total_commission=round(total_comm, 2),
        total_slippage=round(total_slip, 2),
        by_regime=by_regime,
        by_volatility=by_volatility,
        max_win_streak=max_win_streak,
        max_loss_streak=max_loss_streak,
        benchmark_return=round(benchmark_ret, 6),
        alpha=round(alpha, 6),
        beta=round(beta, 4),
        information_ratio=round(information_ratio, 4),
        backtest_date=pd.Timestamp.now().isoformat()[:10],
        elapsed_s=elapsed_s,
    )
