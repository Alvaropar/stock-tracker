// ════════════════════════════════════════════════════════════════════════════
// Portfolio module: dashboard, transactions, rebalance, watchlist.
// Wired into app.js via window.openPortfolio() / window.closePortfolio().
// ════════════════════════════════════════════════════════════════════════════

(function () {
  "use strict";

  const $   = id => document.getElementById(id);
  const show = el => el && el.classList.remove("hidden");
  const hide = el => el && el.classList.add("hidden");
  const toast = (msg, kind) => (window.toast ? window.toast(msg, kind) : console.log(kind, msg));

  // ── State ────────────────────────────────────────────────────────────────
  const PFState = {
    dashboard: null,
    transactions: [],
    targets: {},
    baseCurrency: "USD",
    llmAvailable: false,
  };
  window._pfPositionsCache = window._pfPositionsCache || [];

  // ── Formatting helpers ───────────────────────────────────────────────────
  function fmtMoney(v) {
    if (v === null || v === undefined || isNaN(Number(v))) return "—";
    const n = Number(v);
    const abs = Math.abs(n);
    const sign = n < 0 ? "-" : "";
    if (abs >= 1e9) return sign + "$" + (abs / 1e9).toFixed(2) + "B";
    if (abs >= 1e6) return sign + "$" + (abs / 1e6).toFixed(2) + "M";
    if (abs >= 1e3) return sign + "$" + (abs / 1e3).toFixed(2) + "K";
    return sign + "$" + abs.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  function fmtPnl(v) {
    if (v === null || v === undefined || isNaN(Number(v))) return '<span class="muted">—</span>';
    const n = Number(v);
    const cls = n > 0 ? "pos" : n < 0 ? "neg" : "";
    const sign = n > 0 ? "+" : "";
    return `<span class="${cls}">${sign}${fmtMoney(n)}</span>`;
  }
  function fmtPct(v, digits) {
    if (v === null || v === undefined || isNaN(Number(v))) return '<span class="muted">—</span>';
    const n = Number(v);
    const cls = n > 0 ? "pos" : n < 0 ? "neg" : "";
    const sign = n > 0 ? "+" : "";
    return `<span class="${cls}">${sign}${n.toFixed(digits ?? 2)}%</span>`;
  }
  function fmtWeight(w) {
    return ((Number(w) || 0) * 100).toFixed(1) + "%";
  }
  function fmtSignal(sig) {
    if (!sig) return '<span class="muted" title="no history">—</span>';
    const colour = sig.css_class === "sbuy" || sig.css_class === "buy" ? "#22c55e"
                : sig.css_class === "sell" || sig.css_class === "ssell" ? "#ef4444"
                : "#a78bfa";
    const tip = `score ${sig.score} · tech ${sig.tech_score} · mom ${sig.momentum_score} · risk ${sig.risk_score} · RSI ${sig.rsi ?? '—'}`;
    return `<span title="${tip}" style="color:${colour};font-weight:700;font-size:0.78rem">${sig.action}</span>`;
  }

  // ── Tab switching ────────────────────────────────────────────────────────
  window.pfSwitchTab = function (tab) {
    const tabs = ["dashboard", "transactions", "rebalance", "watchlist"];
    tabs.forEach(t => {
      const btn  = $("pfTab" + t[0].toUpperCase() + t.slice(1));
      const pane = $("pf" + t[0].toUpperCase() + t.slice(1) + "Pane");
      if (btn) btn.classList.toggle("active", t === tab);
      if (pane) (t === tab ? show : hide)(pane);
    });
    if (tab === "transactions") loadTransactions();
    if (tab === "rebalance") { renderTargetsEditor(); }
    if (tab === "watchlist") loadWatchlist();
  };

  // ── Dashboard ────────────────────────────────────────────────────────────
  async function loadDashboard() {
    const btn = $("portfolioRefresh");
    if (btn) btn.textContent = "↻ Loading…";
    try {
      const d = await fetch("/api/portfolio/dashboard").then(r => r.json());
      PFState.dashboard = d;
      PFState.baseCurrency = d.base_currency || "USD";
      PFState.llmAvailable = !!d.llm_available;

      const t = d.totals || {};
      $("pfMarketValue").textContent = fmtMoney(t.market_value);
      $("pfCostBasis").textContent   = fmtMoney(t.cost_basis);
      $("pfUnrealized").innerHTML    = fmtPnl(t.unrealized_pnl);
      $("pfRealized").innerHTML      = fmtPnl(t.realized_pnl);
      $("pfDividends").innerHTML     = fmtPnl(t.dividends);
      $("pfReturnPct").innerHTML     = fmtPct(t.return_pct);

      $("pfBeta").textContent       = d.portfolio_beta == null ? "—" : Number(d.portfolio_beta).toFixed(2);
      $("pfHHI").textContent        = d.concentration_hhi == null ? "—" : Number(d.concentration_hhi).toFixed(0);
      $("pfNPositions").textContent = d.n_positions ?? "—";
      $("pfRegime").textContent     = d.market_regime ? d.market_regime + (d.spy_trend_bull ? " · SPY ↑" : " · SPY ↓") : "—";

      // Expose to legacy callers (e.g. AI review in app.js)
      window._pfPositionsCache = d.positions || [];
      renderPositions(d.positions || []);
      renderBreakdown("pfSectors",    d.breakdowns?.sectors    || []);
      renderBreakdown("pfRegions",    d.breakdowns?.regions    || []);
      renderBreakdown("pfCurrencies", d.breakdowns?.currencies || []);
      renderContributors("pfTopGainers", d.top_gainers || [], "gain");
      renderContributors("pfTopLosers",  d.top_losers  || [], "lose");
    } catch (e) {
      toast("Failed to load portfolio dashboard.", "err");
      console.error(e);
    } finally {
      if (btn) btn.textContent = "↻ Refresh";
    }
  }

  function renderPositions(positions) {
    const c = $("pfPositionsTable");
    if (!positions.length) {
      c.innerHTML = '<div class="empty-state">No open positions.</div>';
      return;
    }
    c.innerHTML = `
      <table class="results-table" style="width:100%">
        <thead><tr>
          <th>Ticker</th><th>Name</th><th>Qty</th><th>Avg Cost</th><th>Current</th>
          <th>Mkt Value</th><th>Weight</th><th>PnL</th><th>Return %</th>
          <th>Annlzd</th><th>Sector</th><th>Region</th><th>β</th><th>Signal</th>
        </tr></thead>
        <tbody>
          ${positions.map(p => `
            <tr>
              <td><strong>${p.ticker}</strong></td>
              <td style="font-size:0.78rem;max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${p.name || ""}">${p.name || "—"}</td>
              <td>${Number(p.quantity).toLocaleString()}</td>
              <td>${fmtMoney(p.avg_cost)}</td>
              <td>${fmtMoney(p.current_price)}</td>
              <td>${fmtMoney(p.market_value)}</td>
              <td>${fmtWeight(p.weight)}</td>
              <td>${fmtPnl(p.unrealized_pnl)}</td>
              <td>${fmtPct(p.unrealized_pct)}</td>
              <td>${p.annualized_return_pct == null ? '<span class="muted">—</span>' : fmtPct(p.annualized_return_pct)}</td>
              <td style="font-size:0.78rem">${p.sector || "—"}</td>
              <td style="font-size:0.78rem">${p.region || "—"}</td>
              <td>${p.beta == null ? '<span class="muted">—</span>' : Number(p.beta).toFixed(2)}</td>
              <td>${fmtSignal(p.signal)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>`;
  }

  function renderBreakdown(id, rows) {
    const c = $(id);
    if (!rows.length) { c.innerHTML = '<div class="muted" style="font-size:0.8rem">No data.</div>'; return; }
    c.innerHTML = rows.map(r => {
      const w = (r.weight || 0) * 100;
      return `<div class="pf-bd-row">
        <div class="pf-bd-row-label">${r.label}</div>
        <div class="pf-bd-row-bar"><div class="pf-bd-row-bar-fill" style="width:${w.toFixed(1)}%"></div></div>
        <div class="pf-bd-row-val">${w.toFixed(1)}%</div>
      </div>`;
    }).join("");
  }

  function renderContributors(id, rows, kind) {
    const c = $(id);
    if (!rows.length) { c.innerHTML = `<div class="muted" style="font-size:0.8rem">No ${kind === "gain" ? "gainers" : "losers"}.</div>`; return; }
    c.innerHTML = rows.map(r => `
      <div class="pf-bd-row">
        <div class="pf-bd-row-label"><strong>${r.ticker}</strong></div>
        <div class="pf-bd-row-val">${fmtPnl(r.unrealized_pnl)} (${fmtPct(r.unrealized_pct)})</div>
      </div>`).join("");
  }

  // ── Transactions ─────────────────────────────────────────────────────────
  async function loadTransactions() {
    try {
      PFState.transactions = await fetch("/api/portfolio/transactions").then(r => r.json());
    } catch { toast("Failed to load transactions.", "err"); return; }
    renderTransactions();
  }

  function renderTransactions() {
    const c = $("pfTxTable");
    const txs = PFState.transactions;
    if (!txs.length) { c.innerHTML = '<div class="empty-state">No transactions yet.</div>'; return; }
    c.innerHTML = `
      <table class="results-table" style="width:100%">
        <thead><tr>
          <th>Date</th><th>Ticker</th><th>Side</th><th>Qty</th><th>Price</th>
          <th>CCY</th><th>FX→base</th><th>Fees</th><th>Notes</th><th></th>
        </tr></thead>
        <tbody>
          ${txs.map(t => `
            <tr>
              <td style="font-size:0.78rem">${t.trade_date}</td>
              <td><strong>${t.ticker}</strong></td>
              <td><span class="pf-side pf-side-${t.side.toLowerCase()}">${t.side}</span></td>
              <td>${Number(t.quantity).toLocaleString()}</td>
              <td>${fmtMoney(t.price)}</td>
              <td>${t.currency}</td>
              <td>${Number(t.fx_rate).toFixed(4)}</td>
              <td>${fmtMoney(t.fees)}</td>
              <td style="font-size:0.75rem;color:var(--text2);max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${(t.notes || '').replace(/"/g, '&quot;')}">${t.notes || ""}</td>
              <td><button class="btn btn-sm btn-danger" onclick="pfDeleteTx(${t.id})">✕</button></td>
            </tr>
          `).join("")}
        </tbody>
      </table>`;
  }

  window.pfDeleteTx = async function (id) {
    if (!confirm("Delete this transaction? Realized P&L will recompute.")) return;
    const r = await fetch(`/api/portfolio/transactions/${id}`, { method: "DELETE" });
    if (!r.ok) { toast("Failed to delete.", "err"); return; }
    await loadTransactions();
    await loadDashboard();
    toast("Transaction removed.", "ok");
  };

  async function addTransaction() {
    const payload = {
      ticker:     ($("pfTxTicker").value || "").trim().toUpperCase(),
      side:       $("pfTxSide").value,
      quantity:   parseFloat($("pfTxQty").value),
      price:      parseFloat($("pfTxPrice").value),
      trade_date: $("pfTxDate").value,
      currency:   ($("pfTxCurrency").value || "USD").toUpperCase(),
      fx_rate:    parseFloat($("pfTxFxRate").value) || 1.0,
    };
    if (!payload.ticker) return toast("Ticker required.", "err");
    if (!(payload.quantity > 0)) return toast("Quantity must be > 0.", "err");
    if (!(payload.price >= 0)) return toast("Price must be ≥ 0.", "err");
    const r = await fetch("/api/portfolio/transactions", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const d = await r.json();
    if (!r.ok) return toast(d.error || "Failed to add.", "err");
    $("pfTxTicker").value = "";
    $("pfTxQty").value = "";
    $("pfTxPrice").value = "";
    await loadTransactions();
    await loadDashboard();
    toast(`${payload.side} ${payload.ticker} recorded.`, "ok");
  }

  // ── Rebalance ────────────────────────────────────────────────────────────
  async function loadTargets() {
    try { PFState.targets = await fetch("/api/portfolio/targets").then(r => r.json()) || {}; }
    catch { PFState.targets = {}; }
  }

  function renderTargetsEditor() {
    const c = $("pfRbTargetsEditor");
    const positions = (PFState.dashboard?.positions) || [];
    // Seed targets with current tickers if empty
    if (!Object.keys(PFState.targets).length && positions.length) {
      positions.forEach(p => { PFState.targets[p.ticker] = 0; });
    }
    const tickers = Array.from(new Set([
      ...Object.keys(PFState.targets || {}),
      ...positions.map(p => p.ticker),
    ])).sort();
    if (!tickers.length) {
      c.innerHTML = '<div class="empty-state">Add positions or targets to use the rebalancer.</div>';
      updateTargetSum();
      return;
    }
    const currentWeights = {};
    positions.forEach(p => { currentWeights[p.ticker] = (p.weight || 0) * 100; });
    c.innerHTML = `
      <table class="results-table" style="width:100%">
        <thead><tr>
          <th>Ticker</th><th>Current %</th><th>Target %</th><th>Drift</th><th></th>
        </tr></thead>
        <tbody>
          ${tickers.map(t => {
            const cur = currentWeights[t] ?? 0;
            const tgt = ((PFState.targets[t] || 0) * 100);
            const drift = (cur - tgt).toFixed(1);
            return `<tr>
              <td><strong>${t}</strong></td>
              <td>${cur.toFixed(1)}%</td>
              <td><input class="input pf-target-input" data-ticker="${t}" type="number" step="any" min="0" max="100" value="${tgt.toFixed(2)}" style="width:90px"/></td>
              <td>${fmtPct(parseFloat(drift), 1)}</td>
              <td><button class="btn btn-sm btn-danger" onclick="pfRemoveTarget('${t}')">✕</button></td>
            </tr>`;
          }).join("")}
        </tbody>
      </table>`;
    c.querySelectorAll(".pf-target-input").forEach(inp => {
      inp.addEventListener("input", () => {
        const t = inp.dataset.ticker;
        PFState.targets[t] = (parseFloat(inp.value) || 0) / 100;
        updateTargetSum();
      });
    });
    updateTargetSum();
  }

  function updateTargetSum() {
    const total = Object.values(PFState.targets || {}).reduce((a, b) => a + (Number(b) || 0), 0) * 100;
    const el = $("pfRbTargetSum");
    if (el) {
      el.textContent = "Σ = " + total.toFixed(1) + "%";
      el.style.color = Math.abs(100 - total) < 0.5 ? "var(--text2)" : (total > 100 ? "#ef4444" : "#eab308");
    }
  }

  window.pfRemoveTarget = function (t) {
    delete PFState.targets[t];
    renderTargetsEditor();
  };

  async function saveTargets() {
    try {
      const r = await fetch("/api/portfolio/targets", {
        method: "PUT", headers: {"Content-Type": "application/json"},
        body: JSON.stringify(PFState.targets),
      });
      const d = await r.json();
      if (!r.ok) return toast(d.error || "Failed to save.", "err");
      toast("Targets saved.", "ok");
    } catch { toast("Failed to save targets.", "err"); }
  }

  function equalWeightAll() {
    const positions = (PFState.dashboard?.positions) || [];
    if (!positions.length) return toast("No positions.", "err");
    const w = 1.0 / positions.length;
    PFState.targets = {};
    positions.forEach(p => { PFState.targets[p.ticker] = w; });
    renderTargetsEditor();
  }

  async function computeRebalance() {
    const params = new URLSearchParams({
      cash:       $("pfRbCash").value || "0",
      drift:      ((parseFloat($("pfRbDrift").value) || 0) / 100).toString(),
      min_trade:  $("pfRbMinTrade").value || "0",
      fractional: $("pfRbFractional").value,
    });
    const r = await fetch("/api/portfolio/rebalance?" + params.toString());
    const d = await r.json();
    if (!r.ok) return toast(d.error || "Rebalance failed.", "err");
    renderRebalance(d);
  }

  function renderRebalance(d) {
    const c = $("pfRbResult");
    if (!d.trades?.length) {
      c.innerHTML = `<div class="empty-state">No trades needed — drift is within threshold (max |drift| = ${(d.drift_summary?.max_abs_drift * 100 || 0).toFixed(2)}%).</div>`;
      return;
    }
    c.innerHTML = `
      <div style="margin-bottom:0.5rem;color:var(--text2);font-size:0.78rem">
        Portfolio value: <strong>${fmtMoney(d.portfolio_value)}</strong> ·
        Cash before: <strong>${fmtMoney(d.cash)}</strong> ·
        After: <strong>${fmtMoney(d.cash_after)}</strong> ·
        Max drift: <strong>${(d.drift_summary.max_abs_drift * 100).toFixed(2)}%</strong>
      </div>
      <table class="results-table" style="width:100%">
        <thead><tr>
          <th>Ticker</th><th>Side</th><th>Qty</th><th>Price</th><th>Notional</th>
          <th>Current %</th><th>Target %</th><th>Drift</th><th></th>
        </tr></thead>
        <tbody>
          ${d.trades.map(t => `
            <tr>
              <td><strong>${t.ticker}</strong></td>
              <td><span class="pf-side pf-side-${t.side.toLowerCase()}">${t.side}</span></td>
              <td>${Number(t.quantity).toLocaleString()}</td>
              <td>${fmtMoney(t.price)}</td>
              <td>${fmtMoney(t.notional)}</td>
              <td>${(t.current_weight * 100).toFixed(1)}%</td>
              <td>${(t.target_weight * 100).toFixed(1)}%</td>
              <td>${fmtPct(t.drift * 100, 1)}</td>
              <td><button class="btn btn-sm btn-outline" onclick="pfExecuteTrade(${JSON.stringify(t).replace(/"/g, '&quot;')})">Record</button></td>
            </tr>
          `).join("")}
        </tbody>
      </table>`;
  }

  window.pfExecuteTrade = async function (t) {
    if (!confirm(`Record ${t.side} ${t.quantity} ${t.ticker} @ ${fmtMoney(t.price)} as a transaction?`)) return;
    const r = await fetch("/api/portfolio/transactions", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        ticker: t.ticker, side: t.side,
        quantity: t.quantity, price: t.price,
        notes: "rebalance",
      }),
    });
    if (!r.ok) return toast("Failed to record trade.", "err");
    await loadDashboard();
    await computeRebalance();
    toast("Trade recorded.", "ok");
  };

  // ── Watchlist ────────────────────────────────────────────────────────────
  async function loadWatchlist() {
    try {
      const items = await fetch("/api/portfolio/watchlist").then(r => r.json());
      renderWatchlist(items);
    } catch { toast("Failed to load watchlist.", "err"); }
  }

  function renderWatchlist(items) {
    const c = $("pfWatchlistTable");
    if (!items.length) { c.innerHTML = '<div class="empty-state">Watchlist is empty.</div>'; return; }
    c.innerHTML = `
      <table class="results-table" style="width:100%">
        <thead><tr><th>Ticker</th><th>Name</th><th>Sector</th><th></th></tr></thead>
        <tbody>
          ${items.map(w => `
            <tr>
              <td><strong>${w.ticker}</strong></td>
              <td>${w.name || "—"}</td>
              <td style="color:var(--text2)">${w.sector || "—"}</td>
              <td><button class="btn btn-sm btn-danger" onclick="pfRemoveWatch('${w.ticker}')">✕</button></td>
            </tr>
          `).join("")}
        </tbody>
      </table>`;
    PFState._watchlistItems = items;
  }

  window.pfRemoveWatch = async function (ticker) {
    await fetch(`/api/portfolio/watchlist/${ticker}`, { method: "DELETE" });
    await loadWatchlist();
    toast(`${ticker} removed.`, "ok");
  };

  // ── Open / close ─────────────────────────────────────────────────────────
  window.openPortfolio = async function () {
    const stepSections = () => document.querySelectorAll(".step:not(#portfolioSection)");
    stepSections().forEach(el => hide(el));
    show($("portfolioSection"));
    document.querySelectorAll(".step-btn[data-step]").forEach(b => b.classList.remove("active", "done"));
    if ($("pfTxDate") && !$("pfTxDate").value) {
      $("pfTxDate").value = new Date().toISOString().slice(0, 10);
    }
    window.pfSwitchTab("dashboard");
    await loadTargets();
    await loadDashboard();
  };

  window.closePortfolio = function () {
    hide($("portfolioSection"));
    if (window.goTo && window.state) window.goTo(window.state.step);
  };

  // ── Wire static event listeners on DOMContentLoaded ──────────────────────
  function wire() {
    const navBtn = $("portfolioNavBtn");
    if (navBtn) navBtn.addEventListener("click", window.openPortfolio);
    const closeBtn = $("portfolioClose");
    if (closeBtn) closeBtn.addEventListener("click", window.closePortfolio);
    const refreshBtn = $("portfolioRefresh");
    if (refreshBtn) refreshBtn.addEventListener("click", loadDashboard);

    const addTxBtn = $("pfTxAdd");
    if (addTxBtn) addTxBtn.addEventListener("click", addTransaction);

    const rbCompute = $("pfRbCompute");
    if (rbCompute) rbCompute.addEventListener("click", computeRebalance);
    const rbSave = $("pfRbSaveTargets");
    if (rbSave) rbSave.addEventListener("click", saveTargets);
    const rbEqual = $("pfRbEqualWeight");
    if (rbEqual) rbEqual.addEventListener("click", equalWeightAll);
    const rbAdd = $("pfRbAddTarget");
    if (rbAdd) rbAdd.addEventListener("click", () => {
      const t = prompt("Ticker for new target:");
      if (!t) return;
      PFState.targets[t.toUpperCase()] = 0;
      renderTargetsEditor();
    });

    const watchAdd = $("pfWatchAdd");
    if (watchAdd) watchAdd.addEventListener("click", async () => {
      const ticker = ($("pfWatchTicker").value || "").trim().toUpperCase();
      const name   = ($("pfWatchName").value || "").trim();
      if (!ticker) return toast("Enter a ticker.", "err");
      const r = await fetch("/api/portfolio/watchlist", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ ticker, name }),
      });
      const d = await r.json();
      if (!r.ok) return toast(d.error || "Error adding.", "err");
      $("pfWatchTicker").value = "";
      $("pfWatchName").value = "";
      await loadWatchlist();
      toast(`${ticker} added.`, "ok");
    });

    const analyzePos = $("pfAnalyzePositions");
    if (analyzePos) analyzePos.addEventListener("click", () => {
      const positions = (PFState.dashboard?.positions) || [];
      if (!positions.length) return toast("No positions.", "err");
      const seen = new Set();
      if (window.state) {
        window.state.assets = [];
        positions.forEach(p => {
          if (!seen.has(p.ticker)) {
            seen.add(p.ticker);
            window.state.assets.push({
              ticker: p.ticker, name: p.name || p.ticker,
              sector: p.sector || "", currency: p.currency || "USD",
            });
          }
        });
        if (window.renderAssets) window.renderAssets();
        window.closePortfolio();
        if (window.goTo) window.goTo(1);
        toast(`Loaded ${window.state.assets.length} positions for analysis.`, "ok");
      }
    });

    const analyzeWl = $("pfAnalyzeWatchlist");
    if (analyzeWl) analyzeWl.addEventListener("click", () => {
      const items = PFState._watchlistItems || [];
      if (!items.length) return toast("Watchlist is empty.", "err");
      if (window.state) {
        window.state.assets = items.map(w => ({
          ticker: w.ticker, name: w.name || w.ticker,
          sector: w.sector || "", currency: w.currency || "USD",
        }));
        if (window.renderAssets) window.renderAssets();
        window.closePortfolio();
        if (window.goTo) window.goTo(1);
        toast(`Loaded ${window.state.assets.length} watchlist assets.`, "ok");
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", wire);
  } else {
    wire();
  }
})();
