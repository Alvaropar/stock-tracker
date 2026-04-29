"use strict";

// ════════════════════════════════════════════════════════════════
//  State
// ════════════════════════════════════════════════════════════════

const state = {
  step:       1,
  assets:     [],          // [{ticker, name, sector, currency}]
  config:     {},          // analysis config built from UI
  results:    null,        // final results array
  taskId:     null,
  sortCol:    null,
  sortDir:    1,           // 1 = asc, -1 = desc
  lastDate:   null,
};

// ════════════════════════════════════════════════════════════════
//  Utilities
// ════════════════════════════════════════════════════════════════

const $ = id => document.getElementById(id);
const show = el => el && el.classList.remove("hidden");
const hide = el => el && el.classList.add("hidden");

function toast(msg, type = "ok", ms = 3500) {
  const t = $("toast");
  t.textContent = msg;
  t.className = `toast toast-${type}`;
  clearTimeout(t._tid);
  t._tid = setTimeout(() => hide(t), ms);
}

function fmt(v, decimals = 2, suffix = "") {
  if (v === null || v === undefined) return '<span class="muted">—</span>';
  const n = Number(v);
  if (isNaN(n)) return '<span class="muted">—</span>';
  const s = n.toFixed(decimals);
  return suffix ? s + suffix : s;
}

function fmtPct(v, d = 1) {
  if (v === null || v === undefined) return '<span class="muted">—</span>';
  const n = Number(v);
  if (isNaN(n)) return '<span class="muted">—</span>';
  const cls = n > 0 ? "pos" : n < 0 ? "neg" : "";
  const sign = n > 0 ? "+" : "";
  return `<span class="${cls}">${sign}${n.toFixed(d)}%</span>`;
}

// RS vs SPY: green if outperforming, red if underperforming; bold when |RS| > 5pp
function fmtRS(v) {
  if (v === null || v === undefined) return '<span class="muted">—</span>';
  const n = Number(v);
  if (isNaN(n)) return '<span class="muted">—</span>';
  const cls = n > 2 ? "pos" : n < -2 ? "neg" : "muted";
  const sign = n > 0 ? "+" : "";
  const bold = Math.abs(n) > 5 ? "font-weight:700;" : "";
  return `<span class="${cls}" style="${bold}">${sign}${n.toFixed(1)}%</span>`;
}

function sigPill(signal, css) {
  const cls = css || "sig-neu";
  const lbl = signal || "NEUTRAL";
  return `<span class="sig-pill ${cls}">${lbl}</span>`;
}

function sentPill(signal) {
  if (!signal) return '<span class="muted">—</span>';
  const s = signal.toUpperCase();
  const cls = (s === "BULLISH" || s === "STRONG_BULLISH") ? "sig-buy"
            : (s === "BEARISH" || s === "STRONG_BEARISH") ? "sig-sell"
            : "sig-neu";
  return `<span class="sig-pill ${cls}">${s.replace("_", " ")}</span>`;
}

function trendBadge(stage) {
  if (!stage) return '<span class="muted">—</span>';
  const colors = { EARLY: "#3b82f6", ESTABLISHED: "#22c55e", EXTENDED: "#f59e0b", PARABOLIC: "#ef4444" };
  const c = colors[stage] || "var(--text3)";
  return `<span style="color:${c};font-size:0.75rem;font-weight:600">${stage}</span>`;
}

function regimeBadge(regime) {
  if (!regime) return '<span class="muted">—</span>';
  const colors = { BULLISH: "#22c55e", BEARISH: "#ef4444", TRANSITION: "#f59e0b" };
  const c = colors[regime] || "var(--text3)";
  return `<span style="color:${c};font-size:0.75rem;font-weight:600">${regime}</span>`;
}

function regimeChgBadge(chg) {
  if (!chg) return '<span class="muted">—</span>';
  const colors = {
    "BULLISH REVERSAL": "#22c55e", "BEARISH REVERSAL": "#ef4444",
    "WEAKENING": "#f97316", "POTENTIAL BOTTOM": "#3b82f6",
  };
  const c = colors[chg] || "var(--text2)";
  return `<span style="color:${c};font-size:0.72rem;font-weight:600">${chg}</span>`;
}

function fmtScore01(v) {
  if (v === null || v === undefined) return '<span class="muted">—</span>';
  const n = Number(v);
  if (isNaN(n)) return '<span class="muted">—</span>';
  const cls = n >= 0.65 ? "pos" : n <= 0.30 ? "neg" : "";
  return `<span class="${cls}">${n.toFixed(2)}</span>`;
}

function fmtRisk(v) {
  if (v === null || v === undefined) return '<span class="muted">—</span>';
  const n = Number(v);
  if (isNaN(n)) return '<span class="muted">—</span>';
  const cls = n >= 2.0 ? "neg" : n >= 1.0 ? "" : "pos";
  return `<span class="${cls}">${n.toFixed(2)}</span>`;
}

function fmtDip(v) {
  if (v === null || v === undefined) return '<span class="muted">—</span>';
  const n = Number(v);
  if (isNaN(n)) return '<span class="muted">—</span>';
  if (n < 0.01) return '<span class="muted">—</span>';
  const cls = n >= 0.55 ? "pos" : n >= 0.35 ? "" : "muted";
  return `<span class="${cls}">${n.toFixed(2)}</span>`;
}

function fmtConf(v) {
  if (v === null || v === undefined) return '<span class="muted">—</span>';
  const n = Number(v);
  if (isNaN(n)) return '<span class="muted">—</span>';
  const cls = n >= 70 ? "pos" : n <= 40 ? "neg" : "";
  return `<span class="${cls}">${n.toFixed(0)}%</span>`;
}

// ── ML Classifier formatters ─────────────────────────────────

function mlRegimeBadge(regime) {
  if (!regime) return '<span class="muted">—</span>';
  const colors = {
    "TREND_UP": "#22c55e", "TREND_DOWN": "#ef4444",
    "REVERSAL_UP": "#06b6d4", "REVERSAL_DOWN": "#f97316",
    "RANGE": "#a78bfa",
  };
  const c = colors[regime] || "var(--text3)";
  const short = regime.replace("_", " ");
  return `<span style="color:${c};font-size:0.72rem;font-weight:600">${short}</span>`;
}

function fmtEntryExit(v, isExit) {
  if (v === null || v === undefined) return '<span class="muted">—</span>';
  const n = Number(v);
  if (isNaN(n)) return '<span class="muted">—</span>';
  const pct = Math.round(n * 100);
  let color;
  if (isExit) {
    color = n >= 0.7 ? "#ef4444" : n >= 0.45 ? "#f97316" : "var(--text3)";
  } else {
    color = n >= 0.7 ? "#22c55e" : n >= 0.55 ? "#3b82f6" : "var(--text3)";
  }
  // Mini bar + value
  const barW = Math.min(pct, 100);
  const bg = isExit ? "rgba(239,68,68,0.15)" : "rgba(34,197,94,0.15)";
  return `<span style="display:inline-flex;align-items:center;gap:4px">` +
    `<span style="display:inline-block;width:30px;height:6px;background:${bg};border-radius:3px;overflow:hidden">` +
    `<span style="display:block;width:${barW}%;height:100%;background:${color};border-radius:3px"></span></span>` +
    `<span style="color:${color};font-size:0.78rem;font-weight:600">${n.toFixed(2)}</span></span>`;
}

function mlSigPill(signal) {
  if (!signal) return '<span class="muted">—</span>';
  const colors = {
    "STRONG ENTRY": "sig-sbuy", "ENTRY": "sig-buy",
    "SPECULATIVE": "sig-neu", "HOLD": "sig-neu",
    "WATCH (REVERSAL)": "sig-neu",
    "REDUCE": "sig-sell", "EXIT": "sig-ssell",
  };
  const cls = colors[signal] || "sig-neu";
  return `<span class="sig-pill ${cls}">${signal}</span>`;
}

function fmtDecision(r) {
  const d = r.ml_decision;
  if (!d) return '<span class="muted">—</span>';
  const actionColors = {
    BUY: "#22c55e", SELL: "#ef4444", REDUCE: "#f97316",
    HOLD: "var(--text3)", WATCH: "#a78bfa",
  };
  const convBadge = {
    HIGH: "▲▲▲", MEDIUM: "▲▲", LOW: "▲", NONE: "—",
  };
  const ac = actionColors[d.action] || "var(--text3)";
  const sizePct = Math.round(d.position_size * 100);
  const convIcon = convBadge[d.conviction] || "";

  // Position size bar
  const barColor = d.action === "SELL" || d.action === "REDUCE"
    ? "rgba(239,68,68,0.3)" : "rgba(34,197,94,0.3)";
  const fillColor = ac;

  let html = `<span style="display:inline-flex;align-items:center;gap:4px;flex-wrap:nowrap">`;
  // Action label
  html += `<span style="color:${ac};font-weight:700;font-size:0.72rem">${d.action}</span>`;
  // Position size bar (only if non-zero)
  if (sizePct > 0) {
    html += `<span style="display:inline-block;width:28px;height:6px;background:${barColor};border-radius:3px;overflow:hidden">` +
      `<span style="display:block;width:${sizePct}%;height:100%;background:${fillColor};border-radius:3px"></span></span>`;
    html += `<span style="color:${ac};font-size:0.68rem">${sizePct}%</span>`;
  }
  // Conviction
  html += `<span style="font-size:0.6rem;color:${ac}" title="Conviction: ${d.conviction}">${convIcon}</span>`;
  html += `</span>`;

  // Tooltip with reasons
  if (d.reasons && d.reasons.length > 0) {
    const tip = d.reasons.join("\n");
    html = `<span title="${tip.replace(/"/g, '&quot;')}">${html}</span>`;
  }
  return html;
}

function fmtUncertainty(r) {
  const u = r.ml_uncertainty;
  const d = r.ml_decision;
  if (!u && !d) return '<span class="muted">—</span>';

  let penalty = d ? d.uncertainty_penalty : 0;
  let label, color;
  if (penalty <= 0.2) { label = "LOW"; color = "#22c55e"; }
  else if (penalty <= 0.5) { label = "MED"; color = "#eab308"; }
  else if (penalty <= 0.7) { label = "HIGH"; color = "#f97316"; }
  else { label = "VERY HIGH"; color = "#ef4444"; }

  const pctBar = Math.round(penalty * 100);
  let html = `<span style="display:inline-flex;align-items:center;gap:3px">`;
  html += `<span style="display:inline-block;width:24px;height:5px;background:rgba(255,255,255,0.08);border-radius:3px;overflow:hidden">`;
  html += `<span style="display:block;width:${pctBar}%;height:100%;background:${color};border-radius:3px"></span></span>`;
  html += `<span style="color:${color};font-size:0.68rem;font-weight:600">${label}</span>`;
  html += `</span>`;
  return html;
}

// ════════════════════════════════════════════════════════════════
//  Navigation
// ════════════════════════════════════════════════════════════════

function goTo(n) {
  // Validate before advancing
  if (n > state.step) {
    if (state.step === 1 && state.assets.length === 0) {
      toast("Add at least one asset first.", "err");
      return;
    }
  }
  document.querySelectorAll(".step").forEach(el => hide(el));
  show($(`step${n}`));

  document.querySelectorAll(".step-btn").forEach(btn => {
    const s = parseInt(btn.dataset.step);
    btn.classList.remove("active", "done");
    if (s === n) btn.classList.add("active");
    else if (s < n) btn.classList.add("done");
  });

  state.step = n;

  // When entering weights step, show/hide sentiment slider
  if (n === 4) updateSentimentWeightRow();
  // When entering backtest step, refresh ticker list
  if (n === 6) typeof btPopulateTickers === "function" && btPopulateTickers();
}

// Step nav click
document.querySelectorAll(".step-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const s = parseInt(btn.dataset.step);
    // Allow backward nav freely; forward nav only if assets are set
    if (s <= state.step || state.assets.length > 0) goTo(s);
  });
});

// ── Step 1 ────────────────────────────────────────────────────
$("step1Next").addEventListener("click", () => goTo(2));

// ── Step 2 ────────────────────────────────────────────────────
$("step2Back").addEventListener("click", () => goTo(1));
$("step2Next").addEventListener("click", () => goTo(3));

// ── Step 3 ────────────────────────────────────────────────────
$("step3Back").addEventListener("click", () => goTo(2));
$("step3Next").addEventListener("click", () => goTo(4));

$("sentEnabled").addEventListener("change", () => {
  const enabled = $("sentEnabled").checked;
  const cfg = $("sentConfig");
  enabled ? show(cfg) : hide(cfg);
  updateSentimentWeightRow();
});

// ── ML toggle ────────────────────────────────────────────────
$("mlEnabled").addEventListener("change", () => {
  const enabled = $("mlEnabled").checked;
  const cfg = $("mlConfig");
  enabled ? show(cfg) : hide(cfg);
});

// Check GPU availability on load
(async () => {
  try {
    const res = await fetch("/api/ml/status");
    if (res.ok) {
      const info = await res.json();
      const hint = $("mlBackendHint");
      if (info.cuda_available) {
        hint.innerHTML = `<span class="pos">GPU detected: ${info.gpu_name}</span> — PyTorch recommended`;
      } else if (info.pytorch_available) {
        hint.textContent = "PyTorch available (CPU only) — sklearn recommended for speed";
      } else {
        hint.textContent = "sklearn available (CPU)";
        // Hide pytorch option
        const ptOpt = document.querySelector('#mlBackend option[value="pytorch"]');
        if (ptOpt) ptOpt.disabled = true;
      }
    }
  } catch (_) {
    const hint = $("mlBackendHint");
    if (hint) hint.textContent = "";
  }
})();

// Clear ML models button
const _clearMLBtn = $("clearMLModels");
if (_clearMLBtn) {
  _clearMLBtn.addEventListener("click", async () => {
    try {
      const r = await fetch("/api/ml/models/clear", { method: "POST" });
      const d = await r.json();
      toast(`Cleared ${d.cleared} cached models`, "ok");
    } catch { toast("Failed to clear models", "err"); }
  });
}

// Toggle cloud vs local fields when provider changes
function updateProviderFields() {
  const provider = document.querySelector("input[name='sentProvider']:checked")?.value || "local";
  const cloudFields = $("cloudApiFields");
  const localFields = $("localModelFields");
  const costEstimate = $("costEstimate");
  if (provider === "local") {
    hide(cloudFields);
    show(localFields);
    if (costEstimate) hide(costEstimate);
  } else if (provider === "compactifai") {
    // Key is pre-configured server-side — hide both key field and cost notice
    hide(cloudFields);
    hide(localFields);
    if (costEstimate) hide(costEstimate);
  } else {
    show(cloudFields);
    hide(localFields);
    if (costEstimate) show(costEstimate);
  }
}
document.querySelectorAll("input[name='sentProvider']").forEach(r => {
  r.addEventListener("change", updateProviderFields);
});
updateProviderFields();

// ── Step 4 ────────────────────────────────────────────────────
$("step4Back").addEventListener("click", () => goTo(3));
$("step5Back").addEventListener("click", () => goTo(4));

$("runAnalysis").addEventListener("click", startAnalysis);
$("newAnalysis").addEventListener("click", () => goTo(1));

// ── Weight sliders ────────────────────────────────────────────
["Tech","Fund","Sent"].forEach(k => {
  const sl = $(`w${k}`);
  const val = $(`w${k}Val`);
  if (!sl) return;
  sl.addEventListener("input", () => {
    val.textContent = sl.value + "%";
    checkWeightTotal();
  });
});

$("resetWeights").addEventListener("click", () => {
  $("wTech").value = 40; $("wTechVal").textContent = "40%";
  $("wFund").value = 40; $("wFundVal").textContent = "40%";
  $("wSent").value = 20; $("wSentVal").textContent = "20%";
  checkWeightTotal();
});

function checkWeightTotal() {
  const t = parseInt($("wTech").value) + parseInt($("wFund").value) + parseInt($("wSent").value);
  const el = $("weightTotal");
  el.textContent = t + "%";
  el.classList.toggle("bad", t !== 100);
  const warn = $("weightWarning");
  t !== 100 ? show(warn) : hide(warn);
  $("runAnalysis").disabled = (t !== 100);
}

function updateSentimentWeightRow() {
  const enabled = $("sentEnabled").checked;
  const row = $("sentWeightRow");
  row.style.opacity = enabled ? "1" : "0.35";
  row.style.pointerEvents = enabled ? "" : "none";
}

checkWeightTotal();

// ════════════════════════════════════════════════════════════════
//  Asset management
// ════════════════════════════════════════════════════════════════

function renderAssets() {
  const list  = $("assetList");
  const empty = $("assetEmpty");
  const count = $("assetCount");

  count.textContent = state.assets.length;

  if (state.assets.length === 0) {
    list.innerHTML = "";
    show(empty);
    return;
  }
  hide(empty);

  list.innerHTML = state.assets.map(a => `
    <div class="asset-chip" data-ticker="${a.ticker}">
      <span class="asset-chip-ticker">${a.ticker}</span>
      <span class="asset-chip-name">${a.name || ""}${a.sector ? " · " + a.sector : ""}</span>
      <button class="asset-chip-remove" onclick="removeAsset('${a.ticker}')" title="Remove">✕</button>
    </div>
  `).join("");
}

function addAsset(asset) {
  if (state.assets.find(a => a.ticker === asset.ticker)) {
    toast(`${asset.ticker} already added.`, "err");
    return;
  }
  state.assets.push(asset);
  renderAssets();
}

window.removeAsset = function(ticker) {
  state.assets = state.assets.filter(a => a.ticker !== ticker);
  renderAssets();
};

$("clearAssets").addEventListener("click", () => {
  state.assets = [];
  renderAssets();
});

// ── Search ────────────────────────────────────────────────────
let searchDebounce = null;

$("searchInput").addEventListener("input", () => {
  clearTimeout(searchDebounce);
  const q = $("searchInput").value.trim();
  if (q.length < 1) {
    hide($("searchResults"));
    return;
  }
  searchDebounce = setTimeout(() => searchAssets(q), 350);
});

$("searchInput").addEventListener("keydown", e => {
  if (e.key === "Enter") searchAssets($("searchInput").value.trim());
});

$("searchBtn").addEventListener("click", () => searchAssets($("searchInput").value.trim()));

async function searchAssets(q) {
  if (!q) return;
  const res = $("searchResults");
  res.innerHTML = '<div style="padding:10px 12px;color:var(--text3)">Searching…</div>';
  show(res);

  try {
    const data = await fetch(`/api/assets/search?q=${encodeURIComponent(q)}&limit=10`)
                       .then(r => r.json());
    if (!data.results || data.results.length === 0) {
      res.innerHTML = '<div style="padding:10px 12px;color:var(--text3)">No results found.</div>';
      return;
    }
    res.innerHTML = data.results.map(a => `
      <div class="search-item" onclick="addAsset({ticker:'${a.ticker}',name:'${(a.name||"").replace(/'/g,"\\'")}',sector:'${(a.sector||"").replace(/'/g,"\\'")}',currency:'${a.currency||"USD"}'})">
        <span class="search-item-ticker">${a.ticker}</span>
        <span class="search-item-name">${a.name || ""}</span>
      </div>
    `).join("");
  } catch {
    res.innerHTML = '<div style="padding:10px 12px;color:var(--red)">Search failed.</div>';
  }
}

// Close search on outside click
document.addEventListener("click", e => {
  if (!e.target.closest(".search-row") && !e.target.closest(".search-results")) {
    hide($("searchResults"));
  }
});

// ── Presets ────────────────────────────────────────────────────
$("presetPortfolio").addEventListener("click",   () => loadPreset("portfolio"));
$("presetCommodities").addEventListener("click", () => loadPreset("commodities"));

async function clearNewsCache() {
  try {
    const r = await fetch("/api/settings/clear-cache", { method: "POST" });
    const d = await r.json();
    toast(d.msg || "Cache cleared", "ok");
  } catch { toast("Failed to clear cache", "err"); }
}

async function loadPreset(name) {
  try {
    const data = await fetch(`/api/assets/preset?name=${name}`).then(r => r.json());
    const list = Array.isArray(data) ? data : (data.assets || []);
    if (!list.length) { toast("No assets in preset.", "err"); return; }
    state.assets = [];
    list.forEach(a => state.assets.push(a));
    renderAssets();
    toast(`Loaded ${state.assets.length} assets.`, "ok");
  } catch {
    toast("Failed to load preset.", "err");
  }
}

// ── TAM Baskets ───────────────────────────────────────────────
async function loadTamBaskets() {
  const container = $("tamBasketBtns");
  if (!container) return;
  try {
    const baskets = await fetch("/api/assets/tam-baskets").then(r => r.json());
    container.innerHTML = baskets.map(b => `
      <button class="btn btn-sm" title="${b.description} (${b.count} stocks)"
        onclick="loadTamBasket('${b.key}', '${b.label.replace(/'/g, "\\'")}')">
        ${b.label} <span style="opacity:0.55;font-size:0.72rem">${b.count}</span>
      </button>
    `).join("");
  } catch {
    if (container) container.innerHTML = '<span style="font-size:0.8rem;color:var(--text3)">Failed to load baskets.</span>';
  }
}

window.loadTamBasket = async function(key, label) {
  try {
    const data = await fetch(`/api/assets/preset?name=${key}`).then(r => r.json());
    const list = Array.isArray(data) ? data : (data.assets || []);
    let added = 0;
    list.forEach(a => {
      if (!state.assets.find(x => x.ticker === a.ticker)) {
        state.assets.push(a);
        added++;
      }
    });
    renderAssets();
    toast(`Added ${added} stocks from ${label}.`, "ok");
  } catch {
    toast("Failed to load basket.", "err");
  }
};

loadTamBaskets();

// ════════════════════════════════════════════════════════════════
//  PORTFOLIO SECTION
// ════════════════════════════════════════════════════════════════

const pfSection = $("portfolioSection");
const pfStepSections = () => document.querySelectorAll(".step:not(#portfolioSection)");

let _pfPositionsCache = [];

function openPortfolio() {
  pfStepSections().forEach(el => hide(el));
  show(pfSection);
  document.querySelectorAll(".step-btn[data-step]").forEach(b => b.classList.remove("active", "done"));
  // Default buy date = today
  const today = new Date().toISOString().slice(0, 10);
  if ($("pfPosDate") && !$("pfPosDate").value) $("pfPosDate").value = today;
  loadPositions();
  loadPfWatchlist();
}

function closePortfolio() {
  hide(pfSection);
  goTo(state.step);
}

$("portfolioNavBtn").addEventListener("click", openPortfolio);
$("portfolioClose").addEventListener("click", closePortfolio);
$("portfolioRefresh").addEventListener("click", loadPositions);

window.pfSwitchTab = function(tab) {
  const isPositions = tab === "positions";
  $("pfTabPositions").classList.toggle("active", isPositions);
  $("pfTabWatchlist").classList.toggle("active", !isPositions);
  isPositions ? show($("pfPositionsPane")) : hide($("pfPositionsPane"));
  isPositions ? hide($("pfWatchlistPane")) : show($("pfWatchlistPane"));
};

function fmtUSD(v) {
  if (v === null || v === undefined) return "—";
  const n = Number(v);
  if (isNaN(n)) return "—";
  return "$" + n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtPnl(v) {
  if (v === null || v === undefined) return '<span class="muted">—</span>';
  const n = Number(v);
  if (isNaN(n)) return '<span class="muted">—</span>';
  const cls = n > 0 ? "pos" : n < 0 ? "neg" : "";
  const sign = n > 0 ? "+" : "";
  return `<span class="${cls}">${sign}${fmtUSD(n)}</span>`;
}

function fmtSignal(sig) {
  if (!sig) return '<span class="muted">—</span>';
  const tip = `RSI ${sig.rsi} · MA50 ${sig.ma50} · MA200 ${sig.ma200}`;
  return `<span title="${tip}" style="color:${sig.color};font-weight:700;font-size:0.75rem">${sig.action}</span>`;
}

async function loadPositions() {
  const btn = $("portfolioRefresh");
  if (btn) btn.textContent = "↻ Refreshing…";
  try {
    const d = await fetch("/api/portfolio/positions").then(r => r.json());
    _pfPositionsCache = d.positions || [];

    // Summary cards
    $("pfMarketValue").textContent = fmtUSD(d.market_value);
    $("pfCostBasis").textContent   = fmtUSD(d.cost_basis);
    $("pfUnrealized").innerHTML    = fmtPnl(d.unrealized_pnl);
    $("pfReturnPct").innerHTML     = fmtPct(d.return_pct);

    // Positions table
    const container = $("pfPositionsTable");
    if (!_pfPositionsCache.length) {
      container.innerHTML = '<div class="empty-state">No positions yet. Add one above to start tracking.</div>';
      return;
    }
    container.innerHTML = `
      <table class="results-table" style="width:100%">
        <thead><tr>
          <th>Ticker</th><th>Qty</th><th>Buy Price</th><th>Buy Date</th>
          <th>Days</th><th>Current</th><th>Mkt Value</th>
          <th>PnL</th><th>Return %</th><th>Annlzd</th>
          <th>Signal</th><th>Notes</th><th></th>
        </tr></thead>
        <tbody>
          ${_pfPositionsCache.map(p => `
            <tr>
              <td><strong>${p.ticker}</strong></td>
              <td>${p.quantity}</td>
              <td>${fmtUSD(p.buy_price)}</td>
              <td style="font-size:0.78rem">${p.buy_date || "—"}</td>
              <td style="color:var(--text2)">${p.days_held ?? "—"}</td>
              <td>${fmtUSD(p.current_price)}</td>
              <td>${fmtUSD(p.market_value)}</td>
              <td>${fmtPnl(p.unrealized_pnl)}</td>
              <td>${fmtPct(p.unrealized_pct)}</td>
              <td>${p.annualized_return_pct == null ? '<span class="muted">—</span>' : fmtPct(p.annualized_return_pct)}</td>
              <td>${fmtSignal(p.signal)}</td>
              <td style="font-size:0.75rem;color:var(--text2);max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${(p.notes || '').replace(/"/g, '&quot;')}">${p.notes || "—"}</td>
              <td><button class="btn btn-sm btn-danger" onclick="pfRemovePosition('${p.id}')">✕</button></td>
            </tr>
          `).join("")}
        </tbody>
      </table>`;
  } catch {
    toast("Failed to load positions.", "err");
  } finally {
    if (btn) btn.textContent = "↻ Refresh";
  }
}

// Add position form
$("pfPosAdd").addEventListener("click", async () => {
  const ticker = ($("pfPosTicker").value || "").trim().toUpperCase();
  const quantity = parseFloat($("pfPosQty").value);
  const buy_price = parseFloat($("pfPosPrice").value);
  const buy_date = $("pfPosDate").value;
  const notes = ($("pfPosNotes").value || "").trim();

  if (!ticker) { toast("Ticker is required.", "err"); return; }
  if (!(quantity > 0)) { toast("Quantity must be > 0.", "err"); return; }
  if (!(buy_price > 0)) { toast("Buy price must be > 0.", "err"); return; }

  try {
    const r = await fetch("/api/portfolio/positions", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ ticker, quantity, buy_price, buy_date, notes }),
    });
    const d = await r.json();
    if (!r.ok) { toast(d.error || "Failed to add.", "err"); return; }
    // Clear inputs (keep date)
    $("pfPosTicker").value = "";
    $("pfPosQty").value = "";
    $("pfPosPrice").value = "";
    $("pfPosNotes").value = "";
    await loadPositions();
    toast(`${ticker} added.`, "ok");
  } catch { toast("Failed to add position.", "err"); }
});

window.pfRemovePosition = async function(id) {
  if (!confirm("Remove this position?")) return;
  try {
    await fetch(`/api/portfolio/positions/${id}`, { method: "DELETE" });
    await loadPositions();
    toast("Position removed.", "ok");
  } catch { toast("Failed to remove.", "err"); }
};

// Analyze positions: load unique tickers into Step 1 and navigate there
$("pfAnalyzePositions").addEventListener("click", () => {
  if (!_pfPositionsCache.length) { toast("No positions to analyze.", "err"); return; }
  const seen = new Set();
  state.assets = [];
  _pfPositionsCache.forEach(p => {
    if (!seen.has(p.ticker)) {
      seen.add(p.ticker);
      state.assets.push({ ticker: p.ticker, name: p.name || p.ticker, sector: "", currency: "USD" });
    }
  });
  renderAssets();
  closePortfolio();
  goTo(1);
  toast(`Loaded ${state.assets.length} tickers for analysis.`, "ok");
});

// ── Watchlist ─────────────────────────────────────────────────

let _pfWatchlist = [];

async function loadPfWatchlist() {
  try {
    _pfWatchlist = await fetch("/api/portfolio/watchlist").then(r => r.json());
    renderPfWatchlist();
  } catch { toast("Failed to load watchlist.", "err"); }
}

function renderPfWatchlist() {
  const container = $("pfWatchlistTable");
  if (!_pfWatchlist.length) {
    container.innerHTML = '<div class="empty-state">Watchlist is empty.</div>';
    return;
  }
  container.innerHTML = `
    <table class="results-table" style="width:100%">
      <thead><tr><th>Ticker</th><th>Name</th><th>Sector</th><th></th></tr></thead>
      <tbody>
        ${_pfWatchlist.map(w => `
          <tr>
            <td><strong>${w.ticker}</strong></td>
            <td>${w.name || "—"}</td>
            <td style="color:var(--text2)">${w.sector || "—"}</td>
            <td><button class="btn btn-sm btn-danger" onclick="pfRemoveWatch('${w.ticker}')">✕</button></td>
          </tr>
        `).join("")}
      </tbody>
    </table>`;
}

$("pfWatchAdd").addEventListener("click", async () => {
  const ticker = ($("pfWatchTicker").value || "").trim().toUpperCase();
  const name   = ($("pfWatchName").value || "").trim();
  if (!ticker) { toast("Enter a ticker.", "err"); return; }
  try {
    const r = await fetch("/api/portfolio/watchlist", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ ticker, name }),
    });
    const d = await r.json();
    if (!r.ok) { toast(d.error || "Error adding.", "err"); return; }
    $("pfWatchTicker").value = "";
    $("pfWatchName").value   = "";
    await loadPfWatchlist();
    toast(`${ticker} added to watchlist.`, "ok");
  } catch { toast("Failed to add.", "err"); }
});

window.pfRemoveWatch = async function(ticker) {
  try {
    await fetch(`/api/portfolio/watchlist/${ticker}`, { method: "DELETE" });
    await loadPfWatchlist();
    toast(`${ticker} removed.`, "ok");
  } catch { toast("Failed to remove.", "err"); }
};

$("pfAnalyzeWatchlist").addEventListener("click", () => {
  if (!_pfWatchlist.length) { toast("Watchlist is empty.", "err"); return; }
  state.assets = _pfWatchlist.map(w => ({
    ticker: w.ticker, name: w.name || w.ticker, sector: w.sector || "", currency: w.currency || "USD",
  }));
  renderAssets();
  closePortfolio();
  goTo(1);
  toast(`Loaded ${state.assets.length} watchlist assets.`, "ok");
});

// ════════════════════════════════════════════════════════════════
//  Run analysis
// ════════════════════════════════════════════════════════════════

async function startAnalysis() {
  // Build config
  const tech = [...document.querySelectorAll("input[name='tech']:checked")].map(c => c.value);
  const fund = [...document.querySelectorAll("input[name='fund']:checked")].map(c => c.value);
  const sentEnabled = $("sentEnabled").checked;

  const provider = document.querySelector("input[name='sentProvider']:checked")?.value || "local";
  const apiKey   = $("apiKey").value.trim();
  const sentModel = $("sentModel").value.trim();
  const modelPath = $("modelPath")?.value.trim() || "";
  const adapterPath = $("adapterPath")?.value.trim() || "";

  // ML config
  const mlEnabled = $("mlEnabled").checked;

  state.config = {
    assets: state.assets.map(a => ({
      ticker:   a.ticker,
      name:     a.name,
      sector:   a.sector,
      currency: a.currency || "USD",
    })),
    indicators: {
      period:      $("dataPeriod").value,
      technical:   tech,
      fundamental: fund,
    },
    sentiment: {
      enabled:      sentEnabled,
      provider:     provider,
      api_key:      apiKey,
      model:        sentModel,
      model_path:   modelPath,
      adapter_path: adapterPath,
      max_articles: parseInt($("maxArticles").value) || 50,
      days: parseInt($("newsDays").value) || 15,
    },
    ml: {
      enabled:           mlEnabled,
      model_type:        $("mlBackend")?.value || "lightgbm",
      backend:           ($("mlBackend")?.value === "mlp") ? "auto" : "cpu",
      training_period:   $("mlTrainPeriod")?.value || "5y",
      forward_horizon:   parseInt($("mlHorizon")?.value) || 10,
      strong_threshold:  parseFloat($("mlThreshold")?.value) || 0.05,
      train_mode:        document.querySelector("input[name='mlTrainMode']:checked")?.value || "per_ticker",
      feature_set:       $("mlFeatureSet")?.value || "full",
      n_trees:           parseInt($("mlTrees")?.value) || 200,
      max_depth:         parseInt($("mlDepth")?.value) || 4,
      epochs:            parseInt($("mlEpochs")?.value) || 100,
      dropout:           parseFloat($("mlDropout")?.value) || 0.3,
    },
    weights: {
      technical:   parseInt($("wTech").value),
      fundamental: parseInt($("wFund").value),
      sentiment:   parseInt($("wSent").value),
    },
  };

  // Persist selected provider + model + local paths (NOT the API key) for next session
  _saveSettings({
    provider: provider,
    sent_model: sentModel,
    model_path: modelPath,
    adapter_path: adapterPath,
  });

  goTo(5);
  show($("progressPanel"));
  hide($("resultsPanel"));
  resetProgress();

  try {
    const res = await fetch("/api/analysis/start", {
      method:  "POST",
      headers: {"Content-Type": "application/json"},
      body:    JSON.stringify(state.config),
    }).then(r => r.json());

    if (!res.ok || !res.task_id) {
      throw new Error(res.error || "Failed to start analysis");
    }
    state.taskId = res.task_id;
    streamProgress(res.task_id);

  } catch (err) {
    toast("Failed to start: " + err.message, "err");
    logLine("Error: " + err.message, "error");
  }
}

// ── Progress / SSE ────────────────────────────────────────────

const STAGE_LABELS = {
  market_data:    "Fetching market data",
  scoring:        "Computing scores",
  ml_training:    "Training ML classifier",
  loading_model:  "Initializing sentiment engine",
  sentiment:      "Analyzing sentiment",
  assembling:     "Assembling results",
};

function resetProgress() {
  $("progressBar").style.width = "0%";
  $("progressPct").textContent  = "0%";
  $("progressStage").textContent = "Initializing…";
  $("progressLog").innerHTML = "";
}

function setProgress(pct, label) {
  $("progressBar").style.width = pct + "%";
  $("progressPct").textContent  = pct + "%";
  if (label) $("progressStage").textContent = label;
}

function logLine(msg, cls = "") {
  const el = document.createElement("div");
  el.className = "progress-log-line " + cls;
  el.textContent = msg;
  const log = $("progressLog");
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
}

function streamProgress(taskId) {
  const es = new EventSource(`/api/analysis/stream/${taskId}`);

  es.onmessage = ev => {
    const data = JSON.parse(ev.data);
    handleEvent(data);
    if (data.type === "complete" || data.type === "error") {
      es.close();
      if (data.type === "complete") fetchResults(taskId);
      else logLine("Analysis failed.", "error");
    }
  };

  es.onerror = () => {
    es.close();
    logLine("Connection lost – fetching results…", "warn");
    fetchResults(taskId);
  };
}

function handleEvent(ev) {
  switch (ev.type) {
    case "start":
      logLine(`Starting analysis of ${ev.total} assets…`);
      break;
    case "progress": {
      const stage = ev.stage || "";
      const lbl   = ev.msg || STAGE_LABELS[stage] || stage;
      const pct   = ev.pct || 0;
      const detail = ev.ticker ? ` [${ev.ticker}]` : "";
      setProgress(pct, lbl + detail);
      if (ev.ticker && stage) logLine(`${lbl}: ${ev.ticker} (${ev.done + 1}/${ev.total})`);
      break;
    }
    case "warn":
      logLine(`⚠ ${ev.ticker || ""} ${ev.msg || ""}`, "warn");
      break;
    case "complete":
      setProgress(100, "Done!");
      logLine("Analysis complete.");
      break;
    case "error":
      logLine("Error: " + (ev.message || "unknown"), "error");
      break;
  }
}

async function fetchResults(taskId) {
  try {
    const data = await fetch(`/api/analysis/results/${taskId}`).then(r => r.json());
    if (data.status === "running") {
      setTimeout(() => fetchResults(taskId), 1000);
      return;
    }
    if (data.status === "error") {
      logLine("Server error: " + data.error, "error");
      return;
    }
    state.results = data.results;
    state.lastDate = new Date().toLocaleString();
    renderResults(data.results);
  } catch (err) {
    logLine("Failed to fetch results: " + err.message, "error");
  }
}

// ════════════════════════════════════════════════════════════════
//  Render results
// ════════════════════════════════════════════════════════════════

const hasSent = () => state.results && state.results.some(r => r.sent_score !== null && r.sent_score !== undefined);
const hasML   = () => state.results && state.results.some(r => r.ml_regime !== null && r.ml_regime !== undefined);

const TABLE_COLS = [
  { key: "#",           label: "#",          render: (r,i) => i+1,             numeric: true  },
  { key: "ticker",      label: "Ticker",      render: r => `<strong>${r.ticker}</strong>`, numeric: false },
  { key: "name",        label: "Company",     render: r => r.name,               numeric: false },
  { key: "sector",      label: "Sector",      render: r => `<span class="muted">${r.sector||"—"}</span>`, numeric: false },
  { key: "price",       label: "Price",       render: r => r.price ? `${r.symbol||"$"}${r.price.toFixed(2)}` : "—", numeric: true },
  { key: "ret_1d",      label: "1D %",        render: r => fmtPct(r.ret_1d),    numeric: true  },
  { key: "ret_1w",      label: "1W %",        render: r => fmtPct(r.ret_1w),    numeric: true  },
  { key: "ret_1m",      label: "1M %",        render: r => fmtPct(r.ret_1m),    numeric: true  },
  { key: "ret_3m",      label: "3M %",        render: r => fmtPct(r.ret_3m),    numeric: true  },
  { key: "rs_1m",       label: "RS 1M",       render: r => fmtRS(r.rs_1m),      numeric: true  },
  { key: "rs_55d",      label: "RS 55D",      render: r => fmtRS(r.rs_55d),     numeric: true  },
  { key: "rs_3m",       label: "RS 3M",       render: r => fmtRS(r.rs_3m),      numeric: true  },
  { key: "w52_pct",     label: "52W Pos",     render: r => fmt(r.w52_pct, 1, "%"), numeric: true },
  { key: "rsi",         label: "RSI 14",      render: r => {
      if (r.rsi === null || r.rsi === undefined) return "—";
      const cls = r.rsi > 70 ? "neg" : r.rsi < 30 ? "pos" : "";
      return `<span class="${cls}">${r.rsi.toFixed(1)}</span>`;
    }, numeric: true },
  { key: "ma_cross",    label: "MA Cross",    render: r => r.ma_cross === "golden"
      ? '<span class="ma-golden">GOLDEN</span>'
      : r.ma_cross === "death"
      ? '<span class="ma-death">DEATH</span>'
      : '<span class="muted">—</span>',            numeric: false },
  { key: "macd_bull",   label: "MACD",        render: r => r.macd_bull === true
      ? '<span class="pos">▲ Bull</span>'
      : r.macd_bull === false
      ? '<span class="neg">▼ Bear</span>'
      : '<span class="muted">—</span>',            numeric: false },
  { key: "bb_pct",      label: "Boll %",      render: r => fmt(r.bb_pct, 1, "%"), numeric: true },
  { key: "pe_trail",    label: "P/E",         render: r => fmt(r.pe_trail, 1),   numeric: true },
  { key: "pe_fwd",      label: "Fwd P/E",     render: r => fmt(r.pe_fwd, 1),     numeric: true },
  { key: "tech_score",  label: "Tech",        render: r => fmt(r.tech_score, 3), numeric: true },
  { key: "fund_score",  label: "Fund",        render: r => fmt(r.fund_score, 3), numeric: true },
  { key: "trend_stage", label: "Trend",       render: r => trendBadge(r.trend_stage), numeric: false },
  { key: "mkt_regime",  label: "Regime",      render: r => regimeBadge(r.mkt_regime), numeric: false },
  { key: "regime_chg",  label: "Chg",         render: r => regimeChgBadge(r.regime_chg), numeric: false },
  { key: "momentum_score", label: "Mom",      render: r => fmtScore01(r.momentum_score), numeric: true },
  { key: "risk_score",  label: "Risk",        render: r => fmtRisk(r.risk_score), numeric: true },
  { key: "dip_score",   label: "Dip",         render: r => fmtDip(r.dip_score), numeric: true },
  { key: "adj_confidence", label: "Conf%",    render: r => fmtConf(r.adj_confidence), numeric: true },
  // Sentiment columns inserted dynamically
  { key: "overall_score", label: "Overall",   render: r => fmt(r.overall_score, 3), numeric: true },
  { key: "ctx_signal",  label: "Signal",      render: r => sigPill(r.ctx_signal || r.signal, r.signal_css), numeric: false },
  { key: "cs_z_score",  label: "CS Z",        render: r => {
      const v = r.cs_z_score;
      if (v === null || v === undefined) return '<span class="muted">—</span>';
      const cls = v >= 1 ? "pos" : v <= -1 ? "neg" : "";
      return `<span class="${cls}" title="Cross-sectional z-score vs. universe">${v >= 0 ? "+" : ""}${v.toFixed(2)}</span>`;
    }, numeric: true },
  { key: "cs_rank_pct", label: "CS Rank",     render: r => {
      const v = r.cs_rank_pct;
      if (v === null || v === undefined) return '<span class="muted">—</span>';
      return `<span title="Percentile rank in current analysis universe">${(v * 100).toFixed(0)}%</span>`;
    }, numeric: true },
  { key: "_llm",        label: "LLM",         render: r => `<button class="btn btn-xs btn-outline llm-validate-btn" data-ticker="${r.ticker}">🤖 Validate</button>`, numeric: false },
  { key: "_altdata",    label: "Alt Data",    render: r => `<button class="btn btn-xs btn-outline alt-data-btn" data-ticker="${r.ticker}" onclick="openAltData('${r.ticker}')">📊 Alt Data</button>`, numeric: false },
];

const ML_COLS = [
  { key: "ml_regime",      label: "ML Regime", render: r => mlRegimeBadge(r.ml_regime), numeric: false },
  { key: "ml_entry",       label: "Entry",     render: r => fmtEntryExit(r.ml_entry, false), numeric: true },
  { key: "ml_exit",        label: "Exit",      render: r => fmtEntryExit(r.ml_exit, true), numeric: true },
  { key: "ml_decision",    label: "Decision",  render: r => fmtDecision(r), numeric: false },
  { key: "ml_uncertainty",  label: "Uncert",   render: r => fmtUncertainty(r), numeric: false },
  { key: "ml_signal",      label: "ML Sig",    render: r => mlSigPill(r.ml_signal), numeric: false },
];

const SENT_COLS = [
  { key: "sent_score",  label: "Sent Score",  render: r => {
      const v = r.sent_score;
      if (v === null || v === undefined) return '<span class="muted">—</span>';
      const cls = v >= 0.2 ? "pos" : v <= -0.2 ? "neg" : "";
      return `<span class="${cls}">${v.toFixed(3)}</span>`;
    }, numeric: true },
  { key: "sent_signal", label: "Sent Signal", render: r => sentPill(r.sent_signal), numeric: false },
  { key: "n_articles",  label: "Articles",    render: r => r.n_articles || "—", numeric: true },
];

function renderResults(results) {
  hide($("progressPanel"));
  show($("resultsPanel"));

  $("resultsCount").textContent = results.length + " assets";
  $("resultsDate").textContent  = state.lastDate || "";

  renderSummaryCards(results);
  buildTable(results);

  // Filter
  $("filterInput").addEventListener("input", filterTable);

  // LLM validate buttons (event delegation)
  $("tableBody").addEventListener("click", (ev) => {
    const btn = ev.target.closest(".llm-validate-btn");
    if (!btn) return;
    const ticker = btn.dataset.ticker;
    const row = state.results.find(r => r.ticker === ticker);
    if (row) openLLMValidator(row);
  });
}

// ── LLM + RAG news validator ────────────────────────────────────────────────
async function openLLMValidator(row) {
  const modal = $("llmModal");
  const body  = $("llmModalBody");
  show(modal);
  body.innerHTML = `<p class="hint">Checking LLM API…</p>`;

  // 1. Status
  let status;
  try { status = await (await fetch("/api/llm/status")).json(); }
  catch (e) { body.innerHTML = `<p class="error">Could not reach backend: ${e}</p>`; return; }
  if (!status.available) {
    body.innerHTML = `
      <p class="error">LLM API not available.</p>
      <p class="hint">Make sure <code>API_KEY</code> is set in the project <code>.env</code> file.</p>
      <p class="hint">Error: ${status.error || ""}</p>`;
    return;
  }

  body.innerHTML = `
    <p style="margin:0 0 0.5rem">
      Running two-turn analysis for <strong>${row.ticker}</strong>
      with <code>${status.model || "gpt-oss-120b"}</code>
    </p>
    <ol class="llm-steps">
      <li id="llm-step-news" class="llm-step-active">Retrieving live news, SEC filings &amp; analyst data…</li>
      <li id="llm-step-t1"   class="llm-step-pending">Turn 1 — independent analysis (thinking…)</li>
      <li id="llm-step-t2"   class="llm-step-pending">Turn 2 — debate with model scores</li>
    </ol>
    <div class="bt-spinner"></div>
    <p class="hint" style="margin-top:0.4rem">~50–90 s total. The LLM reasons step-by-step before scoring.</p>`;

  // 2. Validate (single call, backend handles both turns)
  let res;
  try {
    res = await (await fetch("/api/llm/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ticker: row.ticker,
        company_name: row.name || "",
        analysis: row,
      }),
    })).json();
  } catch (e) {
    body.innerHTML = `<p class="error">Request failed: ${e}</p>`;
    return;
  }

  if (!res.ok && res.error && !res.turn1) {
    body.innerHTML = `<p class="error">${res.error}</p>`;
    return;
  }

  // ── helpers ───────────────────────────────────────────────────────────────
  const esc   = s => String(s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  const list  = arr => (arr?.length)
    ? `<ul>${arr.map(x => `<li>${esc(x)}</li>`).join("")}</ul>`
    : `<p class="hint muted">—</p>`;

  const srcMap = {};
  (res.sources||[]).forEach(s => { srcMap[s.n] = s; });
  const srcLink  = n => { const s = srcMap[n]; return s?.url ? `<a href="${esc(s.url)}" target="_blank">[${n}]</a>` : `[${n}]`; };
  const srcNums  = arr => (arr||[]).map(srcLink).join(" ");

  const badge = (text, color) =>
    `<span class="badge" style="background:${color};color:#fff;padding:0.25rem 0.6rem">${esc(text)}</span>`;

  const verdictColor = { AGREE:"#22c55e", DISAGREE:"#ef4444", MIXED:"#eab308",
                         INSUFFICIENT_NEWS:"#94a3b8", PARSE_ERROR:"#ef4444" };
  const actionColor  = { HOLD_SIGNAL:"#22c55e", UPGRADE:"#3b82f6",
                         DOWNGRADE:"#ef4444", REVIEW_MANUALLY:"#eab308" };
  const sigColor     = { STRONG_BUY:"#16a34a", BUY:"#22c55e", NEUTRAL:"#eab308",
                         SELL:"#f97316", STRONG_SELL:"#ef4444" };
  const riskColor    = { VERY_LOW:"#22c55e", LOW:"#86efac", MODERATE:"#eab308",
                         HIGH:"#f97316", VERY_HIGH:"#ef4444" };

  const t1 = res.turn1 || {};
  const t2 = res.turn2 || {};

  // ── score comparison table ────────────────────────────────────────────────
  const fmtDelta = (d) => {
    if (d == null) return "—";
    const v = Number(d); const s = v >= 0 ? "+" : "";
    const c = Math.abs(v) < 0.1 ? "#22c55e" : Math.abs(v) < 0.25 ? "#eab308" : "#ef4444";
    return `<span style="color:${c}">${s}${v.toFixed(2)}</span>`;
  };
  const fmtScore = v => v != null ? Number(v).toFixed(2) : "—";

  const scoreRows = (cmp) => {
    if (!cmp) return "";
    const dims = [
      ["overall_signal", "Overall Signal"],
      ["technical",      "Technical"],
      ["fundamental",    "Fundamental"],
      ["sentiment",      "Sentiment"],
      ["risk",           "Risk"],
      ["dip",            "Dip Opportunity"],
    ];
    return dims.map(([k, label]) => {
      const d = cmp[k];
      if (!d) return "";
      const vColor = { ALIGNED:"#22c55e", LLM_HIGHER:"#3b82f6", LLM_LOWER:"#f97316" }[d.verdict] || "#94a3b8";
      return `<tr>
        <td>${label}</td>
        <td style="text-align:center">${fmtScore(d.llm)}</td>
        <td style="text-align:center">${fmtScore(d.model)}</td>
        <td style="text-align:center">${fmtDelta(d.delta)}</td>
        <td><span style="color:${vColor};font-size:0.8rem">${d.verdict||""}</span></td>
        <td class="hint" style="font-size:0.8rem">${esc(d.comment||"")}</td>
      </tr>`;
    }).join("");
  };

  const trendRow = (t2.trend_comparison)
    ? `<tr><td>Trend Stage</td><td colspan="2" style="text-align:center">${esc(t1.trend_assessment||"?")} vs ${esc(t2.trend_comparison?.model||"?")}</td>
       <td colspan="2"><span style="color:${t2.trend_comparison.verdict==="ALIGNED"?"#22c55e":"#f97316"};font-size:0.8rem">${t2.trend_comparison.verdict||""}</span></td>
       <td class="hint" style="font-size:0.8rem">${esc(t2.trend_comparison.comment||"")}</td></tr>` : "";

  const regimeRow = (t2.regime_comparison)
    ? `<tr><td>Regime</td><td colspan="2" style="text-align:center">${esc(t1.regime_assessment||"?")} vs ${esc(t2.regime_comparison?.model||"?")}</td>
       <td colspan="2"><span style="color:${t2.regime_comparison.verdict==="ALIGNED"?"#22c55e":"#f97316"};font-size:0.8rem">${t2.regime_comparison.verdict||""}</span></td>
       <td class="hint" style="font-size:0.8rem">${esc(t2.regime_comparison.comment||"")}</td></tr>` : "";

  // ── sources list ──────────────────────────────────────────────────────────
  const catColor = { news:"#3b82f6", earnings:"#22c55e", analysts:"#8b5cf6",
                     bearish:"#ef4444", macro:"#f59e0b", sector:"#06b6d4",
                     insider:"#ec4899", catalysts:"#10b981" };
  const sourcesList = (res.sources||[]).map(s => {
    const cc = catColor[s.category] || "#94a3b8";
    return `<li>
      <span style="color:${cc};font-size:0.7rem;font-weight:600;text-transform:uppercase">${s.category}</span>
      <a href="${esc(s.url)}" target="_blank"> ${esc(s.title)}</a>
      <span class="hint"> ${s.date}${s.source?" · "+esc(s.source):""}</span>
    </li>`;
  }).join("");

  const edgarList = (res.edgar_filings||[]).map(f =>
    `<li><strong>${f.form}</strong> ${f.date} — ${esc(f.title)} <a href="${esc(f.url)}" target="_blank">↗</a></li>`
  ).join("") || "<li class='hint'>None found</li>";

  const analystBlock = (() => {
    const ad = res.analyst_data || {};
    if (!Object.keys(ad).length) return "<p class='hint'>—</p>";
    const rows = [];
    if (ad.analyst_targets) {
      const t = ad.analyst_targets;
      rows.push(`<p>Price targets: mean <strong>$${t.mean}</strong>, range $${t.low}–$${t.high} (${t.n} analysts)</p>`);
    }
    if (ad.consensus) {
      const c = ad.consensus;
      rows.push(`<p>Consensus: SB=${c.strong_buy} B=${c.buy} H=${c.hold} S=${c.sell} SS=${c.strong_sell}</p>`);
    }
    if (ad.next_earnings) rows.push(`<p>Next earnings: <strong>${ad.next_earnings}</strong></p>`);
    if (ad.last_earnings_surprise) {
      const s = ad.last_earnings_surprise;
      const col = s.surprise_pct >= 0 ? "#22c55e" : "#ef4444";
      rows.push(`<p>Last surprise: <span style="color:${col}">${s.surprise_pct >= 0?"+":""}${s.surprise_pct}%</span> on ${s.date}</p>`);
    }
    return rows.join("") || "<p class='hint'>—</p>";
  })();

  // ── final verdict ─────────────────────────────────────────────────────────
  const finalVerdict     = t2.final_verdict     || t1.signal_direction || "?";
  const finalRecommend   = t2.final_recommendation || "REVIEW_MANUALLY";
  const finalConfidence  = t2.final_confidence  != null ? Number(t2.final_confidence) : null;
  const finalSummary     = t2.final_summary     || "";

  body.innerHTML = `
  <!-- header -->
  <div class="llm-header">
    ${badge(finalVerdict, verdictColor[finalVerdict]||"#94a3b8")}
    ${badge(finalRecommend, actionColor[finalRecommend]||"#94a3b8")}
    ${badge(t1.signal_direction||"—", sigColor[t1.signal_direction]||"#94a3b8")}
    <span class="hint">
      conf ${finalConfidence!=null?finalConfidence.toFixed(2):"—"} ·
      ${res.n_sources||0} sources · ${res.model||""}
    </span>
  </div>
  ${finalSummary ? `<p class="llm-summary">${esc(finalSummary)}</p>` : ""}

  <!-- Turn 1: LLM independent assessment -->
  <details class="llm-section" open>
    <summary><strong>Turn 1 — Independent LLM Assessment</strong>
      <span class="hint">(model scores withheld)</span></summary>

    <div class="llm-assessments">
      <div class="llm-assess-card">
        <div class="llm-assess-label">Signal</div>
        <div class="llm-assess-val" style="color:${sigColor[t1.signal_direction]||"inherit"}">${t1.signal_direction||"—"}</div>
        <div class="hint">${fmtScore(t1.signal_score_estimate)}</div>
      </div>
      <div class="llm-assess-card">
        <div class="llm-assess-label">Trend</div>
        <div class="llm-assess-val">${t1.trend_assessment||"—"}</div>
      </div>
      <div class="llm-assess-card">
        <div class="llm-assess-label">Regime</div>
        <div class="llm-assess-val">${t1.regime_assessment||"—"}${t1.regime_change&&t1.regime_change!=="null"?" ("+t1.regime_change+")":""}</div>
      </div>
      <div class="llm-assess-card">
        <div class="llm-assess-label">Sentiment</div>
        <div class="llm-assess-val">${t1.sentiment_assessment||"—"}</div>
        <div class="hint">${fmtScore(t1.sentiment_score_estimate)}</div>
      </div>
      <div class="llm-assess-card">
        <div class="llm-assess-label">Risk</div>
        <div class="llm-assess-val" style="color:${riskColor[t1.risk_level]||"inherit"}">${t1.risk_level||"—"}</div>
        <div class="hint">${fmtScore(t1.risk_score_estimate)}</div>
      </div>
      <div class="llm-assess-card">
        <div class="llm-assess-label">Dip</div>
        <div class="llm-assess-val">${t1.dip_opportunity||"—"}</div>
        <div class="hint">${fmtScore(t1.dip_score_estimate)}</div>
      </div>
    </div>

    ${t1.macro_context||t1.industry_context ? `
    <div class="llm-context-row">
      ${t1.macro_context ? `<div><h4>Macro Context</h4><p>${esc(t1.macro_context)}</p></div>` : ""}
      ${t1.industry_context ? `<div><h4>Industry / Sector</h4><p>${esc(t1.industry_context)}</p></div>` : ""}
    </div>` : ""}

    ${t1.time_horizon_rationale ? `<p class="hint"><em>Horizon: ${esc(t1.time_horizon_rationale)}</em></p>` : ""}

    <div class="llm-grid">
      <div>
        <h4>Bull case ${t1.supporting_sources?.length?"("+srcNums(t1.supporting_sources)+")":""}</h4>
        ${list(t1.bull_case)}
      </div>
      <div>
        <h4>Bear case ${t1.contradicting_sources?.length?"("+srcNums(t1.contradicting_sources)+")":""}</h4>
        ${list(t1.bear_case)}
      </div>
    </div>
    ${t1.key_catalysts?.length ? `<h4>Key Catalysts</h4>${list(t1.key_catalysts)}` : ""}
    ${t1.key_risks?.length ? `<h4>Key Risks</h4>${list(t1.key_risks)}` : ""}
    ${t1.missed_by_quant_model?.length ? `<h4>Likely missed by quant model</h4>${list(t1.missed_by_quant_model)}` : ""}

    ${res.thinking ? `
    <details style="margin-top:0.75rem">
      <summary class="hint">Chain-of-thought reasoning (expand)</summary>
      <pre class="llm-thinking">${esc(res.thinking)}</pre>
    </details>` : ""}
  </details>

  <!-- Turn 2: Debate -->
  <details class="llm-section" open>
    <summary><strong>Turn 2 — Debate with Model Scores</strong></summary>

    ${t2.score_comparison ? `
    <div class="table-wrap" style="margin:0.75rem 0">
      <table class="results-table" style="font-size:0.82rem">
        <thead><tr>
          <th>Dimension</th>
          <th style="text-align:center">LLM est.</th>
          <th style="text-align:center">Model</th>
          <th style="text-align:center">Δ</th>
          <th>Verdict</th>
          <th>Comment</th>
        </tr></thead>
        <tbody>${scoreRows(t2.score_comparison)}${trendRow}${regimeRow}</tbody>
      </table>
    </div>` : ""}

    <div class="llm-grid">
      <div><h4>Agreements</h4>${list(t2.agreements)}</div>
      <div><h4>Disagreements</h4>${list(t2.disagreements)}</div>
    </div>
    ${t2.model_blind_spots?.length ? `<h4>Blind spots &amp; information gaps</h4>${list(t2.model_blind_spots)}` : ""}
  </details>

  <!-- Sources & data used -->
  <details class="llm-section">
    <summary><strong>Sources &amp; Data Used</strong>
      (${res.n_sources||0} web · ${res.edgar_filings?.length||0} SEC filings)</summary>

    <h4>Analyst Data</h4>
    ${analystBlock}

    <h4>SEC EDGAR Filings</h4>
    <ol style="font-size:0.82rem">${edgarList}</ol>

    <h4>Web Sources</h4>
    <ol style="font-size:0.82rem">${sourcesList}</ol>
  </details>

  ${(res.turn1?.verdict==="PARSE_ERROR"||res.turn2?.final_verdict==="PARSE_ERROR") ? `
  <details><summary class="hint">Raw LLM output (debug)</summary>
    <pre class="llm-thinking">${esc((res.turn1?.raw||"")+(res.turn2?.raw||""))}</pre>
  </details>` : ""}
  `;
}

// ── Alternative data panel ────────────────────────────────────────────────────
// Rendered separately so the tweet-count slider can re-render just the social section
function _renderAltSocial(st, fmt2) {
  const src      = st.source === "reddit" ? "Reddit WSB/Stocks" : "StockTwits";
  const nMsgs    = st.n_messages ?? st.n_total ?? 0;
  const bullPct  = st.bull_pct ?? 0;
  const bearPct  = st.bear_pct ?? 0;
  const bullW    = (bullPct * 100).toFixed(1);
  const bearW    = (bearPct * 100).toFixed(1);

  if (!st.ok || !nMsgs) {
    const errMsg = st.error ? `<p class="muted hint" style="font-size:0.78rem">${st.error}</p>` : "";
    return `<p class="muted">No social sentiment data found. ${st.source === "reddit" ? "Reddit" : "StockTwits"} returned 0 posts mentioning this ticker.</p>${errMsg}`;
  }

  const rows = Array.isArray(st.messages) && st.messages.length
    ? `<details style="margin-top:0.6rem"><summary class="hint" style="cursor:pointer;font-size:0.78rem">Show ${st.messages.length} messages</summary>
        <div style="max-height:220px;overflow-y:auto;margin-top:0.4rem">
        ${st.messages.map(m => {
          const sentCls = m.sentiment === "Bullish" ? "pos" : m.sentiment === "Bearish" ? "neg" : "muted";
          const score   = m.score !== undefined ? ` ▲${m.score}` : "";
          return `<div style="padding:0.3rem 0;border-bottom:1px solid var(--border);font-size:0.78rem">
            <span class="${sentCls}">[${m.sentiment || "—"}]</span>
            ${m.body}
            <span class="muted" style="font-size:0.71rem"> — @${m.username}${score}</span>
          </div>`;
        }).join("")}
        </div></details>`
    : "";

  return `
    <div class="st-bar-wrap">
      <div class="st-bull" style="width:${bullW}%" title="Bullish: ${bullW}%">🐂 ${bullW}%</div>
      <div class="st-bear" style="width:${bearW}%" title="Bearish: ${bearW}%">🐻 ${bearW}%</div>
    </div>
    <div class="alt-grid">
      <div class="alt-item"><span class="alt-label">Bull Count</span><span class="alt-val pos">${st.n_bullish ?? "—"}</span></div>
      <div class="alt-item"><span class="alt-label">Bear Count</span><span class="alt-val neg">${st.n_bearish ?? "—"}</span></div>
      <div class="alt-item"><span class="alt-label">Untagged</span><span class="alt-val">${st.n_untagged ?? "—"}</span></div>
      <div class="alt-item"><span class="alt-label">Bull/Bear Ratio</span><span class="alt-val">${fmt2(st.bull_bear_ratio)}</span></div>
    </div>
    ${rows}`;
}

async function openAltData(ticker, stLimit) {
  const modal = $("altDataModal");
  const body  = $("altDataBody");
  const title = $("altDataTicker");
  if (!modal || !body) return;

  stLimit = stLimit || 25;
  if (title) title.textContent = ticker;
  show(modal);
  body.innerHTML = `<div class="bt-spinner"></div><p class="hint" style="text-align:center;margin-top:0.5rem">Loading alternative data…</p>`;

  let data;
  try {
    const resp = await fetch(`/api/altdata/all/${ticker}?st_limit=${stLimit}`);
    data = await resp.json();
  } catch (e) {
    body.innerHTML = `<p class="error">Request failed: ${e}</p>`;
    return;
  }

  const fmt2 = (v, d=2) => (v === null || v === undefined || isNaN(v)) ? "—" : Number(v).toFixed(d);
  const fmtM = v => (v === null || v === undefined) ? "—"
    : v >= 1e9 ? `$${(v/1e9).toFixed(2)}B`
    : v >= 1e6 ? `$${(v/1e6).toFixed(1)}M`
    : `$${v.toLocaleString()}`;

  let html = "";

  // ── Options ──
  const opt = data.options || {};
  const optErr = opt.error && !opt.ok ? `<p class="muted hint">${opt.error}</p>` : "";
  html += `<section class="alt-section">
    <h4>Options Market${opt.expiry ? ` <span class="muted" style="font-weight:400;font-size:0.78rem">exp ${opt.expiry} (${opt.dte}d)</span>` : ""}</h4>
    ${opt.ok ? `<div class="alt-grid">
      <div class="alt-item"><span class="alt-label">ATM IV</span><span class="alt-val">${opt.atm_iv != null ? (opt.atm_iv*100).toFixed(1)+"%" : "—"}</span></div>
      <div class="alt-item"><span class="alt-label">IV / HV30</span><span class="alt-val">${opt.iv_rank != null ? opt.iv_rank.toFixed(2)+"×" : "—"}</span></div>
      <div class="alt-item"><span class="alt-label">P/C Ratio</span><span class="alt-val">${fmt2(opt.put_call_ratio)}</span></div>
      <div class="alt-item"><span class="alt-label">25Δ Skew</span><span class="alt-val">${opt.skew_25d != null ? (opt.skew_25d*100).toFixed(2)+"%" : "—"}</span></div>
      <div class="alt-item"><span class="alt-label">Implied Move</span><span class="alt-val ${opt.earnings_risk_flag ? "neg" : ""}">${opt.implied_move_pct != null ? "±"+opt.implied_move_pct.toFixed(1)+"%" : "—"}</span></div>
      <div class="alt-item"><span class="alt-label">Next Earnings</span><span class="alt-val ${opt.earnings_risk_flag ? "neg" : ""}">${opt.next_earnings || "—"}${opt.earnings_risk_flag ? " ⚠️" : ""}</span></div>
    </div>
    ${opt.earnings_risk_flag ? '<p class="alt-warn">⚠️ Earnings within 7 days — options IV elevated</p>' : ""}
    ` : `<p class="muted">Options data unavailable.</p>${optErr}`}
  </section>`;

  // ── Social Sentiment (StockTwits / Reddit) ──
  const st     = data.stocktwits || {};
  const src    = st.source === "reddit" ? "Reddit (WSB+Stocks)" : "StockTwits";
  const nMsgs  = st.n_messages ?? st.n_total ?? 0;
  html += `<section class="alt-section" id="altSocialSection">
    <h4>Social Sentiment — ${src}
      <span class="muted" style="font-weight:400;font-size:0.78rem">(${nMsgs} posts)</span>
    </h4>
    <div class="alt-tweet-ctrl">
      <label style="font-size:0.78rem">Posts to fetch:
        <input type="range" id="stLimitSlider" min="5" max="50" step="5" value="${stLimit}"
               style="vertical-align:middle;width:120px;margin:0 0.5rem">
        <span id="stLimitVal">${stLimit}</span>
      </label>
      <button class="btn btn-xs btn-outline" onclick="reloadAltSocial('${ticker}')">Refresh</button>
    </div>
    <div id="altSocialBody">${_renderAltSocial(st, fmt2)}</div>
  </section>`;

  // ── Insider transactions ──
  const ins = data.insiders || {};
  const insErr = ins.error && !ins.ok ? `<p class="muted hint">${ins.error}</p>` : "";
  html += `<section class="alt-section">
    <h4>Insider Transactions — EDGAR Form 4 (90 days)</h4>
    ${(ins.n_buys || ins.n_sells) ? `
    <div class="alt-grid">
      <div class="alt-item"><span class="alt-label">Buys</span><span class="alt-val pos">${ins.n_buys ?? 0}</span></div>
      <div class="alt-item"><span class="alt-label">Sells</span><span class="alt-val neg">${ins.n_sells ?? 0}</span></div>
      <div class="alt-item"><span class="alt-label">Buy Value</span><span class="alt-val pos">${fmtM(ins.total_buy_value)}</span></div>
      <div class="alt-item"><span class="alt-label">Sell Value</span><span class="alt-val neg">${fmtM(ins.total_sell_value)}</span></div>
      <div class="alt-item"><span class="alt-label">Buy/Sell Ratio</span><span class="alt-val">${fmt2(ins.buy_sell_ratio)}</span></div>
      <div class="alt-item"><span class="alt-label">Filings</span><span class="alt-val">${ins.n_filings ?? 0}</span></div>
    </div>
    ${Array.isArray(ins.transactions) && ins.transactions.length ? `
    <table class="bt-table" style="margin-top:0.5rem;font-size:0.78rem">
      <thead><tr><th>Date</th><th>Type</th><th>Security</th><th>Shares</th><th>Price</th><th>Value</th></tr></thead>
      <tbody>${ins.transactions.slice(0, 12).map(t => `
        <tr>
          <td>${t.date || t.filing_date || "—"}</td>
          <td><span class="${t.type === "buy" ? "pos" : "neg"}">${(t.type || "").toUpperCase()}</span></td>
          <td style="max-width:90px;overflow:hidden;text-overflow:ellipsis">${t.title || "Common"}</td>
          <td>${t.shares ? Number(t.shares).toLocaleString() : "—"}</td>
          <td>${t.price ? "$"+t.price.toFixed(2) : "—"}</td>
          <td>${fmtM(t.value)}</td>
        </tr>`).join("")}
      </tbody>
    </table>` : ""}
    ` : `<p class="muted">No open-market insider transactions found in the last 90 days.</p>${insErr}`}
  </section>`;

  // ── FINRA Daily Short Volume ──
  const short = data.short_interest || {};
  const shortErr = short.error && !short.ok ? `<p class="muted hint">${short.error}</p>` : "";
  html += `<section class="alt-section">
    <h4>FINRA Short Volume (Daily)</h4>
    ${short.short_pct != null ? `
    <div class="alt-grid">
      <div class="alt-item"><span class="alt-label">Short % of Vol</span>
        <span class="alt-val ${short.short_pct > 0.5 ? "neg" : short.short_pct > 0.35 ? "" : "pos"}">${(short.short_pct*100).toFixed(1)}%</span>
      </div>
      <div class="alt-item"><span class="alt-label">Prev Day</span>
        <span class="alt-val">${short.short_pct_prev != null ? (short.short_pct_prev*100).toFixed(1)+"%" : "—"}</span>
      </div>
      <div class="alt-item"><span class="alt-label">Change</span>
        <span class="alt-val ${(short.short_pct_chg||0) > 0.03 ? "neg" : (short.short_pct_chg||0) < -0.03 ? "pos" : ""}">
          ${short.short_pct_chg != null ? ((short.short_pct_chg*100)>=0?"+":"")+(short.short_pct_chg*100).toFixed(1)+"%" : "—"}
        </span>
      </div>
      <div class="alt-item"><span class="alt-label">Short Volume</span><span class="alt-val">${short.short_volume?.toLocaleString() ?? "—"}</span></div>
      <div class="alt-item"><span class="alt-label">Total Volume</span><span class="alt-val">${short.total_volume?.toLocaleString() ?? "—"}</span></div>
      <div class="alt-item"><span class="alt-label">As of</span><span class="alt-val" style="font-size:0.78rem">${short.latest_date || "—"}</span></div>
    </div>
    <p class="hint" style="font-size:0.73rem;margin-top:0.3rem">Short % > 50% of daily volume is normal for liquid large-caps. Context matters — use trend (change) rather than absolute level.</p>
    ` : `<p class="muted">No FINRA daily short data available for this ticker.</p>${shortErr}`}
  </section>`;

  body.innerHTML = html;

  // Wire up the tweet-count slider live label
  const slider = $("stLimitSlider");
  if (slider) {
    slider.addEventListener("input", () => {
      const lbl = $("stLimitVal");
      if (lbl) lbl.textContent = slider.value;
    });
  }
}

async function reloadAltSocial(ticker) {
  const slider  = $("stLimitSlider");
  const limit   = slider ? parseInt(slider.value) : 25;
  const bodyEl  = $("altSocialBody");
  const hdr     = document.querySelector("#altSocialSection h4");
  if (!bodyEl) return;

  bodyEl.innerHTML = `<div class="bt-spinner" style="margin:0.5rem 0"></div>`;
  try {
    const resp = await fetch(`/api/altdata/stocktwits/${ticker}?limit=${limit}`);
    const st   = await resp.json();
    const fmt2 = (v, d=2) => (v === null || v === undefined || isNaN(v)) ? "—" : Number(v).toFixed(d);
    const src  = st.source === "reddit" ? "Reddit (WSB+Stocks)" : "StockTwits";
    const n    = st.n_messages ?? st.n_total ?? 0;
    if (hdr) hdr.innerHTML = `Social Sentiment — ${src} <span class="muted" style="font-weight:400;font-size:0.78rem">(${n} posts)</span>`;
    bodyEl.innerHTML = _renderAltSocial(st, fmt2);
  } catch (e) {
    bodyEl.innerHTML = `<p class="error">Reload failed: ${e}</p>`;
  }
}

function renderSummaryCards(results) {
  const counts = { sbuy: 0, buy: 0, neu: 0, sell: 0, ssell: 0 };
  results.forEach(r => {
    const css = (r.signal_css || "sig-neu").replace("sig-", "");
    if (counts[css] !== undefined) counts[css]++;
  });

  const labels = { sbuy: "Strong Buy", buy: "Buy", neu: "Neutral", sell: "Sell", ssell: "Strong Sell" };
  $("summaryCards").innerHTML = Object.entries(counts).map(([cls, cnt]) => `
    <div class="summary-card ${cls}">
      <div class="summary-card-label">${labels[cls]}</div>
      <div class="summary-card-val" style="color:var(--${cls === "neu" ? "amber" : cls.startsWith("s") ? "dark-" + (cls === "sbuy" ? "green" : "red") : cls === "buy" ? "green" : "red"})">${cnt}</div>
    </div>
  `).join("");
}

function buildTable(results) {
  const cols = [...TABLE_COLS];
  // Insert ML cols before "overall_score"
  if (hasML()) {
    const oi = cols.findIndex(c => c.key === "overall_score");
    cols.splice(oi, 0, ...ML_COLS);
  }
  // Insert sentiment cols before "overall_score"
  if (hasSent()) {
    const oi = cols.findIndex(c => c.key === "overall_score");
    cols.splice(oi, 0, ...SENT_COLS);
  }

  // Headers
  const thead = $("tableHead");
  thead.innerHTML = cols.map((c, i) => `
    <th data-col="${i}" class="">${c.label}</th>
  `).join("");

  thead.querySelectorAll("th").forEach(th => {
    th.addEventListener("click", () => sortTable(parseInt(th.dataset.col)));
  });

  renderTableBody(results, cols);
  // store cols on state for sorting
  state._cols = cols;
}

function renderTableBody(results, cols) {
  const tbody = $("tableBody");
  tbody.innerHTML = results.map((r, i) => `
    <tr data-ticker="${r.ticker}">
      ${cols.map(c => `<td>${c.render(r, i)}</td>`).join("")}
    </tr>
  `).join("");
}

function sortTable(colIdx) {
  const col = state._cols[colIdx];
  if (!col) return;

  if (state.sortCol === colIdx) state.sortDir *= -1;
  else { state.sortCol = colIdx; state.sortDir = -1; }

  document.querySelectorAll("#tableHead th").forEach((th, i) => {
    th.className = i === colIdx ? (state.sortDir === -1 ? "sort-desc" : "sort-asc") : "";
  });

  const sorted = [...state.results].sort((a, b) => {
    let va = a[col.key];
    let vb = b[col.key];
    if (va === null || va === undefined) va = col.numeric ? -Infinity : "";
    if (vb === null || vb === undefined) vb = col.numeric ? -Infinity : "";
    if (col.numeric) return (va - vb) * state.sortDir;
    return String(va).localeCompare(String(vb)) * state.sortDir;
  });

  renderTableBody(sorted, state._cols);
  filterTable(); // re-apply filter
}

function filterTable() {
  const q = ($("filterInput").value || "").toLowerCase();
  document.querySelectorAll("#tableBody tr").forEach(tr => {
    const ticker = (tr.dataset.ticker || "").toLowerCase();
    const text   = tr.textContent.toLowerCase();
    tr.classList.toggle("filtered-out", q.length > 0 && !ticker.includes(q) && !text.includes(q));
  });
}

// ── Download Excel ────────────────────────────────────────────

/** True when running inside pywebview native window */
function _isDesktop() {
  return !!(window.pywebview && window.pywebview.api);
}

$("downloadExcel").addEventListener("click", async () => {
  if (!state.results) return;

  const btn = $("downloadExcel");
  const defaultName = `stock_analysis_${new Date().toISOString().slice(0,10)}.xlsx`;

  // ── Step 1: Ask user WHERE to save ──────────────────────────
  let savePath = null;

  if (_isDesktop()) {
    // Native "Save As" dialog via pywebview bridge
    try {
      savePath = await window.pywebview.api.save_file_dialog(defaultName);
    } catch (_) { /* dialog cancelled or error */ }
    if (!savePath) return;  // user cancelled
  }

  btn.disabled = true;
  btn.textContent = "⏳ Generating…";

  try {
    if (savePath) {
      // ── Desktop mode: generate + save to chosen path on server side ──
      const res = await fetch("/api/export/excel", {
        method:  "POST",
        headers: {"Content-Type": "application/json"},
        body:    JSON.stringify({
          results:   state.results,
          config:    state.config,
          task_id:   state.taskId,
          save_path: savePath,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Export failed");
      toast("Saved to " + data.path, "ok");

    } else {
      // ── Browser mode: classic blob download ──────────────────────────
      const res = await fetch("/api/export/excel", {
        method:  "POST",
        headers: {"Content-Type": "application/json"},
        body:    JSON.stringify({ results: state.results, config: state.config, task_id: state.taskId }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || "Export failed");
      }
      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement("a");
      a.href = url;
      a.download = defaultName;
      a.click();
      URL.revokeObjectURL(url);
      toast("Excel downloaded!", "ok");
    }

  } catch (err) {
    toast("Export failed: " + err.message, "err");
  } finally {
    btn.disabled = false;
    btn.textContent = "⬇ Download Excel";
  }
});

// ════════════════════════════════════════════════════════════════
//  AI QUANT TRADER — cross-sectional ranking (Step 5)
// ════════════════════════════════════════════════════════════════

const _convictionColors = {
  "STRONG BUY":  "#22c55e",
  "BUY":         "#4ade80",
  "HOLD":        "#a78bfa",
  "REDUCE":      "#f97316",
  "SELL":        "#ef4444",
  "STRONG SELL": "#dc2626",
};

const _actionColors = {
  "HOLD":         "#a78bfa",
  "ADD":          "#22c55e",
  "REDUCE":       "#f97316",
  "EXIT":         "#ef4444",
  "TAKE_PROFIT":  "#eab308",
  "CUT_LOSS":     "#dc2626",
};

const _urgencyColors = { low: "#5a6a88", medium: "#f59e0b", high: "#ef4444" };

function convPill(conviction) {
  const c = _convictionColors[conviction] || "var(--text2)";
  return `<span style="background:${c}22;color:${c};padding:2px 8px;border-radius:12px;font-size:0.72rem;font-weight:700;white-space:nowrap">${conviction}</span>`;
}

function actionPill(action) {
  const c = _actionColors[action] || "var(--text2)";
  return `<span style="background:${c}22;color:${c};padding:2px 8px;border-radius:12px;font-size:0.72rem;font-weight:700">${action}</span>`;
}

window.closeAiRankPanel = function() { hide($("aiRankPanel")); };

$("aiRankBtn").addEventListener("click", async () => {
  if (!state.results || state.results.length === 0) {
    toast("Run analysis first.", "err"); return;
  }
  const btn = $("aiRankBtn");
  btn.disabled = true;
  btn.textContent = "🤖 Ranking…";
  show($("aiRankPanel"));
  $("aiRankBody").innerHTML = `
    <div style="display:flex;align-items:center;gap:0.75rem;padding:1rem 0;color:var(--text2)">
      <div class="bt-spinner" style="width:20px;height:20px;border-width:2px"></div>
      <span>AI is ranking ${state.results.length} stocks — this may take 30–60 s…</span>
    </div>`;
  $("aiRankPanel").scrollIntoView({ behavior: "smooth", block: "nearest" });

  try {
    const r = await fetch("/api/llm/quant-rank", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ results: state.results }),
    });
    const resp = await r.json();
    if (!resp.ok) { $("aiRankBody").innerHTML = `<p class="neg">${resp.error}</p>`; return; }
    renderAiRanking(resp.data);
  } catch (e) {
    $("aiRankBody").innerHTML = `<p class="neg">Request failed: ${e.message}</p>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "🤖 AI Ranking";
  }
});

function renderAiRanking(d) {
  if (!d) { $("aiRankBody").innerHTML = '<p class="neg">No data returned.</p>'; return; }

  // Market context + portfolio suggestion
  let html = "";
  if (d.market_context) {
    html += `<p style="color:var(--text2);font-size:0.88rem;margin-bottom:1rem;line-height:1.6">${d.market_context}</p>`;
  }

  // Top picks / avoid pills
  if ((d.top_picks && d.top_picks.length) || (d.avoid && d.avoid.length)) {
    html += `<div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin-bottom:1rem;align-items:center">`;
    if (d.top_picks && d.top_picks.length) {
      html += `<span style="font-size:0.72rem;color:var(--text2);font-weight:600">TOP PICKS:</span>`;
      d.top_picks.forEach(t => {
        html += `<span style="background:#22c55e22;color:#22c55e;padding:2px 10px;border-radius:12px;font-size:0.78rem;font-weight:700">${t}</span>`;
      });
    }
    if (d.avoid && d.avoid.length) {
      html += `<span style="font-size:0.72rem;color:var(--text2);font-weight:600;margin-left:0.5rem">AVOID:</span>`;
      d.avoid.forEach(t => {
        html += `<span style="background:#ef444422;color:#ef4444;padding:2px 10px;border-radius:12px;font-size:0.78rem;font-weight:700">${t}</span>`;
      });
    }
    html += `</div>`;
  }

  // Rankings table
  if (d.rankings && d.rankings.length) {
    html += `<div style="overflow-x:auto"><table class="results-table" style="width:100%">
      <thead><tr>
        <th style="width:40px">#</th>
        <th>Ticker</th>
        <th>Conviction</th>
        <th>Rationale</th>
        <th>Key Factors</th>
      </tr></thead><tbody>`;
    d.rankings.forEach(row => {
      const factors = (row.key_factors || []).map(f =>
        `<span style="font-size:0.7rem;background:var(--bg4);padding:1px 6px;border-radius:8px;margin-right:3px">${f}</span>`
      ).join("");
      html += `<tr>
        <td style="color:var(--text2);font-weight:700">${row.rank}</td>
        <td><strong>${row.ticker}</strong></td>
        <td>${convPill(row.conviction)}</td>
        <td style="font-size:0.82rem;line-height:1.5">${row.rationale || "—"}</td>
        <td>${factors}</td>
      </tr>`;
    });
    html += `</tbody></table></div>`;
  }

  if (d.portfolio_suggestion) {
    html += `<div style="margin-top:1rem;padding:0.75rem 1rem;background:var(--bg3);border-radius:var(--radius);font-size:0.84rem;line-height:1.6">
      <strong style="color:var(--text2);font-size:0.72rem;text-transform:uppercase;letter-spacing:0.05em">Portfolio Suggestion</strong><br>
      ${d.portfolio_suggestion}
    </div>`;
  }

  $("aiRankBody").innerHTML = html;
}


// ════════════════════════════════════════════════════════════════
//  AI PORTFOLIO MANAGER — portfolio review
// ════════════════════════════════════════════════════════════════

window.closePfAiReview = function() { hide($("pfAiReviewPanel")); };

$("pfAiReviewBtn").addEventListener("click", async () => {
  if (!_pfPositionsCache.length) {
    toast("No positions to review.", "err"); return;
  }
  const btn = $("pfAiReviewBtn");
  btn.disabled = true;
  btn.textContent = "🤖 Reviewing…";
  show($("pfAiReviewPanel"));
  $("pfAiReviewBody").innerHTML = `
    <div style="display:flex;align-items:center;gap:0.75rem;padding:1rem 0;color:var(--text2)">
      <div class="bt-spinner" style="width:20px;height:20px;border-width:2px"></div>
      <span>AI is reviewing your portfolio — this may take 20–40 s…</span>
    </div>`;
  $("pfAiReviewPanel").scrollIntoView({ behavior: "smooth", block: "nearest" });

  try {
    const r = await fetch("/api/llm/portfolio-review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ positions: _pfPositionsCache }),
    });
    const resp = await r.json();
    if (!resp.ok) { $("pfAiReviewBody").innerHTML = `<p class="neg">${resp.error}</p>`; return; }
    renderPfAiReview(resp.data);
  } catch (e) {
    $("pfAiReviewBody").innerHTML = `<p class="neg">Request failed: ${e.message}</p>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "🤖 AI Review";
  }
});

function renderPfAiReview(d) {
  if (!d) { $("pfAiReviewBody").innerHTML = '<p class="neg">No data returned.</p>'; return; }

  const gradeColors = { A: "#22c55e", B: "#4ade80", C: "#f59e0b", D: "#ef4444" };
  const gc = gradeColors[d.grade] || "var(--text2)";

  let html = `<div style="display:flex;align-items:flex-start;gap:1.5rem;margin-bottom:1.25rem;flex-wrap:wrap">`;

  // Grade badge
  if (d.grade) {
    html += `<div style="text-align:center;min-width:60px">
      <div style="font-size:2.5rem;font-weight:900;color:${gc};line-height:1">${d.grade}</div>
      <div style="font-size:0.68rem;color:var(--text2);text-transform:uppercase;margin-top:2px">Grade</div>
    </div>`;
  }

  // Overview + grade rationale
  html += `<div style="flex:1;min-width:200px">`;
  if (d.overview) html += `<p style="font-size:0.88rem;line-height:1.6;margin-bottom:0.4rem">${d.overview}</p>`;
  if (d.grade_rationale) html += `<p style="font-size:0.8rem;color:var(--text2)">${d.grade_rationale}</p>`;
  html += `</div></div>`;

  // Per-position recommendations table
  if (d.positions && d.positions.length) {
    html += `<div style="overflow-x:auto;margin-bottom:1rem"><table class="results-table" style="width:100%">
      <thead><tr><th>Ticker</th><th>Action</th><th>Urgency</th><th>Rationale</th></tr></thead><tbody>`;
    d.positions.forEach(p => {
      const uc = _urgencyColors[p.urgency] || "var(--text2)";
      html += `<tr>
        <td><strong>${p.ticker}</strong></td>
        <td>${actionPill(p.action)}</td>
        <td><span style="color:${uc};font-size:0.75rem;font-weight:600;text-transform:uppercase">${p.urgency || "—"}</span></td>
        <td style="font-size:0.82rem;line-height:1.5">${p.rationale || "—"}</td>
      </tr>`;
    });
    html += `</tbody></table></div>`;
  }

  // Risk alerts
  if (d.risk_alerts && d.risk_alerts.length) {
    html += `<div style="margin-bottom:0.75rem">
      <div style="font-size:0.72rem;color:var(--text2);text-transform:uppercase;letter-spacing:0.05em;font-weight:600;margin-bottom:0.4rem">Risk Alerts</div>`;
    d.risk_alerts.forEach(alert => {
      html += `<div style="display:flex;align-items:flex-start;gap:0.4rem;font-size:0.83rem;margin-bottom:0.25rem">
        <span style="color:#f59e0b">⚠</span><span>${alert}</span></div>`;
    });
    html += `</div>`;
  }

  // Bottom row: rebalancing + key opportunity
  if (d.rebalancing_suggestion || d.key_opportunity) {
    html += `<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;margin-top:0.5rem">`;
    if (d.rebalancing_suggestion) {
      html += `<div style="padding:0.75rem;background:var(--bg3);border-radius:var(--radius);font-size:0.83rem">
        <div style="font-size:0.7rem;color:var(--text2);text-transform:uppercase;letter-spacing:0.05em;font-weight:600;margin-bottom:0.3rem">Rebalancing</div>
        ${d.rebalancing_suggestion}</div>`;
    }
    if (d.key_opportunity) {
      html += `<div style="padding:0.75rem;background:var(--bg3);border-radius:var(--radius);font-size:0.83rem;border-left:2px solid #22c55e">
        <div style="font-size:0.7rem;color:#22c55e;text-transform:uppercase;letter-spacing:0.05em;font-weight:600;margin-bottom:0.3rem">Key Opportunity</div>
        ${d.key_opportunity}</div>`;
    }
    html += `</div>`;
  }

  $("pfAiReviewBody").innerHTML = html;
}

// ════════════════════════════════════════════════════════════════
//  File Browser Modal
// ════════════════════════════════════════════════════════════════
const fileBrowserModal = $("fileBrowserModal");
const browserTitle = $("browserTitle");
const browserBreadcrumb = $("browserBreadcrumb");
const browserEntries = $("browserEntries");
const selectedPathEl = $("selectedPath");
const closeBrowserBtn = $("closeBrowser");
const cancelBrowseBtn = $("cancelBrowse");
const confirmBrowseBtn = $("confirmBrowse");
const browseModelBtn = $("browseModelBtn");
const browseAdapterBtn = $("browseAdapterBtn");

// Debug: log if buttons are found
if (!browseModelBtn) console.error("browseModelBtn not found!");
if (!browseAdapterBtn) console.error("browseAdapterBtn not found!");

let browserCurrentPath = "";
let browserSelectedPath = "";
let browserTargetInput = null; // 'modelPath' or 'adapterPath'

function openFileBrowser(targetInputId, title) {
  browserTargetInput = targetInputId;
  browserTitle.textContent = title;
  fileBrowserModal.classList.remove("hidden");
  browserCurrentPath = "";
  browserSelectedPath = "";
  loadDirectory("");
}

function closeFileBrowser() {
  fileBrowserModal.classList.add("hidden");
  browserTargetInput = null;
}

async function loadDirectory(path) {
  try {
    const res = await fetch(`/api/browse?path=${encodeURIComponent(path)}`);
    if (!res.ok) {
      throw new Error(`Server returned ${res.status}`);
    }
    const data = await res.json();
    if (data.error) {
      toast(data.error, "err");
      return;
    }
    browserCurrentPath = data.current;
    renderBrowser(data);
  } catch (e) {
    console.error("Browse error:", e);
    toast("Server not responding. Make sure Flask is running. Error: " + e.message, "err");
  }
}

function renderBrowser(data) {
  // Update breadcrumb
  const parts = data.current.split(/[\\/]/).filter(p => p);
  browserBreadcrumb.innerHTML = parts.length === 0
    ? '<span class="breadcrumb-root" data-path="">Computer</span>'
    : parts.map((p, i) => {
        const path = parts.slice(0, i + 1).join("/");
        return `<span class="breadcrumb-root" data-path="${path}">${p}</span>`;
      }).join(' <span style="color:var(--text3)">/</span> ');

  // Add click handlers to breadcrumb
  browserBreadcrumb.querySelectorAll("[data-path]").forEach(el => {
    el.addEventListener("click", () => loadDirectory(el.dataset.path));
  });

  // Render entries
  if (data.entries.length === 0) {
    browserEntries.innerHTML = '<div class="browser-entry empty">Empty directory</div>';
    return;
  }

  browserEntries.innerHTML = data.entries.map(entry => {
    const icon = entry.is_dir ? "📁" : "📄";
    const badges = [];
    if (entry.is_model) badges.push('<span class="badge-model">MODEL</span>');
    if (entry.is_adapter) badges.push('<span class="badge-adapter">ADAPTER</span>');
    const selectedClass = entry.path === browserSelectedPath ? "selected" : "";

    return `<div class="browser-entry ${selectedClass}" data-path="${entry.path}" data-is-dir="${entry.is_dir}">
      <span class="icon">${icon}</span>
      <span class="name">${entry.name}</span>
      ${badges.join("")}
    </div>`;
  }).join("");

  // Add click handlers
  browserEntries.querySelectorAll(".browser-entry").forEach(el => {
    el.addEventListener("click", () => {
      const path = el.dataset.path;
      const isDir = el.dataset.isDir === "true";

      // Remove previous selection
      browserEntries.querySelectorAll(".selected").forEach(s => s.classList.remove("selected"));
      el.classList.add("selected");

      browserSelectedPath = path;
      selectedPathEl.textContent = path;

      // Double-click behavior: navigate into directories
      if (isDir) {
        // Use a timer to distinguish single vs double click
        if (el.dataset.pendingDouble) {
          clearTimeout(parseInt(el.dataset.pendingDouble));
          el.dataset.pendingDouble = "";
          loadDirectory(path);
        } else {
          const timer = setTimeout(() => {
            el.dataset.pendingDouble = "";
          }, 250);
          el.dataset.pendingDouble = timer;
        }
      }
    });
  });
}

// Browser button handlers
if (browseModelBtn) {
  browseModelBtn.addEventListener("click", () => openFileBrowser("modelPath", "Select Base Model Directory"));
} else {
  console.error("browseModelBtn element not found - cannot attach event listener");
}

if (browseAdapterBtn) {
  browseAdapterBtn.addEventListener("click", () => openFileBrowser("adapterPath", "Select LoRA Adapter Directory"));
} else {
  console.error("browseAdapterBtn element not found - cannot attach event listener");
}

closeBrowserBtn?.addEventListener("click", closeFileBrowser);
cancelBrowseBtn?.addEventListener("click", closeFileBrowser);

confirmBrowseBtn?.addEventListener("click", () => {
  if (browserSelectedPath && browserTargetInput) {
    $(browserTargetInput).value = browserSelectedPath;
    closeFileBrowser();
  } else {
    toast("Please select a directory", "err");
  }
});

// Close on backdrop click
fileBrowserModal?.querySelector(".modal-backdrop")?.addEventListener("click", closeFileBrowser);

// ════════════════════════════════════════════════════════════════
//  Init
// ════════════════════════════════════════════════════════════════
renderAssets();

// ── Settings persistence (server-side JSON file) ─────────────

async function _saveSettings(data) {
  try {
    await fetch("/api/settings", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(data),
    });
  } catch (_) { /* best-effort */ }
}

// Restore last-used settings on startup
(async () => {
  try {
    const res = await fetch("/api/settings");
    if (res.ok) {
      const s = await res.json();
      // Restore provider selection
      if (s.provider) {
        const radio = document.querySelector(`input[name='sentProvider'][value='${s.provider}']`);
        if (radio) radio.checked = true;
      }
      // Restore model override
      if (s.sent_model && $("sentModel")) $("sentModel").value = s.sent_model;
      // Restore local model paths
      if (s.model_path && $("modelPath")) $("modelPath").value = s.model_path;
      if (s.adapter_path && $("adapterPath")) $("adapterPath").value = s.adapter_path;
      // Update field visibility based on restored provider
      updateProviderFields();
    }
  } catch (_) { /* first run or no saved settings */ }
})();

// ── ML Lab handoff: auto-load ticker + enable ML if redirected from ML Lab ──
(async () => {
  const params = new URLSearchParams(window.location.search);
  const mlLabTicker = params.get("ticker");
  const fromMlLab   = params.get("ml_lab") === "1";
  if (!mlLabTicker || !fromMlLab) return;

  // Check which tickers have a trained ML Lab model ready
  let readyTickers = [];
  try {
    const r = await fetch("/api/ml/dashboard/ready");
    if (r.ok) readyTickers = (await r.json()).tickers || [];
  } catch (_) {}

  const hasModel = readyTickers.includes(mlLabTicker.toUpperCase());

  // Search for the ticker and add it
  try {
    const r = await fetch(`/api/search?q=${encodeURIComponent(mlLabTicker)}&limit=1`);
    if (r.ok) {
      const d = await r.json();
      const asset = d.results?.[0] || d[0];
      if (asset) {
        addAsset(asset);
        toast(`${mlLabTicker} added${hasModel ? " — ML Lab model active" : ""}`, "ok");
      }
    }
  } catch (_) {
    // Fallback: add by ticker only
    addAsset({ ticker: mlLabTicker.toUpperCase(), name: mlLabTicker.toUpperCase(), sector: "", currency: "USD" });
  }

  // Enable ML if the model is ready in the Lab
  if (hasModel && $("mlEnabled")) {
    $("mlEnabled").checked = true;
    $("mlEnabled").dispatchEvent(new Event("change"));
    toast(`ML Lab model for ${mlLabTicker} will be used — no retraining needed`, "ok", 5000);
  }

  // Clean URL so refreshing doesn't re-trigger
  window.history.replaceState({}, "", "/");
})();


// ════════════════════════════════════════════════════════════════
//  STEP 6 — SIGNAL BACKTEST
// ════════════════════════════════════════════════════════════════

const btState = {
  activeTicker: null,
  runTicker:    null,
  results:      {},      // ticker → {dates, prices, scores, verbal_signals, css_classes, fund_score}
  summary:      null,
  chart:        null,    // active Chart.js instance
};
const BT_ALL_TICKERS = "__ALL__";
const BT_FUND_KEYS = [
  "pe_trail", "pe_fwd", "peg", "pb", "ps", "ev_ebitda", "ev_rev",
  "gross_mgn", "op_mgn", "net_mgn", "roe", "roa", "rev_growth",
  "eps_growth", "debt_eq", "curr_ratio", "quick_ratio", "fcf",
  "mkt_cap", "beta", "div_yield", "short_float", "target_px",
  "rec_mean", "n_analysts", "sector", "currency",
];
let btChartDepsPromise = null;

function btLoadScript(src) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[data-src="${src}"]`);
    if (existing?.dataset.loaded === "true") {
      resolve();
      return;
    }
    if (existing) {
      existing.addEventListener("load", () => resolve(), { once: true });
      existing.addEventListener("error", () => reject(new Error(`Failed to load ${src}`)), { once: true });
      return;
    }

    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.dataset.src = src;
    script.onload = () => {
      script.dataset.loaded = "true";
      resolve();
    };
    script.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.body.appendChild(script);
  });
}

function ensureBacktestChartDeps() {
  if (window.Chart) return Promise.resolve();
  if (btChartDepsPromise) return btChartDepsPromise;

  btChartDepsPromise = (async () => {
    await btLoadScript("https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js");
    await btLoadScript("https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.1.0/dist/chartjs-plugin-annotation.min.js");
  })();

  return btChartDepsPromise;
}

// ── Colour mapping (matches scoring.py signal CSS classes) ───────────────────
function btScoreColor(score) {
  if (score >= 0.5)  return "#22c55e";   // sbuy  — strong buy
  if (score >= 0.2)  return "#86efac";   // buy
  if (score >= -0.2) return "#eab308";   // neu   — neutral / hold
  if (score >= -0.5) return "#f97316";   // sell
  return "#ef4444";                       // ssell — strong sell / avoid
}

// ── Populate ticker list in the controls card ─────────────────────────────────
function btPopulateTickers() {
  const container = $("btTickerList");
  if (!container) return;
  if (state.assets.length === 0) {
    container.innerHTML = '<span class="hint">No assets selected — go to Step 1.</span>';
    btState.runTicker = null;
    return;
  }

  const availableTickers = state.assets.map(a => a.ticker);
  if (!btState.runTicker || (btState.runTicker !== BT_ALL_TICKERS && !availableTickers.includes(btState.runTicker))) {
    btState.runTicker = availableTickers[0];
  }

  const chips = [{
    ticker: BT_ALL_TICKERS,
    label: "ALL",
    done: availableTickers.length > 0 && availableTickers.every(t => btState.results[t] && !btState.results[t].error),
  }].concat(state.assets.map(a => ({
    ticker: a.ticker,
    label: a.ticker,
    done: !!(btState.results[a.ticker] && !btState.results[a.ticker].error),
  })));

  container.innerHTML = chips.map(chip => `
    <button type="button"
            class="bt-ticker-chip${chip.done ? " bt-ticker-chip-done" : ""}${chip.ticker === btState.runTicker ? " active" : ""}"
            onclick="btSetRunTicker('${chip.ticker}')">${chip.label}</button>
  `).join("");
}

// ── Populate chart tabs when multiple tickers have results ────────────────────
function btPopulateChartTabs() {
  const tabs = $("btChartTabs");
  if (!tabs) return;
  const tickers = state.assets
    .map(a => a.ticker)
    .filter(t => btState.results[t] && !btState.results[t].error);

  if (tickers.length <= 1) {
    tabs.innerHTML = "";
    return;
  }
  tabs.innerHTML = tickers.map(t =>
    `<button class="btn btn-sm bt-ticker-btn${t === btState.activeTicker ? " active" : ""}"
             onclick="btSelectTicker('${t}')">${t}</button>`
  ).join("");
}

function btSetRunTicker(ticker) {
  btState.runTicker = ticker;
  btPopulateTickers();
}

window.btSetRunTicker = btSetRunTicker;

function btSelectTicker(ticker) {
  btState.activeTicker = ticker;
  btPopulateChartTabs();
  btRenderChart(ticker);
  btRenderStats(ticker);
}

function btRequestedTickers() {
  if (btState.runTicker === BT_ALL_TICKERS) {
    return state.assets.map(a => a.ticker);
  }
  return btState.runTicker ? [btState.runTicker] : [];
}

function btBuildFundamentalsMap(tickers) {
  const out = {};
  const rows = Array.isArray(state.results) ? state.results : [];
  tickers.forEach(ticker => {
    const row = rows.find(r => r.ticker === ticker);
    if (!row) return;
    const fund = {};
    BT_FUND_KEYS.forEach(key => {
      const value = row[key];
      if (value !== null && value !== undefined && value !== "") {
        fund[key] = value;
      }
    });
    if (Object.keys(fund).length > 0) out[ticker] = fund;
  });
  return out;
}

// ── Navigation ────────────────────────────────────────────────────────────────
const _step6BackBtn = $("step6Back");
if (_step6BackBtn) _step6BackBtn.addEventListener("click", () => goTo(5));

const _goToBtBtn = $("goToBacktest");
if (_goToBtBtn) _goToBtBtn.addEventListener("click", () => goTo(6));

// ── Calibrate IC weights button ───────────────────────────────────────────────
// Stores the last-saved IC config_id per ticker so Rerun can use it
const _lastICConfigId = {};

const _btCalibrateBtn   = $("btCalibrateIC");
const _btRerunICBtn     = $("btRerunIC");
const _btSaveWeightsBtn = $("btSaveWeights");

if (_btCalibrateBtn) {
  _btCalibrateBtn.addEventListener("click", async () => {
    const ticker = btState.activeTicker;
    if (!ticker || ticker === BT_ALL_TICKERS) {
      toast("Select a single ticker in the chart before calibrating.", "warn");
      return;
    }
    const data = btState.results[ticker];
    if (!data || data.error) {
      toast("Run the backtest for this ticker first.", "warn");
      return;
    }

    const statusEl      = $("btCalibrateStatus");
    _btCalibrateBtn.disabled = true;
    if (_btRerunICBtn) _btRerunICBtn.disabled = true;
    if (statusEl) statusEl.innerHTML = `<span class="hint">Calibrating IC weights for <strong>${ticker}</strong>…</span>`;

    // Read periods: backtest_period from btPeriod, calib_period from btCalibPeriod
    const backtestPeriod = $("btPeriod")?.value  || "2y";
    const calibPeriod    = $("btCalibPeriod")?.value || "5y";
    const techSel = Array.from(
      document.querySelectorAll("#techGroup input[type=checkbox]:checked")
    ).map(el => el.value).filter(Boolean);

    try {
      const resp = await fetch("/api/thresholds/calibrate-ic", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ticker,
          backtest_period: backtestPeriod,
          calib_period:    calibPeriod,
          tech_sel: techSel.length ? techSel : ["ma20","ma50","ma200","cross","rsi","macd","bb"],
          name: `${ticker} IC (bt:${backtestPeriod} cal:${calibPeriod})`,
        }),
      });
      const res = await resp.json();
      if (!res.ok) throw new Error(res.error || "Calibration failed");

      const wStr = Object.entries(res.ic_weights || {})
        .filter(([,v]) => v > 0)
        .sort(([,a],[,b]) => b - a)
        .map(([k,v]) => `${k}: ${v.toFixed(2)}×`)
        .join(" · ");

      if (statusEl) statusEl.innerHTML =
        `<span style="color:#22c55e">✓ Saved: <strong>${res.config.name}</strong></span>
         <br><span class="hint">Calib window: up to ${res.calib_end_date} (${res.n_calib_bars} bars)
         · Backtest starts: ${res.backtest_start}</span>
         <br><span class="hint">${wStr}</span>`;

      // Enable Rerun button
      _lastICConfigId[ticker] = res.config.config_id;
      if (_btRerunICBtn) _btRerunICBtn.disabled = false;

      toast(`IC weights saved for ${ticker}`, "ok");
      loadICWeightConfigs();   // refresh Step 4 list
    } catch (e) {
      if (statusEl) statusEl.innerHTML = `<span class="error">Error: ${e.message}</span>`;
      toast(`Calibration failed: ${e.message}`, "err");
    } finally {
      _btCalibrateBtn.disabled = false;
    }
  });
}

// ── Rerun backtest with IC weights ────────────────────────────────────────────
if (_btRerunICBtn) {
  _btRerunICBtn.addEventListener("click", async () => {
    const ticker = btState.activeTicker;
    if (!ticker || ticker === BT_ALL_TICKERS) {
      toast("Select a single ticker first.", "warn");
      return;
    }
    const configId = _lastICConfigId[ticker];
    if (!configId) {
      toast("Calibrate IC weights for this ticker first.", "warn");
      return;
    }

    const period  = $("btPeriod")?.value || "2y";
    const techSel = [...document.querySelectorAll('input[name="tech"]:checked')].map(el => el.value);
    const wTech   = parseInt($("wTech")?.value || "60");
    const wFund   = parseInt($("wFund")?.value || "40");

    const payload = {
      tickers:              [ticker],
      period,
      technical:            techSel.length ? techSel : ["ma20","ma50","ma200","cross","rsi","macd","bb"],
      fundamental:          [],
      use_fundamentals:     false,
      weights:              { technical: wTech, fundamental: wFund, sentiment: 0 },
      fundamentals_map:     {},
      threshold_config_id:  configId,
    };

    hide($("btChartPanel"));
    hide($("btStats"));
    show($("btLoading"));
    _btRerunICBtn.disabled = true;
    if ($("btLoadingMsg")) $("btLoadingMsg").textContent = `Rerunning backtest with IC weights for ${ticker}…`;

    try {
      const resp = await fetch("/api/backtest/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const res = await resp.json();
      if (!res.ok) throw new Error(res.error || "Backtest failed");

      // Store results under a special IC key so we can compare
      const tickerData = res.results?.[ticker];
      if (tickerData) {
        btState.results[ticker + "_ic"] = { ...tickerData, _ic_rerun: true };
        btState.results[ticker]._baseline_signal_quality = btState.results[ticker].signal_quality;
        btState.results[ticker]._ic_signal_quality       = tickerData.signal_quality;
        btState.results[ticker]._ic_config_id            = configId;
        // Show the IC results in the main view
        btState.results[ticker] = { ...tickerData, _ic_rerun: true,
          _baseline_signal_quality: btState.results[ticker]._baseline_signal_quality,
          _ic_signal_quality:       tickerData.signal_quality,
          _ic_config_id:            configId,
        };
      }

      hide($("btLoading"));
      btState.activeTicker = ticker;
      show($("btChartPanel"));
      show($("btStats"));
      btPopulateChartTabs();
      btRenderChart(ticker);
      btRenderStats(ticker);
      btPopulateTickers();

      const sq    = tickerData?.signal_quality;
      const sqStr = sq != null ? ` Signal quality: ${(sq * 100).toFixed(1)}%` : "";
      const statusEl = $("btCalibrateStatus");
      if (statusEl) {
        statusEl.innerHTML += `<br><span style="color:#3b82f6">↻ Rerun complete with IC weights.${sqStr}</span>`;
      }
      toast(`Rerun with IC weights done for ${ticker}`, "ok");
    } catch (e) {
      hide($("btLoading"));
      show($("btStats"));
      toast(`Rerun failed: ${e.message}`, "err");
    } finally {
      _btRerunICBtn.disabled = false;
    }
  });
}

// ── Save Weights Config button ────────────────────────────────────────────────
// Saves the current weights config (the calibrated one if available, otherwise
// a default-weights placeholder tagged to the active ticker) so it shows up in
// the Step 4 weights UI list grouped by stock.
if (_btSaveWeightsBtn) {
  _btSaveWeightsBtn.addEventListener("click", async () => {
    const ticker = btState.activeTicker;
    if (!ticker || ticker === BT_ALL_TICKERS) {
      toast("Select a single ticker first.", "warn");
      return;
    }

    const defaultName = `${ticker} weights ${new Date().toISOString().slice(0,10)}`;
    const name = prompt("Name for this weights config:", defaultName);
    if (!name) return;

    const statusEl = $("btCalibrateStatus");
    _btSaveWeightsBtn.disabled = true;

    try {
      let payload;
      const lastId = _lastICConfigId[ticker];

      if (lastId) {
        // Calibrated already — fetch the existing config and resave under new name
        const r = await fetch(`/api/thresholds/${lastId}`);
        const j = await r.json();
        if (!j.ok) throw new Error(j.error || "Failed to load calibrated config");
        const src = j.config || {};
        payload = {
          name,
          description: `Saved from backtest page for ${ticker} (calibrated)`,
          ic_weights:             src.ic_weights || {},
          ic_calibration_ticker:  ticker,
          ic_calibration_meta:    src.ic_calibration_meta || {},
          rsi_mr_bp:        src.rsi_mr_bp,
          rsi_trend_bp:     src.rsi_trend_bp,
          bb_bp:            src.bb_bp,
          ma200_dist_bp:    src.ma200_dist_bp,
          score_thresholds: src.score_thresholds,
        };
      } else {
        // No calibration — save a placeholder tagged to this ticker (default weights)
        payload = {
          name,
          description: `Saved from backtest page for ${ticker} (default weights, no calibration)`,
          ic_weights:             {},
          ic_calibration_ticker:  ticker,
          ic_calibration_meta:    { source: "manual_save", saved_at: new Date().toISOString() },
        };
      }

      const resp = await fetch("/api/thresholds", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const res = await resp.json();
      if (!res.ok) throw new Error(res.error || "Save failed");

      if (statusEl) {
        statusEl.innerHTML += `<br><span style="color:#22c55e">💾 Saved <strong>${res.config.name}</strong> for ${ticker}</span>`;
      }
      toast(`Weights config saved for ${ticker}`, "ok");
      // Refresh the Step 4 list so it shows up immediately
      loadICWeightConfigs();
    } catch (e) {
      toast(`Save failed: ${e.message}`, "err");
      if (statusEl) {
        statusEl.innerHTML += `<br><span class="error">Save error: ${e.message}</span>`;
      }
    } finally {
      _btSaveWeightsBtn.disabled = false;
    }
  });
}

// ── IC weight configs (Step 4) ────────────────────────────────────────────────
// Currently selected IC config per ticker: {ticker: config_id}
const _selectedICConfigs = {};

async function loadICWeightConfigs() {
  const el = $("icWeightConfigList");
  if (!el) return;

  let configs;
  try {
    const r = await fetch("/api/thresholds");
    const j = await r.json();
    configs  = (j.configs || []).filter(c => c.config_id !== "default" && c.ic_calibration_ticker);
  } catch (e) {
    el.innerHTML = `<p class="error">Failed to load: ${e}</p>`;
    return;
  }

  if (!configs.length) {
    el.innerHTML = `<p class="hint muted">No IC weight configs saved yet. Run a backtest and click "⚙ Calibrate IC Weights".</p>`;
    return;
  }

  // Group by ticker
  const byTicker = {};
  configs.forEach(c => {
    const t = c.ic_calibration_ticker || "—";
    (byTicker[t] = byTicker[t] || []).push(c);
  });

  let html = `<table class="bt-table" style="font-size:0.79rem">
    <thead><tr><th>Ticker</th><th>Config Name</th><th>Created</th><th>Use for analysis</th><th></th></tr></thead>
    <tbody>`;

  for (const [ticker, cfgs] of Object.entries(byTicker)) {
    cfgs.forEach((c, idx) => {
      const selected = _selectedICConfigs[ticker] === c.config_id;
      html += `<tr>
        ${idx === 0 ? `<td rowspan="${cfgs.length}" style="font-weight:600">${ticker}</td>` : ""}
        <td>${c.name}</td>
        <td class="muted">${c.created_at || "—"}</td>
        <td>
          <input type="radio" name="icsel_${ticker}"
            value="${c.config_id}"
            ${selected ? "checked" : ""}
            onchange="_selectedICConfigs['${ticker}']='${c.config_id}';updateICActiveDisplay()">
          <label style="margin-left:0.3rem;font-size:0.77rem">Apply</label>
        </td>
        <td>
          <button class="btn btn-xs" style="color:#ef4444"
            onclick="deleteICConfig('${c.config_id}','${ticker}')">✕</button>
        </td>
      </tr>`;
    });
    // "None" option
    html += `<tr>
      <td colspan="2" class="muted" style="font-size:0.77rem">No IC weights (default regime weights)</td>
      <td></td>
      <td>
        <input type="radio" name="icsel_${ticker}"
          value=""
          ${!_selectedICConfigs[ticker] ? "checked" : ""}
          onchange="delete _selectedICConfigs['${ticker}'];updateICActiveDisplay()">
        <label style="margin-left:0.3rem;font-size:0.77rem">Use default</label>
      </td>
      <td></td>
    </tr>`;
  }

  html += `</tbody></table>`;
  el.innerHTML = html;
  updateICActiveDisplay();
}

function updateICActiveDisplay() {
  const el = $("icWeightActive");
  if (!el) return;
  const entries = Object.entries(_selectedICConfigs);
  if (!entries.length) { el.innerHTML = ""; return; }
  el.innerHTML = `<p class="hint" style="font-size:0.78rem">
    Active IC configs: ${entries.map(([t,id]) => `<strong>${t}</strong>: ${id}`).join(" · ")}
    <br>These will be passed as <code>threshold_config_id</code> per-ticker when you run the analysis.
  </p>`;
}

async function deleteICConfig(configId, ticker) {
  try {
    await fetch(`/api/thresholds/${configId}`, { method: "DELETE" });
    if (_selectedICConfigs[ticker] === configId) delete _selectedICConfigs[ticker];
    loadICWeightConfigs();
    toast("Config deleted", "ok");
  } catch (e) {
    toast(`Delete failed: ${e}`, "err");
  }
}

// Auto-load IC configs when entering Step 4
document.addEventListener("DOMContentLoaded", () => {
  const step4Btn = document.querySelector('[data-step="4"]');
  if (step4Btn) step4Btn.addEventListener("click", () => loadICWeightConfigs());
});

// ── Run backtest ──────────────────────────────────────────────────────────────
const _runBtBtn = $("runBacktest");
if (_runBtBtn) {
  _runBtBtn.addEventListener("click", async () => {
    if (state.assets.length === 0) {
      toast("No assets selected — go to Step 1 first.", "err");
      return;
    }

    const period  = $("btPeriod")?.value || "2y";
    const techSel = [...document.querySelectorAll('input[name="tech"]:checked')].map(el => el.value);
    const wTech   = parseInt($("wTech")?.value || "60");
    const wFund   = parseInt($("wFund")?.value || "40");
    const requestedTickers = btRequestedTickers();

    if (requestedTickers.length === 0) {
      toast("Pick a ticker to backtest first.", "err");
      return;
    }

    const payload = {
      tickers:     requestedTickers,
      period,
      technical:   techSel.length ? techSel : ["ma20","ma50","ma200","cross","rsi","macd","bb"],
      fundamental: [],
      use_fundamentals: false,
      weights:     { technical: wTech, fundamental: wFund, sentiment: 0 },
      fundamentals_map: {},
    };

    // Show loading, hide chart/stats
    hide($("btChartPanel"));
    hide($("btStats"));
    show($("btLoading"));
    _runBtBtn.disabled = true;
    const tickerStr = payload.tickers.join(", ");
    $("btLoadingMsg").textContent = `Running backtest for ${tickerStr} (${period})…`;

    try {
      const res = await fetch("/api/backtest/run", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      if (!json.ok) throw new Error(json.error || "Backtest failed");

      btState.results = json.results || {};
      btState.summary = json.summary || null;

      // Find first valid ticker result
      const firstValid = payload.tickers.find(t =>
        btState.results[t] && !btState.results[t].error &&
        (btState.results[t].n_points || 0) > 0
      );

      hide($("btLoading"));

      if (firstValid) {
        btState.activeTicker = firstValid;
        show($("btChartPanel"));
        show($("btStats"));
        btPopulateChartTabs();
        btRenderChart(firstValid);
        btRenderStats(firstValid);
        btPopulateTickers();   // refresh chips with "done" state
        toast(`Backtest complete — ${payload.tickers.length} ticker(s)`, "ok");
      } else {
        toast("No valid backtest data returned. Try a longer period.", "err");
      }

      // Report per-ticker errors
      payload.tickers.forEach(t => {
        const r = btState.results[t];
        if (r && r.error) toast(`${t}: ${r.error}`, "warn", 6000);
      });

    } catch (err) {
      hide($("btLoading"));
      toast(`Backtest error: ${err.message}`, "err");
      console.error("Backtest error:", err);
    } finally {
      _runBtBtn.disabled = false;
    }
  });
}

// ── Chart rendering ───────────────────────────────────────────────────────────
async function btRenderChart(ticker) {
  const data = btState.results[ticker];
  if (!data || data.error || !data.dates || !data.dates.length) return;

  // Update title
  const asset = state.assets.find(a => a.ticker === ticker);
  const label = asset ? `${ticker} — ${asset.name || ticker}` : ticker;
  const titleEl = $("btChartTitle");
  if (titleEl) titleEl.textContent = label;

  // Destroy previous chart instance
  if (btState.chart) {
    btState.chart.destroy();
    btState.chart = null;
  }

  const canvas = $("btChart");
  if (!canvas) return;

  try {
    await ensureBacktestChartDeps();
  } catch (err) {
    toast(`Chart library failed to load: ${err.message}`, "err", 6000);
    console.error("Backtest chart dependency error:", err);
    return;
  }

  // Capture for tooltip closure
  const verbalSignals = data.verbal_signals;
  const scoreArr      = data.scores;

  btState.chart = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: {
      labels: data.dates,
      datasets: [
        // ── Dataset 0: Price ─────────────────────────────────────────────
        {
          label:                   "Price",
          data:                    data.prices,
          yAxisID:                 "yPrice",
          borderColor:             "#3b82f6",
          backgroundColor:         "rgba(59,130,246,0.06)",
          borderWidth:             2,
          pointRadius:             0,
          pointHoverRadius:        4,
          pointHoverBackgroundColor: "#3b82f6",
          fill:                    false,
          tension:                 0.1,
          order:                   1,
        },
        // ── Dataset 1: Signal score (segments coloured by signal zone) ───
        {
          label:           "Signal Score",
          data:            data.scores,
          yAxisID:         "yScore",
          borderColor:     "#8b9ab4",     // fallback (overridden by segment)
          backgroundColor: "transparent",
          borderWidth:     1.5,
          pointRadius:     0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: ctx => btScoreColor(scoreArr[ctx.dataIndex]),
          fill:            false,
          tension:         0.15,
          order:           2,
          segment: {
            // Each segment coloured by the score at its right endpoint
            borderColor: ctx => btScoreColor(ctx.p1.parsed.y),
          },
        },
      ],
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },

      plugins: {
        legend: { display: false },

        // ── Tooltip ───────────────────────────────────────────────────────
        tooltip: {
          backgroundColor: "rgba(22,27,39,0.96)",
          borderColor:     "#2a3350",
          borderWidth:     1,
          titleColor:      "#e2e8f4",
          bodyColor:       "#8b9ab4",
          padding:         10,
          callbacks: {
            title: items => items[0]?.label || "",
            label: item => {
              if (item.datasetIndex === 0) {
                const sym = (state.assets.find(a => a.ticker === ticker)?.currency === "EUR") ? "€" : "$";
                return ` Price: ${sym}${item.parsed.y.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
              }
              // Signal score dataset
              const idx    = item.dataIndex;
              const score  = item.parsed.y;
              const verbal = verbalSignals[idx] || "";
              const sign   = score >= 0 ? "+" : "";
              return [
                ` Score:  ${sign}${score.toFixed(3)}`,
                ` Signal: ${verbal}`,
              ];
            },
            labelColor: item => {
              if (item.datasetIndex === 0)
                return { borderColor: "#3b82f6", backgroundColor: "#3b82f6" };
              const c = btScoreColor(scoreArr[item.dataIndex]);
              return { borderColor: c, backgroundColor: c };
            },
          },
        },

        // ── Annotation: reference bands and zero line ─────────────────────
        annotation: {
          annotations: {
            buyZone: {
              type:            "box",
              yScaleID:        "yScore",
              yMin:             0.2,
              yMax:             1.1,
              backgroundColor: "rgba(34,197,94,0.05)",
              borderWidth:      0,
            },
            sellZone: {
              type:            "box",
              yScaleID:        "yScore",
              yMin:            -1.1,
              yMax:            -0.2,
              backgroundColor: "rgba(239,68,68,0.05)",
              borderWidth:      0,
            },
            zeroLine: {
              type:        "line",
              yScaleID:    "yScore",
              yMin:         0,
              yMax:         0,
              borderColor: "rgba(139,154,180,0.25)",
              borderWidth:  1,
              borderDash:  [4, 4],
            },
            buyLine: {
              type:        "line",
              yScaleID:    "yScore",
              yMin:         0.2,
              yMax:         0.2,
              borderColor: "rgba(34,197,94,0.20)",
              borderWidth:  1,
              borderDash:  [3, 5],
            },
            sellLine: {
              type:        "line",
              yScaleID:    "yScore",
              yMin:        -0.2,
              yMax:        -0.2,
              borderColor: "rgba(239,68,68,0.20)",
              borderWidth:  1,
              borderDash:  [3, 5],
            },
          },
        },
      },

      scales: {
        x: {
          ticks: {
            maxTicksLimit: 14,
            color:         "#5a6a88",
            font:          { size: 11 },
            maxRotation:   0,
          },
          grid:   { color: "rgba(42,51,80,0.35)" },
          border: { color: "#2a3350" },
        },
        yPrice: {
          type:     "linear",
          position: "left",
          ticks: {
            color: "#3b82f6",
            font:  { size: 11 },
            callback: v => v.toLocaleString(undefined, { maximumFractionDigits: 2 }),
          },
          grid:   { color: "rgba(42,51,80,0.35)" },
          border: { color: "#2a3350" },
          title:  { display: true, text: "Price", color: "#3b82f6", font: { size: 11 } },
        },
        yScore: {
          type:     "linear",
          position: "right",
          min:      -1.1,
          max:       1.1,
          ticks: {
            color:    "#8b9ab4",
            font:     { size: 11 },
            stepSize: 0.25,
            callback: v => v.toFixed(2),
          },
          grid:   { drawOnChartArea: false },
          border: { color: "#2a3350" },
          title:  { display: true, text: "Score", color: "#8b9ab4", font: { size: 11 } },
        },
      },
    },
  });
}

// ── Stats / signal distribution ───────────────────────────────────────────────
function btRenderStats(ticker) {
  const data = btState.results[ticker];
  const el   = $("btStatsInner");
  if (!data || data.error || !el) return;

  const n = data.scores.length;
  if (!n) { el.innerHTML = '<span class="muted">No data</span>'; return; }

  // Signal distribution count
  const dist   = { sbuy: 0, buy: 0, neu: 0, sell: 0, ssell: 0 };
  const labels = { sbuy: "STRONG BUY", buy: "BUY", neu: "NEUTRAL / HOLD", sell: "SELL", ssell: "STRONG SELL / AVOID" };
  const colors = { sbuy: "#22c55e",    buy: "#86efac", neu: "#eab308",    sell: "#f97316", ssell: "#ef4444" };
  data.css_classes.forEach(c => { if (c in dist) dist[c]++; });

  const latest      = data.scores[n - 1];
  const latestLabel = data.verbal_signals[n - 1];
  const fundScore   = data.fund_score || 0;
  const fundMode    = data.fundamentals_source || "disabled";
  const fsColor     = fundScore >= 0.3 ? "#22c55e" : fundScore >= 0 ? "#eab308" : "#ef4444";
  const latestColor = btScoreColor(latest);
  const sign        = latest >= 0 ? "+" : "";
  const quality     = data.signal_quality || {};
  const qualityVal  = quality.signal_quality_score;
  const qualityClr  = qualityVal === null || qualityVal === undefined
    ? "var(--text3)"
    : qualityVal >= 60 ? "#22c55e" : qualityVal >= 45 ? "#eab308" : "#ef4444";
  const hitRate     = quality.actionable_hit_rate_21d;
  const avgQuality  = btState.summary ? btState.summary.avg_signal_quality_score : null;
  const fundText    = fundMode === "disabled"
    ? "Disabled"
    : `${fundScore >= 0 ? "+" : ""}${fundScore.toFixed(3)}`;
  const fundTextClr = fundMode === "disabled" ? "var(--text3)" : fsColor;

  // ── Summary stat cards ────────────────────────────────────────────────────
  let html = `<div class="bt-stats-row">
    <div class="bt-stat-card">
      <div class="bt-stat-label">Current Signal</div>
      <div class="bt-stat-value" style="color:${latestColor};font-size:0.82rem">${latestLabel}</div>
    </div>
    <div class="bt-stat-card">
      <div class="bt-stat-label">Current Score</div>
      <div class="bt-stat-value" style="color:${latestColor}">${sign}${latest.toFixed(3)}</div>
    </div>
    <div class="bt-stat-card">
      <div class="bt-stat-label">Signal Quality</div>
      <div class="bt-stat-value" style="color:${qualityClr}">${qualityVal !== null && qualityVal !== undefined ? qualityVal.toFixed(1) : "—"}</div>
    </div>
    <div class="bt-stat-card">
      <div class="bt-stat-label">1M Hit Rate</div>
      <div class="bt-stat-value">${hitRate !== null && hitRate !== undefined ? `${(hitRate * 100).toFixed(1)}%` : "—"}</div>
    </div>
    <div class="bt-stat-card">
      <div class="bt-stat-label">Run Avg Quality</div>
      <div class="bt-stat-value">${avgQuality !== null && avgQuality !== undefined ? avgQuality.toFixed(1) : "—"}</div>
    </div>
    <div class="bt-stat-card">
      <div class="bt-stat-label">Fundamentals</div>
      <div class="bt-stat-value" style="color:${fundTextClr}">${fundText}</div>
    </div>
    <div class="bt-stat-card">
      <div class="bt-stat-label">Trading Days</div>
      <div class="bt-stat-value">${n.toLocaleString()}</div>
    </div>
  </div>`;

  // ── Distribution bar chart ─────────────────────────────────────────────────
  html += `<div class="bt-dist-row">`;
  for (const [key, count] of Object.entries(dist)) {
    if (!count) continue;
    const pct    = (count / n * 100);
    const pctStr = pct.toFixed(1);
    const barH   = Math.max(2, Math.round(pct));   // px height (out of 60)
    html += `
      <div class="bt-dist-item" title="${labels[key]}: ${count} days (${pctStr}%)">
        <div class="bt-dist-bar-wrap">
          <div class="bt-dist-bar" style="height:${barH}%;background:${colors[key]}"></div>
        </div>
        <div class="bt-dist-label" style="color:${colors[key]}">${labels[key]}</div>
        <div class="bt-dist-pct">${pctStr}%</div>
      </div>`;
  }
  html += `</div>`;

  const horizonBits = Array.isArray(quality.horizons)
    ? quality.horizons
        .map(h => `${h.label.toUpperCase()} IC: ${h.information_coefficient !== null && h.information_coefficient !== undefined ? h.information_coefficient.toFixed(3) : "—"}`)
        .join(" · ")
    : "";

  html += `<p class="bt-note" style="margin-top:0.9rem">
    Signal quality measures how well the score aligned with future returns: a weighted Spearman rank correlation of score vs forward returns over 5D, 21D, and 63D, plus a 21D actionable hit rate for BUY/SELL days.
    ${horizonBits ? `<br>${horizonBits}` : ""}
  </p>`;

  // ── Walk-forward OOS IC ────────────────────────────────────────────────────
  const wf = data.walk_forward || {};
  if (wf.n_windows > 0) {
    const inIC   = quality.composite_ic;
    const oosIC  = wf.oos_mean_ic;
    const oosIR  = wf.oos_icir;
    const pctPos = wf.pct_positive_windows;
    const overfit = inIC !== null && oosIC !== null ? inIC - oosIC : null;
    const overfitCls = overfit === null ? "" : overfit > 0.05 ? "neg" : overfit < -0.02 ? "pos" : "";

    html += `<div class="bt-section-title" style="margin-top:1.4rem">Walk-Forward OOS Validation
      <span class="bt-section-hint">${wf.n_windows} windows × ${wf.n_test_bars}d test / ${wf.n_train_bars}d burn-in</span>
    </div>
    <div class="bt-stats-row">
      <div class="bt-stat-card">
        <div class="bt-stat-label">In-Sample IC</div>
        <div class="bt-stat-value">${inIC !== null && inIC !== undefined ? inIC.toFixed(4) : "—"}</div>
      </div>
      <div class="bt-stat-card">
        <div class="bt-stat-label">OOS Mean IC</div>
        <div class="bt-stat-value ${oosIC !== null && oosIC > 0 ? "pos-text" : oosIC !== null && oosIC < 0 ? "neg-text" : ""}">${oosIC !== null && oosIC !== undefined ? oosIC.toFixed(4) : "—"}</div>
      </div>
      <div class="bt-stat-card">
        <div class="bt-stat-label">OOS ICIR</div>
        <div class="bt-stat-value">${oosIR !== null && oosIR !== undefined ? oosIR.toFixed(3) : "—"}</div>
      </div>
      <div class="bt-stat-card">
        <div class="bt-stat-label">Overfit Gap</div>
        <div class="bt-stat-value ${overfitCls}">${overfit !== null ? (overfit >= 0 ? "+" : "") + overfit.toFixed(4) : "—"}</div>
      </div>
      <div class="bt-stat-card">
        <div class="bt-stat-label">% Positive Windows</div>
        <div class="bt-stat-value">${pctPos !== null && pctPos !== undefined ? (pctPos * 100).toFixed(0) + "%" : "—"}</div>
      </div>
    </div>
    <div class="wf-windows">
      ${(wf.windows || []).map(w =>
        `<div class="wf-win ${w.ic > 0.02 ? "wf-pos" : w.ic < -0.02 ? "wf-neg" : "wf-neu"}"
              title="Window ${w.window_idx+1}: bars ${w.bar_start}–${w.bar_end}, n=${w.n_obs}">
          ${w.ic >= 0 ? "+" : ""}${w.ic.toFixed(3)}
        </div>`
      ).join("")}
    </div>`;
  }

  // ── Per-indicator IC ──────────────────────────────────────────────────────
  const pic = data.per_indicator_ic || {};
  if (Array.isArray(pic.indicators) && pic.indicators.length) {
    const maxAbsIC = Math.max(...pic.indicators.map(x => Math.abs(x.ic)), 0.001);
    html += `<div class="bt-section-title" style="margin-top:1.4rem">Per-Indicator IC
      <span class="bt-section-hint">Spearman corr vs ${pic.forward_days || 21}D forward return</span>
    </div>
    <div class="ic-table">`;
    for (const ind of pic.indicators) {
      const barW = Math.round(Math.abs(ind.ic) / maxAbsIC * 100);
      const cls  = ind.ic >= 0.02 ? "pos-text" : ind.ic <= -0.02 ? "neg-text" : "";
      html += `
      <div class="ic-row">
        <div class="ic-name">${ind.name}</div>
        <div class="ic-bar-wrap">
          <div class="ic-bar ${ind.ic >= 0 ? "ic-bar-pos" : "ic-bar-neg"}" style="width:${barW}%"></div>
        </div>
        <div class="ic-val ${cls}">${ind.ic >= 0 ? "+" : ""}${ind.ic.toFixed(4)}</div>
        <div class="ic-n muted">${ind.n_obs}obs</div>
      </div>`;
    }
    html += `</div>`;
  }

  // ── Signal correlation matrix ─────────────────────────────────────────────
  const cm = data.signal_correlation || {};
  if (Array.isArray(cm.columns) && cm.columns.length >= 2 && Array.isArray(cm.matrix)) {
    html += `<div class="bt-section-title" style="margin-top:1.4rem">Indicator Correlation (Spearman)
      <span class="bt-section-hint">High off-diagonal = redundant signals</span>
    </div>
    <div class="corr-matrix-wrap">
    <table class="corr-table">
      <thead><tr><th></th>${cm.columns.map(c => `<th>${c}</th>`).join("")}</tr></thead>
      <tbody>
      ${cm.matrix.map((row, i) => `
        <tr>
          <th>${cm.columns[i]}</th>
          ${row.map((v, j) => {
            if (v === null || v === undefined) return `<td class="muted">—</td>`;
            const abs = Math.abs(v);
            const hue = v > 0 ? "52,211,153" : "239,68,68";   // green / red
            const alpha = i === j ? 0.15 : Math.min(0.85, abs * 1.2);
            return `<td style="background:rgba(${hue},${alpha.toFixed(2)})" title="${cm.columns[i]} vs ${cm.columns[j]}: ${v.toFixed(3)}">${v.toFixed(2)}</td>`;
          }).join("")}
        </tr>`).join("")}
      </tbody>
    </table>
    </div>`;
  }

  // ── Factor exposure ───────────────────────────────────────────────────────
  const fe = data.factor_exposure || {};
  if (fe.r_squared !== null && fe.r_squared !== undefined && Object.keys(fe.factors || {}).length) {
    html += `<div class="bt-section-title" style="margin-top:1.4rem">Factor Exposure
      <span class="bt-section-hint">OLS beta of forward 21D returns on style factors (n=${fe.n_obs})</span>
    </div>
    <div class="bt-stats-row">
      <div class="bt-stat-card">
        <div class="bt-stat-label">Momentum β</div>
        <div class="bt-stat-value">${fe.factors.momentum_12_1m !== undefined ? (fe.factors.momentum_12_1m >= 0 ? "+" : "") + fe.factors.momentum_12_1m.toFixed(3) : "—"}</div>
      </div>
      <div class="bt-stat-card">
        <div class="bt-stat-label">Trend 50D β</div>
        <div class="bt-stat-value">${fe.factors.trend_50d !== undefined ? (fe.factors.trend_50d >= 0 ? "+" : "") + fe.factors.trend_50d.toFixed(3) : "—"}</div>
      </div>
      <div class="bt-stat-card">
        <div class="bt-stat-label">Low-Vol β</div>
        <div class="bt-stat-value">${fe.factors.low_vol !== undefined ? (fe.factors.low_vol >= 0 ? "+" : "") + fe.factors.low_vol.toFixed(3) : "—"}</div>
      </div>
      <div class="bt-stat-card">
        <div class="bt-stat-label">Factor R²</div>
        <div class="bt-stat-value">${(fe.r_squared * 100).toFixed(1)}%</div>
      </div>
    </div>`;
  }

  el.innerHTML = html;
}
