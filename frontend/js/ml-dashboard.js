/* ════════════════════════════════════════════════════════════════
   ML Lab Dashboard — Interactive ML training monitor
   ════════════════════════════════════════════════════════════════ */

const REGIME_COLORS = {
  TREND_UP:      { bg: "rgba(34,197,94,0.15)",  line: "#22c55e" },
  TREND_DOWN:    { bg: "rgba(239,68,68,0.15)",   line: "#ef4444" },
  REVERSAL_UP:   { bg: "rgba(6,182,212,0.15)",   line: "#06b6d4" },
  REVERSAL_DOWN: { bg: "rgba(249,115,22,0.15)",  line: "#f97316" },
  RANGE:         { bg: "rgba(167,139,250,0.08)", line: "#a78bfa" },
};

// ── State ─────────────────────────────────────────────────────────────────
const S = {
  charts: {},
  taskId: null,
  evtSource: null,
  result: null,
};

// ── Helpers ───────────────────────────────────────────────────────────────
function toast(msg, type = "info") {
  const c = document.getElementById("toastContainer");
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  c.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

function logLine(text, cls = "") {
  const term = document.getElementById("logTerminal");
  const line = document.createElement("div");
  line.className = cls;
  line.textContent = text;
  term.appendChild(line);
  term.scrollTop = term.scrollHeight;
}

function clearLog() {
  document.getElementById("logTerminal").innerHTML = "";
}

function $(id) { return document.getElementById(id); }

// ── Chart.js default config ──────────────────────────────────────────────
Chart.defaults.color = "#8b9ab4";
Chart.defaults.borderColor = "rgba(42,51,80,0.5)";
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size = 11;

// ── Validation Mode ───────────────────────────────────────────────────────
function initValModeTabs() {
  const tabs = document.querySelectorAll(".val-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      const mode = tab.dataset.mode;
      document.querySelectorAll(".val-panel").forEach(p => p.classList.add("hidden"));
      $(`panel-${mode}`).classList.remove("hidden");
      S.valMode = mode;
    });
  });
}

function buildValidationPayload() {
  const mode = S.valMode || "wf-preset";
  // wf_trade_cost in bps (UI shows whole bps, backend needs fraction)
  const wfCostBps = parseFloat($("wfTradeCost")?.value ?? 10);
  const wfTradeCost = (isNaN(wfCostBps) ? 10 : wfCostBps) / 10000;

  if (mode === "wf-preset") {
    return {
      validation_mode: "walkforward",
      period: $("wfPresetPeriod").value,
      cv_splits: parseInt($("wfPresetFolds").value),
      wf_gap: parseInt($("wfPresetGap").value),
      wf_window: $("wfPresetWindow").value,
      holdout_months: parseInt($("wfPresetHoldout").value) || 0,
      wf_trade_cost: wfTradeCost,
    };
  }
  if (mode === "wf-custom") {
    const start = $("wfCustomStart").value;
    const end   = $("wfCustomEnd").value;
    if (!start || !end) { toast("Set start and end date for custom walk-forward", "err"); return null; }
    return {
      validation_mode: "walkforward",
      start_date: start,
      end_date: end,
      cv_splits: parseInt($("wfCustomFolds").value),
      wf_gap: parseInt($("wfCustomGap").value),
      wf_window: $("wfCustomWindow").value,
      wf_trade_cost: wfTradeCost,
    };
  }
  if (mode === "chrono") {
    const trainStart = $("chronoTrainStart").value;
    const trainEnd   = $("chronoTrainEnd").value;
    const testStart  = $("chronoTestStart").value;
    const testEnd    = $("chronoTestEnd").value;
    if (!trainStart || !trainEnd || !testStart || !testEnd) {
      toast("Fill all four dates for chronological split", "err"); return null;
    }
    if (testStart <= trainEnd) {
      toast("Test start must be after train end", "err"); return null;
    }
    return {
      validation_mode: "chrono",
      train_start: trainStart,
      train_end: trainEnd,
      test_start: testStart,
      test_end: testEnd,
    };
  }
  return {};
}

// ── Init ──────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  S.valMode = "wf-preset";
  loadGPUStatus();
  initValModeTabs();
  $("trainBtn").addEventListener("click", startTraining);
  $("loadDataBtn").addEventListener("click", loadDataInfo);
  $("tickerInput").addEventListener("keydown", e => {
    if (e.key === "Enter") startTraining();
  });
  initCharts();
});

async function loadGPUStatus() {
  try {
    const r = await fetch("/api/ml/status");
    const d = await r.json();
    const badge = $("gpuBadge");
    if (d.cuda_available) {
      badge.textContent = `GPU: ${d.gpu_name || "CUDA"}`;
      badge.className = "ml-gpu-badge gpu-on";
    } else {
      badge.textContent = "CPU only";
      badge.className = "ml-gpu-badge gpu-off";
    }
  } catch (e) { /* ignore */ }
}

// ── Chart initialization ──────────────────────────────────────────────────

function initCharts() {
  // Loss chart
  S.charts.loss = new Chart($("lossChart"), {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { label: "Train Loss", data: [], borderColor: "#3b82f6", borderWidth: 2, pointRadius: 0, tension: 0.3 },
        { label: "Val Loss", data: [], borderColor: "#f59e0b", borderWidth: 2, pointRadius: 0, tension: 0.3 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: "top", labels: { boxWidth: 12 } }, annotation: { annotations: {} } },
      scales: { x: { title: { display: true, text: "Epoch" } }, y: { title: { display: true, text: "Loss" } } },
      animation: false,
    },
  });

  // Class distribution
  S.charts.classDist = new Chart($("classDistChart"), {
    type: "bar",
    data: { labels: [], datasets: [{ label: "Count", data: [], backgroundColor: [] }] },
    options: {
      responsive: true, maintainAspectRatio: false, indexAxis: "y",
      plugins: { legend: { display: false } },
    },
  });

  // Price chart
  S.charts.price = new Chart($("priceChart"), {
    type: "line",
    data: { labels: [], datasets: [] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        annotation: { annotations: {} },
        tooltip: {
          callbacks: {
            afterBody: function(ctx) {
              const idx = ctx[0]?.dataIndex;
              if (idx === undefined || !S.result?.timeseries) return "";
              const ts = S.result.timeseries;
              const regime = ts.regimes?.[idx] || "—";
              const entry = ts.entry_scores?.[idx]?.toFixed(3) || "—";
              const exit_ = ts.exit_scores?.[idx]?.toFixed(3) || "—";
              return `Regime: ${regime}\nEntry: ${entry}  Exit: ${exit_}`;
            }
          }
        }
      },
      scales: {
        x: { type: "category", ticks: { maxTicksLimit: 20 } },
        y: { title: { display: true, text: "Price" } },
      },
      animation: false,
    },
  });

  // Entry/Exit chart
  S.charts.entryExit = new Chart($("entryExitChart"), {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { label: "Entry Score", data: [], borderColor: "#22c55e", borderWidth: 1.5, pointRadius: 0, tension: 0.2, fill: { target: "origin", above: "rgba(34,197,94,0.08)" } },
        { label: "Exit Score", data: [], borderColor: "#ef4444", borderWidth: 1.5, pointRadius: 0, tension: 0.2, fill: { target: "origin", above: "rgba(239,68,68,0.08)" } },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: "top", labels: { boxWidth: 12 } } },
      scales: {
        x: { ticks: { maxTicksLimit: 20 } },
        y: { min: 0, max: 1, title: { display: true, text: "Score" } },
      },
      animation: false,
    },
  });

  // Feature importance
  S.charts.features = new Chart($("featureChart"), {
    type: "bar",
    data: { labels: [], datasets: [{ label: "Importance", data: [], backgroundColor: "#3b82f6" }] },
    options: {
      responsive: true, maintainAspectRatio: false, indexAxis: "y",
      plugins: { legend: { display: false } },
    },
  });

  // Regime probabilities
  S.charts.regimeProb = new Chart($("regimeProbChart"), {
    type: "line",
    data: { labels: [], datasets: [] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: "top", labels: { boxWidth: 10, font: { size: 10 } } } },
      scales: {
        x: { ticks: { maxTicksLimit: 20 } },
        y: { min: 0, max: 1, stacked: true, title: { display: true, text: "Probability" } },
      },
      animation: false,
    },
  });

  // Backtest equity chart
  S.charts.btEquity = new Chart($("btEquityChart"), {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { label: "Strategy", data: [], borderColor: "#22c55e", borderWidth: 2, pointRadius: 0, tension: 0.2, fill: false, yAxisID: "y" },
        { label: "Buy & Hold", data: [], borderColor: "#64748b", borderWidth: 1.5, pointRadius: 0, tension: 0.1, borderDash: [5,5], fill: false, yAxisID: "y" },
        { label: "Stock Price", data: [], borderColor: "#4fc3f7", borderWidth: 1.5, pointRadius: 0, tension: 0.1, borderDash: [3,3], fill: false, yAxisID: "y2" },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: "top", labels: { boxWidth: 12 } } },
      scales: {
        x: { ticks: { maxTicksLimit: 12 } },
        y:  { position: "left",  title: { display: true, text: "Equity ($)" } },
        y2: { position: "right", title: { display: true, text: "Price ($)" }, grid: { drawOnChartArea: false } },
      },
      animation: false,
    },
  });
}

// ── Training ──────────────────────────────────────────────────────────────

async function startTraining() {
  const ticker = $("tickerInput").value.trim().toUpperCase();
  if (!ticker) { toast("Enter a ticker first", "err"); return; }

  $("trainBtn").disabled = true;
  $("trainBtn").textContent = "Training...";
  clearLog();
  resetCharts();

  logLine(`Starting training for ${ticker}...`, "log-line-status");

  const valPayload = buildValidationPayload();
  if (!valPayload) { resetTrainBtn(); return; }

  const modelType = $("modelSelect").value;
  const body = {
    ticker,
    model_type: modelType,
    backend: modelType === "mlp" ? "auto" : "cpu",
    feature_set: $("featureSet").value,
    ...valPayload,
  };

  const _holdoutNote = valPayload.holdout_months > 0 ? `  |  holdout: last ${valPayload.holdout_months}mo` : "";
  logLine(`Mode: ${valPayload.validation_mode === "chrono" ? "Chronological split" : `Walk-forward (${valPayload.cv_splits ?? 5} folds)`}${_holdoutNote}`, "log-line-status");

  try {
    const r = await fetch("/api/ml/dashboard/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const d = await r.json();
    if (d.error) { toast(d.error, "err"); resetTrainBtn(); return; }

    S.taskId = d.task_id;
    logLine(`Task ${d.task_id} created, streaming epochs...`, "log-line-status");

    // Open SSE
    if (S.evtSource) S.evtSource.close();
    S.evtSource = new EventSource(`/api/ml/dashboard/stream/${d.task_id}`);

    S.evtSource.onmessage = (e) => {
      const ev = JSON.parse(e.data);
      handleStreamEvent(ev);
    };

    S.evtSource.onerror = () => {
      S.evtSource.close();
      // Fetch final result
      fetchResult(d.task_id);
    };

  } catch (e) {
    toast("Failed to start training: " + e.message, "err");
    resetTrainBtn();
  }
}

function handleStreamEvent(ev) {
  switch (ev.type) {
    case "status":
      logLine(ev.message, "log-line-status");
      break;

    case "epoch": {
      // Update loss chart in real time
      const chart = S.charts.loss;
      chart.data.labels.push(ev.epoch);
      chart.data.datasets[0].data.push(ev.train_loss);
      chart.data.datasets[1].data.push(ev.val_loss);
      chart.update("none");

      if (ev.epoch % 10 === 0 || ev.epoch < 5) {
        logLine(
          `Epoch ${ev.epoch}: train=${ev.train_loss.toFixed(4)} val=${ev.val_loss.toFixed(4)} lr=${ev.lr?.toFixed(6) || "—"}`,
          "log-line-epoch"
        );
      }
      break;
    }

    case "fold": {
      const acc = (ev.accuracy * 100).toFixed(1);
      const timeStr = ev.elapsed_s != null ? ` (${ev.elapsed_s}s)` : "";
      logLine(
        `  Fold ${ev.fold + 1}: train=${ev.train_size} rows → test=${ev.test_size} rows | acc=${acc}% | trades=${ev.n_trades}${timeStr}`,
        "log-line-epoch"
      );
      break;
    }

    case "complete":
      logLine("Training complete!", "log-line-ok");
      if (S.evtSource) S.evtSource.close();
      fetchResult(S.taskId);
      break;

    case "error":
      logLine("ERROR: " + ev.message, "log-line-err");
      toast("Training failed", "err");
      if (S.evtSource) S.evtSource.close();
      resetTrainBtn();
      break;
  }
}

async function fetchResult(taskId) {
  try {
    const r = await fetch(`/api/ml/dashboard/result/${taskId}`);
    const d = await r.json();
    if (d.status === "error") {
      logLine("Error: " + d.error, "log-line-err");
      toast("Training failed", "err");
      resetTrainBtn();
      return;
    }
    if (d.status === "running") {
      setTimeout(() => fetchResult(taskId), 500);
      return;
    }
    S.result = d.result;
    renderResults(d.result);
    toast("Training complete!", "ok");
  } catch (e) {
    logLine("Failed to fetch result: " + e.message, "log-line-err");
  }
  resetTrainBtn();
}

function resetTrainBtn() {
  $("trainBtn").disabled = false;
  $("trainBtn").textContent = "Train Model";
}

function resetCharts() {
  // Clear loss chart
  const lc = S.charts.loss;
  lc.data.labels = [];
  lc.data.datasets[0].data = [];
  lc.data.datasets[1].data = [];
  lc.options.plugins.annotation.annotations = {};
  lc.update();
}

// ── Render Results ────────────────────────────────────────────────────────

function renderResults(result) {
  const tlog = result.training_log;

  // Training info chips
  $("infoBackend").textContent = `Backend: ${result.backend}`;
  $("infoRows").textContent = `Samples: ${result.n_samples}`;
  $("infoFeatures").textContent = `Features: ${result.n_features}`;
  $("infoTime").textContent = `Time: ${result.training_time_s}s`;

  // Calibration chip — shows method + temperature for MLP
  const calChip = $("infoCalibration");
  if (calChip && tlog) {
    const calStatus = tlog.calibration_status || "";
    const calN      = tlog.calibration_samples ?? 0;
    const calT      = tlog.temperature ?? 1.0;
    if (calStatus === "isotonic") {
      calChip.textContent = `Cal: isotonic (${calN} samples)`;
      calChip.style.color = "#22c55e";
    } else if (calStatus === "raw_lgbm") {
      calChip.textContent = `Cal: raw LightGBM (${calN} < 50)`;
      calChip.style.color = "#f59e0b";
    } else if (calStatus === "temperature_scaling") {
      calChip.textContent = `Cal: T-scaling T=${calT.toFixed(3)} (${calN} samples)`;
      calChip.style.color = calT !== 1.0 ? "#22c55e" : "#f59e0b";
    } else {
      calChip.textContent = "Cal: —";
      calChip.style.color = "";
    }
  }

  if (tlog) {
    if (tlog.early_stop_epoch != null) {
      $("infoEarlyStop").textContent = `Early stop: epoch ${tlog.early_stop_epoch}`;
      $("infoEarlyStop").style.color = "#f59e0b";
      logLine(`Early stopping triggered at epoch ${tlog.early_stop_epoch}`, "log-line-warn");

      // Add annotation line on loss chart
      S.charts.loss.options.plugins.annotation.annotations.earlyStop = {
        type: "line", xMin: tlog.early_stop_epoch, xMax: tlog.early_stop_epoch,
        borderColor: "#f59e0b", borderWidth: 2, borderDash: [6, 3],
        label: { display: true, content: "Early Stop", position: "start", color: "#f59e0b", font: { size: 10 } },
      };
      S.charts.loss.update();
    } else {
      $("infoEarlyStop").textContent = `Full ${tlog.epochs.length} epochs`;
    }

    // If the epoch log wasn't streamed (loaded from cache), render it now
    if (S.charts.loss.data.labels.length === 0 && tlog.epochs.length > 0) {
      const lc = S.charts.loss;
      lc.data.labels = tlog.epochs.map(e => e.epoch);
      lc.data.datasets[0].data = tlog.epochs.map(e => e.train_loss);
      lc.data.datasets[1].data = tlog.epochs.map(e => e.val_loss);
      lc.update();
    }
  }

  // Metrics
  $("metricAcc").textContent = (result.regime_accuracy * 100).toFixed(1) + "%";
  $("metricAcc").style.color = result.regime_accuracy > 0.4 ? "#22c55e" : result.regime_accuracy > 0.25 ? "#f59e0b" : "#ef4444";
  $("metricEntryMAE").textContent = result.entry_mae.toFixed(4);
  $("metricExitMAE").textContent = result.exit_mae.toFixed(4);
  $("metricBestVL").textContent = tlog?.best_val_loss?.toFixed(4) || "—";

  // F1 bars
  renderF1Bars(result.regime_f1);

  // CV scores
  renderCVScores(result.cv_scores);

  // Data splits
  if (tlog) renderSplits(tlog, result.walk_forward?.fold_results ?? null);

  // Class distribution
  if (tlog?.class_distribution) renderClassDist(tlog.class_distribution);

  // Timeseries charts
  if (result.timeseries) {
    const wfFolds = result.walk_forward?.fold_results ?? null;
    renderPriceChart(result.timeseries, tlog, wfFolds);
    renderEntryExitChart(result.timeseries);
    renderRegimeProbChart(result.timeseries);
  }

  // Feature importance
  if (result.feature_importances) renderFeatureImportance(result.feature_importances);

  // Walk-forward metrics
  if (result.walk_forward) renderWalkForward(result.walk_forward);

  // Cache best policy thresholds for backtest
  const _pol = result.walk_forward?.policy_opt;
  if (_pol) {
    _bestEntryThreshold = _pol.best_entry_threshold;
    _bestExitThreshold  = _pol.best_exit_threshold;
    // Pre-fill hidden policy inputs if they exist
    const inpEntry = $("btEntryThreshold"), inpExit = $("btExitThreshold");
    if (inpEntry) inpEntry.value = _bestEntryThreshold;
    if (inpExit)  inpExit.value  = _bestExitThreshold;
  } else {
    _bestEntryThreshold = null;
    _bestExitThreshold  = null;
  }

  // Show post-train action bar
  $("postTrainActions").style.display = "";
  $("ptModelLabel").textContent = `${result.model_type?.toUpperCase() || "—"} · ${result.ticker} · ${result.n_samples} samples`;

  // Pre-fill backtest date range: holdout window if set, else last ~30%
  if (result.holdout_start) {
    $("btStart").value = result.holdout_start;
    $("btEnd").value   = result.holdout_end || "";
    toast(`Holdout period pre-filled: ${result.holdout_start} → ${result.holdout_end}`, "ok");
  } else if (result.timeseries?.dates?.length > 10) {
    const dates = result.timeseries.dates;
    const cutIdx = Math.floor(dates.length * 0.7);
    const suggestStart = dates[cutIdx]?.slice(0, 10);
    const suggestEnd   = dates[dates.length - 1]?.slice(0, 10);
    if (suggestStart) $("btStart").value = suggestStart;
    if (suggestEnd)   $("btEnd").value   = suggestEnd;
  }
  _updateBtRangeHint();

  // Log summary
  logLine(`\nResult Summary:`, "log-line-ok");
  logLine(`  Model: ${result.model_type || "—"} | Backend: ${result.backend}`);
  logLine(`  Accuracy: ${(result.regime_accuracy * 100).toFixed(1)}%`);
  logLine(`  Entry MAE: ${result.entry_mae.toFixed(4)} | Exit MAE: ${result.exit_mae.toFixed(4)}`);
  logLine(`  Samples: ${result.n_samples} | Features: ${result.n_features}`);
  if (result.cv_scores.length > 0) {
    logLine(`  WF Fold Accuracies: [${result.cv_scores.join(", ")}]`);
  }
  if (result.walk_forward) {
    const wf = result.walk_forward;
    logLine(`  All folds — Sharpe: ${wf.sharpe_ratio} | IR: ${(wf.information_ratio ?? 0).toFixed(2)} | CAGR: ${(wf.cagr * 100).toFixed(1)}% | MaxDD: ${(wf.max_drawdown * 100).toFixed(1)}% | Trades: ${wf.n_trades}`);
    const lf = wf.last_fold_metrics;
    if (lf) logLine(`  Last fold — Sharpe: ${lf.sharpe_ratio} | CAGR: ${(lf.cagr * 100).toFixed(1)}% | MaxDD: ${(lf.max_drawdown * 100).toFixed(1)}% | Trades: ${lf.n_trades}`);
    if (wf.policy_opt) {
      const p = wf.policy_opt;
      logLine(`  Policy grid: best entry=${p.best_entry_threshold} exit=${p.best_exit_threshold} Sharpe=${p.best_sharpe} (${p.improvement_pct >= 0 ? "+" : ""}${p.improvement_pct}% vs 0.60/0.60)`, "log-line-ok");
    }
  }
}

// ── Walk-Forward Metrics Rendering ────────────────────────────────────────

function renderWalkForward(wf) {
  // Core metrics
  const colorizePct = (el, val, goodPositive = true) => {
    el.textContent = (val * 100).toFixed(1) + "%";
    el.style.color = (goodPositive ? val > 0 : val < 0) ? "#22c55e" : val === 0 ? "var(--text2)" : "#ef4444";
  };

  const sharpeEl = $("wfSharpe");
  sharpeEl.textContent = wf.sharpe_ratio.toFixed(2);
  sharpeEl.style.color = wf.sharpe_ratio > 1 ? "#22c55e" : wf.sharpe_ratio > 0 ? "#f59e0b" : "#ef4444";

  // Information Ratio (annualised return / annualised vol, rf=0)
  const irEl = $("wfIR");
  if (irEl) {
    const ir = wf.information_ratio ?? 0;
    irEl.textContent = ir.toFixed(2);
    irEl.style.color = ir > 1 ? "#22c55e" : ir > 0 ? "#f59e0b" : "#ef4444";
  }

  // RANGE regime Sharpe — gate status
  const rangeSharpeEl = $("wfRangeSharpe");
  if (rangeSharpeEl) {
    const rs = wf.range_regime_sharpe ?? 0;
    const hasRangeTrades = wf.by_regime?.RANGE?.n_trades > 0;
    if (!hasRangeTrades) {
      rangeSharpeEl.textContent = "no trades";
      rangeSharpeEl.style.color = "var(--text3)";
    } else {
      rangeSharpeEl.textContent = rs.toFixed(2) + (rs < 0 ? " 🔒" : "");
      rangeSharpeEl.style.color = rs > 0.5 ? "#22c55e" : rs > 0 ? "#f59e0b" : "#ef4444";
      rangeSharpeEl.title = rs < 0
        ? "Negative — RANGE gate is ACTIVE: speculative RANGE entries are disabled"
        : "Positive — RANGE entries are allowed";
    }
  }

  colorizePct($("wfCAGR"), wf.cagr);
  colorizePct($("wfMaxDD"), wf.max_drawdown, false);
  colorizePct($("wfHitRate"), wf.hit_rate);
  colorizePct($("wfTotalReturn"), wf.total_return);

  const pfEl = $("wfProfitFactor");
  pfEl.textContent = wf.profit_factor.toFixed(2);
  pfEl.style.color = wf.profit_factor > 1.5 ? "#22c55e" : wf.profit_factor > 1 ? "#f59e0b" : "#ef4444";

  $("wfTrades").textContent = wf.n_trades;
  $("wfTradesMonth").textContent = wf.avg_trades_per_month.toFixed(1);
  $("wfAvgHold").textContent = wf.avg_holding_period.toFixed(1);

  const atrEl = $("wfAvgTradeRet");
  atrEl.textContent = (wf.avg_trade_return * 100).toFixed(2) + "%";
  atrEl.style.color = wf.avg_trade_return > 0 ? "#22c55e" : "#ef4444";

  // Last-fold-only metrics
  const lf = wf.last_fold_metrics;
  if (lf) {
    const lfSharpeEl = $("lfSharpe");
    lfSharpeEl.textContent = lf.sharpe_ratio.toFixed(2);
    lfSharpeEl.style.color = lf.sharpe_ratio > 1 ? "#22c55e" : lf.sharpe_ratio > 0 ? "#f59e0b" : "#ef4444";

    colorizePct($("lfCAGR"), lf.cagr);
    colorizePct($("lfMaxDD"), lf.max_drawdown, false);
    colorizePct($("lfHitRate"), lf.hit_rate);
    colorizePct($("lfTotalReturn"), lf.total_return);

    const lfPfEl = $("lfProfitFactor");
    lfPfEl.textContent = lf.profit_factor.toFixed(2);
    lfPfEl.style.color = lf.profit_factor > 1.5 ? "#22c55e" : lf.profit_factor > 1 ? "#f59e0b" : "#ef4444";

    $("lfTrades").textContent = lf.n_trades;
    $("lfTradesMonth").textContent = lf.avg_trades_per_month.toFixed(1);
    $("lfAvgHold").textContent = lf.avg_holding_period.toFixed(1);

    const lfAtrEl = $("lfAvgTradeRet");
    lfAtrEl.textContent = (lf.avg_trade_return * 100).toFixed(2) + "%";
    lfAtrEl.style.color = lf.avg_trade_return > 0 ? "#22c55e" : "#ef4444";
  }

  // By Regime table
  renderPerfTable("byRegimeTable", wf.by_regime, "Regime");

  // By Volatility table
  renderPerfTable("byVolTable", wf.by_volatility, "Vol Bucket");

  // Fold consistency summary
  const pctProfEl = $("wfPctProfitable");
  const pctProf = (wf.pct_folds_profitable ?? 0) * 100;
  pctProfEl.textContent = pctProf.toFixed(0) + "%";
  pctProfEl.style.color = pctProf >= 70 ? "#22c55e" : pctProf >= 50 ? "#f59e0b" : "#ef4444";

  $("wfSharpeStd").textContent = (wf.fold_sharpe_std ?? 0).toFixed(3);
  $("wfReturnStd").textContent = ((wf.fold_return_std ?? 0) * 100).toFixed(2) + "%";

  const worstEl = $("wfWorstFold");
  const wfIdx = wf.worst_fold_idx ?? -1;
  if (wfIdx >= 0 && wf.fold_results && wf.fold_results[wfIdx]) {
    const wf_fold = wf.fold_results[wfIdx];
    worstEl.textContent = `#${wfIdx + 1} (${(wf_fold.total_return * 100).toFixed(1)}%)`;
    worstEl.style.color = "#ef4444";
  } else {
    worstEl.textContent = "—";
  }

  // Window type badge
  const winType = wf.window_type || "expanding";
  const titleEl = document.querySelector(".card-title");  // walk-forward card title
  // update mode hint in log
  logLine(`  Window: ${winType} | Gap: ${wf.fold_results?.length > 0 ? "applied" : "n/a"} | Consistency: ${pctProf.toFixed(0)}% profitable folds`, "log-line-status");

  // Fold results
  renderFoldTable(wf.fold_results, wf.worst_fold_idx ?? -1);

  // Policy optimisation
  const polSec = $("policyOptSection");
  const pol = wf.policy_opt;
  if (pol && polSec) {
    polSec.style.display = "";

    // Best thresholds
    setText("polBestEntry",  pol.best_entry_threshold?.toFixed(2) ?? "—");
    setText("polBestExit",   pol.best_exit_threshold?.toFixed(2)  ?? "—");
    setText("polDefaultEntry", "0.60");
    setText("polDefaultExit",  "0.60");

    // Sharpe comparison
    const bsEl = $("polBestSharpe");
    if (bsEl) {
      bsEl.textContent = pol.best_sharpe?.toFixed(3) ?? "—";
      bsEl.style.color = pol.best_sharpe > 0 ? "#22c55e" : "#ef4444";
    }
    const dsEl = $("polDefaultSharpe");
    if (dsEl) {
      dsEl.textContent = pol.default_sharpe?.toFixed(3) ?? "—";
      dsEl.style.color = pol.default_sharpe > 0 ? "#22c55e" : "#ef4444";
    }

    // Improvement badge
    const impEl = $("polImprovement");
    if (impEl) {
      const imp = pol.improvement_pct ?? 0;
      impEl.textContent = (imp >= 0 ? "+" : "") + imp.toFixed(1) + "% vs default";
      impEl.style.color  = imp > 5 ? "#22c55e" : imp > 0 ? "#f59e0b" : "#ef4444";
    }

    // Locked policy banner
    const lockEl = $("polLockedBanner");
    if (lockEl) {
      lockEl.textContent =
        `Locked: Entry ≥ ${pol.best_entry_threshold?.toFixed(2)} · Exit ≥ ${pol.best_exit_threshold?.toFixed(2)} · Applied to backtest`;
    }

    // Grid heatmap (4×4)
    renderPolicyHeatmap(pol.grid_scores);

    // Log
    logLine(`  Policy opt: best entry=${pol.best_entry_threshold} exit=${pol.best_exit_threshold} ` +
            `Sharpe=${pol.best_sharpe} (${pol.improvement_pct >= 0 ? "+" : ""}${pol.improvement_pct}% vs default)`, "log-line-ok");
  } else if (polSec) {
    polSec.style.display = "none";
  }
}

function setText(id, val) {
  const el = $(id);
  if (el) el.textContent = val;
}

function resetPolicyToOptimal() {
  const inpEntry = $("btEntryThreshold"), inpExit = $("btExitThreshold");
  if (inpEntry && _bestEntryThreshold != null) inpEntry.value = _bestEntryThreshold;
  if (inpExit  && _bestExitThreshold  != null) inpExit.value  = _bestExitThreshold;
}

function renderPolicyHeatmap(gridScores) {
  const container = $("policyHeatmap");
  if (!container || !gridScores) return;

  const entryThrs = [...new Set(gridScores.map(s => s.entry_threshold))].sort();
  const exitThrs  = [...new Set(gridScores.map(s => s.exit_threshold))].sort();

  // Find min/max Sharpe for colour scaling
  const sharpes = gridScores.map(s => s.sharpe);
  const minS = Math.min(...sharpes), maxS = Math.max(...sharpes);
  const range = maxS - minS || 1;

  const colourFor = sharpe => {
    const t = (sharpe - minS) / range;   // 0 = worst, 1 = best
    // red → yellow → green
    if (t < 0.5) {
      const r = 239, g = Math.round(68 + (159 - 68) * (t * 2)), b = 68;
      return `rgb(${r},${g},${b})`;
    } else {
      const r = Math.round(234 - (234 - 34) * ((t - 0.5) * 2)), g = Math.round(179 + (197 - 179) * ((t - 0.5) * 2)), b = 68;
      return `rgb(${r},${g},${b})`;
    }
  };

  let html = `<table class="heatmap-table">
    <thead><tr><th>Entry↓ / Exit→</th>`;
  exitThrs.forEach(e => { html += `<th>${e.toFixed(2)}</th>`; });
  html += `</tr></thead><tbody>`;

  entryThrs.forEach(en => {
    html += `<tr><th>${en.toFixed(2)}</th>`;
    exitThrs.forEach(ex => {
      const cell = gridScores.find(s =>
        Math.abs(s.entry_threshold - en) < 1e-9 && Math.abs(s.exit_threshold - ex) < 1e-9
      );
      const s = cell?.sharpe ?? 0;
      const bg = colourFor(s);
      const textCol = s > (minS + maxS) / 2 ? "#000" : "#fff";
      html += `<td style="background:${bg};color:${textCol}" title="${cell?.n_trades ?? 0} trades">${s.toFixed(2)}</td>`;
    });
    html += `</tr>`;
  });

  html += `</tbody></table>`;
  container.innerHTML = html;
}

function renderPerfTable(containerId, data, firstColLabel) {
  const container = $(containerId);
  if (!data || Object.keys(data).length === 0) {
    container.innerHTML = '<span class="info-chip">No data</span>';
    return;
  }
  let html = `<table><thead><tr><th>${firstColLabel}</th><th>Trades</th><th>Hit Rate</th><th>Avg Ret</th></tr></thead><tbody>`;
  for (const [key, val] of Object.entries(data)) {
    const hrCls = val.hit_rate > 0.5 ? "pos" : val.hit_rate < 0.4 ? "neg" : "neu";
    const retCls = val.avg_return > 0 ? "pos" : val.avg_return < 0 ? "neg" : "neu";
    html += `<tr>
      <td>${key.replace("_", " ")}</td>
      <td>${val.n_trades}</td>
      <td class="${hrCls}">${(val.hit_rate * 100).toFixed(1)}%</td>
      <td class="${retCls}">${(val.avg_return * 100).toFixed(2)}%</td>
    </tr>`;
  }
  html += '</tbody></table>';
  container.innerHTML = html;
}

function renderFoldTable(folds, worstIdx = -1) {
  const container = $("foldTable");
  if (!folds || folds.length === 0) {
    container.innerHTML = '<span class="info-chip">No folds</span>';
    return;
  }
  let html = `<table style="font-size:0.75rem">
    <thead><tr>
      <th>Fold</th><th>Train</th><th>Test</th>
      <th>Acc</th><th>Sharpe</th><th>Return</th>
      <th>MaxDD</th><th>Win%</th><th>Trades</th><th>Vol</th>
    </tr></thead><tbody>`;
  for (const f of folds) {
    const isWorst = f.fold === worstIdx;
    const rowStyle = isWorst ? ' style="background:rgba(239,68,68,0.08)"' : '';
    const worstBadge = isWorst ? ' ⚠' : '';
    const accCls = f.accuracy > 0.35 ? "pos" : f.accuracy < 0.25 ? "neg" : "neu";
    const sharpeCls = (f.sharpe ?? 0) > 1 ? "pos" : (f.sharpe ?? 0) > 0 ? "neu" : "neg";
    const retCls = (f.total_return ?? 0) > 0 ? "pos" : "neg";
    const ddCls = (f.max_drawdown ?? 0) < -0.15 ? "neg" : (f.max_drawdown ?? 0) < -0.08 ? "neu" : "pos";
    const winCls = (f.win_rate ?? 0) > 0.5 ? "pos" : (f.win_rate ?? 0) < 0.4 ? "neg" : "neu";
    html += `<tr${rowStyle}>
      <td>${f.fold + 1}${worstBadge}</td>
      <td>${f.train_size}</td>
      <td>${f.test_size}</td>
      <td class="${accCls}">${(f.accuracy * 100).toFixed(1)}%</td>
      <td class="${sharpeCls}">${(f.sharpe ?? 0).toFixed(2)}</td>
      <td class="${retCls}">${((f.total_return ?? 0) * 100).toFixed(1)}%</td>
      <td class="${ddCls}">${((f.max_drawdown ?? 0) * 100).toFixed(1)}%</td>
      <td class="${winCls}">${((f.win_rate ?? 0) * 100).toFixed(0)}%</td>
      <td>${f.n_trades}</td>
      <td>${((f.volatility ?? 0) * 100).toFixed(1)}%</td>
    </tr>`;
  }
  html += '</tbody></table>';
  container.innerHTML = html;
}

function renderF1Bars(f1) {
  const container = $("f1Bars");
  container.innerHTML = "";
  const classes = Object.keys(REGIME_COLORS);
  for (const cls of classes) {
    const val = f1[cls] || 0;
    const color = REGIME_COLORS[cls].line;
    container.innerHTML += `
      <div class="f1-row">
        <span class="f1-label">${cls.replace("_", " ")}</span>
        <div class="f1-bar-bg">
          <div class="f1-bar-fill" style="width:${val * 100}%;background:${color}"></div>
        </div>
        <span class="f1-val">${val.toFixed(2)}</span>
      </div>`;
  }
}

function renderCVScores(scores) {
  const container = $("cvScores");
  container.innerHTML = "";
  if (!scores || scores.length === 0) {
    container.innerHTML = '<span class="info-chip">No CV (PyTorch)</span>';
    return;
  }
  for (const s of scores) {
    const color = s > 0.4 ? "#22c55e" : s > 0.25 ? "#f59e0b" : "#ef4444";
    container.innerHTML += `<span class="cv-chip" style="color:${color}">${(s * 100).toFixed(1)}%</span>`;
  }
}

function renderSplits(tlog, wfFolds) {
  const total = tlog.n_train_rows + tlog.n_test_rows;
  const valMode = S.valMode || "wf-preset";

  if (valMode !== "chrono" && wfFolds && wfFolds.length > 1) {
    // Walk-forward: show expanding train windows + fixed test slices
    const n = total;
    let html = "";
    wfFolds.forEach((fold, fi) => {
      const trainPct = (fold.train_size / n * 100).toFixed(1);
      const testPct  = (fold.test_size  / n * 100).toFixed(1);
      html += `<div class="split-seg split-train" style="flex:${trainPct}" title="Fold ${fi+1} train: ${fold.train_size} rows">▶ F${fi+1} train</div>`;
      html += `<div class="split-seg split-test"  style="flex:${testPct}"  title="Fold ${fi+1} test: ${fold.test_size} rows (acc ${(fold.accuracy*100).toFixed(1)}%)">test</div>`;
    });
    $("splitTimeline").innerHTML = html;
    const lastFold = wfFolds[wfFolds.length - 1];
    $("splitTrain").textContent = `Walk-forward: ${wfFolds.length} folds · final train window ${lastFold.train_size} rows`;
    $("splitTest").textContent  = `Test window per fold: ~${lastFold.test_size} rows`;
  } else {
    // Chrono or single split
    const trainPct = total > 0 ? (tlog.n_train_rows / total * 100) : 80;
    const testPct  = 100 - trainPct;
    $("splitTimeline").innerHTML = `
      <div class="split-seg split-train" style="flex:${trainPct}" title="Train: ${tlog.train_date_range?.[0]} → ${tlog.train_date_range?.[1]}">
        Train ${trainPct.toFixed(0)}%
      </div>
      <div class="split-seg split-test" style="flex:${testPct}" title="Test: ${tlog.test_date_range?.[0]} → ${tlog.test_date_range?.[1]}">
        Test ${testPct.toFixed(0)}%
      </div>`;
    $("splitTrain").textContent = `Train: ${tlog.n_train_rows} rows (${tlog.train_date_range?.[0] || "?"} → ${tlog.train_date_range?.[1] || "?"})`;
    $("splitTest").textContent  = `Test: ${tlog.n_test_rows} rows (${tlog.test_date_range?.[0] || "?"} → ${tlog.test_date_range?.[1] || "?"})`;
  }
  $("splitTotal").textContent = `Total: ${total} rows · ${tlog.data_date_range?.[0] || "?"} → ${tlog.data_date_range?.[1] || "?"}`;
}

function renderClassDist(dist) {
  const chart = S.charts.classDist;
  const labels = Object.keys(dist);
  const data = Object.values(dist);
  const colors = labels.map(l => REGIME_COLORS[l]?.line || "#8b9ab4");

  chart.data.labels = labels.map(l => l.replace("_", " "));
  chart.data.datasets[0].data = data;
  chart.data.datasets[0].backgroundColor = colors;
  chart.update();
}

// ── Price Chart with Regime Regions ───────────────────────────────────────

function renderPriceChart(ts, tlog, wfFolds) {
  const chart = S.charts.price;

  chart.data.labels = ts.dates;
  chart.data.datasets = [{
    label: "Price",
    data: ts.prices,
    borderColor: "#e2e8f4",
    borderWidth: 1.5,
    pointRadius: 0,
    tension: 0.1,
    fill: false,
  }];

  // Build regime region annotations
  const annotations = {};
  let regionStart = 0;
  let currentRegime = ts.regimes[0];

  for (let i = 1; i <= ts.regimes.length; i++) {
    if (i === ts.regimes.length || ts.regimes[i] !== currentRegime) {
      const rc = REGIME_COLORS[currentRegime];
      if (rc && currentRegime !== "RANGE") {
        annotations[`region_${regionStart}`] = {
          type: "box",
          xMin: regionStart, xMax: i - 1,
          backgroundColor: rc.bg,
          borderWidth: 0,
        };
      }
      if (i < ts.regimes.length) {
        regionStart = i;
        currentRegime = ts.regimes[i];
      }
    }
  }

  // ── Split lines depend on validation mode ────────────────────────────
  const valMode = S.valMode || "wf-preset";

  if (valMode === "chrono") {
    // Single train | test boundary line
    if (tlog?.train_date_range?.[1]) {
      const splitIdx = ts.dates.indexOf(tlog.train_date_range[1]);
      if (splitIdx >= 0) {
        annotations.trainTestSplit = {
          type: "line", xMin: splitIdx, xMax: splitIdx,
          borderColor: "#f59e0b", borderWidth: 2, borderDash: [8, 4],
          label: { display: true, content: "Train | Test", position: "start",
                   color: "#f59e0b", font: { size: 10 } },
        };
      }
    }
  } else if (wfFolds && wfFolds.length > 0) {
    // Walk-forward: draw a test-window highlight box for each fold
    const n = ts.dates.length;
    const nFolds = wfFolds.length;
    // Approximate each fold's test window as the last equal slice
    // Use fold train_size if available to position boundaries
    wfFolds.forEach((fold, fi) => {
      const trainSize = fold.train_size ?? Math.round(n * (fi + 1) / (nFolds + 1));
      const testSize  = fold.test_size  ?? Math.round(n / (nFolds + 1));
      const testStart = Math.min(trainSize, n - 1);
      const testEnd   = Math.min(trainSize + testSize - 1, n - 1);

      // Shaded test window
      annotations[`wf_test_${fi}`] = {
        type: "box",
        xMin: testStart, xMax: testEnd,
        backgroundColor: "rgba(245,158,11,0.07)",
        borderColor: "rgba(245,158,11,0.35)",
        borderWidth: 1,
      };
      // Fold boundary line
      annotations[`wf_line_${fi}`] = {
        type: "line", xMin: testStart, xMax: testStart,
        borderColor: "rgba(245,158,11,0.5)", borderWidth: 1, borderDash: [4, 4],
        label: { display: true, content: `F${fi + 1}`, position: "start",
                 color: "#f59e0b", font: { size: 9 } },
      };
    });
  }

  // Mark strong entry / exit signals
  for (let i = 0; i < ts.entry_scores.length; i++) {
    if (ts.entry_scores[i] > 0.75) {
      annotations[`entry_${i}`] = {
        type: "point", xValue: i, yValue: ts.prices[i],
        radius: 4, backgroundColor: "#22c55e", borderColor: "#22c55e",
      };
    }
    if (ts.exit_scores[i] > 0.75) {
      annotations[`exit_${i}`] = {
        type: "point", xValue: i, yValue: ts.prices[i],
        radius: 4, backgroundColor: "#ef4444", borderColor: "#ef4444",
      };
    }
  }

  chart.options.plugins.annotation.annotations = annotations;
  chart.options.scales.x.ticks.maxTicksLimit = 20;
  chart.update();
}

// ── Entry/Exit Score Timeline ────────────────────────────────────────────

function renderEntryExitChart(ts) {
  const chart = S.charts.entryExit;
  chart.data.labels = ts.dates;
  chart.data.datasets[0].data = ts.entry_scores;
  chart.data.datasets[1].data = ts.exit_scores;
  chart.update();
}

// ── Feature Importance ───────────────────────────────────────────────────

function renderFeatureImportance(imp) {
  const chart = S.charts.features;
  const sorted = Object.entries(imp).sort((a, b) => b[1] - a[1]);
  chart.data.labels = sorted.map(e => e[0]);
  chart.data.datasets[0].data = sorted.map(e => e[1]);
  chart.update();
}

// ── Regime Probability Timeline ──────────────────────────────────────────

function renderRegimeProbChart(ts) {
  const chart = S.charts.regimeProb;
  chart.data.labels = ts.dates;

  const classes = ts.classes || Object.keys(ts.regime_probs);
  chart.data.datasets = classes.map(cls => ({
    label: cls.replace("_", " "),
    data: ts.regime_probs[cls] || [],
    borderColor: REGIME_COLORS[cls]?.line || "#8b9ab4",
    backgroundColor: REGIME_COLORS[cls]?.bg || "rgba(139,154,180,0.1)",
    borderWidth: 1.2,
    pointRadius: 0,
    tension: 0.3,
    fill: true,
  }));

  chart.update();
}

// ── Load Data Info ────────────────────────────────────────────────────────

async function loadDataInfo() {
  const ticker = $("tickerInput").value.trim().toUpperCase();
  if (!ticker) { toast("Enter a ticker first", "err"); return; }

  try {
    const valPayload = buildValidationPayload();
    if (!valPayload) return;
    const body = { ticker, ...valPayload };
    const r = await fetch("/api/ml/dashboard/data-info", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const d = await r.json();
    if (d.error) { toast(d.error, "err"); return; }

    // Update split display
    const total = d.total_rows;
    const trainPct = (d.train_rows / total * 100);

    if (d.validation_mode === "walkforward" && d.cv_splits > 1) {
      // Show expanding folds
      const folds = d.cv_splits;
      const foldSize = d.approx_fold_size || Math.round(total / (folds + 1));
      let html = "";
      for (let i = 0; i < folds; i++) {
        const trainW = ((i + 1) * foldSize / total * 100).toFixed(1);
        const testW = (foldSize / total * 100).toFixed(1);
        html += `<div class="split-seg split-train" style="flex:${trainW}" title="Fold ${i+1} train">▶ F${i+1}</div>`;
        html += `<div class="split-seg split-test" style="flex:${testW}" title="Fold ${i+1} test">✦</div>`;
      }
      $("splitTimeline").innerHTML = html;
      $("splitTrain").textContent = `Walk-forward: ${folds} folds, ~${foldSize} rows/fold`;
    } else {
      $("splitTimeline").innerHTML = `
        <div class="split-seg split-train" style="flex:${trainPct}">Train ${trainPct.toFixed(0)}% (${d.train_rows})</div>
        <div class="split-seg split-test" style="flex:${100 - trainPct}">Test ${(100 - trainPct).toFixed(0)}% (${d.test_rows})</div>`;
      $("splitTrain").textContent = `Train: ${d.train_rows} rows (${d.train_range[0]} → ${d.train_range[1]})`;
    }

    $("splitTest").textContent = `Test: ${d.test_rows} rows (${d.test_range[0]} → ${d.test_range[1]})`;
    $("splitTotal").textContent = `Total: ${total} rows (${d.date_range[0]} → ${d.date_range[1]}) | $${d.price_range[0]} – $${d.price_range[1]}`;

    toast(`Loaded ${ticker}: ${total} rows`, "ok");
  } catch (e) {
    toast("Failed: " + e.message, "err");
  }
}

// ── Post-train actions ────────────────────────────────────────────────────

function openBacktestPanel() {
  $("backtestConfig").classList.toggle("hidden");
  // Pre-fill date range hint based on training data
  _updateBtRangeHint();
}

function _updateBtRangeHint() {
  const s = $("btStart").value, e = $("btEnd").value;
  const hint = $("btRangeHint");
  if (!hint) return;
  if (s || e) {
    hint.textContent = `Custom range: ${s || "start"} → ${e || "end"}`;
  } else {
    hint.textContent = "No range = full dataset (in-sample)";
  }
}
document.addEventListener("DOMContentLoaded", () => {
  const s = document.getElementById("btStart"), e = document.getElementById("btEnd");
  if (s) s.addEventListener("change", _updateBtRangeHint);
  if (e) e.addEventListener("change", _updateBtRangeHint);
});

// ── Live Inference ────────────────────────────────────────────────────────

let _lastInferResult = null;  // stored for Excel export
let _bestEntryThreshold = null;  // set after training if policy grid search ran
let _bestExitThreshold  = null;

async function runInference() {
  const ticker = $("tickerInput").value.trim().toUpperCase();
  if (!ticker) { toast("No ticker selected", "err"); return; }
  const btn = $("inferBtn");
  btn.disabled = true; btn.textContent = "Fetching…";
  try {
    const r = await fetch("/api/ml/dashboard/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker }),
    });
    const d = await r.json();
    if (d.error) { toast(d.error.split("\n")[0], "err"); return; }
    _lastInferResult = d.result;
    renderInference(d.result);
  } catch (e) {
    toast("Inference failed: " + e.message, "err");
  } finally {
    btn.disabled = false; btn.textContent = "⚡ Live Signal";
  }
}

function renderInference(res) {
  $("inferPanel").style.display = "";
  $("inferAsOf").textContent = `As of ${res.as_of}`;

  // Decision badge
  const sig = (res.ml_signal || "HOLD").toUpperCase();
  const badge = $("inferDecision");
  badge.textContent = sig;
  badge.className = "infer-badge " + {
    BUY: "infer-buy", SELL: "infer-sell", REDUCE: "infer-reduce", HOLD: "infer-hold"
  }[sig] || "infer-hold";

  // Regime
  const regColors = {
    TREND_UP:      "#22c55e", TREND_DOWN:    "#ef4444",
    REVERSAL_UP:   "#86efac", REVERSAL_DOWN: "#fca5a5",
    RANGE:         "#94a3b8",
  };
  $("inferRegime").textContent = (res.regime || "—").replace(/_/g, " ");
  $("inferRegime").style.color = regColors[res.regime] || "var(--text1)";
  $("inferRegimeConf").textContent = res.regime_confidence != null
    ? `${(res.regime_confidence * 100).toFixed(1)}% confidence` : "";

  // Scores
  const ep = res.entry_score ?? 0, xp = res.exit_score ?? 0;
  $("inferEntryBar").style.width = (ep * 100) + "%";
  $("inferExitBar").style.width  = (xp * 100) + "%";
  $("inferEntryVal").textContent = (ep * 100).toFixed(0) + "%";
  $("inferExitVal").textContent  = (xp * 100).toFixed(0) + "%";

  // Regime probability bars
  const probs = res.regime_probs || {};
  $("inferRegimeProbs").innerHTML = Object.entries(probs)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `
      <div class="infer-prob-row">
        <span class="infer-prob-label">${k.replace(/_/g, " ")}</span>
        <div class="infer-prob-bar"><div class="infer-prob-fill" style="width:${(v*100).toFixed(0)}%;background:${regColors[k]||'#6366f1'}"></div></div>
        <span class="infer-prob-val">${(v*100).toFixed(0)}%</span>
      </div>`).join("");

  // Key indicators
  const ind = res.indicators || {};
  const fmtInd = (label, val, dec=2, suffix="") =>
    val != null ? `<div class="infer-ind">${label}: <span>${parseFloat(val).toFixed(dec)}${suffix}</span></div>` : "";
  $("inferIndicators").innerHTML = [
    fmtInd("Close",    ind.close,   2, "$".padStart(0)),
    fmtInd("RSI",      ind.rsi,     1),
    fmtInd("BB%",      ind.bb_pct ? ind.bb_pct * 100 : null, 1, "%"),
    fmtInd("ADX",      ind.adx,     1),
    fmtInd("ATR%",     ind.atr_pct, 2, "%"),
    fmtInd("VolRatio", ind.vol_ratio, 2, "x"),
  ].filter(Boolean).join("");

  // Fund quality overlay score
  const fundGroup = $("inferFundGroup");
  const fundEl    = $("inferFundScore");
  const fundLabel = $("inferFundLabel");
  if (fundGroup && fundEl) {
    const fqs = res.fund_quality_score;
    if (fqs != null) {
      fundGroup.style.display = "";
      fundEl.textContent = fqs.toFixed(2);
      if (fqs < -0.5) {
        fundEl.style.color = "#ef4444";
        if (fundLabel) fundLabel.textContent = "⛔ balance sheet veto";
      } else if (fqs < -0.2) {
        fundEl.style.color = "#f97316";
        if (fundLabel) fundLabel.textContent = "⚠ size capped 60%";
      } else if (fqs > 0.3) {
        fundEl.style.color = "#22c55e";
        if (fundLabel) fundLabel.textContent = "+10% size bonus";
      } else {
        fundEl.style.color = "#f59e0b";
        if (fundLabel) fundLabel.textContent = "neutral";
      }
    } else {
      fundGroup.style.display = "";
      fundEl.textContent = "N/A";
      fundEl.style.color = "var(--text3)";
      if (fundLabel) fundLabel.textContent = "no fundamentals";
    }
  }

  // RANGE gate status
  const rangeGroup = $("inferRangeGroup");
  const rangeEl    = $("inferRangeGate");
  if (rangeGroup && rangeEl) {
    rangeGroup.style.display = "";
    const active = res.range_gate_active ?? false;
    rangeEl.textContent = active ? "ACTIVE 🔒" : "OFF";
    rangeEl.style.color = active ? "#f59e0b" : "#22c55e";
  }

  // Decision reasons
  const reasonsDiv  = $("inferReasons");
  const reasonsText = $("inferReasonsText");
  const reasons = res.decision?.reasons;
  if (reasonsDiv && reasonsText && reasons && reasons.length > 0) {
    reasonsDiv.style.display = "";
    reasonsText.textContent = reasons.join(" · ");
  } else if (reasonsDiv) {
    reasonsDiv.style.display = "none";
  }
}

function goToAnalysis() {
  if (!_lastInferResult) { toast("Run inference first", "err"); return; }
  const ticker = _lastInferResult.ticker;
  // Navigate to main analysis page with ticker pre-filled and ml_lab flag set
  window.location.href = `/?ticker=${encodeURIComponent(ticker)}&ml_lab=1`;
}

async function saveToRegistry() {
  const ticker = $("tickerInput").value.trim().toUpperCase();
  if (!ticker) { toast("No ticker selected", "err"); return; }
  const notes = prompt("Optional notes for this model version:", "") ?? "";
  const btn = $("saveModelBtn");
  btn.disabled = true;
  btn.textContent = "Saving…";
  try {
    const r = await fetch("/api/ml/dashboard/registry/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker, notes }),
    });
    const d = await r.json();
    if (d.error) { toast(d.error.split("\n")[0], "err"); return; }
    toast(`Saved: ${d.version_id}`, "ok");
    loadRegistry();
  } catch (e) {
    toast("Save failed: " + e.message, "err");
  } finally {
    btn.disabled = false;
    btn.textContent = "💾 Save to Registry";
  }
}

async function runBacktest() {
  const ticker = $("tickerInput").value.trim().toUpperCase();
  if (!ticker) { toast("No ticker selected", "err"); return; }

  const btn = $("runBtBtn");
  btn.disabled = true;
  btn.textContent = "Running…";

  const slRaw = $("btStopLoss").value;
  const tpRaw = $("btTakeProfit").value;

  const btStart = $("btStart").value || null;
  const btEnd   = $("btEnd").value   || null;

  // Read policy thresholds: prefer explicit input fields, fall back to grid-search best
  const _inpEntry = $("btEntryThreshold"), _inpExit = $("btExitThreshold");
  const _entryThrVal = _inpEntry?.value ? parseFloat(_inpEntry.value) : _bestEntryThreshold;
  const _exitThrVal  = _inpExit?.value  ? parseFloat(_inpExit.value)  : _bestExitThreshold;

  const body = {
    ticker,
    initial_capital: parseFloat($("btCapital").value) || 100000,
    commission_pct: parseFloat($("btCommission").value) / 100 || 0.001,
    slippage_pct: parseFloat($("btSlippage").value) / 100 || 0.0005,
    stop_loss_pct: slRaw ? parseFloat(slRaw) / 100 : null,
    take_profit_pct: tpRaw ? parseFloat(tpRaw) / 100 : null,
    start_date: btStart,
    end_date: btEnd,
    entry_threshold: _entryThrVal,
    exit_threshold:  _exitThrVal,
  };

  try {
    const r = await fetch("/api/ml/dashboard/backtest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const d = await r.json();
    if (d.error) { toast(d.error.split("\n")[0], "err"); return; }
    renderBacktest(d.result);
    $("backtestConfig").classList.add("hidden");
  } catch (e) {
    toast("Backtest failed: " + e.message, "err");
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Backtest";
  }
}

function renderBacktest(bt) {
  $("backtestPanel").style.display = "";
  $("btModelLabel").textContent = `${bt.ticker} · ${bt.dates?.[0]} → ${bt.dates?.[bt.dates?.length-1]}`;

  const colorize = (el, val, goodPos = true) => {
    el.textContent = (val * 100).toFixed(2) + "%";
    el.style.color = (goodPos ? val > 0 : val < 0) ? "#22c55e" : val === 0 ? "var(--text2)" : "#ef4444";
  };
  const colorizeNum = (el, val, good = ">0") => {
    el.textContent = val?.toFixed(3) ?? "—";
    el.style.color = (good === ">0" ? val > 0 : val > 1) ? "#22c55e" : val === 0 ? "var(--text2)" : "#ef4444";
  };

  colorize($("btTotalReturn"), bt.total_return);
  colorize($("btCAGR"), bt.cagr);
  colorizeNum($("btSharpe"), bt.sharpe_ratio);
  colorizeNum($("btSortino"), bt.sortino_ratio);
  colorizeNum($("btCalmar"), bt.calmar_ratio);
  colorize($("btMaxDD"), bt.max_drawdown, false);
  colorize($("btHitRate"), bt.hit_rate);
  const pfEl = $("btProfitFactor");
  pfEl.textContent = bt.profit_factor?.toFixed(2) ?? "—";
  pfEl.style.color = bt.profit_factor > 1.5 ? "#22c55e" : bt.profit_factor > 1 ? "#f59e0b" : "#ef4444";
  $("btTrades").textContent = `${bt.n_trades} (W:${bt.n_winning} L:${bt.n_losing})`;
  $("btAvgHold").textContent = (bt.avg_holding_days ?? 0).toFixed(1) + "d";
  $("btWinStreak").textContent = `${bt.max_win_streak}W / ${bt.max_loss_streak}L`;
  $("btComm").textContent = `$${(bt.total_commission ?? 0).toLocaleString(undefined, {maximumFractionDigits:0})}`;

  // Equity curve + stock price
  const chart = S.charts.btEquity;
  const step = Math.max(1, Math.floor((bt.equity_curve?.length || 1) / 300));
  const labels = (bt.dates || []).filter((_, i) => i % step === 0);
  const equity = (bt.equity_curve || []).filter((_, i) => i % step === 0);
  const initial = equity[0] || 100000;
  const bnh = equity.map(() => initial);  // flat buy-and-hold baseline

  // Stock price on secondary axis
  const rawPrices = (bt.close_prices || []).filter((_, i) => i % step === 0);

  chart.data.labels = labels;
  chart.data.datasets[0].data = equity;
  chart.data.datasets[1].data = bnh;
  chart.data.datasets[2].data = rawPrices;
  // Show/hide price axis depending on data availability
  chart.options.scales.y2.display = rawPrices.length > 0;
  chart.update();

  // Trade log
  renderTradeLog(bt.trades || []);
}

function renderTradeLog(trades) {
  const el = $("btTradeLog");
  if (!trades.length) {
    el.innerHTML = '<span class="info-chip">No trades in backtest period</span>';
    return;
  }
  let html = `<table>
    <thead><tr>
      <th>#</th><th>Entry</th><th>Exit</th><th>Hold</th>
      <th>Regime</th><th>Entry Score</th><th>Exit Score</th>
      <th>P&L %</th><th>Reason</th><th>Commission</th>
    </tr></thead><tbody>`;
  trades.forEach((t, i) => {
    const cls = t.pnl_pct > 0 ? "trade-win" : "trade-loss";
    const pct = (t.pnl_pct * 100).toFixed(2);
    html += `<tr>
      <td>${i + 1}</td>
      <td>${t.entry_date}</td>
      <td>${t.exit_date}</td>
      <td>${t.holding_days}d</td>
      <td>${t.regime?.replace("_"," ") || "—"}</td>
      <td>${(t.entry_score * 100).toFixed(0)}%</td>
      <td>${(t.exit_score * 100).toFixed(0)}%</td>
      <td class="${cls}">${pct > 0 ? "+" : ""}${pct}%</td>
      <td>${t.exit_reason || "—"}</td>
      <td>$${(t.commission_paid || 0).toFixed(0)}</td>
    </tr>`;
  });
  html += "</tbody></table>";
  el.innerHTML = html;
}

// ── Model Registry ────────────────────────────────────────────────────────

async function loadRegistry() {
  const ticker = $("tickerInput").value.trim().toUpperCase() || undefined;
  try {
    const url = "/api/ml/dashboard/registry" + (ticker ? `?ticker=${ticker}` : "");
    const r = await fetch(url);
    const d = await r.json();
    if (d.error) { toast(d.error, "err"); return; }
    renderRegistry(d.versions || []);
  } catch (e) {
    toast("Failed to load registry: " + e.message, "err");
  }
}

function renderRegistry(versions) {
  const el = $("registryTable");
  if (!versions.length) {
    el.innerHTML = '<span class="info-chip">No saved versions yet. Train a model and click Save to Registry.</span>';
    return;
  }
  const tagClass = { lightgbm: "tag-lgbm", mlp: "tag-mlp", logistic: "tag-logistic" };
  let html = `<table>
    <thead><tr>
      <th>Version</th><th>Type</th><th>Backend</th><th>Train Period</th><th>Samples</th>
      <th>Acc</th><th>Sharpe</th><th>Max DD</th><th>CAGR</th>
      <th>Hit Rate</th><th>Saved</th><th>Notes</th><th>Actions</th>
    </tr></thead><tbody>`;
  for (const v of versions) {
    const tc = tagClass[v.model_type] || "";
    const fmt = (x, pct=false) => x == null ? "—" : pct ? `${(x*100).toFixed(1)}%` : x.toFixed(3);
    const sharpeColor = (v.sharpe_ratio ?? 0) > 1 ? "#22c55e" : (v.sharpe_ratio ?? 0) > 0 ? "#f59e0b" : "#ef4444";
    const backendLabel = v.backend === "pytorch" ? "GPU" : (v.backend || v.model_type || "—").toUpperCase();
    const backendColor = v.backend === "pytorch" ? "#22c55e" : "var(--text3)";
    html += `<tr>
      <td><strong>${v.version_id}</strong></td>
      <td><span class="tag ${tc}">${v.model_type?.toUpperCase()}</span></td>
      <td style="color:${backendColor};font-size:0.75rem">${backendLabel}</td>
      <td>${v.train_period || "—"}</td>
      <td>${v.n_samples ?? "—"}</td>
      <td>${fmt(v.regime_accuracy, true)}</td>
      <td style="color:${sharpeColor}">${fmt(v.sharpe_ratio)}</td>
      <td style="color:#ef4444">${fmt(v.max_drawdown, true)}</td>
      <td>${fmt(v.cagr, true)}</td>
      <td>${fmt(v.hit_rate, true)}</td>
      <td>${(v.created_at || "").slice(0,10)}</td>
      <td style="color:var(--text3);font-style:italic">${v.notes || ""}</td>
      <td>
        <button class="btn btn-sm" style="font-size:0.7rem;padding:0.15rem 0.4rem" onclick="loadVersion('${v.version_id}')">Load</button>
        <button class="btn btn-sm" style="font-size:0.7rem;padding:0.15rem 0.4rem;color:#ef4444" onclick="deleteVersion('${v.version_id}')">✕</button>
      </td>
    </tr>`;
  }
  html += "</tbody></table>";
  el.innerHTML = html;
}

async function loadVersion(versionId) {
  const btn = event.target;
  btn.disabled = true;
  btn.textContent = "Loading…";
  try {
    const r = await fetch("/api/ml/dashboard/registry/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ version_id: versionId }),
    });
    const d = await r.json();
    if (d.error) { toast(d.error.split("\n")[0], "err"); return; }

    // Update ticker input and show post-train bar
    $("tickerInput").value = d.ticker;
    $("postTrainActions").style.display = "";
    $("ptModelLabel").textContent = `${d.model_type?.toUpperCase()} · ${d.ticker} · loaded from registry`;

    // Pre-fill backtest dates from the model's train period if available
    // train_period looks like "2022-01-01 → 2024-12-31"
    if (d.train_period) {
      const parts = d.train_period.split(/\s*→\s*/);
      if (parts.length === 2) {
        // Suggest last ~30% as out-of-sample backtest window
        const t0 = new Date(parts[0].trim()), t1 = new Date(parts[1].trim());
        const cutMs = t0.getTime() + (t1.getTime() - t0.getTime()) * 0.7;
        $("btStart").value = new Date(cutMs).toISOString().slice(0, 10);
        $("btEnd").value   = parts[1].trim().slice(0, 10);
        _updateBtRangeHint();
      }
    }
    toast(`Loaded ${versionId} — ready to backtest`, "ok");
  } catch (e) {
    toast("Load failed: " + e.message, "err");
  } finally {
    btn.disabled = false;
    btn.textContent = "Load";
  }
}

async function deleteVersion(versionId) {
  if (!confirm(`Delete ${versionId}?`)) return;
  try {
    const r = await fetch(`/api/ml/dashboard/registry/${versionId}`, { method: "DELETE" });
    const d = await r.json();
    if (d.error) { toast(d.error, "err"); return; }
    toast(`Deleted ${versionId}`, "ok");
    loadRegistry();
  } catch (e) {
    toast("Delete failed: " + e.message, "err");
  }
}
