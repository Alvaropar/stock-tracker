/* Sentiment Pipeline — frontend application
 * Extracted from inline HTML for proper CSP support.
 */
"use strict";

const $ = s => document.querySelector(s);
let CFG = {};
let POLL_TIMER = null;
let priceChart = null;
let candleSeries = null;

// ── Live price state ──
let PRICE_TIMER = null;
let PRICE_INTERVAL = 30;
let PRICE_MARKET = '';
let PRICE_ASSET_TYPE = '';
let PRICE_ASSET_ID = '';
let PRICE_CHART_BUILT = false;

// ── File Browser state ──
let FB_TARGET_INPUT = null;
let FB_CURRENT_PATH = '';

function log(msg) {
  const el = $('#log');
  const ts = new Date().toLocaleTimeString();
  el.textContent += `[${ts}] ${msg}\n`;
  el.scrollTop = el.scrollHeight;
}

function setBusy(btnId, on) {
  const b = $(btnId);
  if (on) { b.disabled = true; b.dataset.orig = b.textContent; b.innerHTML = '<span class="spinner"></span> Working...'; }
  else    { b.disabled = false; b.textContent = b.dataset.orig || b.textContent; }
}

async function api(path, opts) {
  const resp = await fetch('/api' + path, opts);
  const data = await resp.json();
  if (!data.ok && data.error) throw new Error(data.error);
  return data;
}
async function post(path, body={}) {
  return api(path, { method:'POST', headers:{'Content-Type':'application/json'},
                     body:JSON.stringify(body) });
}
async function get(path) { return api(path); }

// ── Toast notifications ──
function toast(msg, type='info') {
  const el = document.createElement('div');
  el.className = 'toast ' + type;
  el.textContent = msg;
  const container = $('#toasts');
  container.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity 0.3s'; setTimeout(() => el.remove(), 300); }, 4000);
}

// ── HTML escaping ──
function escHtml(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

// ── Pagination state ──
let PAGE_SIZE = 20;
let CURRENT_PAGE = 0;
let ALL_ARTICLES = [];
let SHOW_SENTIMENT = false;
let SEARCH_QUERY = '';

// ── init ──
async function init() {
  CFG = await get('/config');

  if (CFG.version) {
    const v = $('#app-version');
    if (v) v.textContent = 'v' + CFG.version;
  }

  const ms = $('#sel-market');
  ms.innerHTML = CFG.markets.map(m =>
    `<option value="${m.name}">${m.display_name}</option>`).join('');

  const lc = CFG.last_config || {};

  if (lc.last_market) {
    ms.value = lc.last_market;
  }

  const bases = CFG.base_models || [];
  const adapters = CFG.adapters || [];

  if (lc.last_filter_base) {
    $('#inp-filter-base').value = lc.last_filter_base;
  } else if (bases.length > 0) {
    $('#inp-filter-base').value = bases[0].path;
  }
  if (lc.last_filter_adapter) {
    $('#inp-filter-adapter').value = lc.last_filter_adapter;
  }
  if (lc.last_sentiment_base) {
    $('#inp-sent-base').value = lc.last_sentiment_base;
  } else if (bases.length > 0) {
    $('#inp-sent-base').value = bases[0].path;
  }
  if (lc.last_sentiment_adapter) {
    $('#inp-sent-adapter').value = lc.last_sentiment_adapter;
  } else {
    for (const a of adapters) {
      if (a.name.includes('sft-qwen3')) {
        $('#inp-sent-adapter').value = a.path;
        break;
      }
    }
  }

  updateAssetDropdown();
  ms.addEventListener('change', updateAssetDropdown);
  $('#sel-asset-type').addEventListener('change', updateAssetDropdown);

  if (lc.last_asset_type) {
    $('#sel-asset-type').value = lc.last_asset_type;
    updateAssetDropdown();
    if (lc.last_asset_id) {
      if (lc.last_asset_type === 'commodity') {
        $('#sel-asset').value = lc.last_asset_id;
      } else {
        $('#inp-asset').value = lc.last_asset_id;
      }
    }
  }

  const isFirstRun = !lc.last_market && !lc.last_filter_base;
  if (isFirstRun) {
    setTimeout(showWizard, 500);
  }

  // ── Bind event listeners (replaces inline onclick/onchange/oninput) ──
  _bind('#btn-configure', 'click', configure);
  _bind('#btn-apply-models', 'click', applyModels);
  _bind('#sel-refresh-interval', 'change', changeRefreshInterval);
  _bind('.modal-close', 'click', closeBrowser);
  _bind('#fb-cancel-btn', 'click', closeBrowser);
  _bind('#fb-select-btn', 'click', selectBrowserDir);
  _bind('#btn-fetch', 'click', () => fetchNews());
  _bind('#btn-filter', 'click', startFilter);
  _bind('#btn-sentiment', 'click', startSentiment);
  _bind('#btn-export', 'click', exportCSV);
  _bind('#btn-history', 'click', toggleHistory);
  _bind('#inp-search', 'input', function() { onSearchInput(this.value); });
  _bind('[title="Refresh now"]', 'click', refreshPriceNow);

  // Browse buttons — use data-target to identify which input to fill
  document.querySelectorAll('.btn-browse').forEach(btn => {
    btn.addEventListener('click', function() {
      openBrowser(this.dataset.target);
    });
  });
}

function _bind(sel, evt, fn) {
  const el = document.querySelector(sel);
  if (el) el.addEventListener(evt, fn);
}

function updateAssetDropdown() {
  const atype = $('#sel-asset-type').value;
  const market = $('#sel-market').value;
  const selAsset = $('#sel-asset');
  const inpAsset = $('#inp-asset');

  if (atype === 'commodity') {
    selAsset.style.display = '';
    inpAsset.style.display = 'none';
    const clist = market === 'US' ? CFG.us_commodities : CFG.china_commodities;
    selAsset.innerHTML = clist.map(c => `<option value="${c}">${c}</option>`).join('');
  } else {
    if (market === 'US') {
      selAsset.style.display = '';
      inpAsset.style.display = '';
      inpAsset.placeholder = 'or type ticker';
      selAsset.innerHTML = '<option value="">Popular tickers...</option>'
        + (CFG.us_stocks||[]).map(s => `<option value="${s}">${s}</option>`).join('');
    } else {
      selAsset.style.display = 'none';
      inpAsset.style.display = '';
      inpAsset.placeholder = 'Stock code, e.g. 600547';
    }
  }
}

// ── configure ──
async function configure() {
  const market = $('#sel-market').value;
  const atype = $('#sel-asset-type').value;
  let assetId;
  if (atype === 'commodity') {
    assetId = $('#sel-asset').value;
  } else if (market === 'US') {
    assetId = $('#inp-asset').value.trim() || $('#sel-asset').value;
  } else {
    assetId = $('#inp-asset').value.trim();
  }
  if (!assetId) { toast('Enter an asset first.', 'error'); return; }

  try {
    await post('/market', { market });
    await post('/asset', { asset_type: atype, asset_id: assetId });
    log(`Configured: ${market} / ${atype} / ${assetId}`);
    toast(`Configured: ${market} / ${assetId}`, 'success');
    updateBadges(market, `${atype}: ${assetId}`, 0, 0);
    $('#btn-fetch').disabled = false;
    $('#btn-filter').disabled = true;
    $('#btn-sentiment').disabled = true;
    startPriceRefresh(market, atype, assetId);
  } catch(e) { log('Error: ' + e.message); toast('Configuration failed: ' + e.message, 'error'); }
}

// ── Price + Chart ──

function startPriceRefresh(market, assetType, assetId) {
  PRICE_MARKET = market;
  PRICE_ASSET_TYPE = assetType;
  PRICE_ASSET_ID = assetId;
  PRICE_CHART_BUILT = false;
  stopPriceRefresh();
  fetchPrice(false);
  scheduleRefresh();
}

function stopPriceRefresh() {
  if (PRICE_TIMER) { clearInterval(PRICE_TIMER); PRICE_TIMER = null; }
}

function scheduleRefresh() {
  stopPriceRefresh();
  const secs = parseInt($('#sel-refresh-interval').value) || 0;
  PRICE_INTERVAL = secs;
  if (secs > 0 && PRICE_ASSET_ID) {
    PRICE_TIMER = setInterval(() => fetchPrice(true), secs * 1000);
  }
}

function changeRefreshInterval() {
  scheduleRefresh();
  const secs = parseInt($('#sel-refresh-interval').value) || 0;
  if (secs > 0) log(`Price auto-refresh: every ${secs}s`);
  else log('Price auto-refresh: off (manual)');
}

function refreshPriceNow() {
  if (!PRICE_ASSET_ID) return;
  fetchPrice(true);
  scheduleRefresh();
}

async function fetchPrice(priceOnly) {
  const card = $('#price-card');
  card.style.display = '';

  if (!priceOnly || !PRICE_CHART_BUILT) {
    $('#price-info').innerHTML = '<p class="price-loading"><span class="spinner"></span> Loading price data...</p>';
    $('#chart-container').innerHTML = '';
  }

  const params = `market=${encodeURIComponent(PRICE_MARKET)}&asset_type=${encodeURIComponent(PRICE_ASSET_TYPE)}&asset_id=${encodeURIComponent(PRICE_ASSET_ID)}`;

  try {
    if (priceOnly && PRICE_CHART_BUILT) {
      const resp = await fetch(`/api/price/live?${params}`);
      const data = await resp.json();
      if (data.error) {
        $('#price-updated').textContent = 'Refresh error ' + new Date().toLocaleTimeString();
        return;
      }
      updatePriceDisplay(data);
      $('#price-updated').textContent = 'Updated ' + new Date().toLocaleTimeString();
    } else {
      const resp = await fetch(`/api/price?${params}`);
      const data = await resp.json();

      if (data.error) {
        $('#price-info').innerHTML = `<p class="price-loading" style="color:var(--red)">Price data unavailable: ${escHtml(data.error)}</p>`;
        return;
      }

      updatePriceDisplay(data);
      $('#price-updated').textContent = 'Updated ' + new Date().toLocaleTimeString();
      buildChart(data.candles || []);
    }
  } catch(e) {
    if (priceOnly && PRICE_CHART_BUILT) {
      $('#price-updated').textContent = 'Refresh failed ' + new Date().toLocaleTimeString();
    } else {
      $('#price-info').innerHTML = `<p class="price-loading" style="color:var(--red)">Failed to load price: ${escHtml(e.message)}</p>`;
    }
  }
}

function updatePriceDisplay(data) {
  const isUp = data.change >= 0;
  const arrow = isUp ? '\u25B2' : '\u25BC';
  const sign = isUp ? '+' : '';
  $('#price-info').innerHTML = `
    <div class="price-header">
      <span class="price-big" id="price-value">${data.current_price}</span>
      <span class="price-change ${isUp ? 'up' : 'down'}">${arrow} ${sign}${data.change} (${sign}${data.change_pct}%)</span>
      <span class="price-currency">${data.currency}</span>
    </div>
  `;
  const el = $('#price-value');
  if (el) { el.style.transition = 'color 0.3s'; el.style.color = isUp ? '#22c55e' : '#ef4444';
    setTimeout(() => { el.style.color = ''; }, 800); }
}

function buildChart(candles) {
  if (!candles.length || typeof LightweightCharts === 'undefined') return;

  const container = $('#chart-container');
  if (priceChart) { priceChart.remove(); priceChart = null; }

  priceChart = LightweightCharts.createChart(container, {
    width: container.clientWidth,
    height: 380,
    layout: { background: { color: '#1a1d27' }, textColor: '#71717a' },
    grid: { vertLines: { color: '#2a2d3a' }, horzLines: { color: '#2a2d3a' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#2a2d3a' },
    timeScale: { borderColor: '#2a2d3a', timeVisible: false },
  });

  candleSeries = priceChart.addCandlestickSeries({
    upColor: '#22c55e', downColor: '#ef4444',
    borderUpColor: '#22c55e', borderDownColor: '#ef4444',
    wickUpColor: '#22c55e', wickDownColor: '#ef4444',
  });

  candleSeries.setData(candles);
  priceChart.timeScale().fitContent();

  const ro = new ResizeObserver(() => {
    if (priceChart) priceChart.applyOptions({ width: container.clientWidth });
  });
  ro.observe(container);
  PRICE_CHART_BUILT = true;
}

// ── File Browser (XSS-safe: uses data-path + addEventListener) ──
function openBrowser(inputId) {
  FB_TARGET_INPUT = inputId;
  const currentVal = document.getElementById(inputId).value.trim();
  $('#fb-modal').classList.add('active');
  loadBrowserDir(currentVal || '');
}

function closeBrowser() {
  $('#fb-modal').classList.remove('active');
  FB_TARGET_INPUT = null;
}

function selectBrowserDir() {
  if (FB_TARGET_INPUT && FB_CURRENT_PATH) {
    document.getElementById(FB_TARGET_INPUT).value = FB_CURRENT_PATH;
  }
  closeBrowser();
}

function _makeFbItem(icon, name, dirPath) {
  const el = document.createElement('div');
  el.className = 'fb-item';
  const iconSpan = document.createElement('span');
  iconSpan.className = 'fb-icon';
  iconSpan.textContent = icon;
  const nameSpan = document.createElement('span');
  nameSpan.className = 'fb-name';
  nameSpan.textContent = name;
  el.appendChild(iconSpan);
  el.appendChild(nameSpan);
  el.dataset.path = dirPath;
  el.addEventListener('click', function() { loadBrowserDir(this.dataset.path); });
  return el;
}

async function loadBrowserDir(path) {
  const body = $('#fb-body');
  body.innerHTML = '<div style="padding:20px;color:var(--muted)"><span class="spinner"></span> Loading...</div>';

  try {
    const url = '/api/browse' + (path ? `?path=${encodeURIComponent(path)}` : '');
    const resp = await fetch(url);
    const data = await resp.json();

    FB_CURRENT_PATH = data.current || '';
    $('#fb-path').textContent = FB_CURRENT_PATH;

    if (data.error) {
      body.innerHTML = `<div style="padding:20px;color:var(--red)">${escHtml(data.error)}</div>`;
      return;
    }

    const frag = document.createDocumentFragment();

    if (data.parent) {
      frag.appendChild(_makeFbItem('\uD83D\uDCC1', '..', data.parent));
    }

    for (const e of data.entries) {
      if (e.is_dir) {
        const el = _makeFbItem('\uD83D\uDCC2', e.name, e.path);
        if (e.is_model) {
          const tag = document.createElement('span');
          tag.className = 'fb-tag fb-tag-model';
          tag.textContent = 'Model';
          el.appendChild(tag);
        }
        if (e.is_adapter) {
          const tag = document.createElement('span');
          tag.className = 'fb-tag fb-tag-adapter';
          tag.textContent = 'Adapter';
          el.appendChild(tag);
        }
        frag.appendChild(el);
      } else {
        const el = document.createElement('div');
        el.className = 'fb-item';
        el.style.opacity = '0.5';
        el.style.cursor = 'default';
        const fi = document.createElement('span');
        fi.className = 'fb-icon';
        fi.textContent = '\uD83D\uDCC4';
        const fn = document.createElement('span');
        fn.className = 'fb-name';
        fn.textContent = e.name;
        el.appendChild(fi);
        el.appendChild(fn);
        frag.appendChild(el);
      }
    }

    if (!data.entries || data.entries.length === 0) {
      const empty = document.createElement('div');
      empty.style.cssText = 'padding:20px;color:var(--muted);text-align:center';
      empty.textContent = 'Empty directory';
      frag.appendChild(empty);
    }

    body.innerHTML = '';
    body.appendChild(frag);
  } catch(e) {
    body.innerHTML = `<div style="padding:20px;color:var(--red)">Error: ${escHtml(e.message)}</div>`;
  }
}

// ── apply model settings ──
async function applyModels() {
  try {
    const fb = $('#inp-filter-base').value.trim();
    const fa = $('#inp-filter-adapter').value.trim() || null;
    const sb = $('#inp-sent-base').value.trim();
    const sa = $('#inp-sent-adapter').value.trim() || null;
    if (!fb || !sb) { toast('Enter base model paths first.', 'error'); return; }
    await post('/filter_model', { base_model_path: fb, adapter_path: fa });
    await post('/sentiment_model', { base_model_path: sb, adapter_path: sa });
    log(`Filter model: ${fb.split(/[/\\]/).pop()}` + (fa ? ` + ${fa.split(/[/\\]/).pop()}` : ' (base only)'));
    log(`Sentiment model: ${sb.split(/[/\\]/).pop()}` + (sa ? ` + ${sa.split(/[/\\]/).pop()}` : ' (base only)'));
    toast('Model settings applied successfully', 'success');
  } catch(e) { log('Error: ' + e.message); toast('Failed to apply models: ' + e.message, 'error'); }
}

// ── fetch (background) ──
async function fetchNews() {
  setBusy('#btn-fetch', true);
  setBusy('#btn-filter', true);
  setBusy('#btn-sentiment', true);
  log('Fetching news (background)...');
  try {
    await post('/fetch');
    startPolling('fetch');
  } catch(e) {
    log('Error: ' + e.message);
    toast('Fetch failed: ' + e.message, 'error');
    setBusy('#btn-fetch', false);
    setBusy('#btn-filter', false);
    setBusy('#btn-sentiment', false);
  }
}

// ── async filter ──
async function startFilter() {
  setBusy('#btn-filter', true);
  setBusy('#btn-sentiment', true);
  setBusy('#btn-fetch', true);
  log('Starting filter (background)...');
  try {
    await post('/filter');
    startPolling('filter');
  } catch(e) {
    log('Error: ' + e.message);
    setBusy('#btn-filter', false);
    setBusy('#btn-sentiment', false);
    setBusy('#btn-fetch', false);
  }
}

// ── async sentiment ──
async function startSentiment() {
  setBusy('#btn-sentiment', true);
  setBusy('#btn-filter', true);
  setBusy('#btn-fetch', true);
  log('Starting sentiment analysis (background)...');
  try {
    await post('/sentiment');
    startPolling('sentiment');
  } catch(e) {
    log('Error: ' + e.message);
    setBusy('#btn-sentiment', false);
    setBusy('#btn-filter', false);
    setBusy('#btn-fetch', false);
  }
}

// ── poll for task completion ──
function startPolling(taskType) {
  if (POLL_TIMER) clearInterval(POLL_TIMER);
  let lastMsg = '';
  POLL_TIMER = setInterval(async () => {
    try {
      const t = await get('/task');
      if (t.message && t.message !== lastMsg) {
        log(t.message);
        lastMsg = t.message;
      }
      if (t.status === 'done') {
        clearInterval(POLL_TIMER); POLL_TIMER = null;
        if (taskType === 'fetch') {
          const cnt = t.result ? t.result.length : 0;
          log(`Fetched ${cnt} articles`);
          toast(`Fetched ${cnt} articles`, 'success');
          updateBadges(null, null, cnt, 0);
          renderArticles(t.result || [], false);
          setBusy('#btn-fetch', false);
          setBusy('#btn-filter', false);
          setBusy('#btn-sentiment', false);
          $('#btn-filter').disabled = false;
          $('#btn-sentiment').disabled = false;
        } else if (taskType === 'filter') {
          const cnt = t.result ? t.result.length : 0;
          updateBadges(null, null, null, cnt);
          renderArticles(t.result || [], false);
          toast(`Filter complete: ${cnt} relevant articles`, 'success');
          setBusy('#btn-filter', false);
          setBusy('#btn-sentiment', false);
          setBusy('#btn-fetch', false);
        } else {
          renderArticles(t.result || [], true);
          toast('Sentiment analysis complete', 'success');
          setBusy('#btn-sentiment', false);
          setBusy('#btn-filter', false);
          setBusy('#btn-fetch', false);
        }
      } else if (t.status === 'error') {
        clearInterval(POLL_TIMER); POLL_TIMER = null;
        log('Error: ' + (t.error || 'Unknown error'));
        toast(taskType + ' failed: ' + (t.error || 'Unknown error'), 'error');
        setBusy('#btn-filter', false);
        setBusy('#btn-sentiment', false);
        setBusy('#btn-fetch', false);
      }
    } catch(e) {
      // Network blip — keep polling
    }
  }, 1500);
}

// ── rendering ──
function updateBadges(market, asset, articles, filtered) {
  if (market !== null) { $('#badge-market').textContent = market; $('#badge-market').classList.add('active'); }
  if (asset  !== null) { $('#badge-asset').textContent  = asset;  $('#badge-asset').classList.add('active'); }
  if (articles !== null) $('#badge-articles').textContent = articles + ' articles';
  if (filtered !== null) $('#badge-filtered').textContent = filtered + ' filtered';
}

function renderArticles(articles, showSentiment) {
  ALL_ARTICLES = articles || [];
  SHOW_SENTIMENT = showSentiment;
  CURRENT_PAGE = 0;
  SEARCH_QUERY = '';
  if ($('#inp-search')) $('#inp-search').value = '';

  const exportBtn = $('#btn-export');
  if (exportBtn) exportBtn.style.display = showSentiment ? '' : 'none';
  if (showSentiment) showDashboard(articles);
  else $('#dashboard-card').style.display = 'none';
  const sb = $('#search-bar');
  if (sb) sb.style.display = ALL_ARTICLES.length > 0 ? '' : 'none';

  renderPage();
}

function getFilteredArticles() {
  if (!SEARCH_QUERY) return ALL_ARTICLES;
  const q = SEARCH_QUERY.toLowerCase();
  return ALL_ARTICLES.filter(a => {
    const title = (a.title || a.headline || '').toLowerCase();
    const src = (a.source || '').toLowerCase();
    return title.includes(q) || src.includes(q);
  });
}

let _searchTimer = null;
function onSearchInput(val) {
  clearTimeout(_searchTimer);
  _searchTimer = setTimeout(() => {
    SEARCH_QUERY = val.trim();
    CURRENT_PAGE = 0;
    renderPage();
  }, 150);
}

function renderPage() {
  const area = $('#results-area');
  const filtered = getFilteredArticles();

  if (!filtered.length) {
    area.innerHTML = SEARCH_QUERY
      ? '<p class="empty">No articles match your search.</p>'
      : '<p class="empty">No articles.</p>';
    $('#pagination').style.display = 'none';
    return;
  }

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const start = CURRENT_PAGE * PAGE_SIZE;
  const page = filtered.slice(start, start + PAGE_SIZE);

  let html = '<table class="tbl"><thead><tr><th>Date</th><th>Source</th><th>Headline</th>';
  if (SHOW_SENTIMENT) html += '<th>Sentiment</th>';
  html += '</tr></thead><tbody>';
  for (const a of page) {
    const date = (a.date || a.datetime || '').slice(0, 10);
    const src = (a.source || '').slice(0, 25);
    const title = a.title || a.headline || '';
    html += `<tr><td style="white-space:nowrap">${escHtml(date)}</td><td>${escHtml(src)}</td><td>${escHtml(title)}</td>`;
    if (SHOW_SENTIMENT) {
      const s = (a.sentiment || 'neutral').toLowerCase();
      html += `<td><span class="sent sent-${s}">${s}</span></td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';
  area.innerHTML = html;

  const pag = $('#pagination');
  if (totalPages <= 1) { pag.style.display = 'none'; return; }
  pag.style.display = '';
  pag.innerHTML = '';
  const pagBtns = [
    { label: '\u00AB', page: 0, disabled: CURRENT_PAGE === 0 },
    { label: '\u2039', page: CURRENT_PAGE - 1, disabled: CURRENT_PAGE === 0 },
    { label: null, text: `Page ${CURRENT_PAGE+1} of ${totalPages} (${filtered.length} items)` },
    { label: '\u203A', page: CURRENT_PAGE + 1, disabled: CURRENT_PAGE >= totalPages - 1 },
    { label: '\u00BB', page: totalPages - 1, disabled: CURRENT_PAGE >= totalPages - 1 },
  ];
  for (const pb of pagBtns) {
    if (pb.label === null) {
      const sp = document.createElement('span');
      sp.textContent = pb.text;
      pag.appendChild(sp);
    } else {
      const btn = document.createElement('button');
      btn.textContent = pb.label;
      btn.disabled = pb.disabled;
      const target = pb.page;
      btn.addEventListener('click', () => goPage(target));
      pag.appendChild(btn);
    }
  }
}

function goPage(p) {
  const filtered = getFilteredArticles();
  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  CURRENT_PAGE = Math.max(0, Math.min(p, totalPages - 1));
  renderPage();
}

// ── Onboarding Wizard ──
const WIZARD_STEPS = [
  {
    icon: '\uD83D\uDC4B',
    title: 'Welcome to Sentiment Analyzer',
    body: 'This tool scrapes financial news, filters relevant articles using AI, and classifies their sentiment \u2014 all running locally on your machine.',
  },
  {
    icon: '\uD83E\uDD16',
    title: 'Set Up Your Models',
    body: 'The pipeline uses a local LLM for filtering and sentiment analysis. Point the model paths to your Qwen3-8B base model, and optionally a LoRA adapter for fine-tuned sentiment.',
    check: () => !!$('#inp-filter-base').value && !!$('#inp-sent-base').value,
  },
  {
    icon: '\uD83D\uDCC8',
    title: 'Choose Market & Asset',
    body: 'Select a market (US or China), an asset type (commodity or stock), and the specific asset you want to analyze. Then click <b>Set</b> to configure.',
  },
  {
    icon: '\u2705',
    title: "You're All Set!",
    body: 'Click <b>Fetch News</b> to scrape headlines, <b>Filter</b> to find relevant ones, and <b>Analyze Sentiment</b> to classify them. Results are saved automatically.',
  },
];

let WIZARD_CURRENT = 0;

function showWizard() {
  WIZARD_CURRENT = 0;
  $('#wizard-overlay').classList.add('active');
  renderWizardStep();
}

function closeWizard() {
  $('#wizard-overlay').classList.remove('active');
  fetch('/api/market', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({market: 'US'})}).catch(()=>{});
}

function renderWizardStep() {
  const s = WIZARD_STEPS[WIZARD_CURRENT];
  const total = WIZARD_STEPS.length;

  let dots = '';
  for (let i = 0; i < total; i++) {
    let cls = 'step-dot';
    if (i < WIZARD_CURRENT) cls += ' done';
    else if (i === WIZARD_CURRENT) cls += ' active';
    dots += `<div class="${cls}"></div>`;
  }
  $('#wizard-steps').innerHTML = dots;

  $('#wizard-body').innerHTML = `
    <div class="wizard-icon">${s.icon}</div>
    <h2>${s.title}</h2>
    <p>${s.body}</p>
  `;

  const btnsEl = $('#wizard-btns');
  btnsEl.innerHTML = '';
  if (WIZARD_CURRENT > 0) {
    const back = document.createElement('button');
    back.className = 'btn btn-sm';
    back.style.cssText = 'background:var(--border);color:var(--text)';
    back.textContent = 'Back';
    back.addEventListener('click', wizardPrev);
    btnsEl.appendChild(back);
  }
  if (WIZARD_CURRENT < total - 1) {
    const next = document.createElement('button');
    next.className = 'btn btn-sm btn-primary';
    next.textContent = 'Next';
    next.addEventListener('click', wizardNext);
    btnsEl.appendChild(next);
  } else {
    const start = document.createElement('button');
    start.className = 'btn btn-sm btn-green';
    start.textContent = 'Get Started';
    start.addEventListener('click', closeWizard);
    btnsEl.appendChild(start);
  }
  const skip = document.createElement('button');
  skip.className = 'btn btn-sm';
  skip.style.cssText = 'background:transparent;color:var(--muted);font-size:0.75rem';
  skip.textContent = 'Skip';
  skip.addEventListener('click', closeWizard);
  btnsEl.appendChild(skip);
}

function wizardNext() {
  const step = WIZARD_STEPS[WIZARD_CURRENT];
  if (step.check && !step.check()) {
    toast('Please complete this step before continuing.', 'error');
    return;
  }
  if (WIZARD_CURRENT < WIZARD_STEPS.length - 1) { WIZARD_CURRENT++; renderWizardStep(); }
}
function wizardPrev() { if (WIZARD_CURRENT > 0) { WIZARD_CURRENT--; renderWizardStep(); } }

// ── Sentiment Dashboard ──
let sentimentChart = null;

function showDashboard(articles) {
  if (!articles || !articles.length) {
    $('#dashboard-card').style.display = 'none';
    return;
  }
  $('#dashboard-card').style.display = '';

  const pos = articles.filter(a => (a.sentiment||'').toLowerCase() === 'positive').length;
  const neg = articles.filter(a => (a.sentiment||'').toLowerCase() === 'negative').length;
  const neu = articles.filter(a => (a.sentiment||'').toLowerCase() === 'neutral').length;
  const total = articles.length;

  $('#dashboard-stats').innerHTML = `
    <div class="stat-card">
      <div class="stat-value" style="color:var(--green)">${pos}</div>
      <div class="stat-label">Positive</div>
    </div>
    <div class="stat-card">
      <div class="stat-value" style="color:var(--red)">${neg}</div>
      <div class="stat-label">Negative</div>
    </div>
    <div class="stat-card">
      <div class="stat-value" style="color:var(--yellow)">${neu}</div>
      <div class="stat-label">Neutral</div>
    </div>
  `;

  const pPct = total ? (pos/total*100).toFixed(1) : 0;
  const nPct = total ? (neg/total*100).toFixed(1) : 0;
  const uPct = total ? (neu/total*100).toFixed(1) : 0;
  $('#sentiment-bar').innerHTML = `
    <div class="seg seg-pos" style="width:${pPct}%" title="Positive ${pPct}%"></div>
    <div class="seg seg-neg" style="width:${nPct}%" title="Negative ${nPct}%"></div>
    <div class="seg seg-neu" style="width:${uPct}%" title="Neutral ${uPct}%"></div>
  `;
  $('#sentiment-labels').innerHTML = `
    <span style="color:var(--green)">\u25CF Positive ${pPct}%</span>
    <span style="color:var(--red)">\u25CF Negative ${nPct}%</span>
    <span style="color:var(--yellow)">\u25CF Neutral ${uPct}%</span>
  `;

  loadSentimentTrend();
}

async function loadSentimentTrend() {
  try {
    const resp = await fetch('/api/sentiment_history?days=30');
    const data = await resp.json();
    const history = data.data || [];
    if (!history.length) {
      $('#sentiment-chart-container').innerHTML = '<p style="color:var(--muted);font-size:0.8rem;padding:12px">Not enough historical data for trend chart. Run more analyses to build history.</p>';
      return;
    }

    const container = $('#sentiment-chart-container');
    if (sentimentChart) { sentimentChart.remove(); sentimentChart = null; }
    if (typeof LightweightCharts === 'undefined') return;

    sentimentChart = LightweightCharts.createChart(container, {
      width: container.clientWidth,
      height: 200,
      layout: { background: { color: '#1a1d27' }, textColor: '#71717a' },
      grid: { vertLines: { color: '#2a2d3a' }, horzLines: { color: '#2a2d3a' } },
      rightPriceScale: { borderColor: '#2a2d3a' },
      timeScale: { borderColor: '#2a2d3a', timeVisible: false },
    });

    const posSeries = sentimentChart.addLineSeries({ color: '#22c55e', lineWidth: 2, title: 'Positive' });
    const negSeries = sentimentChart.addLineSeries({ color: '#ef4444', lineWidth: 2, title: 'Negative' });

    posSeries.setData(history.map(d => ({ time: d.date, value: d.positive || 0 })));
    negSeries.setData(history.map(d => ({ time: d.date, value: d.negative || 0 })));

    sentimentChart.timeScale().fitContent();
    new ResizeObserver(() => {
      if (sentimentChart) sentimentChart.applyOptions({ width: container.clientWidth });
    }).observe(container);
  } catch(e) {
    $('#sentiment-chart-container').innerHTML = '';
  }
}

// ── Export & History ──
function exportCSV() {
  window.open('/api/export/csv', '_blank');
}

let HISTORY_VISIBLE = false;
async function toggleHistory() {
  HISTORY_VISIBLE = !HISTORY_VISIBLE;
  const card = $('#history-card');
  if (!HISTORY_VISIBLE) { card.style.display = 'none'; return; }
  card.style.display = '';
  const area = $('#history-area');
  area.innerHTML = '<p class="empty"><span class="spinner"></span> Loading...</p>';
  try {
    const resp = await fetch('/api/history');
    const data = await resp.json();
    const sessions = data.sessions || [];
    if (!sessions.length) {
      area.innerHTML = '<p class="empty">No analysis history yet.</p>';
      return;
    }
    let html = '<table class="tbl"><thead><tr><th>Date</th><th>Market</th><th>Asset</th><th>Articles</th><th style="color:var(--green)">+</th><th style="color:var(--red)">-</th><th style="color:var(--yellow)">~</th></tr></thead><tbody>';
    for (const s of sessions) {
      const dt = (s.created_at || '').slice(0, 16).replace('T', ' ');
      html += `<tr>
        <td style="white-space:nowrap;font-size:0.8rem">${escHtml(dt)}</td>
        <td>${escHtml(s.market)}</td>
        <td>${escHtml(s.asset_type)}: ${escHtml(s.asset_id)}</td>
        <td>${s.article_count}</td>
        <td style="color:var(--green)">${s.positive_count}</td>
        <td style="color:var(--red)">${s.negative_count}</td>
        <td style="color:var(--yellow)">${s.neutral_count}</td>
      </tr>`;
    }
    html += '</tbody></table>';
    area.innerHTML = html;
  } catch(e) {
    area.innerHTML = `<p class="empty" style="color:var(--red)">Error: ${escHtml(e.message)}</p>`;
  }
}

// ── Keyboard shortcuts ──
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;

  if (e.ctrlKey && e.key === 'Enter') {
    e.preventDefault();
    const fetchBtn = $('#btn-fetch'), filter = $('#btn-filter'), sent = $('#btn-sentiment');
    if (!sent.disabled) startSentiment();
    else if (!filter.disabled) startFilter();
    else if (!fetchBtn.disabled) fetchNews();
  }
  else if (e.ctrlKey && e.key === 'e') {
    e.preventDefault();
    const btn = $('#btn-export');
    if (btn && btn.style.display !== 'none') exportCSV();
  }
  else if (e.key === 'Escape') {
    if ($('#fb-modal').classList.contains('active')) closeBrowser();
    else if ($('#wizard-overlay').classList.contains('active')) closeWizard();
  }
});

// ── Re-fetch confirmation ──
let HAS_RESULTS = false;
const _origFetchNews = fetchNews;
fetchNews = async function() {
  if (HAS_RESULTS) {
    if (!confirm('You have existing results. Fetching again will replace them. Continue?')) return;
  }
  await _origFetchNews();
  HAS_RESULTS = true;
};

init();
