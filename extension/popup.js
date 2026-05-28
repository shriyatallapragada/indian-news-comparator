// ── Utilities ──────────────────────────────────────────────────────────────
function escHtml(str) {
  if (!str) return "";
  return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function escAttr(str) {
  if (!str) return "";
  return String(str).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function getDomain(url) {
  try { return new URL(url).hostname.replace(/^www\./, ""); } catch (_) { return url; }
}

function formatBiasScore(score) {
  const n = parseFloat(score);
  if (Number.isNaN(n)) return "…";
  return (n >= 0 ? "+" : "") + n.toFixed(1);
}

function sendRuntimeMessage(message) {
  return new Promise(function (resolve) {
    chrome.runtime.sendMessage(message, function (response) {
      if (chrome.runtime.lastError) {
        resolve({ ok: false, error: chrome.runtime.lastError.message });
        return;
      }
      resolve(response);
    });
  });
}

// Send a runtime message but fail fast after `timeoutMs` milliseconds.
function sendWithTimeout(message, timeoutMs = 13000) {
  return Promise.race([
    sendRuntimeMessage(message),
    new Promise(resolve => setTimeout(() => resolve({ ok: false, error: 'Request timed out' }), timeoutMs))
  ]);
}

function buildNewsSearchParams(tab, entities, coreSlug) {
  var title = (tab && tab.title ? tab.title : "news").replace(/\s*[-|]\s*[^-|]+$/, "").trim();
  var params = new URLSearchParams({ q: title });
  if (entities && entities.length) params.set("keywords", entities.join(","));
  if (coreSlug) params.set("source_event", coreSlug);
  return params.toString();
}

function searchNewsAndRender(biasData, tab, entities, coreSlug) {
  const params = buildNewsSearchParams(tab, entities, coreSlug);
  setLoading("Searching news sources…");

  return sendWithTimeout({ action: "news", params: params })
    .then(newsRes => {
      if (!newsRes || !newsRes.ok) {
        const msg = (newsRes && newsRes.error) ? newsRes.error : "News search failed";
        console.warn("News search failed:", msg);
        renderResults(biasData, null, tab);
        return;
      }
      renderResults(biasData, newsRes.data, tab);
    })
    .catch(err => {
      console.error("News search error:", err);
      renderResults(biasData, null, tab);
    });
}

// ── Navigation State ───────────────────────────────────────────────────────
const startScreen = document.getElementById("start-screen");
const resultScreen = document.getElementById("result");
const aboutScreen = document.getElementById("about-screen");

// Holds the article text currently being analysed so tab clicks can re-use it.
var _currentArticleText = "";
// Holds the real bias score from the engine so the bar can be updated
var _engineBiasScore = null;
// Holds the already-rendered related articles so comparison can degrade gracefully.
var _currentBiasData = null;
var _currentNewsData = null;

document.getElementById("btnStart").addEventListener("click", getNews);
document.getElementById("btnWhatIs").addEventListener("click", toggleAbout);

function toggleAbout() {
  if (aboutScreen.style.display === "block") {
    aboutScreen.style.display = "none";
    if (resultScreen.innerHTML.trim() !== "") {
      resultScreen.style.display = "block";
    } else {
      startScreen.style.display = "flex";
    }
  } else {
    startScreen.style.display = "none";
    resultScreen.style.display = "none";
    aboutScreen.style.display = "block";
  }
}

// ── Tab switching via event delegation ─────────────────────────────────────
document.getElementById("result").addEventListener("click", function (e) {
  const tab = e.target.closest(".perspective-tab");
  if (tab) {
    const side = tab.getAttribute("data-tab");
    if (side) switchTab(side);
  }
  const card = e.target.closest(".article-link-card[data-url]");
  if (card) {
    const url = card.getAttribute("data-url");
    if (url) window.open(url, "_blank");
  }
});

// ── Loading helper ─────────────────────────────────────────────────────────
function setLoading(msg) {
  startScreen.style.display = "none";
  aboutScreen.style.display = "none";
  resultScreen.style.display = "block";
  resultScreen.innerHTML =
    '<div class="loader-wrap">' +
      '<div class="spinner"></div>' +
      '<span class="loader-msg">' + escHtml(msg) + '</span>' +
    '</div>';
}

// ── Main flow ──────────────────────────────────────────────────────────────
function getNews() {
  setLoading("Extracting article text…");

  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    const tab = tabs[0];

    if (!tab || !tab.url || tab.url.startsWith("chrome://") || tab.url.startsWith("edge://") || tab.url.startsWith("about:")) {
      resultScreen.innerHTML = '<div class="error-msg">Navigate to a news article first.</div>';
      return;
    }

    chrome.scripting.executeScript(
      { target: { tabId: tab.id }, files: ["content.js"] },
      function () {
        if (chrome.runtime.lastError) {
          resultScreen.innerHTML = '<div class="error-msg">Could not inject script.<br><small>' + escHtml(chrome.runtime.lastError.message) + '</small></div>';
          return;
        }
        chrome.tabs.sendMessage(tab.id, { action: "extractText" }, function (response) {
          const articleText = response && response.text ? response.text : "";

          if (!articleText) {
            resultScreen.innerHTML = '<div class="error-msg">Could not extract article text from this page.</div>';
            return;
          }

          setLoading("Analysing bias with AI…");
          _currentArticleText = articleText;
          chrome.runtime.sendMessage({ action: "analyze", text: articleText }, function (res) {
            if (chrome.runtime.lastError || !res || !res.ok) {
              const msg = (res && res.error) || (chrome.runtime.lastError && chrome.runtime.lastError.message) || "Unknown error";
              resultScreen.innerHTML = '<div class="error-msg">Error: ' + escHtml(msg) + '<br><small>Is the backend running?</small></div>';
              return;
            }

            const biasData = res.data;
            if (biasData.error) {
              resultScreen.innerHTML = '<div class="error-msg">Analysis error: ' + escHtml(biasData.error) + '</div>';
              return;
            }

            const entities = biasData.named_entities || [];
            const summary  = biasData.article_summary || "";
            const coreSlug = biasData.core_event_slug || "";

            setLoading("Finding related perspectives…");

            // Use the Promise-based helper and race with a timeout so the UI
            // doesn't hang if the background service worker or backend stalls.
            const relatedPromise = sendRuntimeMessage({
              action: "related",
              payload: {
                summary: summary,
                named_entities: entities,
                published_at: "",
                allow_live_fetch: false
              }
            });

            const timed = Promise.race([
              relatedPromise,
              new Promise(resolve => setTimeout(() => resolve({ ok: false, error: 'Related lookup timed out' }), 13000))
            ]);

            timed.then(relRes => {
              const vectorResults = relRes && relRes.ok ? relRes.data : null;
              const hasVector = vectorResults && (vectorResults.left || vectorResults.center || vectorResults.right);

              if (hasVector) {
                renderResults(biasData, vectorResults, tab);
                return;
              }

              searchNewsAndRender(biasData, tab, entities, coreSlug);
            }).catch(err => {
              // Fallback to news search on any unexpected error
              console.error('Related lookup error:', err);
              searchNewsAndRender(biasData, tab, entities, coreSlug);
            });

            ingestOpenedArticle(articleText, biasData, tab);
          });
        });
      }
    );
  });
}

function ingestOpenedArticle(articleText, biasData, tab) {
  if (!articleText || !biasData) return;

  const title = tab ? (tab.title || "").replace(/\s*[-|]\s*[^-|]+$/, "").trim() : "";
  const url = tab && tab.url ? tab.url : "";

  sendRuntimeMessage({
    action: "ingest",
    payload: {
      summary: articleText,
      title: title,
      url: url,
      source: getDomain(url),
      published_at: "",
      bias: biasData.bias_classification || "Center",
      named_entities: biasData.named_entities || [],
      core_event_slug: biasData.core_event_slug || "",
    },
  }).then(function (res) {
    if (!res || !res.ok) {
      console.warn("Article ingest failed:", res && res.error);
    }
  });
}

// ── Render full UI ─────────────────────────────────────────────────────────
function renderResults(biasData, newsData, tab) {
  try {
    renderResultsInner(biasData, newsData, tab);
  } catch (err) {
    console.error("renderResults failed:", err);
    resultScreen.innerHTML =
      '<div class="error-msg">Could not display results.<br><small>' +
      escHtml(err.message || String(err)) +
      "</small></div>";
  }
}

function renderResultsInner(biasData, newsData, tab) {
  _currentBiasData = biasData || null;
  _currentNewsData = newsData || null;

  const biasRaw   = (biasData.bias_classification || "Center");
  const bias      = biasRaw.toLowerCase();
  const summary   = biasData.article_summary || "Summary not available.";
  const target    = biasData.step_1_target_analysis || "Target entities not identified.";
  const reasoning = biasData.step_2_alignment_logic || "Reasoning not provided.";
  const tabTitle  = tab ? (tab.title || "Article Analysis").replace(/\s*[-|]\s*[^-|]+$/, "").trim() : "Article Analysis";

  const initialScore = (biasData && biasData.bias_score !== undefined && biasData.bias_score !== null)
    ? parseFloat(biasData.bias_score)
    : null;
  const percentage = initialScore !== null
    ? Math.min(100, Math.max(0, ((initialScore + 5) / 10) * 100))
    : 50;
  const scoreLabel = initialScore !== null
    ? formatBiasScore(initialScore)
    : "…";

  // ── 1. Original Article Analysis ─────────────────────────────────────
  let html = `<div class="card">
    <div class="card-header">📄 ORIGINAL ARTICLE ANALYSIS</div>
    <div class="article-title">${escHtml(tabTitle)}</div>
    <div class="analysis-text"><span class="analysis-label">Summary:</span> ${escHtml(summary)}</div>
    <div class="analysis-text"><span class="analysis-label">Target:</span> ${escHtml(target)}</div>
    <div class="analysis-text"><span class="analysis-label">Reasoning:</span> ${escHtml(reasoning)}</div>
  </div>`;

  // ── 2. Detected Article Bias (Slider) ────────────────────────────────
  html += `<div class="card">
    <div class="card-header">🧭 DETECTED ARTICLE BIAS</div>
    <div class="bias-bar-container">
      <div class="bias-score-bubble" id="bias-bubble" style="left: ${percentage}%;">${scoreLabel}</div>
      <div class="bias-bar-track"></div>
      <div class="bias-bar-marker" id="bias-marker" style="left: ${percentage}%;"></div>
    </div>
    <div class="bias-bar-labels">
      <span class="lbl-left">-5 Left</span>
      <span class="lbl-center">0 Center</span>
      <span class="lbl-right">+5 Right</span>
    </div>
    <div class="lean-text" id="lean-text">Lean Score: ${scoreLabel} (${biasRaw})</div>
  </div>`;

  // ── 3. Perspectives & Related Articles ───────────────────────────────
  const left   = newsData && newsData.left   ? newsData.left   : null;
  const center = newsData && newsData.center ? newsData.center : null;
  const right  = newsData && newsData.right  ? newsData.right  : null;

  html += `<div class="card" style="padding: 0; background: transparent; border: none; box-shadow: none;">
    <div class="perspective-tabs">
      <button class="perspective-tab active-center" data-tab="left">LEFT</button>
      <button class="perspective-tab active-center" style="background: #3b2c15; color: #f39c12;" data-tab="center">CENTER</button>
      <button class="perspective-tab active-center" data-tab="right">RIGHT</button>
    </div>

    <div class="card perspective-panel" id="panel-left">
      <div class="card-header" style="color: #e74c3c;">LEFT PERSPECTIVE</div>
      ${left ? `<div class="perspective-summary">${escHtml(left.summary || left.description || "Perspective summary...")}</div>${buildArticleLinkCard(left)}` : `<div style="color:#9aa0a6; font-size: 12px;">No left-leaning coverage found.</div>`}
    </div>

    <div class="card perspective-panel active" id="panel-center">
      <div class="card-header" style="color: #f39c12;">CENTER PERSPECTIVE</div>
      ${center ? `<div class="perspective-summary">${escHtml(center.summary || center.description || "Perspective summary...")}</div>${buildArticleLinkCard(center)}` : `<div style="color:#9aa0a6; font-size: 12px;">No center coverage found.</div>`}
    </div>

    <div class="card perspective-panel" id="panel-right">
      <div class="card-header" style="color: #3498db;">RIGHT PERSPECTIVE</div>
      ${right ? `<div class="perspective-summary">${escHtml(right.summary || right.description || "Perspective summary...")}</div>${buildArticleLinkCard(right)}` : `<div style="color:#9aa0a6; font-size: 12px;">No right-leaning coverage found.</div>`}
    </div>
  </div>`;

  // ── 4. Cross-Article Topic Comparison ────────────────────────────────
  html += `<div class="card">
    <div class="card-header">🔍 CROSS-ARTICLE TOPIC COMPARISON</div>
    <div class="analysis-text" style="margin-bottom: 8px;">What other sources focus on (Context missing from this article):</div>
    <ul id="comparison-list" class="comparison-list">
      <li style="color:#9aa0a6;">⏳ Loading context analysis…</li>
    </ul>
  </div>`;

  resultScreen.innerHTML = html;

  // Reset engine cache for new article
  _engineCache = { Left: null, Center: null, Right: null };

  // Clear any stale comparison content immediately, then fetch fresh
  var listEl = document.getElementById("comparison-list");
  if (listEl) listEl.innerHTML = '<li style="color:#9aa0a6;">⏳ Loading context analysis…</li>';

  // Auto-fetch engine data for the default active tab (center)
  fetchPerspectiveData(_currentArticleText, "Center");
}

// ── Tab switcher ───────────────────────────────────────────────────────────
function switchTab(side) {
  const tabs = {
    left: { bg: "#3d2222", col: "#e74c3c" },
    center: { bg: "#3b2c15", col: "#f39c12" },
    right: { bg: "#1a2c3f", col: "#3498db" }
  };

  ["left", "center", "right"].forEach(function (s) {
    var tabEl   = document.querySelector(`.perspective-tab[data-tab="${s}"]`);
    var panelEl = document.getElementById("panel-" + s);
    if (!tabEl || !panelEl) return;
    
    if (s === side) {
      tabEl.style.background = tabs[s].bg;
      tabEl.style.color = tabs[s].col;
      panelEl.className = "card perspective-panel active";
    } else {
      tabEl.style.background = "transparent";
      tabEl.style.color = "#9aa0a6";
      panelEl.className = "card perspective-panel";
    }
  });

  // Fetch engine perspective for the selected tab
  fetchPerspectiveData(_currentArticleText, capitalise(side));
}

// ── Article link card ──────────────────────────────────────────────────────
function buildArticleLinkCard(article) {
  if (!article) return "";
  // source can be a string (vector_store) or {name: "..."} (NewsAPI)
  var source = (article.source && typeof article.source === "object")
    ? (article.source.name || getDomain(article.url || ""))
    : (article.source || getDomain(article.url || ""));
  var title  = article.title  || "Read full perspective article";
  var url    = article.url    || "";
  if (!url || url === "#") return "";
  return `<div class="article-link-card" data-url="${escAttr(url)}">
    <div class="alc-source"><span>${escHtml(source)}</span> <span>↗</span></div>
    <div class="alc-title">${escHtml(title)}</div>
  </div>`;
}

// Cache engine results per tab so switching doesn't re-fetch
var _engineCache = { Left: null, Center: null, Right: null };

async function fetchPerspectiveData(userText, targetLean) {
  const comparisonList = document.getElementById("comparison-list");
  const selectedArticle = getSelectedComparisonArticle(targetLean);

  if (!selectedArticle) {
    renderFallbackComparison(targetLean);
    return;
  }

  // If we already have a result for this tab, just re-render it
  if (_engineCache[targetLean]) {
    updateUI(_engineCache[targetLean], targetLean);
    return;
  }

  if (comparisonList) {
    comparisonList.innerHTML = '<li style="color:#9aa0a6;">⏳ Loading context analysis…</li>';
  }

  try {
    const response = await sendRuntimeMessage({
      action: "perspective",
      payload: buildPerspectivePayload(userText, targetLean, selectedArticle),
    });

    if (!response || !response.ok) {
      throw new Error((response && response.error) || "Engine unavailable");
    }

    const data = response.data;
    console.log("Engine response:", data);

    // Cache and render
    _engineCache[targetLean] = data;
    updateUI(data, targetLean);

  } catch (err) {
    console.error("Engine fetch error:", err);
    renderFallbackComparison(targetLean, err.message);
  }
}

function getSelectedComparisonArticle(targetLean) {
  if (!_currentNewsData) return null;
  return _currentNewsData[targetLean.toLowerCase()] || null;
}

function articleComparisonText(article) {
  if (!article) return "";
  const parts = [article.title, article.summary || article.description || article.content]
    .filter(Boolean);
  return parts.filter((part, index) => parts.indexOf(part) === index).join(". ");
}

function articleSourceName(article) {
  if (!article) return "";
  if (article.source && typeof article.source === "object") {
    return article.source.name || "";
  }
  return article.source || "";
}

function buildPerspectivePayload(userText, targetLean, article) {
  const text = articleComparisonText(article);
  return {
    user_text: userText,
    target_lean: targetLean,
    alternative_text: text,
    alternative_source: articleSourceName(article),
    alternative_url: article && article.url ? article.url : "",
    allow_live_seed: false,
  };
}

function updateUI(data, targetLean) {
  // ── Update bias bar with real engine score (only on first valid score) ──
  if (data.bias_score !== undefined && data.bias_score !== null) {
    const score = parseFloat(data.bias_score);
    const pct   = Math.min(100, Math.max(0, ((score + 5) / 10) * 100));
    const label = formatBiasScore(score);

    const bubble  = document.getElementById("bias-bubble");
    const marker  = document.getElementById("bias-marker");
    const leanTxt = document.getElementById("lean-text");

    if (bubble)  { bubble.style.left = pct + "%"; bubble.textContent = label; }
    if (marker)  { marker.style.left = pct + "%"; }
    if (leanTxt) {
      const biasLabel = _currentBiasData && _currentBiasData.bias_classification
        ? _currentBiasData.bias_classification
        : targetLean;
      leanTxt.textContent = "Lean Score: " + label + " (" + biasLabel + ")";
    }
  }

  // ── Update Cross-Article Comparison list (engine LLM output only) ──
  const listEl = document.getElementById("comparison-list");
  if (!listEl) return;

  listEl.innerHTML = "";

  // Engine found no matching article in ChromaDB for this topic
  if (!data.reasoning && !data.missing_context && !data.perspective_summary) {
    renderFallbackComparison(targetLean);
    return;
  }

  if (data.reasoning) {
    const li = document.createElement("li");
    li.innerHTML = "<strong>Why it leans this way:</strong> " + escHtml(data.reasoning);
    listEl.appendChild(li);
  }

  if (data.missing_context) {
    const li = document.createElement("li");
    li.innerHTML = "<strong>What's missing:</strong> " + escHtml(data.missing_context);
    listEl.appendChild(li);
  }
}

function renderFallbackComparison(targetLean, errorMessage) {
  const listEl = document.getElementById("comparison-list");
  if (!listEl) return;

  const suffix = errorMessage ? " (" + errorMessage + ")" : "";
  listEl.innerHTML = '<li style="color:#9aa0a6;">No closely related ' +
    escHtml(targetLean) + '-leaning article found for this topic' +
    escHtml(suffix) + '.</li>';
}

function capitalise(str) {
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
}
