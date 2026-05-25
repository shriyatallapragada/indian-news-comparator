// background.js — service worker handles all fetch calls (stable context in MV3)

// ── Backend URLs ──────────────────────────────────────────────────────────
// For Hugging Face Spaces, replace the placeholder with:
// https://[username]-[spacename].hf.space
const HF_SPACE_BASE = "https://shriyat-indian-news-comparator.hf.space";
const USE_HF_SPACE = !HF_SPACE_BASE.includes("[username]");

// Hugging Face exposes one port, so both route groups live under the same base.
const API_BASE = USE_HF_SPACE ? HF_SPACE_BASE : "http://127.0.0.1:8000";
const ENGINE_BASE = USE_HF_SPACE ? HF_SPACE_BASE : "http://127.0.0.1:8001";

function fetchJsonWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, { ...(options || {}), signal: controller.signal })
    .then(r => {
      if (!r.ok) throw new Error("Backend returned " + r.status);
      return r.json();
    })
    .catch(err => {
      if (err.name === "AbortError") {
        throw new Error("Backend took too long to respond");
      }
      throw err;
    })
    .finally(() => clearTimeout(timer));
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "analyze") {
    fetchJsonWithTimeout(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: message.text }),
    }, 45000)
      .then(data => sendResponse({ ok: true, data }))
      .catch(err => sendResponse({ ok: false, error: err.message }));
    return true; // keep channel open for async response
  }

  if (message.action === "related") {
    fetchJsonWithTimeout(`${API_BASE}/api/related`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(message.payload),
    }, 12000)
      .then(data => sendResponse({ ok: true, data }))
      .catch(err => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (message.action === "news") {
    fetchJsonWithTimeout(`${API_BASE}/news?${message.params}`, {}, 15000)
      .then(data => sendResponse({ ok: true, data }))
      .catch(err => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (message.action === "perspective") {
    fetchJsonWithTimeout(`${ENGINE_BASE}/analyze_perspective`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(message.payload),
    }, 12000)
      .then(data => sendResponse({ ok: true, data }))
      .catch(err => sendResponse({ ok: false, error: err.message }));
    return true;
  }
});
