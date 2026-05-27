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

function isNetworkError(err) {
  return err && (
    err.name === "TypeError" ||
    /failed to fetch|network|load failed/i.test(err.message || "")
  );
}

function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function wakeBackend(baseUrl) {
  try {
    await fetchJsonWithTimeout(`${baseUrl}/healthz`, {}, 15000);
  } catch (_) {
    // The real request below will return the useful error if wake-up fails.
  }
}

async function fetchJsonWithRetry(baseUrl, path, options, timeoutMs) {
  const url = `${baseUrl}${path}`;
  try {
    return await fetchJsonWithTimeout(url, options, timeoutMs);
  } catch (err) {
    if (!isNetworkError(err)) throw err;
    await wakeBackend(baseUrl);
    await wait(1500);
    return fetchJsonWithTimeout(url, options, timeoutMs);
  }
}

function publicBackendError(err) {
  if (isNetworkError(err)) {
    return "Could not reach the hosted backend. Open https://shriyat-indian-news-comparator.hf.space/healthz in Chrome on this computer, then reload the extension.";
  }
  return err.message;
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "analyze") {
    fetchJsonWithRetry(API_BASE, "/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: message.text }),
    }, 45000)
      .then(data => sendResponse({ ok: true, data }))
      .catch(err => sendResponse({ ok: false, error: publicBackendError(err) }));
    return true; // keep channel open for async response
  }

  if (message.action === "related") {
    fetchJsonWithRetry(API_BASE, "/api/related", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(message.payload),
    }, 12000)
      .then(data => sendResponse({ ok: true, data }))
      .catch(err => sendResponse({ ok: false, error: publicBackendError(err) }));
    return true;
  }

  if (message.action === "news") {
    fetchJsonWithRetry(API_BASE, `/news?${message.params}`, {}, 15000)
      .then(data => sendResponse({ ok: true, data }))
      .catch(err => sendResponse({ ok: false, error: publicBackendError(err) }));
    return true;
  }

  if (message.action === "ingest") {
    fetchJsonWithRetry(API_BASE, "/api/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(message.payload),
    }, 20000)
      .then(data => sendResponse({ ok: true, data }))
      .catch(err => sendResponse({ ok: false, error: publicBackendError(err) }));
    return true;
  }

  if (message.action === "perspective") {
    fetchJsonWithRetry(ENGINE_BASE, "/analyze_perspective", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(message.payload),
    }, 12000)
      .then(data => sendResponse({ ok: true, data }))
      .catch(err => sendResponse({ ok: false, error: publicBackendError(err) }));
    return true;
  }
});
