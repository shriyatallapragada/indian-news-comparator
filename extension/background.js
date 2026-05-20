// background.js — service worker handles all fetch calls (stable context in MV3)

// ── Backend URLs — update these when deploying ─────────────────────────────
const API_BASE    = "http://127.0.0.1:8000";  // main.py  (port 8000)
const ENGINE_BASE = "http://127.0.0.1:8001";  // engine.py (port 8001)

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "analyze") {
    fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: message.text }),
    })
      .then(r => r.json())
      .then(data => sendResponse({ ok: true, data }))
      .catch(err => sendResponse({ ok: false, error: err.message }));
    return true; // keep channel open for async response
  }

  if (message.action === "related") {
    fetch(`${API_BASE}/api/related`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(message.payload),
    })
      .then(r => r.json())
      .then(data => sendResponse({ ok: true, data }))
      .catch(err => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (message.action === "news") {
    fetch(`${API_BASE}/news?${message.params}`)
      .then(r => r.json())
      .then(data => sendResponse({ ok: true, data }))
      .catch(err => sendResponse({ ok: false, error: err.message }));
    return true;
  }
});
