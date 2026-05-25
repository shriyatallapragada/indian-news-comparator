// background.js — service worker handles all fetch calls (stable context in MV3)

// ── Backend URLs ──────────────────────────────────────────────────────────
// For Hugging Face Spaces, replace the placeholder with:
// https://[username]-[spacename].hf.space
const HF_SPACE_BASE = "https://[username]-[spacename].hf.space";
const USE_HF_SPACE = !HF_SPACE_BASE.includes("[username]");

// Hugging Face exposes one port, so both route groups live under the same base.
const API_BASE = USE_HF_SPACE ? HF_SPACE_BASE : "http://127.0.0.1:8000";
const ENGINE_BASE = USE_HF_SPACE ? HF_SPACE_BASE : "http://127.0.0.1:8001";

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

  if (message.action === "perspective") {
    fetch(`${ENGINE_BASE}/analyze_perspective`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(message.payload),
    })
      .then(r => {
        if (!r.ok) throw new Error("Engine returned " + r.status);
        return r.json();
      })
      .then(data => sendResponse({ ok: true, data }))
      .catch(err => sendResponse({ ok: false, error: err.message }));
    return true;
  }
});
