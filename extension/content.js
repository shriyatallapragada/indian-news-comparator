// Check if the script has already been injected to prevent redeclaration errors
if (!window.newsComparatorContentLoaded) {
  window.newsComparatorContentLoaded = true;

  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "extractText") {
      sendResponse({ text: extractArticleText() });
    }
    return true;
  });

  const MAX_CHARS = 3000;

  function extractArticleText() {
    const headline = extractHeadline();
    const container = findArticleContainer();
    const lines = headline ? [headline] : [];

    if (container) {
      // THE FIX: Use a TreeWalker. It is 100x faster than querySelectorAll.
      // It bypasses HTML tags entirely and just vacuums up the raw text nodes.
      // This natively solves the Times of India <br> issue and Economic Times <div> issue.
      const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null, false);
      let node;
      
      while ((node = walker.nextNode())) {
        const parent = node.parentNode;
        if (isNoise(parent)) continue; // Skip hidden or sidebar text
        
        const text = clean(node.nodeValue);
        // If the text chunk is a real sentence (not just a date or UI label)
        if (text.length > 40 && !lines.includes(text)) {
          lines.push(text);
        }
      }
    }

    return lines.join(" ").slice(0, MAX_CHARS);
  }

  // ─── HEADLINE EXTRACTOR ──────────────────────────────────────────────────────
  function extractHeadline() {
    for (const s of document.querySelectorAll('script[type="application/ld+json"]')) {
      try {
        const items = [].concat(JSON.parse(s.textContent));
        for (const item of items) {
          const h = item.headline || item.name;
          if (h && h.length > 10) return h.trim();
        }
      } catch (_) {}
    }

    const og = document.querySelector('meta[property="og:title"]');
    if (og && og.content && og.content.length > 10) {
      return og.content.replace(/\s*[-|]\s*[^-|]{2,40}$/, "").trim();
    }

    const h1 = document.querySelector("h1");
    if (h1) return clean(h1.textContent);

    return "";
  }

  // ─── CONTAINER FINDER ────────────────────────────────────────────────────────
  function findArticleContainer() {
    // 1. O(1) Fast lookups: Check for standard developer tags first
    const selectors = [
      ".artText", ".Normal", "div[data-artid]", // Economic Times
      "[data-articlebody]", "._s30J", ".xf8Hc", // Times of India
      "[itemprop='articleBody']", "article",
      "[id*='content-body']", "[class*='article-body']",
      "[class*='story-body']", "[class*='post-content']"
    ];

    for (const sel of selectors) {
      try {
        const el = document.querySelector(sel);
        // Use textContent instead of innerText to avoid browser reflows
        if (el && el.textContent.trim().length > 300) return el;
      } catch (_) {}
    }

    // 2. Fallback: Mini-Readability Scoring Algorithm
    return findBestContainerByScore();
  }

  // ─── ALGORITHMIC SCORING (SUPER FAST) ────────────────────────────────────────
  // Instead of checking every div, this finds paragraph-like nodes, gives them 
  // points, and awards those points to their parent containers. The container 
  // with the most points is the article.
  function findBestContainerByScore() {
    const candidates = new Map();
    const nodes = document.querySelectorAll("p, div > text, span");

    nodes.forEach(node => {
      if (isNoise(node)) return;

      const text = clean(node.textContent);
      if (text.length < 40) return; // Ignore small menu links

      // Base score on text length
      const score = 1 + (text.length / 50);

      // Award points to parent
      const parent = node.parentNode;
      if (parent && !isNoise(parent)) {
        candidates.set(parent, (candidates.get(parent) || 0) + score);
      }

      // Award half points to grandparent
      const grandParent = parent ? parent.parentNode : null;
      if (grandParent && !isNoise(grandParent)) {
        candidates.set(grandParent, (candidates.get(grandParent) || 0) + (score / 2));
      }
    });

    let topCandidate = document.body;
    let topScore = 0;
    
    for (let [node, score] of candidates.entries()) {
      if (score > topScore) {
        topScore = score;
        topCandidate = node;
      }
    }
    
    return topCandidate;
  }

  // ─── AGGRESSIVE NOISE FILTER ──────────────────────────────────────────────────
  function isNoise(el) {
    if (!el || !el.tagName) return true;
    
    const tag = el.tagName.toUpperCase();
    if (["SCRIPT", "STYLE", "NOSCRIPT", "IFRAME", "BUTTON", "FORM", "INPUT", "SELECT", "TEXTAREA", "HEADER", "FOOTER", "NAV", "ASIDE", "FIGURE"].includes(tag)) return true;

    // Check visibility without triggering reflow (mostly)
    if (el.style && (el.style.display === "none" || el.style.visibility === "hidden")) return true;

    const role = (el.getAttribute?.("role") || "").toLowerCase();
    if (["navigation", "banner", "contentinfo", "complementary", "search", "dialog"].includes(role)) return true;

    const cls = (typeof el.className === "string" ? el.className : "").toLowerCase();
    const id = (el.id || "").toLowerCase();
    
    // The ultimate blocklist for Indian/Financial news sites
    const noiseWords = [
      "sidebar", "related", "more-from", "top-stories", "trending", 
      "briefs", "read-also", "also-read", "newsletter", "subscribe", 
      "comment", "social", "share", "cookie", "advertisement", "sponsored", 
      "most-popular", "recommended", "taboola", "outbrain", "promo",
      "ticker", "market-data", "stock-quotes", "latest-news", "widget", "author-bio"
    ];
    
    return noiseWords.some(w => cls.includes(w) || id.includes(w));
  }

  // ─── UTILS ───────────────────────────────────────────────────────────────────
  function clean(text) {
    return (text || "").replace(/\s+/g, " ").trim();
  }
}