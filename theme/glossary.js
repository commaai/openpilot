(function() {
  const DATA_SCRIPT_ID = "openpilot-glossary-data";
  const IGNORED_TAGS = new Set(["A", "BUTTON", "CODE", "KBD", "NOSCRIPT", "PRE", "SAMP", "SCRIPT", "STYLE", "SVG", "TEXTAREA"]);

  function escapeRegex(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function isWordBoundary(text, index) {
    if (index < 0 || index >= text.length) {
      return true;
    }
    return !/[A-Za-z0-9_]/.test(text[index]);
  }

  function isInsideParentheses(text, index) {
    let depth = 0;

    for (let i = 0; i < index; i += 1) {
      if (text[i] === "(") {
        depth += 1;
      } else if (text[i] === ")" && depth > 0) {
        depth -= 1;
      }
    }

    return depth > 0;
  }

  function shouldIgnoreNode(node, root) {
    let current = node.parentElement;
    while (current && current !== root) {
      if (IGNORED_TAGS.has(current.tagName) || current.classList.contains("glossary-term") || current.dataset.glossaryIgnore !== undefined) {
        return true;
      }
      current = current.parentElement;
    }

    return false;
  }

  function createTooltip(document, visibleText, entry) {
    const wrapper = document.createElement("a");
    wrapper.className = "glossary-term";
    wrapper.href = `${window.path_to_root || ""}concepts/glossary.html#${entry.anchor}`;
    wrapper.setAttribute("aria-label", `${visibleText}: ${entry.description}`);

    const tooltip = document.createElement("span");
    tooltip.className = "tooltip-content";
    tooltip.textContent = `${entry.description} `;

    const hint = document.createElement("span");
    hint.className = "tooltip-glossary-hint";
    hint.textContent = "Open glossary entry";

    wrapper.append(document.createTextNode(visibleText));
    tooltip.append(hint);
    wrapper.append(tooltip);
    return wrapper;
  }

  function replaceTextNode(node, matcher, glossaryByTerm) {
    const text = node.nodeValue;
    const fragment = document.createDocumentFragment();
    const matches = [];
    let cursor = 0;

    matcher.lastIndex = 0;
    for (let match = matcher.exec(text); match !== null; match = matcher.exec(text)) {
      const start = match.index;
      const end = start + match[0].length;
      const matchedText = match[0];

      if (!isWordBoundary(text, start - 1) || !isWordBoundary(text, end) || isInsideParentheses(text, start)) {
        continue;
      }

      matches.push({
        entry: glossaryByTerm.get(matchedText.toLowerCase()),
        end,
        start,
        text: matchedText,
      });
    }

    if (matches.length === 0) {
      return;
    }

    for (const match of matches) {
      fragment.append(document.createTextNode(text.slice(cursor, match.start)));
      fragment.append(createTooltip(document, match.text, match.entry));
      cursor = match.end;
    }

    fragment.append(document.createTextNode(text.slice(cursor)));
    node.parentNode.replaceChild(fragment, node);
  }

  function applyGlossary(root, glossary) {
    const terms = [];
    const glossaryByTerm = new Map();

    for (const entry of glossary) {
      for (const term of entry.terms) {
        const normalized = term.toLowerCase();
        if (glossaryByTerm.has(normalized)) {
          continue;
        }

        glossaryByTerm.set(normalized, entry);
        terms.push(term);
      }
    }

    if (terms.length === 0) {
      return;
    }

    terms.sort((a, b) => b.length - a.length);
    const matcher = new RegExp(terms.map(escapeRegex).join("|"), "gi");

    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        if (!node.nodeValue || !node.nodeValue.trim() || shouldIgnoreNode(node, root)) {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    });

    const nodes = [];
    while (walker.nextNode()) {
      nodes.push(walker.currentNode);
    }

    for (const node of nodes) {
      replaceTextNode(node, matcher, glossaryByTerm);
    }
  }

  function main() {
    const dataScript = document.getElementById(DATA_SCRIPT_ID);
    if (!dataScript) {
      return;
    }

    const root = document.querySelector("#mdbook-content main");
    if (!root) {
      return;
    }

    const glossary = JSON.parse(dataScript.textContent || "[]");
    applyGlossary(root, glossary);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", main, { once: true });
  } else {
    main();
  }
})();
