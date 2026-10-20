// UI lint over what is on screen right now. Every issue has a rule, a selector, a viewport box [x, y, w, h] and a detail.
// cfg: {ignore: css selector list whose subtrees are skipped, shiftsFrom: index into __probe.shifts already reported}
(cfg) => {
  const W = document.documentElement.clientWidth, H = innerHeight;
  const issues = [];
  const box = (r) => [r.left, r.top, r.width, r.height].map(Math.round);
  const snippet = (el) => (el.innerText || el.getAttribute("aria-label") || "").trim().replace(/\s+/g, " ").slice(0, 60);

  const part = (el) => {
    if (el.id) return `#${el.id}`;
    let s = el.localName + [...el.classList].slice(0, 2).map((c) => `.${CSS.escape(c)}`).join("");
    const twins = el.parentElement ? [...el.parentElement.children].filter((c) => c.localName === el.localName && c.className === el.className) : [];
    if (twins.length > 1) s += `:nth-of-type(${[...el.parentElement.children].filter((c) => c.localName === el.localName).indexOf(el) + 1})`;
    return s;
  };
  const selector = (el) => {
    const parts = [];
    for (let e = el; e && e !== document.body && parts.length < 4; e = e.parentElement) {
      parts.unshift(part(e));
      if (e.id) break;
    }
    return parts.join(" > ");
  };
  const add = (rule, el, rect, detail) => issues.push({rule, selector: selector(el), box: box(rect), detail});

  const onScreen = (r) => r.width > 0 && r.height > 0 && r.right > 0 && r.bottom > 0 && r.left < W && r.top < H;
  const els = [];
  for (const el of document.body.querySelectorAll("*")) {
    if (el.ownerSVGElement || (cfg.ignore && el.closest(cfg.ignore))) continue;
    const r = el.getBoundingClientRect();
    if (!onScreen(r) || !el.checkVisibility({opacityProperty: true, visibilityProperty: true})) continue;
    els.push({el, r, cs: getComputedStyle(el)});
  }
  const info = new Map(els.map((e) => [e.el, e]));
  const ownText = (el) => [...el.childNodes].some((n) => n.nodeType === Node.TEXT_NODE && n.textContent.trim());
  const clips = (cs) => /hidden|clip/.test(cs.overflowX + cs.overflowY);
  const clipAncestor = (el) => {
    for (let a = el.parentElement; a && a !== document.body; a = a.parentElement) {
      const cs = getComputedStyle(a);
      if (cs.overflowX !== "visible" || cs.overflowY !== "visible") return a;
      if (cs.position === "fixed") return null;
    }
    return null;
  };

  // content wider than the page: only the outermost offender, and only if nothing between it and the page clips it
  for (const {el, r} of els) {
    if (r.right <= W + 1 && r.left >= -1) continue;
    const parent = info.get(el.parentElement);
    if (parent && (parent.r.right > W + 1 || parent.r.left < -1)) continue;
    if (clipAncestor(el)) continue;
    add("overflow-viewport", el, r, `extends ${Math.round(Math.max(r.right - W, -r.left))}px past the viewport edge`);
  }

  // text: cut off by its own box (ellipsis or hard clip), or partly hidden by a clipping ancestor
  for (const {el, r, cs} of els) {
    if (!ownText(el) || cs.display === "inline") continue;
    const dx = el.scrollWidth - el.clientWidth, dy = el.scrollHeight - el.clientHeight;
    if (clips(cs) && (dx > 1 || dy > 1)) {
      const rule = cs.textOverflow === "ellipsis" && dx > 1 ? "text-ellipsis" : "text-clipped";
      add(rule, el, r, `"${snippet(el)}" needs ${dx > 1 ? `${el.scrollWidth}px width, has ${el.clientWidth}` : `${el.scrollHeight}px height, has ${el.clientHeight}`}`);
      continue;
    }
    const a = clipAncestor(el);
    if (!a || !clips(getComputedStyle(a))) continue;
    const ar = a.getBoundingClientRect();
    const cut = Math.max(ar.left - r.left, r.right - ar.right, ar.top - r.top, r.bottom - ar.bottom);
    if (cut > 1) add("text-clipped", el, r, `"${snippet(el)}" cut ${Math.round(cut)}px by ${selector(a)}`);
  }

  // interactive elements: native controls, plus the outermost element of anything with a pointer cursor
  const NATIVE = "a[href], button, input:not([type=hidden]), select, textarea, summary, [role=button], [tabindex]:not([tabindex='-1'])";
  const targets = els.filter(({el, cs}) => el.matches(NATIVE) ||
    (cs.cursor === "pointer" && getComputedStyle(el.parentElement).cursor !== "pointer" && !el.parentElement.closest(NATIVE)));
  for (const {el, r, cs} of targets) {
    const inlineLink = cs.display === "inline" && ownText(el.parentElement);
    if (!inlineLink && (r.width < 24 || r.height < 24)) add("small-target", el, r, `${Math.round(r.width)}x${Math.round(r.height)}px, under 24x24: "${snippet(el)}"`);
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    if (cx < 0 || cy < 0 || cx >= W || cy >= H) continue;
    const hit = document.elementFromPoint(cx, cy);
    if (hit && !el.contains(hit) && !hit.contains(el)) add("obscured", el, r, `center covered by ${selector(hit)}`);
  }
  for (let i = 0; i < targets.length; i++) {
    for (let j = i + 1; j < targets.length; j++) {
      const a = targets[i], b = targets[j];
      if (a.el.contains(b.el) || b.el.contains(a.el)) continue;
      const w = Math.min(a.r.right, b.r.right) - Math.max(a.r.left, b.r.left), hgt = Math.min(a.r.bottom, b.r.bottom) - Math.max(a.r.top, b.r.top);
      if (w > 2 && hgt > 2) add("overlap", a.el, a.r, `overlaps ${selector(b.el)} by ${Math.round(w)}x${Math.round(hgt)}px`);
    }
  }

  for (const {el, r} of els) {
    if (el.localName === "img" && el.complete && el.naturalWidth === 0 && el.currentSrc) add("broken-image", el, r, el.currentSrc.slice(0, 120));
  }

  // layout shifts since the previous shot, boxed where the shifted node ended up
  const shifts = window.__probe?.shifts || [];
  for (const s of shifts.slice(cfg.shiftsFrom)) {
    for (const {node, rect} of s.sources) {
      if (!(node instanceof Element) || s.value < 0.005) continue;
      add("layout-shift", node, rect, `shift ${s.value.toFixed(3)} at ${Math.round(s.t)}ms`);
    }
  }
  return {issues, shifts: shifts.length, cls: shifts.reduce((a, s) => a + s.value, 0)};
}
