// UI lint over what is on screen right now. Every issue has a rule, a selector, a viewport box [x, y, w, h] and a detail.
// cfg: {ignore: css selector list whose subtrees are skipped, shiftsFrom: index into __probe.shifts already reported}
(cfg) => {
  const W = document.documentElement.clientWidth, H = innerHeight;
  const issues = [];
  const rect = (l, t, r, b) => ({left: l, top: t, right: r, bottom: b, width: Math.max(0, r - l), height: Math.max(0, b - t)});
  const meet = (a, b) => rect(Math.max(a.left, b.left), Math.max(a.top, b.top), Math.min(a.right, b.right), Math.min(a.bottom, b.bottom));
  const ALL = rect(-Infinity, -Infinity, Infinity, Infinity), VIEW = rect(0, 0, W, H);
  const box = (r) => [r.left, r.top, r.width, r.height].map(Math.round);
  const snippet = (el) => (el.selectedOptions?.[0]?.text || el.innerText || el.getAttribute("aria-label") || el.title || "").trim().replace(/\s+/g, " ").slice(0, 60);

  const part = (el) => {
    if (el.id) return `#${el.id}`;
    let s = el.localName + [...el.classList].slice(0, 2).map((c) => `.${CSS.escape(c)}`).join("");
    const same = el.parentElement ? [...el.parentElement.children].filter((c) => c.localName === el.localName) : [];
    if (same.filter((c) => c.className === el.className).length > 1) s += `:nth-of-type(${same.indexOf(el) + 1})`;
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
  const add = (rule, el, r, detail) => issues.push({rule, selector: selector(el), box: box(r), detail});

  const styles = new Map();
  const style = (el) => styles.get(el) || styles.set(el, getComputedStyle(el)).get(el);
  // the rect an element's content is visible within, from every overflow-clipping ancestor (fixed elements escape them)
  const clips = new Map();
  const clipOf = (el) => {
    if (!el || el === document.documentElement) return ALL;
    if (clips.has(el)) return clips.get(el);
    const s = style(el);
    let r = s.position === "fixed" ? ALL : clipOf(el.parentElement);
    if (el !== document.body && (s.overflowX !== "visible" || s.overflowY !== "visible")) r = meet(r, el.getBoundingClientRect());
    clips.set(el, r);
    return r;
  };

  const els = [];
  for (const el of document.body.querySelectorAll("*")) {
    if (el.ownerSVGElement || (cfg.ignore && el.closest(cfg.ignore))) continue;
    const r = el.getBoundingClientRect();
    const clip = style(el).position === "fixed" ? ALL : clipOf(el.parentElement);
    const vis = meet(meet(r, clip), VIEW);
    if (vis.width < 1 || vis.height < 1 || !el.checkVisibility({opacityProperty: true, visibilityProperty: true})) continue;
    els.push({el, r, vis, clip, cs: style(el)});
  }
  const ownText = (el) => [...el.childNodes].some((n) => n.nodeType === Node.TEXT_NODE && n.textContent.trim());
  // the nearest ancestor whose clipping actually cuts r
  const clipper = (el, r) => {
    for (let a = el.parentElement; a && a !== document.body; a = a.parentElement) {
      if (clips.get(a) === clipOf(a.parentElement)) continue;
      const c = a.getBoundingClientRect();
      if (r.left < c.left - 1 || r.top < c.top - 1 || r.right > c.right + 1 || r.bottom > c.bottom + 1) return a;
    }
    return null;
  };

  // content past the page edge that nothing clips: only the outermost offender
  const wide = (r, clip) => (r.right > W + 1 && clip.right > W + 1) || (r.left < -1 && clip.left < -1);
  for (const {el, r, clip} of els) {
    if (!wide(r, clip)) continue;
    const p = el.parentElement;
    if (p && p !== document.body && wide(p.getBoundingClientRect(), clipOf(p.parentElement))) continue;
    add("overflow-viewport", el, r, `"${snippet(el)}" extends ${Math.round(Math.max(r.right - W, -r.left))}px past the viewport edge`);
  }

  // text cut off by its own box (ellipsis or hard clip), or partly hidden by a clipping ancestor
  for (const {el, r, vis, clip, cs} of els) {
    if (!ownText(el) || cs.display === "inline") continue;
    const dx = el.scrollWidth - el.clientWidth, dy = el.scrollHeight - el.clientHeight;
    if (cs.overflowX !== "visible" && (dx > 1 || dy > 1)) {
      const [need, has] = dx > 1 ? [el.scrollWidth, el.clientWidth] : [el.scrollHeight, el.clientHeight];
      const rule = cs.textOverflow !== "ellipsis" || dx <= 1 ? "text-clipped" : has < need / 2 ? "text-squeezed" : "text-ellipsis";
      add(rule, el, vis, `"${snippet(el)}" needs ${need}px ${dx > 1 ? "width" : "height"}, has ${has}`);
    } else {
      // cut by a hidden/clip ancestor; content scrolled partly out of a scroller, or below the fold, is fine
      const shown = meet(r, clip), a = clipper(el, r);
      if ((shown.width >= r.width - 1 && shown.height >= r.height - 1) || !a || /auto|scroll/.test(style(a).overflowX + style(a).overflowY)) continue;
      add("text-clipped", el, vis, `"${snippet(el)}" shows ${Math.round(shown.width)}x${Math.round(shown.height)} of ${Math.round(r.width)}x${Math.round(r.height)}px, clipped by ${selector(a)}`);
    }
  }

  // interactive: native controls, plus the outermost element of anything with a pointer cursor
  const NATIVE = "a[href], button, input:not([type=hidden]), select, textarea, summary, [role=button], [tabindex]:not([tabindex='-1'])";
  const targets = els.filter(({el, cs}) => el.matches(NATIVE) ||
    (cs.cursor === "pointer" && style(el.parentElement).cursor !== "pointer" && !el.parentElement.closest(NATIVE)));
  const center = (r) => [r.left + r.width / 2, r.top + r.height / 2];
  const gap = (r, [x, y]) => Math.hypot(Math.max(r.left - x, 0, x - r.right), Math.max(r.top - y, 0, y - r.bottom));
  const small = (t) => t.vis.width < 24 || t.vis.height < 24;
  // a control sitting on a larger target (a button on a map, a chip on a card) is layering, not a collision
  const within = (a, b) => a.left >= b.left - 1 && a.top >= b.top - 1 && a.right <= b.right + 1 && a.bottom <= b.bottom + 1;
  const apart = (a, b) => a.el.contains(b.el) || b.el.contains(a.el) || within(a.vis, b.vis) || within(b.vis, a.vis);
  for (const t of targets) {
    const {el, vis, cs} = t;
    // WCAG 2.5.8: an undersized target passes if a 24px circle on its center touches no other target (or another undersized one's circle)
    if (small(t) && !(cs.display === "inline" && ownText(el.parentElement))) {
      const c = center(vis);
      const near = targets.find((o) => o !== t && !apart(o, t) &&
        (small(o) ? Math.hypot(c[0] - center(o.vis)[0], c[1] - center(o.vis)[1]) < 24 : gap(o.vis, c) < 12));
      if (near) add("small-target", el, vis, `${Math.round(vis.width)}x${Math.round(vis.height)}px "${snippet(el)}", ${selector(near.el)} within the 24px spacing circle`);
    }
    // covered: most of five probe points land on something unrelated
    const pts = [[0.5, 0.5], [0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8]].map(([fx, fy]) => [vis.left + vis.width * fx, vis.top + vis.height * fy]);
    const hits = pts.map(([x, y]) => document.elementFromPoint(x, y)).filter((hit) => hit && !el.contains(hit) && !hit.contains(el));
    if (hits.length >= 3) add("obscured", el, vis, `"${snippet(el)}" is under ${selector(hits[0])}`);
  }
  for (let i = 0; i < targets.length; i++) {
    for (let j = i + 1; j < targets.length; j++) {
      const a = targets[i], b = targets[j];
      if (apart(a, b)) continue;
      const o = meet(a.vis, b.vis);
      if (o.width > 2 && o.height > 2) add("overlap", a.el, o, `"${snippet(a.el)}" overlaps ${selector(b.el)} by ${Math.round(o.width)}x${Math.round(o.height)}px`);
    }
  }

  for (const {el, vis} of els) {
    if (el.localName === "img" && el.complete && el.naturalWidth === 0 && el.currentSrc) add("broken-image", el, vis, el.currentSrc.slice(0, 120));
  }

  // layout shifts since the previous shot, boxed where the shifted node ended up
  const shifts = window.__probe?.shifts || [];
  for (const s of shifts.slice(cfg.shiftsFrom)) {
    for (const {node, rect: r} of s.sources) {
      if (node instanceof Element && s.value >= 0.005) add("layout-shift", node, r, `shift ${s.value.toFixed(3)} at ${Math.round(s.t)}ms`);
    }
  }
  return {issues, shifts: shifts.length, cls: shifts.reduce((a, s) => a + s.value, 0)};
}
