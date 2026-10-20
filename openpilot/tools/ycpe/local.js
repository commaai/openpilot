// the local daemon's top-bar badge and its panel: modules, the shared disk cache (with prefetch and evict for the open
// route) and jobs. Prefetch jobs also show as a "disk cache" lane in the load status.
import {h, css} from "/dom.js";
import {daemon} from "/daemon.js";
import {status, fmtBytes} from "/status.js";

const ROUTE_RE = /([0-9a-f]{16})[|/_]([0-9a-f]{8}--[0-9a-f]{10}|\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})/i;
const openRoute = () => { const m = ROUTE_RE.exec(decodeURIComponent(location.pathname)); return m && `${m[1]}|${m[2]}`; };

const jobs = new Map();  // id → job, from /local/jobs/events
const subs = new Set();
export const onJobs = (fn) => { subs.add(fn); return () => subs.delete(fn); };
export const allJobs = () => [...jobs.values()];

const lanes = new Map();  // job id → status source
function lane(job) {
  if (job.kind !== "prefetch" || !job.meta?.segs) return;
  let src = lanes.get(job.id);
  if (!src) lanes.set(job.id, src = status.source(`prefetch-${job.id}`, {label: "disk cache", order: 5}));
  const segs = job.meta.segs, n = Math.max(0, ...Object.keys(segs).map((k) => +k + 1));
  src.count(n);
  for (let i = 0; i < n; i++) src.seg(i, segs[i] ?? "missing");
  src.rate(job.state === "running" && job.done && job.ended == null ? job.done / Math.max(1, Date.now() / 1000 - job.started) : 0);
  if (job.state !== "running") setTimeout(() => { lanes.get(job.id)?.remove(); lanes.delete(job.id); }, 5000);
}

if (daemon.present) {
  daemon.events("/local/jobs/events", (ev, job) => {
    if (ev !== "job") return;
    jobs.set(job.id, job);
    lane(job);
    for (const fn of subs) fn(job);
  });
}

export function localBadge() {
  if (!daemon.present) return null;
  css("/local.css");
  const el = h("button", {class: "local-badge", title: "Local daemon (serve.py): disk cache and jobs"}, "local");
  let panel = null;
  onJobs(() => { el.classList.toggle("busy", allJobs().some((j) => j.state === "running")); panel?.update(); });
  el.addEventListener("click", () => {
    if (panel) { panel.close(); panel = null; } else panel = localPanel(el, () => { panel = null; });
  });
  return el;
}

const KINDS = {logs: ["log"], video: ["qcamera", "fcamera", "ecamera", "dcamera"]};

function localPanel(anchor, onClose) {
  const mods = h("div", {class: "local-mods muted"}, Object.entries(daemon.caps.modules).map(([k, v]) => `${k} ${v.version}`).join(" · "));
  const cache = h("div", {class: "local-cache"});
  const here = h("div", {class: "local-route"});
  const list = h("div", {class: "local-jobs"});
  const el = h("div", {class: "local-panel"},
    h("div", {class: "local-head"}, h("b", {}, "Local daemon"), h("button", {class: "link", onclick: () => close()}, "close")),
    mods, cache, here, list);
  const r = anchor.getBoundingClientRect();
  el.style.top = `${r.bottom + 6}px`;
  el.style.right = `${Math.max(8, innerWidth - r.right)}px`;
  document.body.append(el);
  const route = openRoute();

  const act = (label, fn) => h("button", {class: "link", onclick: async (e) => {
    e.target.disabled = true;
    try { await fn(); } catch (err) { here.append(h("div", {class: "local-err"}, String(err.message || err))); }
    e.target.disabled = false;
    stats();
  }}, label);

  async function stats() {
    if (!daemon.has("cache")) return;
    const s = await daemon.json(`/local/cache${route ? `?route=${encodeURIComponent(route)}` : ""}`).catch(() => null);
    if (!s || !el.isConnected) return;
    cache.replaceChildren(
      h("div", {}, `shared cache ${fmtBytes(s.bytes)} of ${fmtBytes(s.budget)} · ${s.files} files`),
      h("div", {class: "local-bar"}, h("i", {style: `width: ${Math.min(100, 100 * s.bytes / s.budget).toFixed(1)}%`})),
      h("div", {class: "muted mono"}, s.dir));
    if (!route || !s.route) return here.replaceChildren();
    const segs = Object.values(s.route.segments), n = segs.length;
    const count = (kinds) => segs.filter((seg) => kinds.some((k) => seg[k])).length;
    const bytes = segs.reduce((a, seg) => a + Object.values(seg).reduce((b, f) => b + f.size, 0), 0);
    here.replaceChildren(
      h("div", {}, `this route: logs ${count(["rlog", "qlog"])}/${n} · video ${count(KINDS.video)}/${n} segments · ${fmtBytes(bytes)}`),
      h("div", {class: "local-acts"},
        act("cache logs", () => daemon.json("/local/cache/prefetch", {method: "POST", body: {route, kinds: KINDS.logs}})),
        act("cache video", () => daemon.json("/local/cache/prefetch", {method: "POST", body: {route, kinds: KINDS.video}})),
        act("evict", () => daemon.json("/local/cache/evict", {method: "POST", body: {route}}))));
  }

  function update() {
    const js = allJobs().slice(-6).reverse();
    list.replaceChildren(...js.map((j) => h("div", {class: `local-job ${j.state}`},
      h("span", {class: "grow"}, j.title),
      h("span", {class: "muted"}, j.state === "running"
        ? `${j.total ? Math.round(100 * j.done / j.total) : 0}%${j.unit === "bytes" && j.total ? ` of ${fmtBytes(j.total)}` : ""}${j.text ? " · " + j.text : ""}`
        : j.error || (j.result?.seconds != null ? `${j.state} in ${j.result.seconds} s` : j.state)),
      j.state === "running" && h("button", {class: "link", onclick: () => daemon.fetch(`/local/jobs/${j.id}/cancel`, {method: "POST"})}, "cancel"))));
  }

  const close = () => { el.remove(); onClose(); };
  setTimeout(() => document.addEventListener("pointerdown", function off(e) {
    if (!el.contains(e.target) && !anchor.contains(e.target)) { document.removeEventListener("pointerdown", off); if (el.isConnected) close(); }
  }));
  stats();
  update();
  let last = 0;
  return {update() { update(); if (performance.now() - last > 1000) { last = performance.now(); stats(); } }, close};
}
