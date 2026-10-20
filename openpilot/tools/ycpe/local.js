// the local daemon's top-bar badge and its panel: modules, the shared disk cache, and running jobs
import {h, css} from "/dom.js";
import {daemon} from "/daemon.js";
import {fmtBytes} from "/status.js";

const jobs = new Map();  // id → job, from /local/jobs/events
const subs = new Set();
export const onJobs = (fn) => { subs.add(fn); return () => subs.delete(fn); };
export const allJobs = () => [...jobs.values()];
if (daemon.present) {
  daemon.events("/local/jobs/events", (ev, job) => {
    if (ev !== "job") return;
    jobs.set(job.id, job);
    for (const fn of subs) fn(job);
  });
}

export function localBadge() {
  if (!daemon.present) return null;
  css("/local.css");
  const el = h("button", {class: "local-badge", title: "Local daemon (serve.py): click for cache and jobs"}, "local");
  let panel = null;
  const running = () => allJobs().filter((j) => j.state === "running");
  onJobs(() => { el.classList.toggle("busy", running().length > 0); panel?.update(); });
  el.addEventListener("click", () => {
    if (panel) { panel.close(); panel = null; } else panel = localPanel(el, () => { panel = null; });
  });
  return el;
}

function localPanel(anchor, onClose) {
  const mods = h("div", {class: "local-mods muted"});
  const cache = h("div", {class: "local-cache"});
  const list = h("div", {class: "local-jobs"});
  const el = h("div", {class: "local-panel"},
    h("div", {class: "local-head"}, h("b", {}, "Local daemon"), h("button", {class: "link", onclick: () => close()}, "close")),
    mods, cache, list);
  const r = anchor.getBoundingClientRect();
  el.style.top = `${r.bottom + 6}px`;
  el.style.right = `${Math.max(8, innerWidth - r.right)}px`;
  document.body.append(el);
  mods.textContent = Object.entries(daemon.caps.modules).map(([k, v]) => `${k} ${v.version}`).join(" · ");

  async function stats() {
    if (!daemon.has("cache")) return;
    const s = await daemon.json("/local/cache").catch(() => null);
    if (!s) return;
    cache.replaceChildren(
      h("div", {}, `cache ${fmtBytes(s.bytes)} of ${fmtBytes(s.budget)} · ${s.files} files · ${s.dir}`),
      h("div", {class: "local-bar"}, h("i", {style: `width: ${Math.min(100, 100 * s.bytes / s.budget).toFixed(1)}%`})));
  }
  function update() {
    const js = allJobs().slice(-8).reverse();
    list.replaceChildren(...(js.length ? js.map((j) => h("div", {class: `local-job ${j.state}`},
      h("span", {class: "grow"}, j.title),
      h("span", {class: "muted"}, j.state === "running" ? `${j.total ? Math.round(100 * j.done / j.total) : 0}%${j.text ? " · " + j.text : ""}` : j.error || j.state),
      j.state === "running" && h("button", {class: "link", onclick: () => daemon.fetch(`/local/jobs/${j.id}/cancel`, {method: "POST"})}, "cancel")))
      : [h("div", {class: "muted"}, "no jobs")]));
  }
  const close = () => { el.remove(); onClose(); };
  setTimeout(() => document.addEventListener("pointerdown", function off(e) {
    if (!el.contains(e.target) && !anchor.contains(e.target)) { document.removeEventListener("pointerdown", off); if (el.isConnected) close(); }
  }));
  stats();
  update();
  return {update: () => { update(); }, close};
}
