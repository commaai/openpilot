// the local daemon's top-bar badge and its panel: modules, the shared disk cache (with prefetch and evict for the open
// route) and jobs. Prefetch jobs also show as a "disk cache" lane in the load status.
import {h, css, listeners, anchoredPanel} from "/dom.js";
import {daemon} from "/daemon.js";
import {status, fmtBytes} from "/status.js";
import {link} from "/app.js";

const ROUTE_RE = /([0-9a-f]{16})[|/_]([0-9a-f]{8}--[0-9a-f]{10}|\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})/i;
const openRoute = () => { const m = ROUTE_RE.exec(decodeURIComponent(location.pathname)); return m && `${m[1]}|${m[2]}`; };

const jobs = new Map();  // id → job, from /local/jobs/events
const subs = listeners();
export const onJobs = subs.on;
export const allJobs = () => [...jobs.values()];

const lanes = new Map();  // job id → status source
function lane(job) {
  const label = {prefetch: "disk cache", pull: "device pull"}[job.kind];
  if (!label || !job.meta?.segs) return;
  let src = lanes.get(job.id);
  if (!src) lanes.set(job.id, src = status.source(`${job.kind}-${job.id}`, {label, order: 5}));
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
    subs.emit(job);
  });
}

export function localBadge() {
  if (!daemon.present) return null;
  css("/local.css");
  const el = h("button", {class: "local-badge", title: "Local daemon (serve.py): disk cache and jobs"}, "local"), panel = anchoredPanel(el, localPanel);
  onJobs(() => { el.classList.toggle("busy", allJobs().some((j) => j.state === "running")); panel.update(); });
  return el;
}

const KINDS = {logs: ["log"], video: ["qcamera", "fcamera", "ecamera", "dcamera"]};

// a comma device on the LAN: its routes over ssh, each pulled (read-only) into the shared cache
function devicePart() {
  const out = h("div", {class: "local-routes"});
  const host = h("input", {class: "local-host", placeholder: "device host or IP", title: "Comma device on your network (ssh)", "aria-label": "Device host or IP", spellcheck: false});
  const err = (e) => out.replaceChildren(h("div", {class: "local-err"}, String(e.message || e)));
  async function list() {
    out.replaceChildren(h("div", {class: "muted"}, "listing…"));
    try {
      if (host.value.trim()) await daemon.json("/local/device", {method: "POST", body: {host: host.value.trim()}});
      const {dongle, routes} = await daemon.json("/local/device/routes");
      out.replaceChildren(...routes.slice(0, 8).map((r) => {
        const segs = Object.values(r.segments), bytes = segs.reduce((a, s) => a + (s["rlog.zst"] ?? s["rlog.bz2"] ?? 0) + (s["qlog.zst"] ?? 0), 0);
        const name = r.route || `${dongle}|${r.log_id}`;
        return h("div", {class: "local-job"}, h("span", {class: "grow mono"}, r.log_id),
          h("span", {class: "muted"}, `${segs.length} seg · ${fmtBytes(bytes)}${r.active.length ? " · recording" : ""}`),
          h("button", {class: "link", onclick: () => daemon.json("/local/device/pull", {method: "POST", body: {route: name}}).catch(err)}, "pull"));
      }), routes.length ? "" : h("div", {class: "muted"}, "no routes on the device"));
    } catch (e) { err(e); }
  }
  daemon.json("/local/device").then((d) => { host.value = d.host || ""; }, () => {});
  return h("div", {class: "local-device"},
    h("div", {class: "local-acts"}, h("span", {title: "Comma device on your network (ssh)"}, "device"), host, h("button", {class: "link", onclick: list}, "list routes")), out);
}

function localPanel(close) {
  const intro = h("div", {class: "local-intro muted", title: Object.entries(daemon.caps.modules).map(([k, v]) => `${k} ${v.version}`).join(" · ")},
    "serve.py on this machine keeps downloaded logs and videos in a disk cache shared by every tab. ",
    h("a", {href: "/local", onclick: (e) => { close(); link("/local")(e); }}, "Open local logs…"));
  const cache = h("div", {class: "local-cache"});
  const here = h("div", {class: "local-route"});
  const list = h("div", {class: "local-jobs"});
  const dev = daemon.has("device") ? devicePart() : null;
  const closeBtn = h("button", {class: "link", onclick: () => close()}, "close");
  const el = h("div", {class: "local-panel", role: "dialog", "aria-label": "Local daemon"},
    h("div", {class: "local-head"}, h("b", {}, "Local daemon"), closeBtn),
    intro, cache, here, dev, list);
  const route = openRoute();

  const act = (label, title, fn, disabled = false) => h("button", {class: "link", title, disabled, onclick: async (e) => {
    e.target.disabled = true;
    const error = await fn().then(() => null, (err) => err);
    e.target.disabled = false;
    await stats();  // after the refresh, which replaces the route's lines
    if (error) here.append(h("div", {class: "local-err"}, String(error.message || error)));
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
    const logs = count(["rlog", "qlog"]);
    here.replaceChildren(
      h("div", {}, `this route: logs ${logs}/${n} · video ${count(KINDS.video)}/${n} segments · ${fmtBytes(bytes)}`),
      h("div", {class: "local-acts"},
        act("cache logs", "Download this route's logs to disk", () => daemon.json("/local/cache/prefetch", {method: "POST", body: {route, kinds: KINDS.logs}}), logs === n),
        act("cache video", "Download its videos", () => daemon.json("/local/cache/prefetch", {method: "POST", body: {route, kinds: KINDS.video}})),
        act("evict", "Remove this route from the disk cache", () => daemon.json("/local/cache/evict", {method: "POST", body: {route}}))));
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

  let last = 0;  // the cache's stats, asked again at most once a second
  return {el, focus: closeBtn, update() { update(); if (performance.now() - last > 1000) { last = performance.now(); stats(); } }};
}
