// user mode's Files and More info menus, as connect's Media.jsx draws them over the button: the playhead segment's
// files to download or ask the device for, all logs or files of the range, the upload queue (Files/UploadQueue.jsx),
// the segment name to copy, sharing, useradmin, and the public and preserved switches. Loaded with the first click.
import {api, memo, segmentAtTime} from "/api.js";
import {h, css, openModal} from "/dom.js";
import {isOnline} from "/app.js";
import {canManage} from "/pages/home.js";
import {API_KIND} from "/data/sources.js";
import {uploads, keysIn, urlKey, parseUploadUrl} from "/pages/files.js";
import {cnIcon} from "/pages/connect/icons.js";
import {progress} from "/pages/cnui.js";

await css(new URL("./connect-menus.css", import.meta.url));

const USERADMIN = "https://useradmin.comma.ai/";
const PUBLIC_TIP = "Making a route public allows anyone with the route name or link to access it.";
const PRESERVED_TIP = "Preserving a route will prevent it from being deleted. You can preserve up to 10 routes, or 100 if you have comma prime.";
const DCAM_TIP = "make sure to enable the “Record and Upload Driver Camera” toggle";
// connect's rows: its name for each kind and ours
const FILES = [["Road camera", "fcamera"], ["Wide road camera", "ecamera"], ["Driver camera", "dcamera"], ["Log data", "rlog"]];
const CAMERAS = ["fcamera", "ecamera", "dcamera"];

// MUI's CircularProgress, white
const spinner = (size) => progress(size, "#fff");

// a MUI menu: a transparent backdrop that closes it, and the paper growing in where place(anchor, paper) puts it,
// kept 16 px inside the window. Returns {paper, list, close}
function menu(anchor, items, place, onClose) {
  const list = h("ul", {class: "cm-list", role: "menu"}, items);
  const paper = h("div", {class: "cm-paper", tabindex: -1}, list);
  const el = h("div", {class: "cm-modal", onkeydown: (e) => e.key === "Escape" && close()}, h("div", {class: "cm-backdrop", onclick: () => close()}), paper);
  document.body.append(el);
  // read once as it opens, a click away from any render
  const a = anchor.getBoundingClientRect(), r = {width: paper.offsetWidth, height: paper.offsetHeight};  // unscaled: it grows in
  const [x, y, ox, oy] = place(a, r);
  const left = Math.max(16, Math.min(innerWidth - 16 - r.width, x)), top = Math.max(16, Math.min(innerHeight - 16 - r.height, y));
  Object.assign(paper.style, {left: `${left}px`, top: `${top}px`, transformOrigin: `${ox + x - left}px ${oy + y - top}px`});
  paper.classList.add("open");
  (list.querySelector("li[tabindex='0']") || paper).focus({preventScroll: true});
  let closed = false;
  function close() {
    if (closed) return;
    closed = true;
    onClose?.();
    paper.classList.remove("open");
    el.classList.add("closing");
    setTimeout(() => el.remove(), 300);
  }
  return {paper, list, close};
}
const item = (attrs, ...children) => h("li", {class: "cm-item", role: "menuitem", ...attrs}, children);
const button = (text, onclick, disabled = false) => h("button", {class: "cm-upload", disabled, onclick}, text);
const fake = (...children) => h("div", {class: "cm-fake"}, children);

// Files: opened over the button's label, from its top right, as connect's menu-download is
export function filesMenu(anchor, ctx) {
  const {route, clock} = ctx;
  const [dongle] = route.fullname.split("|");
  let files = null, device = null, up = null, alive = true, kept = false;  // files: "<n>/<kind>" → url, once fetched
  const watchers = new Set([() => draw()]);  // what the uploads' changes redraw: this menu, then the queue it opened
  const n = segmentAtTime(route, clock.t, clock.duration);
  const rows = FILES.map(([name]) => item({"aria-disabled": "true"}, h("span", {}, name)));
  const allLogs = item({"aria-disabled": "true"}, h("span", {}, "All logs"));
  const allFiles = item({"aria-disabled": "true"}, h("span", {}, "All files"));
  const foot = h("div", {class: "cm-foot"});
  const loading = h("div", {class: "cm-loading"}, spinner(36));
  // opened loading, and placed as it is then: connect places it before its file list arrives, so it grows from there,
  // with its spinner's middle (MUI's content anchor, 21 px below the list's) level with the label's
  draw();
  const m = menu(anchor, [loading, ...rows, h("hr"), allLogs, allFiles, h("hr"), foot],
    (a, r) => [a.right - r.width, a.top + a.height / 2 - (r.height / 2 + 21), r.width, r.height / 2 + 21], () => { alive = false; if (!kept) up?.dispose(); });

  // what's in the zoomed range: connect counts the segments the loop overlaps
  const inRange = (kinds) => keysIn(route, clock.loop || [0, clock.duration], kinds);
  const stats = (keys) => {
    let uploaded = 0, uploading = 0, paused = 0, requested = 0;
    for (const k of keys) {
      const s = up?.state(k);
      uploaded += files.has(k) || !!s?.notFound;
      uploading += !!s?.queued;
      paused += !!s?.paused;
      requested += !!s?.requested;
    }
    const left = keys.length - uploaded - uploading;
    return {canRequest: left - requested, uploading: !left, uploaded: !(keys.length - uploaded), paused, active: uploading};
  };

  // what a file's row offers: its download, how its upload is going, or a request for it
  function action(key, kind) {
    const s = up?.state(key);
    if (files.has(key)) return button("download", () => { location.href = files.get(key); });
    if (s?.queued) return fake(s.current ? `${Math.floor(s.progress * 100)}%` : s.paused ? "paused" : "pending");
    if (s?.requested) return fake(spinner(17));
    if (s?.notFound && kind === "dcamera") return tip(fake("not found", cnIcon("info", 16)), DCAM_TIP);
    if (s?.notFound) return fake("not found");
    if (s?.error) return fake("failed");
    if (!up?.canUpload) return button("download", null, true);
    return button(innerWidth < 425 ? "upload" : "request upload", () => up.request([key]));
  }

  function row(li, key, kind) {
    li.classList.toggle("cm-wait", !files);
    li.replaceChildren(li.firstChild, ...files ? [action(key, kind)] : []);
  }

  function all(li, kinds, what) {
    li.classList.toggle("cm-wait", !files);
    if (!files || !up?.canUpload) return li.replaceChildren(li.firstChild);
    const keys = inRange(kinds), st = stats(keys), off = st.uploaded || st.uploading || !st.canRequest;
    li.replaceChildren(li.firstChild, off ? fake(st.uploaded ? "uploaded" : st.uploading ? "pending" : spinner(17))
      : button(`upload ${st.canRequest} ${what}`, () => up.request(keys)));
  }

  function draw() {
    if (!alive) return;
    loading.hidden = !!files;
    FILES.forEach(([, kind], i) => row(rows[i], `${n}/${kind}`, kind));
    all(allLogs, ["rlog"], "logs");
    all(allFiles, ["rlog", ...CAMERAS], "files");
    const st = files && up ? stats(inRange(["rlog", ...CAMERAS])) : null;
    foot.replaceChildren(...[
      device && !isOnline(device) && files
        ? item({class: "cm-item cm-note", "aria-disabled": "true"}, h("div", {}, cnIcon("warning"), "Device offline"), h("span", {}, "uploading will resume when device is online"))
        : item({class: `cm-item${files ? "" : " cm-wait"}`, tabindex: files ? 0 : -1, "aria-disabled": String(!files), onclick: () => {
            if (!files || !up) return;
            kept = true;  // the queue's now: it ends the uploads' polling as it closes
            m.close();
            watchers.clear();
            queueDialog(up, dongle, watchers);
          }}, "View upload queue"),
      st && st.paused && st.paused === st.active && up.metered
        && item({class: "cm-item cm-note", "aria-disabled": "true"}, h("div", {}, cnIcon("warning"), "Connect to WiFi"), h("span", {}, "uploading paused on cellular connection")),
    ].filter(Boolean));
  }

  // as connect, the file list is fetched afresh each time the menu opens
  api.routeFiles.forget?.(route.fullname);
  Promise.all([api.routeFiles(route.fullname).catch(() => ({})), api.device(dongle).catch(() => null)]).then(([list, d]) => {
    if (!alive) return;
    files = new Map(Object.entries(API_KIND).flatMap(([kind, apiKind]) => (list[apiKind] || []).map((u) => [urlKey(u, kind), u])));
    device = d;
    up = uploads(route, d, (k) => files.has(k), () => { for (const fn of watchers) fn(); });
    draw();
  });
}

// whether the drive is preserved: connect asks for the device's preserved routes (rather than trusting the record) as
// the drive opens, for those who manage the device, so More info opens knowing
const isPreserved = memo((fullname) => api.preservedRoutes(fullname.split("|")[0]).then((list) => Array.isArray(list) && list.some((r) => r.fullname === fullname)));
export const prefetch = ({route}) => api.device(route.fullname.split("|")[0]).then(canManage).then((ok) => ok && isPreserved(route.fullname)).catch(() => {});

// More info: opened from the label's left, 260 px (300 on a phone) to its left, as connect's menu-info is
export function infoMenu(anchor, ctx) {
  const {route, clock} = ctx;
  const [dongle] = route.fullname.split("|");
  const name = `${route.fullname.replace("|", "/")}/${segmentAtTime(route, clock.t, clock.duration)}`;
  const toggles = h("div", {class: "cm-toggles"});
  const m = menu(anchor, [
    item({class: "cm-item cm-copy", tabindex: 0, style: `font-size: ${innerWidth > 400 ? 12.8 : 11.2}px`,
      onclick: async () => { await navigator.clipboard?.writeText(name); m.close(); }}, h("div", {}, name), cnIcon("copy")),
    navigator.share && item({class: "cm-item cm-share", tabindex: 0, onclick: () => navigator.share({title: "comma connect", url: location.href}).catch(() => {})},
      "Share this route", cnIcon("share")),
    h("hr"),
    item({tabindex: 0, onclick: () => window.open(`${USERADMIN}?onebox=${encodeURIComponent(route.fullname)}`, "_blank")?.focus()}, "View in useradmin"),
    toggles,
  ], (a) => { const ox = innerWidth > 400 ? 260 : 300; return [a.left - ox, a.top + a.height / 2 - 32, ox, 32]; });

  api.device(dongle).then(async (device) => {
    if (!await canManage(device)) return;
    const preserved = switchRow("Preserved", PRESERVED_TIP, null, (v) => api.setRoutePreserved(route.fullname, v).then((r) => {
      if (!r?.success) throw new Error("unable to update");
      isPreserved.prime(v, route.fullname);
    }));
    toggles.replaceChildren(h("hr"),
      switchRow("Public access", PUBLIC_TIP, !!route.is_public, async (v) => {
        const r = await api.setRoutePublic(route.fullname, v);
        if (r?.fullname === route.fullname) route.is_public = r.is_public;
        if (r && r.is_public !== v) throw new Error("unable to update");
      }).el,
      preserved.el);
    isPreserved(route.fullname).then(preserved.set, () => {});
  }).catch(() => {});
}

// a MUI switch with its label, info tooltip, and the error it failed with. checked null: loading
function switchRow(label, about, checked, save) {
  const input = h("input", {type: "checkbox", class: "cm-switch-input"});
  const sw = h("span", {class: "cm-switch"}, input, h("span", {class: "cm-thumb"}), h("span", {class: "cm-track"}));
  const info = cnIcon("info", 18);
  info.classList.add("cm-info");
  tip(info, about, true);
  const errorEl = h("span", {class: "cm-error", hidden: true}, cnIcon("error"));
  const el = h("li", {class: "cm-item cm-toggle"}, h("label", {}, sw, h("span", {}, label)), info, errorEl);
  let busy = false;
  const set = (v) => {
    checked = v;
    input.checked = !!v;
    sw.classList.toggle("loading", v === null);
    input.disabled = v === null;
  };
  input.addEventListener("change", async () => {
    if (busy) return void (input.checked = !!checked);
    const v = input.checked;
    busy = true;
    sw.classList.add("loading");
    errorEl.hidden = true;
    try {
      await save(v);
      checked = v;
    } catch (err) {
      input.checked = !!checked;
      errorEl.hidden = false;
      tip(errorEl, err.status ? "could not update" : err.message || "could not update");
    }
    busy = false;
    sw.classList.remove("loading");
  });
  set(checked);
  return {el, set};
}

// a tooltip over el, which it returns: connect's grey bubble, with an arrow above (info) or a plain popper below (hover)
function tip(el, text, above = false) {
  let pop = null;
  const show = () => {
    if (pop) return;
    pop = h("div", {class: `cm-tip${above ? " above" : ""}`, role: "tooltip"}, text);
    document.body.append(pop);
    const r = el.getBoundingClientRect(), p = pop.getBoundingClientRect();
    const x = Math.max(8, Math.min(innerWidth - 8 - p.width, r.left + r.width / 2 - p.width / 2));
    pop.style.left = `${x}px`;
    pop.style.top = `${above ? r.top - p.height - 8 : r.bottom + 12}px`;
    pop.style.setProperty("--arrow", `${r.left + r.width / 2 - x}px`);
  };
  const hide = () => { pop?.remove(); pop = null; };
  el.addEventListener("pointerenter", show);
  el.addEventListener("pointerleave", hide);
  el.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); if (pop) hide(); else show(); });
  return el;
}

// connect's upload queue dialog, over a route's uploads: it owns them from here, closing it ends their polling
function queueDialog(up, dongle, watchers) {
  const body = h("div", {class: "cm-queue-body"});
  const cancelAll = h("button", {class: "cm-dbutton", onclick: () => up.cancel(up.waiting())}, "Cancel All");
  const dialog = openModal({class: "cm-dialog", "aria-labelledby": "upload-queue-modal", onclose: () => { watchers.clear(); up.dispose(); }},
    h("div", {class: "cm-dialog-title"}, h("span", {class: "cm-dialog-h", id: "upload-queue-modal"}, "Upload queue"), h("span", {class: "cm-caption"}, dongle)),
    h("hr"), body,
    h("div", {class: "cm-dialog-buttons"}, cancelAll, h("button", {class: "cm-dbutton", onclick: () => dialog.close()}, "Close")));
  function draw() {
    cancelAll.disabled = !up.cancel || !up.waiting().length;
    body.replaceChildren(queueTable(up));
  }
  watchers.add(draw);
  draw();
}

// connect's upload queue table (Files/UploadQueue.jsx), for its dialog here and settings': the device's queue (this
// drive's and others', pages/files.js deviceQueue) with the one uploading first, its progress, and a cancel for each
// of the rest
export function queueTable(q) {
  const narrow = innerWidth < 600;
  if (q.offline()) return h("p", {}, "device offline");
  if (q.error) return h("p", {}, "unable to load the upload queue");
  if (!q.loaded) return spinner(17);
  if (!q.queue.length) return h("p", {}, "no uploads");
  return h("table", {class: "cm-queue"},
    h("thead", {}, h("tr", {}, h("th", {}, "segment"), h("th", {}, "type"), h("th", {}, "progress"), !narrow && h("th"))),
    h("tbody", {}, [...q.queue].sort((a, b) => b.current - a.current).map((u) => {
      const {route, n, file} = parseUploadUrl(u.url), seg = `${route}--${n}`;
      return h("tr", {},
        h("td", {}, h("div", {class: "cm-seg"}, h("span", {}, seg.slice(0, 12)), h("span", {}, seg.slice(12)))),
        h("td", {}, file.split(".")[0].slice(0, narrow ? 4 : 64)),
        u.current
          ? h("td", {}, h("div", {class: "cm-progress"}, h("div", {style: `transform: translateX(${u.progress * 100 - 100}%)`})))
          : [!narrow && h("td", {}, q.metered && !u.allow_cellular ? "paused" : "pending"),
            h("td", {class: "cm-cancel"}, q.cancelling.has(u.id) ? spinner(15)
              : q.cancel && h("button", {"aria-label": "Cancel upload", onclick: () => q.cancel([u.id])}, cnIcon("cancel", 18)))]);
    })));
}
