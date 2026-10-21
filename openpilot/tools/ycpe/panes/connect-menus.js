// user mode's Files and More info menus, as connect's Media.jsx draws them over the button: the playhead segment's
// files to download or ask the device for, all logs or files of the range, the upload queue (Files/UploadQueue.jsx),
// the segment name to copy, sharing, useradmin, and the public and preserved switches. Loaded with the first click.
import {SEGMENT, api, segmentAtTime} from "/api.js";
import {h, css, openModal} from "/dom.js";
import {isOnline} from "/app.js";
import {canManage} from "/pages/home.js";
import {API_KIND} from "/data/sources.js";
import {uploads} from "/pages/files.js";
import {cnIcon} from "./connect.js";

await css(new URL("./connect-menus.css", import.meta.url));

const USERADMIN = "https://useradmin.comma.ai/";
const PUBLIC_TIP = "Making a route public allows anyone with the route name or link to access it.";
const PRESERVED_TIP = "Preserving a route will prevent it from being deleted. You can preserve up to 10 routes, or 100 if you have comma prime.";
const DCAM_TIP = "make sure to enable the “Record and Upload Driver Camera” toggle";
// connect's rows: its name for each kind and ours
const FILES = [["Road camera", "fcamera"], ["Wide road camera", "ecamera"], ["Driver camera", "dcamera"], ["Log data", "rlog"]];
const CAMERAS = ["fcamera", "ecamera", "dcamera"];

// the material icons connect draws in these menus (its src/icons, MIT): [viewBox, path]
const ICONS = {
  copy: ["0 -960 960 960", "M180-81q-24 0-42-18t-18-42v-603h60v603h474v60H180Zm120-120q-24 0-42-18t-18-42v-560q0-24 18-42t42-18h440q24 0 42 18t18 42v560q0 24-18 42t-42 18H300Zm0-60h440v-560H300v560Zm0 0v-560 560Z"],
  share: ["0 0 24 24", "M18 16.08c-.76 0-1.44.3-1.96.77L8.91 12.7c.05-.23.09-.46.09-.7s-.04-.47-.09-.7l7.05-4.11c.54.5 1.25.81 2.04.81 1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3c0 .24.04.47.09.7L8.04 9.81C7.5 9.31 6.79 9 6 9c-1.66 0-3 1.34-3 3s1.34 3 3 3c.79 0 1.5-.31 2.04-.81l7.12 4.16c-.05.21-.08.43-.08.65 0 1.61 1.31 2.92 2.92 2.92 1.61 0 2.92-1.31 2.92-2.92s-1.31-2.92-2.92-2.92z"],
  warning: ["0 0 24 24", "M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"],
  info: ["0 -960 960 960", "M453-280h60v-240h-60v240Zm27-314q14 0 23-9t10-23q0-14-9-24-10-10-24-10t-23 10-10 24q0 14 9 23 10 9 24 9Zm0 514q-82 0-155-31t-128-86q-54-55-86-128-31-73-31-155 0-83 32-156 31-73 86-127t127-85q73-32 156-32 82 0 155 32 73 31 127 85t86 127q31 73 31 156 0 82-31 155t-86 127q-54 55-127 86-73 32-156 32Zm0-60q142 0 241-99t99-241q0-142-99-241t-241-99q-141 0-240 99-100 99-100 241 0 141 100 241t241 99Zm0-340Z"],
  error: ["0 -960 960 960", "M480-280q14 0 24-9 9-10 9-24t-9-23q-10-10-24-10t-23 9q-10 10-10 24t9 23q10 10 24 10Zm-27-153h60v-253h-60v253Zm27 353q-82 0-155-31t-127-86q-55-55-86-128-32-73-32-155 0-83 32-156 31-73 86-127t127-85q73-32 156-32 82 0 155 32 73 31 127 85t86 127q31 73 31 156 0 82-31 155t-86 127q-54 55-127 86-73 32-156 32Zm1-60q141 0 240-99t99-241q0-142-99-241t-241-99q-141 0-240 99-100 99-100 241 0 141 100 241t241 99Zm-1-340Z"],
  cancel: ["0 0 24 24", "M14.59 8 12 10.59 9.41 8 8 9.41 10.59 12 8 14.59 9.41 16 12 13.41 14.59 16 16 14.59 13.41 12 16 9.41 14.59 8zM12 2C6.47 2 2 6.47 2 12s4.47 10 10 10 10-4.47 10-10S17.53 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"],
};
function glyph(name, size = 24) {
  const svg = cnIcon("play", size), [box, d] = ICONS[name];
  svg.setAttribute("viewBox", box);
  svg.firstChild.setAttribute("d", d);
  return svg;
}
// MUI's CircularProgress, white
const spinner = (size) => h("span", {class: "cm-spinner", style: `width:${size}px;height:${size}px`});

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
  const segs = route.segment_numbers || Array.from({length: (route.maxqlog ?? -1) + 1}, (_, i) => i);
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
  const inRange = (kinds) => {
    const [r0, r1] = clock.loop || [0, clock.duration];
    return segs.filter((s, i) => { const slot = route.range ? i : s; return slot * SEGMENT < r1 && (slot + 1) * SEGMENT > r0; })
      .flatMap((s) => kinds.map((kind) => `${s}/${kind}`));
  };
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

  function row(li, key, kind) {
    const narrow = innerWidth < 425, s = up?.state(key);
    let b = null;
    if (!files) b = null;
    else if (files.has(key)) b = button("download", () => { location.href = files.get(key); });
    else if (s?.queued) b = fake(s.current ? `${Math.floor(s.progress * 100)}%` : s.paused ? "paused" : "pending");
    else if (s?.requested) b = fake(spinner(17));
    else if (s?.notFound) {
      b = fake("not found", kind === "dcamera" && glyph("info", 16));
      if (kind === "dcamera") tip(b, DCAM_TIP);
    } else if (s?.error) b = fake("failed");
    else if (!up?.canUpload) b = button("download", null, true);
    else b = button(narrow ? "upload" : "request upload", () => up.request([key]));
    li.classList.toggle("cm-wait", !files);
    li.replaceChildren(li.firstChild, ...[b].filter(Boolean));
  }

  function all(li, kinds, what) {
    li.classList.toggle("cm-wait", !files);
    if (!files || !up) return li.replaceChildren(li.firstChild);
    const keys = inRange(kinds), st = stats(keys), off = st.uploaded || st.uploading || !st.canRequest;
    const b = !up.canUpload ? null : !off ? button(`upload ${st.canRequest} ${what}`, () => up.request(keys))
      : fake(st.uploaded ? "uploaded" : st.uploading ? "pending" : spinner(17));
    li.replaceChildren(li.firstChild, ...[b].filter(Boolean));
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
        ? item({class: "cm-item cm-note", "aria-disabled": "true"}, h("div", {}, glyph("warning"), "Device offline"), h("span", {}, "uploading will resume when device is online"))
        : item({class: `cm-item${files ? "" : " cm-wait"}`, tabindex: files ? 0 : -1, "aria-disabled": String(!files), onclick: () => {
            if (!files || !up) return;
            kept = true;  // the queue's now: it ends the uploads' polling as it closes
            m.close();
            watchers.clear();
            queueDialog(up, dongle, watchers);
          }}, "View upload queue"),
      st && st.paused && st.paused === st.active && up.metered
        && item({class: "cm-item cm-note", "aria-disabled": "true"}, h("div", {}, glyph("warning"), "Connect to WiFi"), h("span", {}, "uploading paused on cellular connection")),
    ].filter(Boolean));
  }

  // as connect, the file list is fetched afresh each time the menu opens
  api.routeFiles.forget?.(route.fullname);
  Promise.all([api.routeFiles(route.fullname).catch(() => ({})), api.device(dongle).catch(() => null)]).then(([list, d]) => {
    if (!alive) return;
    files = new Map(Object.entries(API_KIND).flatMap(([kind, apiKind]) => (list[apiKind] || []).map((u) => [`${new URL(u).pathname.split("/").at(-2)}/${kind}`, u])));
    device = d;
    up = uploads(route, d, (k) => files.has(k), () => { for (const fn of watchers) fn(); });
    draw();
  });
}

// whether the drive is preserved: connect asks for the device's preserved routes (rather than trusting the record) as
// the drive opens, for those who manage the device, so More info opens knowing
const preservedOf = new Map();
const isPreserved = ({route}) => {
  if (!preservedOf.has(route.fullname)) {
    preservedOf.set(route.fullname, api.preservedRoutes(route.fullname.split("|")[0]).then((list) => Array.isArray(list) && list.some((r) => r.fullname === route.fullname))
      .catch((e) => { preservedOf.delete(route.fullname); throw e; }));
  }
  return preservedOf.get(route.fullname);
};
export const prefetch = (ctx) => api.device(ctx.route.fullname.split("|")[0]).then(canManage).then((ok) => ok && isPreserved(ctx)).catch(() => {});

// More info: opened from the label's left, 260 px (300 on a phone) to its left, as connect's menu-info is
export function infoMenu(anchor, ctx) {
  const {route, clock} = ctx;
  const [dongle] = route.fullname.split("|");
  const name = `${route.fullname.replace("|", "/")}/${segmentAtTime(route, clock.t, clock.duration)}`;
  const toggles = h("div", {class: "cm-toggles"});
  const m = menu(anchor, [
    item({class: "cm-item cm-copy", tabindex: 0, style: `font-size: ${innerWidth > 400 ? 12.8 : 11.2}px`,
      onclick: async () => { await navigator.clipboard?.writeText(name); m.close(); }}, h("div", {}, name), glyph("copy")),
    navigator.share && item({class: "cm-item cm-share", tabindex: 0, onclick: () => navigator.share({title: "comma connect", url: location.href}).catch(() => {})},
      "Share this route", glyph("share")),
    h("hr"),
    item({tabindex: 0, onclick: () => window.open(`${USERADMIN}?onebox=${encodeURIComponent(route.fullname)}`, "_blank")?.focus()}, "View in useradmin"),
    toggles,
  ], (a) => { const ox = innerWidth > 400 ? 260 : 300; return [a.left - ox, a.top + a.height / 2 - 32, ox, 32]; });

  api.device(dongle).then(async (device) => {
    if (!await canManage(device)) return;
    const preserved = switchRow("Preserved", PRESERVED_TIP, null, (v) => api.setRoutePreserved(route.fullname, v).then((r) => {
      if (!r?.success) throw new Error("unable to update");
      preservedOf.set(route.fullname, Promise.resolve(v));
    }));
    toggles.replaceChildren(h("hr"),
      switchRow("Public access", PUBLIC_TIP, !!route.is_public, async (v) => {
        const r = await api.setRoutePublic(route.fullname, v);
        if (r?.fullname === route.fullname) route.is_public = r.is_public;
        if (r && r.is_public !== v) throw new Error("unable to update");
      }).el,
      preserved.el);
    isPreserved(ctx).then(preserved.set, () => {});
  }).catch(() => {});
  void m;
}

// a MUI switch with its label, info tooltip, and the error it failed with. checked null: loading
function switchRow(label, about, checked, save) {
  const input = h("input", {type: "checkbox", class: "cm-switch-input"});
  const sw = h("span", {class: "cm-switch"}, input, h("span", {class: "cm-thumb"}), h("span", {class: "cm-track"}));
  const info = glyph("info", 18);
  info.classList.add("cm-info");
  tip(info, about, true);
  const errorEl = h("span", {class: "cm-error", hidden: true}, glyph("error"));
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

// a tooltip over el: connect's grey bubble, with an arrow above (info) or a plain popper below (hover)
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
}

// connect's upload queue dialog: the device's queue (this drive's and others'), progress, cancel one or all. It owns
// the uploads from here: closing it ends their polling
function queueDialog(up, dongle, watchers) {
  const body = h("div", {class: "cm-queue-body", style: `max-height: ${innerHeight * 0.9 - 98}px`});
  const cancelAll = h("button", {class: "cm-dbutton", onclick: () => up.cancel?.(up.queue.filter((q) => !q.current).map((q) => q.id))}, "Cancel All");
  const dialog = openModal({class: "cm-dialog", "aria-labelledby": "upload-queue-modal", onclose: () => { watchers.clear(); up.dispose(); }},
    h("div", {class: "cm-dialog-title"}, h("span", {class: "cm-dialog-h", id: "upload-queue-modal"}, "Upload queue"), h("span", {class: "cm-caption"}, dongle)),
    h("hr"), body,
    h("div", {class: "cm-dialog-buttons"}, cancelAll, h("button", {class: "cm-dbutton", onclick: () => dialog.close()}, "Close")));
  const empty = up.queue;  // replaced by the first answer
  function draw() {
    const narrow = innerWidth < 600, offline = up.offline(), queue = offline ? [] : up.queue;
    cancelAll.disabled = !queue.some((q) => !q.current);
    if (!queue.length) return body.replaceChildren(offline ? h("p", {}, "device offline") : up.queue !== empty ? h("p", {}, "no uploads") : spinner(17));
    body.replaceChildren(h("table", {class: "cm-queue"},
      h("thead", {}, h("tr", {}, h("th", {}, "segment"), h("th", {}, "type"), h("th", {}, "progress"), !narrow && h("th"))),
      h("tbody", {}, [...queue].sort((a, b) => b.current - a.current).map((q) => {
        const [route, n, file] = new URL(q.url).pathname.split("/").slice(-3), seg = `${route}--${n}`;
        return h("tr", {},
          h("td", {}, h("div", {class: "cm-seg"}, h("span", {}, seg.slice(0, 12)), h("span", {}, seg.slice(12)))),
          h("td", {}, file.split(".")[0].slice(0, narrow ? 4 : 64)),
          q.current
            ? h("td", {}, h("div", {class: "cm-progress"}, h("div", {style: `transform: translateX(${q.progress * 100 - 100}%)`})))
            : [!narrow && h("td", {}, up.metered && !q.allow_cellular ? "paused" : "pending"),
              h("td", {class: "cm-cancel"}, up.cancelling.has(q.id) ? spinner(15)
                : up.cancel && h("button", {"aria-label": "Cancel upload", onclick: () => up.cancel([q.id])}, glyph("cancel", 18)))]);
      }))));
  }
  watchers.add(draw);
  draw();
}
