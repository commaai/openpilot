// user mode's Files and More info menus, as connect's Media.jsx draws them over the button: the playhead segment's
// files to download or ask the device for, all logs or files of the range, the upload queue (pages/settings.js),
// the segment name to copy, sharing, useradmin, and the public and preserved switches. Loaded with the first click.
// The files are the workspace's model of them (files.js), which developer mode's Files panel draws too.
import {USERADMIN, api, segmentAtTime} from "/api.js";
import {deviceRecord} from "/devices.js";
import {h, css} from "/dom.js";
import {isOnline} from "/app.js";
import {canManage} from "/pages/home.js";
import {preservedNow, setPreserved} from "/pages/device.js";
import {keysIn} from "/pages/files.js";
import {CAMERA_NAMES} from "/data/sources.js";
import {uploadsModal} from "/pages/settings.js";
import {cnIcon} from "/pages/connect/icons.js";
import {cnMenu, cnTip, progress} from "/pages/cnui.js";

await css(new URL("./connect-menus.css", import.meta.url));

const PUBLIC_TIP = "Making a route public allows anyone with the route name or link to access it.";
const PRESERVED_TIP = "Preserving a route will prevent it from being deleted. You can preserve up to 10 routes, or 100 if you have comma prime.";
const DCAM_TIP = "make sure to enable the “Record and Upload Driver Camera” toggle";
const CAMERAS = ["fcamera", "ecamera", "dcamera"];
// connect's rows: its name for each kind and ours
const FILES = [...CAMERAS.map((kind) => [`${CAMERA_NAMES[kind].connect} camera`, kind]), ["Log data", "rlog"]];

// a MUI menu of items (cnui.js cnMenu), where place(anchor, paper) puts it
const menu = (anchor, items, place, onclose) => cnMenu(anchor, "cm-paper", h("ul", {class: "cm-list", role: "menu"}, items), place, {onclose});
const item = (attrs, ...children) => h("li", {class: "cm-item", role: "menuitem", ...attrs}, children);
const button = (text, onclick, disabled = false) => h("button", {class: "cm-upload", disabled, onclick}, text);
const fake = (...children) => h("div", {class: "cm-fake"}, children);

// Files: opened over the button's label, from its top right, as connect's menu-download is
export function filesMenu(anchor, ctx) {
  const {route, clock} = ctx;
  const [dongle] = route.fullname.split("|");
  let files = null, device = null, up = null, model = null, off = null, alive = true;  // files: "<n>/<kind>" → url, once fetched
  const n = segmentAtTime(route, clock.t, clock.duration);
  const rows = FILES.map(([name]) => item({"aria-disabled": "true"}, h("span", {}, name)));
  const allLogs = item({"aria-disabled": "true"}, h("span", {}, "All logs"));
  const allFiles = item({"aria-disabled": "true"}, h("span", {}, "All files"));
  const foot = h("div", {class: "cm-foot"});
  const loading = h("div", {class: "cm-loading"}, progress(36, "#fff"));
  // opened loading, and placed as it is then: connect places it before its file list arrives, so it grows from there,
  // with its spinner's middle (MUI's content anchor, 21 px below the list's) level with the label's
  draw();
  const m = menu(anchor, [loading, ...rows, h("hr"), allLogs, allFiles, h("hr"), foot],
    (a, r) => [a.right - r.width, a.top + a.height / 2 - (r.height / 2 + 21), r.width, r.height / 2 + 21], () => { alive = false; off?.(); });

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
    if (s?.requested) return fake(progress(17, "#fff"));
    if (s?.notFound && kind === "dcamera") return cnTip(fake("not found", cnIcon("info", 16)), DCAM_TIP);
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
    li.replaceChildren(li.firstChild, off ? fake(st.uploaded ? "uploaded" : st.uploading ? "pending" : progress(17, "#fff"))
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
            m.close();
            uploadsModal(up, dongle, (draw) => model.on(draw));
          }}, "View upload queue"),
      st && st.paused && st.paused === st.active && up.metered
        && item({class: "cm-item cm-note", "aria-disabled": "true"}, h("div", {}, cnIcon("warning"), "Connect to WiFi"), h("span", {}, "uploading paused on cellular connection")),
    ].filter(Boolean));
  }

  // as connect, the file list is fetched afresh each time the menu opens
  ctx.routeFiles().then(async (fm) => {
    await Promise.all([fm.refresh(), fm.ready]);
    if (!alive) return;
    model = fm;
    files = fm.uploaded;
    device = fm.device;
    up = fm.up;
    off = fm.on(draw);
    draw();
  });
}

// whether the drive is preserved, asked as the drive opens for those who manage the device, so More info opens knowing
export const prefetch = ({route}) => deviceRecord(route.fullname.split("|")[0]).then(canManage).then((ok) => ok && preservedNow(route.fullname)).catch(() => {});

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

  deviceRecord(dongle).then(async (device) => {
    if (!await canManage(device)) return;
    const preserved = switchRow("Preserved", PRESERVED_TIP, null, (v) => setPreserved(route, v));
    toggles.replaceChildren(h("hr"),
      switchRow("Public access", PUBLIC_TIP, !!route.is_public, async (v) => {
        const r = await api.setRoutePublic(route.fullname, v);
        if (r?.fullname === route.fullname) route.is_public = r.is_public;
        if (r && r.is_public !== v) throw new Error("unable to update");
      }).el,
      preserved.el);
    preservedNow(route.fullname).then(preserved.set, () => {});
  }).catch(() => {});
}

// a MUI switch with its label, info tooltip, and the error it failed with. checked null: loading
function switchRow(label, about, checked, save) {
  const input = h("input", {type: "checkbox", class: "cm-switch-input"});
  const sw = h("span", {class: "cm-switch"}, input, h("span", {class: "cm-thumb"}), h("span", {class: "cm-track"}));
  const info = cnIcon("info", 18);
  info.classList.add("cm-info");
  cnTip(info, about, "above");
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
      cnTip(errorEl, err.status ? "could not update" : err.message || "could not update");
    }
    busy = false;
    sw.classList.remove("loading");
  });
  set(checked);
  return {el, set};
}
