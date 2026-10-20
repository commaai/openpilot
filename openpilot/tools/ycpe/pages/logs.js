// boot and crash logs: the signed file list, newest first. Crash logs open in a text dialog; boot logs (capnp) download.
import {api} from "../api.js";
import {h} from "../dom.js";
import {icon} from "../icons.js";

const PAGE = 50;

// "2026-08-16--02-07-04_e8fd3854_openpilot_system_loggerd_loggerd" → when, commit, executable
function crashName(file) {
  const m = file.match(/^(\d{4}-\d\d-\d\d)--(\d\d)-(\d\d)-(\d\d)_([0-9a-f]+)_(.*)$/);
  return m ? [`${m[1]} ${m[2]}:${m[3]}:${m[4]}`, m[5], m[6].replaceAll("_", "/")] : [file, "", ""];
}

function openCrash(url, name) {
  const pre = h("pre", {class: "log-text"}, "Loading…");
  fetch(url).then((r) => r.ok ? r.text() : Promise.reject(new Error(`HTTP ${r.status}`)))
    .then((text) => { pre.textContent = text; }, (err) => { pre.textContent = String(err); });
  const dialog = h("dialog", {class: "log-dialog", onclose: () => dialog.remove()},
    h("div", {class: "log-head"},
      h("span", {class: "mono"}, name),
      h("a", {class: "icon-btn", href: url, download: name, "aria-label": "Download"}, icon("download", 20)),
      h("button", {class: "icon-btn", "aria-label": "Close", onclick: () => dialog.close()}, icon("close", 20))),
    pre);
  dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); });
  document.body.append(dialog);
  dialog.showModal();
}

export function logsSection(dongleId, kind) {
  const el = h("div", {class: "section"}, h("p", {class: "muted"}, "Loading…"));
  (kind === "boot" ? api.bootlogs : api.crashlogs)(dongleId).then((urls) => {
    if (!urls.length) return el.replaceChildren(h("p", {class: "muted"}, `No ${kind} logs.`));
    const files = urls.map((url) => ({url, file: new URL(url).pathname.split("/").at(-1)})).reverse();
    const table = h("table", {class: kind === "boot" ? "list" : "list crashes"}, kind === "boot"
      ? h("tr", {}, h("th", {}, "boot"), h("th"))
      : h("tr", {}, h("th", {}, "time"), h("th", {}, "commit"), h("th", {}, "executable"), h("th")));
    const more = h("button", {class: "button", onclick: show}, "Show more");
    let shown = 0;
    function show() {
      for (const {url, file} of files.slice(shown, shown + PAGE)) {
        const cells = kind === "boot" ? [file.replace(/\.zst$/, "")] : crashName(file);
        const dl = h("a", {class: "dl", href: url, download: file, "aria-label": "Download", onclick: (e) => e.stopPropagation()}, icon("download", 18));
        table.append(h("tr", {class: "clickable", onclick: () => kind === "boot" ? dl.click() : openCrash(url, file)},
          cells.map((c, i) => h("td", {class: i ? "mono muted" : "mono nowrap"}, c)),
          h("td", {class: "right"}, dl)));
      }
      shown += PAGE;
      more.hidden = shown >= files.length;
    }
    show();
    el.replaceChildren(h("p", {class: "muted small"}, `${files.length} ${kind} logs, newest first`), table, h("p", {class: "center"}, more));
  }).catch((err) => el.replaceChildren(h("p", {class: "error"}, err.status === 403 ? "Only the device owner can see its logs." : String(err))));
  return el;
}
