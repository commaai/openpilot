// logs that aren't in the API: /local opens files or a folder from this machine (or drop them anywhere), and a route
// that's only on the LAN device (the daemon's) can be pulled onto this machine to open
import {h} from "../dom.js";
import {daemon} from "../daemon.js";
import {onJobs} from "../local.js";
import {fmtBytes} from "../status.js";
import {parseRange} from "../data/sources.js";

export function localPage(open) {
  const picker = (label, attrs) => {
    const input = h("input", {type: "file", multiple: true, hidden: true, ...attrs, onchange: () => open([...input.files])});
    return h("label", {class: "button"}, label, input);
  };
  return h("div", {class: "page-msg"},
    h("h2", {}, "Open local logs"),
    h("p", {class: "muted"}, "rlog, qlog and camera files, or a folder of segments. You can also drop them anywhere on the page."),
    h("p", {}, picker("Choose files", {accept: ".zst,.bz2,.ts,.hevc"}), picker("Choose a folder", {webkitdirectory: true})));
}

// for a route no source has: a card to pull it from the daemon's device when the device has it, else null. The
// pull's progress also shows as a load-status lane; opened() runs once it's done.
export async function pullCard(name, opened) {
  if (!daemon.has("device")) return null;
  const sr = parseRange(name);
  const {dongle, routes} = await daemon.json("/local/device/routes").catch(() => ({routes: []}));
  const r = dongle === sr.dongle && routes.find((r) => r.log_id === sr.logId);
  if (!r) return null;
  const segs = Object.values(r.segments);
  const bytes = segs.reduce((sum, files) => sum + (files["rlog.zst"] ?? files["rlog.bz2"] ?? 0) + (files["qlog.zst"] ?? files["qlog.bz2"] ?? 0), 0);
  const state = h("p", {class: "muted"});
  const button = h("button", {class: "button primary", onclick: pull}, "Pull to open");

  async function pull() {
    button.disabled = true;
    state.textContent = "starting…";
    try {
      const {id} = await daemon.json("/local/device/pull", {method: "POST", body: {route: sr.route}});
      const off = onJobs((job) => {
        if (job.id !== id) return;
        if (job.state === "running") return void (state.textContent = `pulling ${job.total ? Math.round(100 * job.done / job.total) : 0}%`);
        off();
        if (job.state === "done") return opened();
        state.textContent = job.error || job.state;
        button.disabled = false;
      });
    } catch (e) {
      state.textContent = e.message;
      button.disabled = false;
    }
  }
  return h("div", {class: "page-msg"},
    h("h2", {}, "On the device"),
    h("p", {}, `${segs.length} segments · ${fmtBytes(bytes)}${r.active.length ? " · still recording" : ""}`),
    h("p", {}, button), state);
}
