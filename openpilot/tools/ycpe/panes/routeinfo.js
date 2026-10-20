// route info pane: what the route is, its public/preserved toggles, link sharing and bookmarks (files have their own pane)
import {api, routeEvents} from "../api.js";
import {h, fmt, css} from "../dom.js";
import {canManage} from "../pages/home.js";
import {schemaRow} from "./schema.js";

await css(new URL("./routeinfo.css", import.meta.url));

function toggle(label, value, enabled, set) {
  const box = h("input", {type: "checkbox", checked: value, disabled: !enabled});
  const status = h("span", {class: "small error"});
  box.addEventListener("change", async () => {
    box.disabled = true;
    status.textContent = "";
    try {
      await set(box.checked);
    } catch (err) {
      box.checked = !box.checked;
      status.textContent = err.status === 403 ? "not allowed" : err.message;
    }
    box.disabled = false;
  });
  return h("label", {class: "check"}, box, label, status);
}

export function create(ctx) {
  const {route, clock} = ctx;
  const [dongleId, routeId] = route.fullname.split("|");
  const start = new Date(route.start_time_utc_millis);
  const commit = route.git_commit?.slice(0, 8);
  const github = route.git_remote?.includes("github.com") && commit ? `https://${route.git_remote.replace(/\.git$/, "")}/commit/${route.git_commit}` : null;
  const schema = schemaRow(ctx);
  const rows = [
    ["route", h("span", {class: "mono"}, route.fullname)],
    ["start", `${start.toLocaleString()} (${fmt.duration(route.end_time_utc_millis - route.start_time_utc_millis)})`],
    ["car", [route.platform, route.make && `(${route.make})`].filter(Boolean).join(" ")],
    ["openpilot", route.version],
    ["branch", route.git_branch && [route.git_branch, " @ ", github ? h("a", {href: github, target: "_blank", rel: "noopener"}, commit) : commit, route.git_dirty ? " (dirty)" : ""]],
    ["remote", route.git_remote],
    ["schema", schema.el],
    ["segments", `${(route.maxqlog ?? -1) + 1}, ${fmt.miles(route.distance ?? 0)}`],
  ];

  const copied = h("span", {class: "small muted"});
  const shareUrl = () => `${location.origin}/${dongleId}/${routeId}/drive${clock.t ? `?t=${clock.t.toFixed(1)}` : ""}`;
  const copy = async (text, what) => {
    await navigator.clipboard.writeText(text);
    copied.textContent = `copied ${what}`;
    setTimeout(() => { copied.textContent = ""; }, 2000);
  };
  const toggles = h("span", {class: "toggles"});
  const bookmarks = h("div", {class: "bookmarks"});
  const device = api.device(dongleId).catch(() => null);
  device.then(async (d) => {
    const owner = !!d && await canManage(d);
    toggles.replaceChildren(
      toggle("public", route.is_public, owner, async (v) => { await api.setRoutePublic(route.fullname, v); route.is_public = v; }),
      toggle("preserved", route.is_preserved, owner, async (v) => { await api.setRoutePreserved(route.fullname, v); route.is_preserved = v; }));
  });
  routeEvents(route).then((spans) => bookmarks.replaceChildren(...spans.filter((s) => s.kind === "bookmark").map((s) =>
    h("button", {class: "chip", title: "Seek to bookmark", onclick: () => clock.seek(s.start / 1000)}, `bookmark ${fmt.clock(s.start / 1000)}`)))).catch(() => {});

  const el = h("div", {class: "info-pane"},
    h("table", {class: "kv-table"}, rows.filter(([, v]) => v).map(([k, v]) => h("tr", {}, h("th", {}, k), h("td", {}, v)))),
    h("div", {class: "actions"},
      h("button", {class: "button small", title: "Link to this route at the playhead", onclick: () => copy(shareUrl(), "link")}, "Copy link"),
      h("button", {class: "button small", onclick: () => copy(route.fullname, "route name")}, "Copy name"),
      navigator.share && h("button", {class: "button small", onclick: () => navigator.share({title: route.fullname, url: shareUrl()}).catch(() => {})}, "Share"),
      toggles, copied),
    bookmarks);
  return {name: "info", el, render() {}, resize() {}, dispose: schema.dispose, save: () => ({})};
}
