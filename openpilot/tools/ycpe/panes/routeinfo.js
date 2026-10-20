// route info pane: what the route is, its public/preserved toggles, link sharing and bookmarks (files have their own pane)
import {api, routeEvents} from "/api.js";
import {h, fmt, css} from "/dom.js";
import {parseRemote} from "/data/github.js";
import {canManage} from "/pages/home.js";
import {schemaRow} from "./schema.js";
import {shareUrl} from "/layouts/picker.js";

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
  const [dongleId] = route.fullname.split("|");
  const start = new Date(route.start_time_utc_millis);
  const commit = route.git_commit?.slice(0, 8);
  const repo = parseRemote(route.git_remote);  // the device says what its remote is: only a real GitHub repo is linked
  const github = repo && commit ? `https://github.com/${repo}/commit/${route.git_commit}` : null;
  const schema = schemaRow(ctx);
  const rows = [
    ["route", h("span", {class: "mono"}, route.fullname)],
    ["start", `${start.toLocaleString()} (${fmt.duration(route.end_time_utc_millis - route.start_time_utc_millis)})`],
    ["car", [route.platform, route.make && `(${route.make})`].filter(Boolean).join(" ")],
    ["openpilot", route.version],
    ["branch", route.git_branch && [route.git_branch, " @ ", github ? h("a", {href: github, target: "_blank", rel: "noopener"}, commit) : commit, route.git_dirty ? " (dirty)" : ""]],
    ["remote", route.git_remote],
    ["schema", schema.el],
    // a range shows its own segments; the distance is the whole route's
    ["segments", route.range ? `${route.segment_numbers.length} of ${(route.maxqlog ?? -1) + 1}` : `${(route.maxqlog ?? -1) + 1}, ${fmt.distance(route.distance ?? 0)}`],
  ];

  const copied = h("span", {class: "small muted"});
  // the open view at the playhead and zoom; with plots, the Plot view as this browser has it
  const link = () => location.origin + ctx.url();
  const plotsLink = () => shareUrl(ctx.layoutOf("plot"), ctx.url("plot"));
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
    toggles.replaceChildren(toggle("public", route.is_public, owner, async (v) => { await api.setRoutePublic(route.fullname, v); route.is_public = v; }));
    // a public route's record, as seen signed out, doesn't say whether it's preserved
    if (route.is_preserved !== undefined) {
      toggles.append(toggle("preserved", route.is_preserved, owner, async (v) => { await api.setRoutePreserved(route.fullname, v); route.is_preserved = v; }));
    }
  });
  routeEvents(route).then((spans) => bookmarks.replaceChildren(...spans.filter((s) => s.kind === "bookmark").map((s) =>
    h("button", {class: "chip", title: "Seek to bookmark", onclick: () => clock.seek(s.start / 1000)}, `bookmark ${fmt.clock(s.start / 1000)}`)))).catch(() => {});

  const el = h("div", {class: "info-pane"},
    h("table", {class: "kv-table"}, rows.filter(([, v]) => v).map(([k, v]) => h("tr", {}, h("th", {}, k), h("td", {}, v)))),
    h("div", {class: "actions"},
      h("button", {class: "button small", title: "Link to this view of the route at the playhead, zoomed as it is", onclick: () => copy(link(), "link")}, "Copy link to this time"),
      h("button", {class: "button small", title: "Link to the Plot view at the playhead with your plots (custom series ask before they run)",
        onclick: () => copy(plotsLink(), "link with your plots")}, "Copy link with my plots"),
      h("button", {class: "button small", onclick: () => copy(route.fullname, "route name")}, "Copy name"),
      navigator.share && h("button", {class: "button small", onclick: () => navigator.share({title: route.fullname, url: link()}).catch(() => {})}, "Share"),
      toggles, copied),
    bookmarks);
  return {name: "info", el, render() {}, resize() {}, dispose: schema.dispose, save: () => ({})};
}
