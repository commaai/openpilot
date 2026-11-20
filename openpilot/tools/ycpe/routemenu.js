// the route crumb's menu, the open route's one home in developer mode: its name part by part, every field of its API
// record useradmin shows (grouped as it does), links to copy, its public/preserved toggles, and its files. Every tab has it, so none needs a pane for it; the device's other drives are the navigator's
import {api, segmentAtTime} from "/api.js";
import {deviceRecord} from "/devices.js";
import {h, fmt, css, ask, copy, copyButton, keyValues} from "/dom.js";
import {parseRemote} from "/data/github.js";
import {canManage} from "/pages/home.js";
import {isPreserved, setPreserved} from "/pages/device.js";
import {routeAnatomy} from "/routeid.js";

css(new URL("./routemenu.css", import.meta.url));

// a checkbox that saves when changed, with what it means beside it; ask(v) may call the change off, and refused(v)
// says why the API answered 403
function toggle(label, value, enabled, set, {about, ask = () => true, refused = () => "not allowed"}) {
  const box = h("input", {type: "checkbox", checked: value, disabled: !enabled});
  const status = h("span", {class: "small error"});
  box.addEventListener("change", async () => {
    const v = box.checked;
    status.textContent = "";
    if (!await ask(v)) return void (box.checked = !v);
    box.disabled = true;
    try {
      await set(v);
    } catch (err) {
      box.checked = !v;
      status.textContent = err.status === 403 ? refused(v) : err.message;
    }
    box.disabled = false;
  });
  return h("label", {class: "check"}, box, label, h("span", {class: "small muted"}, about), status);
}

// the API's times: unix seconds, or UTC without its "Z"
const when = (t) => t && new Date(typeof t === "number" ? t * 1000 : `${t}Z`).toLocaleString();
const place = (lat, lng) => lat != null && lng != null && `${lat}, ${lng}`;

// ctx: the workspace's; segment: the playhead's, as the crumb shows it; files() opens the timeline's Files panel
export function routeMenu(ctx, {segment, files}) {
  const {route, clock} = ctx;
  const [dongleId] = route.fullname.split("|");
  const start = new Date(route.start_time_utc_millis);
  const commit = route.git_commit?.slice(0, 8);
  const repo = parseRemote(route.git_remote);  // the device says what its remote is: only a real GitHub repo is linked
  const github = repo && commit ? `https://github.com/${repo}/commit/${route.git_commit}` : null;
  const segments = (route.maxqlog ?? -1) + 1;
  const groups = {
    Route: [
      ["route", h("span", {class: "mono"}, route.fullname, copyButton(() => route.fullname, "Copy the route's name", 12))],
      ["start", `${start.toLocaleString()} (${fmt.duration(route.end_time_utc_millis - route.start_time_utc_millis)})`],
      // a range shows its own segments; the distance is the whole route's
      ["segments", route.range ? `${route.segment_numbers.length} of ${segments}` : `${segments}, ${fmt.distance(route.distance ?? 0)}`],
      // comma's servers process each qlog as it arrives (events, coords, thumbnails)
      ["processed", route.procqlog !== undefined && (route.procqlog === route.maxqlog ? "all segments" : `${(route.procqlog ?? -1) + 1} of ${segments} segments`)],
      ["uploaded", route.create_time && h("span", {title: "when the route's first file reached comma's servers"}, when(route.create_time))],
      ["from", place(route.start_lat, route.start_lng)],
      ["to", place(route.end_lat, route.end_lng)],
      ["user id", route.user_id && h("span", {class: "mono"}, route.user_id)],
    ],
    Car: [
      ["car", [route.platform, route.make && `(${route.make})`].filter(Boolean).join(" ")],
      ["vin", route.vin && h("span", {class: "mono"}, route.vin)],
      ["car id", route.car_id],
    ],
    Version: [
      ["openpilot", route.version],
      ["branch", route.git_branch && [route.git_branch, " @ ", github ? h("a", {href: github, target: "_blank", rel: "noopener"}, commit) : commit, route.git_dirty ? " (dirty)" : ""]],
      ["committed", when(route.git_commit_date)],
      ["remote", route.git_remote],
      ["version id", route.version_id],
    ],
  };

  // "<dongle id>/<log id>/<segment>" at the playhead, as connect copies it
  const segmentName = () => `${route.fullname.replace("|", "/")}/${segmentAtTime(route, clock.t, clock.duration)}`;
  const link = () => location.origin + ctx.url();
  const toggles = h("div", {class: "toggles"});
  deviceRecord(dongleId).catch(() => null).then(async (d) => {
    const owner = !!d && await canManage(d);
    // useradmin's confirmation, connect's explanations, and useradmin's word for the preserve limit
    toggles.replaceChildren(toggle("public", route.is_public, owner, async (v) => { await api.setRoutePublic(route.fullname, v); route.is_public = v; }, {
      about: "anyone with the route name or link can read all its files",
      ask: (v) => !v || ask(`Make ${route.fullname} public? This will allow any comma user with the route name to access all route files.`, "Make public")}));
    // a public route's record, as seen signed out, doesn't say whether it's preserved
    if (route.is_preserved !== undefined) {
      toggles.append(toggle("preserved", isPreserved(route), owner, (v) => setPreserved(route, v), {
        about: "kept from deletion; up to 10 routes, or 100 with comma prime",
        refused: (v) => v ? "Maximum number of drives have been preserved. Unpreserve a drive and try again." : "not allowed"}));
    }
  });

  return h("div", {class: "route-pop"},
    routeAnatomy(route, segment),
    h("div", {class: "rm-body"},
      h("div", {class: "groups"}, Object.entries(groups).map(([title, rows]) => h("section", {},
        h("h4", {}, title),
        keyValues(rows, "kv-table")))),
      // a link with the layout's tabs is the layout menu's (⋯)
      h("div", {class: "actions"},
        h("button", {class: "button small", title: "Link to this view of the route at the playhead, zoomed as it is", onclick: () => copy(link(), "the link")}, "Copy link"),
        h("button", {class: "button small", title: "The name of the segment at the playhead: dongle id/log id/segment",
          onclick: () => { const name = segmentName(); copy(name, name); }}, "Copy segment name"),
        navigator.share && h("button", {class: "button small", onclick: () => navigator.share({title: route.fullname, url: link()}).catch(() => {})}, "Share"),
        !route.loose && h("button", {class: "button small", title: "Every file of every segment: downloads and upload requests", onclick: files}, "Files")),
      toggles));
}
