// the device's drives as a dense list; clicking one swaps the route under the current view
import {api, auth} from "/api.js";
import {h, fmt} from "/dom.js";

const DAY = 86400000, LIMIT = 50;

export function create(ctx) {
  const dongle = ctx.route.dongle_id || ctx.route.fullname.split("|")[0];
  const list = h("div", {class: "route-list"});
  const more = h("button", {class: "link more", onclick: () => load()}, "older drives");
  const failed = h("p", {class: "error center small", hidden: true});
  const el = h("div", {class: "routes-pane"}, list, failed, more);
  let end = Math.ceil(Date.now() / 3600000) * 3600000;
  const seen = new Set();

  const row = (r, note = "") => {
    const start = new Date(r.start_time_utc_millis);
    const current = r.fullname === ctx.route.fullname;
    return h("a", {class: `route-row ${current ? "current" : ""}`, href: `/${dongle}/${r.fullname.split("|")[1]}`, "aria-current": current && "page",
      onclick: (e) => { if (e.button || e.metaKey || e.ctrlKey) return; e.preventDefault(); ctx.openRoute(r.fullname); }},
      h("img", {class: "route-thumb", loading: "lazy", alt: "", src: api.segmentAsset(r, 0, "sprite.jpg")}),
      h("span", {class: "route-when"},
        h("b", {}, start.toLocaleDateString(undefined, {month: "short", day: "numeric"}), " ", start.toLocaleTimeString(undefined, {hour: "2-digit", minute: "2-digit", hour12: false})),
        h("span", {class: "muted"}, `${fmt.minutes(r.end_time_utc_millis - r.start_time_utc_millis)} · ${fmt.miles(r.distance ?? 0)}${note}`)));
  };

  // an open drive older than the first page is pinned on top, so it's always in the list
  const pinned = ctx.route.start_time_utc_millis < end - 14 * DAY;
  if (pinned) {
    seen.add(ctx.route.fullname);
    list.append(row(ctx.route, " · this drive"));
  }

  // the device's list isn't visible (a public route opened signed out, or a device not shared with us): just this one
  const alone = (why) => {
    if (!pinned) list.append(row(ctx.route));
    list.append(h("p", {class: "muted center small"}, why));
    more.hidden = true;
  };

  // windows of 14 days back from `end`; a full window ends at its oldest drive, so the next one picks up there
  async function load() {
    if (!auth.isAuthenticated()) return alone("Sign in to see this device's other drives.");
    more.disabled = true;
    failed.hidden = true;
    try {
      for (let tries = 0; tries < 6; tries++) {
        const start = end - 14 * DAY;
        const routes = await api.routes(dongle, start, end, LIMIT);
        end = routes.length === LIMIT ? Math.min(...routes.map((r) => r.start_time_utc_millis)) : start;
        const fresh = routes.filter((r) => !seen.has(r.fullname));
        for (const r of fresh) { seen.add(r.fullname); api.route.prime(r, r.fullname); }
        list.append(...fresh.map((r) => row(r)));
        if (fresh.length) break;
      }
    } catch (err) {
      if (err.status === 401 || err.status === 403) return alone("This device's other drives aren't shared with you.");
      failed.replaceChildren(`Couldn't load drives: ${err.message} `, h("button", {class: "link", onclick: () => load()}, "Retry"));
      failed.hidden = false;
    }
    more.disabled = false;
  }
  load();
  return {name: "routes", el, render() {}};
}
