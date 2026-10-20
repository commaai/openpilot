// the device's drives as a dense list; clicking one swaps the route under the current view
import {api} from "/api.js";
import {h, fmt} from "/dom.js";

const DAY = 86400000;

export function create(ctx) {
  const dongle = ctx.route.dongle_id || ctx.route.fullname.split("|")[0];
  const list = h("div", {class: "route-list"});
  const more = h("button", {class: "link more", onclick: () => load()}, "older drives");
  const el = h("div", {class: "routes-pane"}, list, more);
  let end = Math.ceil(Date.now() / 3600000) * 3600000;
  const seen = new Set();

  const row = (r) => {
    const start = new Date(r.start_time_utc_millis);
    const current = r.fullname === ctx.route.fullname;
    return h("a", {class: `route-row ${current ? "current" : ""}`, href: `/${dongle}/${r.fullname.split("|")[1]}`,
      onclick: (e) => { if (e.button || e.metaKey || e.ctrlKey) return; e.preventDefault(); ctx.openRoute(r.fullname); }},
      h("img", {class: "route-thumb", loading: "lazy", alt: "", src: api.segmentAsset(r, 0, "sprite.jpg")}),
      h("span", {class: "route-when"},
        h("b", {}, start.toLocaleDateString(undefined, {month: "short", day: "numeric"}), " ", start.toLocaleTimeString(undefined, {hour: "2-digit", minute: "2-digit", hour12: false})),
        h("span", {class: "muted"}, `${fmt.minutes(r.end_time_utc_millis - r.start_time_utc_millis)} · ${fmt.miles(r.distance ?? 0)}`)));
  };

  async function load() {
    more.disabled = true;
    for (let tries = 0; tries < 6; tries++) {
      const start = end - 14 * DAY;
      const routes = await api.routes(dongle, start, end).catch(() => []);
      end = start;
      const fresh = routes.filter((r) => !seen.has(r.fullname));
      for (const r of fresh) { seen.add(r.fullname); api.route.prime(r, r.fullname); }
      list.append(...fresh.map(row));
      if (fresh.length) break;
    }
    // a public route opened signed out: the device's list isn't visible, so show just this one
    if (!list.children.length) list.append(row(ctx.route), h("p", {class: "muted center small"}, "Sign in to see this device's other drives."));
    more.disabled = false;
  }
  load();
  return {name: "routes", el, render() {}};
}
