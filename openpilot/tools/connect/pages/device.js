import {api} from "/api.js";
import {h, fmt} from "/dom.js";
import {link} from "/app.js";

const DAY = 86400000;

export function routeCard(r) {
  const path = `/${r.dongle_id}/${r.fullname.split("|")[1]}`;
  const duration = r.end_time_utc_millis - r.start_time_utc_millis;
  return h("a", {class: "card route", href: path, onclick: link(path)},
    h("img", {class: "thumb", src: api.segmentAsset(r, 0, "sprite.jpg"), alt: "", loading: "lazy", onerror: (e) => { e.target.style.visibility = "hidden"; }}),
    h("div", {},
      h("div", {class: "title"}, fmt.date(r.start_time_utc_millis)),
      h("div", {class: "muted"}, fmt.duration(duration), " · ", fmt.miles(r.distance ?? 0), r.is_preserved ? " · preserved" : "", r.is_public ? " · public" : "")));
}

export async function devicePage(dongleId) {
  const [device, location] = await Promise.all([api.device(dongleId), api.deviceLocation(dongleId).catch(() => null)]);
  const list = h("div", {class: "cards"});
  const more = h("button", {}, "Load older drives");
  let end = Date.now();
  let seen = new Set();

  async function loadMore() {
    more.disabled = true;
    more.textContent = "loading…";
    const start = end - 14 * DAY;
    const routes = await api.routes(dongleId, start, end);
    for (const r of routes) {
      if (seen.has(r.fullname)) continue;
      seen.add(r.fullname);
      list.append(routeCard(r));
    }
    end = start;
    more.disabled = false;
    more.textContent = list.children.length ? "Load older drives" : "No drives yet, load older";
    if (!routes.length && list.children.length === 0 && Date.now() - end < 180 * DAY) loadMore();  // keep looking back
  }
  loadMore();

  return h("div", {class: "page"},
    h("div", {class: "device-head"},
      h("h2", {}, device.alias || dongleId),
      h("div", {class: "muted"}, dongleId, " · ", device.device_type, " · ", device.openpilot_version || "", " · last seen ", fmt.ago(device.last_athena_ping),
        location && location.lat ? [" · ", h("a", {href: `https://www.google.com/maps?q=${location.lat},${location.lng}`, target: "_blank"}, "last location")] : null)),
    list, h("p", {}, more));
}
