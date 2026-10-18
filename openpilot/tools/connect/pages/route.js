import {api, routeEvents} from "/api.js";
import {h, fmt} from "/dom.js";
import {link} from "/app.js";

// a bar of engaged / alert / override spans that follows and seeks the video
function timeline(route, spans, video) {
  const duration = route.end_time_utc_millis - route.start_time_utc_millis;
  const pct = (ms) => `${(100 * ms / duration).toFixed(3)}%`;
  const cursor = h("div", {class: "cursor"});
  const bar = h("div", {class: "timeline", onclick: (e) => {
    const r = bar.getBoundingClientRect();
    video.currentTime = (e.clientX - r.left) / r.width * duration / 1000;
  }},
    spans.map((s) => h("div", {class: `span ${s.kind}`, style: `left:${pct(s.start)};width:${pct(s.end - s.start)}`})),
    cursor);
  video.addEventListener("timeupdate", () => { cursor.style.left = pct(video.currentTime * 1000); });
  return bar;
}

function player(route) {
  const video = h("video", {controls: true, playsinline: true, preload: "metadata"});
  const src = api.qcameraUrl(route);
  // Chrome answers "maybe" to native HLS but can't play it; hls.js first, native for Safari
  if (window.Hls && Hls.isSupported()) {
    const hls = new Hls({maxBufferLength: 60});
    hls.loadSource(src);
    hls.attachMedia(video);
  } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
    video.src = src;
  } else {
    return h("p", {class: "error"}, "This browser can't play HLS video.");
  }
  const t = Number(new URLSearchParams(location.search).get("t"));
  if (t > 0) video.addEventListener("loadedmetadata", () => { video.currentTime = t; }, {once: true});
  return video;
}

function filesTable(files) {
  // group the signed urls by segment number, from the path .../<segment>/<name>
  const kinds = {logs: "rlog", qlogs: "qlog", cameras: "fcamera", ecameras: "ecamera", dcameras: "dcamera", qcameras: "qcamera"};
  const segments = {};
  for (const [kind, name] of Object.entries(kinds)) {
    for (const url of files[kind] || []) {
      const n = Number(new URL(url).pathname.split("/").at(-2));
      (segments[n] ||= {})[name] = url;
    }
  }
  return h("table", {class: "files"},
    h("tr", {}, h("th", {}, "segment"), Object.values(kinds).map((k) => h("th", {}, k))),
    Object.keys(segments).sort((a, b) => a - b).map((n) => h("tr", {}, h("td", {}, n),
      Object.values(kinds).map((k) => h("td", {}, segments[n][k] ? h("a", {href: segments[n][k]}, "↓") : "")))));
}

export async function routePage(routeName, dongleId) {
  const route = await api.route(routeName);
  const duration = route.end_time_utc_millis - route.start_time_utc_millis;
  const video = player(route);
  const bar = h("div", {class: "timeline placeholder"});
  routeEvents(route).then((spans) => bar.replaceWith(timeline(route, spans, video))).catch(console.error);

  const toggle = (label, field, setter) => {
    const b = h("button", {}, "");
    const refresh = () => { b.textContent = route[field] ? `${label}: on` : `${label}: off`; };
    b.addEventListener("click", async () => {
      b.disabled = true;
      try {
        await setter(!route[field]);
        route[field] = !route[field];
      } catch (e) {
        alert(e.status === 403 ? "Not allowed" : String(e));
      }
      b.disabled = false;
      refresh();
    });
    refresh();
    return b;
  };
  const files = h("div", {class: "muted"}, "loading files…");
  api.routeFiles(routeName).then((f) => files.replaceWith(filesTable(f))).catch(() => files.replaceWith(h("p", {class: "muted"}, "Files not available.")));

  return h("div", {class: "page route-page"},
    h("div", {class: "route-head"},
      h("a", {href: `/${dongleId}`, onclick: link(`/${dongleId}`)}, "← device"),
      h("h2", {}, fmt.date(route.start_time_utc_millis)),
      h("div", {class: "muted"}, route.fullname, " · ", fmt.duration(duration), " · ", fmt.miles(route.distance ?? 0),
        route.platform ? ` · ${route.platform}` : "", route.version ? ` · ${route.version}` : ""),
      h("div", {class: "actions"},
        toggle("public", "is_public", (v) => api.setRoutePublic(routeName, v)),
        toggle("preserved", "is_preserved", (v) => api.setRoutePreserved(routeName, v)),
        h("button", {onclick: () => navigator.clipboard.writeText(location.origin + location.pathname)}, "copy link"))),
    video, bar,
    h("h3", {}, "Files"), files);
}
