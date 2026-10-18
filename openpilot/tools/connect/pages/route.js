import {api} from "/api.js";
import {h, fmt} from "/dom.js";
import {icon} from "/icons.js";
import {link, go} from "/app.js";
import {driveMap} from "/map.js";
import {timeline, routeDuration, segmentNumber} from "/timeline.js";

function player(route) {
  const video = h("video", {playsinline: true, preload: "metadata", muted: true});
  const src = api.qcameraUrl(route);
  // Chrome answers "maybe" to native HLS but can't play it; hls.js first, native for Safari
  if (window.Hls && Hls.isSupported()) {
    const hls = new Hls({maxBufferLength: 60});
    hls.loadSource(src);
    hls.attachMedia(video);
  } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
    video.src = src;
  }
  const t = Number(new URLSearchParams(location.search).get("t"));
  if (t > 0) video.addEventListener("loadedmetadata", () => { video.currentTime = t; }, {once: true});
  return video;
}

// connect's control bar: ±10 s, clock and segment, speed, mute, play/pause
function controls(route, video) {
  const SPEEDS = [0.1, 0.25, 0.5, 1, 2, 3, 4, 5];
  let speed = 3;
  const time = h("span", {class: "time"}, "--:--:--");
  const rate = h("span", {class: "rate"}, "1×");
  const play = h("button", {class: "ctl big", "aria-label": "Play"}, icon("play", 32));
  const mute = h("button", {class: "ctl", "aria-label": "Mute"}, icon("volumeOff"));
  const setSpeed = (i) => { speed = Math.max(0, Math.min(SPEEDS.length - 1, i)); video.playbackRate = SPEEDS[speed]; rate.textContent = `${SPEEDS[speed]}×`; };
  play.addEventListener("click", () => video.paused ? video.play() : video.pause());
  mute.addEventListener("click", () => { video.muted = !video.muted; mute.replaceChildren(icon(video.muted ? "volumeOff" : "volumeUp")); });
  video.addEventListener("play", () => { play.replaceChildren(icon("pause", 32)); });
  video.addEventListener("pause", () => { play.replaceChildren(icon("play", 32)); });
  const showTime = () => {
    const ms = video.currentTime * 1000;
    time.textContent = `${new Date(route.start_time_utc_millis + ms).toLocaleTimeString(undefined, {hour12: false})} – ${segmentNumber(route, ms)}`;
  };
  video.addEventListener("timeupdate", showTime);
  showTime();
  return h("div", {class: "controls"},
    h("button", {class: "ctl", "aria-label": "Back 10 seconds", onclick: () => { video.currentTime -= 10; }}, icon("replay10")),
    h("button", {class: "ctl", "aria-label": "Forward 10 seconds", onclick: () => { video.currentTime += 10; }}, icon("forward10")),
    time,
    h("div", {class: "speed"},
      h("button", {class: "ctl tiny", "aria-label": "Faster", onclick: () => setSpeed(speed + 1)}, icon("up", 18)), rate,
      h("button", {class: "ctl tiny", "aria-label": "Slower", onclick: () => setSpeed(speed - 1)}, icon("down", 18))),
    mute, play);
}

function pills(items) {
  return h("div", {class: "pills"}, items.map(([label, onclick, selected]) =>
    h("div", {class: `pill-item ${selected === false ? "dim" : ""}`, onclick}, label)));
}

function filesMenu(routeName) {
  const kinds = {logs: "rlog", qlogs: "qlog", cameras: "fcamera", ecameras: "ecamera", dcameras: "dcamera", qcameras: "qcamera"};
  const body = h("div", {class: "muted"}, "Loading…");
  api.routeFiles(routeName).then((files) => {
    const segments = {};
    for (const [kind, name] of Object.entries(kinds)) {
      for (const url of files[kind] || []) {
        const n = Number(new URL(url).pathname.split("/").at(-2));
        (segments[n] ||= {})[name] = url;
      }
    }
    body.replaceWith(h("table", {class: "files"},
      h("tr", {}, h("th", {}, "seg"), Object.values(kinds).map((k) => h("th", {}, k))),
      Object.keys(segments).sort((a, b) => a - b).map((n) => h("tr", {}, h("td", {}, n),
        Object.values(kinds).map((k) => h("td", {}, segments[n][k] ? h("a", {href: segments[n][k], class: "dl", title: `${k} ${n}`}, icon("download", 18)) : ""))))));
  }).catch(() => body.replaceWith(h("p", {class: "muted"}, "Files not available.")));
  return h("div", {class: "menu-panel"}, body);
}

function moreInfoMenu(route, routeName) {
  const toggle = (label, field, setter) => {
    const b = h("div", {class: "menu-item"});
    const refresh = () => { b.replaceChildren(h("span", {class: "tick"}, route[field] ? icon("check", 18) : ""), label); };
    b.addEventListener("click", async () => {
      try { await setter(!route[field]); route[field] = !route[field]; } catch (e) { alert(e.status === 403 ? "Not allowed" : String(e)); }
      refresh();
    });
    refresh();
    return b;
  };
  return h("div", {class: "menu-panel"},
    h("div", {class: "menu-item muted mono"}, route.fullname),
    h("div", {class: "menu-item muted"}, [route.platform, route.version, route.git_branch].filter(Boolean).join(" · ")),
    h("div", {class: "menu-item", onclick: () => navigator.clipboard.writeText(route.fullname)}, "Copy route name"),
    h("div", {class: "menu-item", onclick: () => navigator.clipboard.writeText(location.origin + location.pathname)}, "Copy link"),
    toggle("Public", "is_public", (v) => api.setRoutePublic(routeName, v)),
    toggle("Preserved", "is_preserved", (v) => api.setRoutePreserved(routeName, v)));
}

async function routeCoords(route) {
  const n = (route.maxqlog ?? -1) + 1;
  const lists = await Promise.all(Array.from({length: n}, async (_, i) => {
    const resp = await fetch(api.segmentAsset(route, i, "coords.json"));
    const seg = route.segment_start_times ? (route.segment_start_times[i] - route.start_time_utc_millis) / 1000 : i * 60;
    return resp.ok ? (await resp.json()).map((c) => ({...c, t: seg + c.t})) : [];
  }));
  return lists.flat();
}

export async function routePage(routeName, dongleId) {
  const route = await api.route(routeName);
  const duration = routeDuration(route);
  const start = new Date(route.start_time_utc_millis);
  const sameYear = start.getFullYear() === new Date().getFullYear();
  const hm = (d) => d.toLocaleTimeString(undefined, {hour: "2-digit", minute: "2-digit", hour12: false});
  const title = `${start.toLocaleDateString(undefined, {month: "short", day: "numeric", ...(sameYear ? {} : {year: "numeric"})})} @ ${hm(start)} - ${hm(new Date(route.end_time_utc_millis))}`;

  const video = player(route);
  const media = h("div", {class: "media"}, video);
  let map = null;
  const show = (which) => {
    videoPill.classList.toggle("dim", which !== "video");
    mapPill.classList.toggle("dim", which !== "map");
    if (which === "map") {
      if (!map) { map = h("div", {class: "map-holder"}); routeCoords(route).then((c) => map.append(driveMap(c, video))); }
      media.replaceChildren(map);
    } else {
      media.replaceChildren(video);
    }
  };
  const panel = h("div", {class: "panel-holder"});
  const openPanel = (el) => { panel.replaceChildren(panel.firstChild === el ? "" : el); };
  panel.addEventListener("click", (e) => e.stopPropagation());
  document.addEventListener("click", () => panel.replaceChildren());
  const files = filesMenu(routeName), info = moreInfoMenu(route, routeName);
  const videoPill = h("div", {class: "pill-item", onclick: () => show("video")}, "Video");
  const mapPill = h("div", {class: "pill-item dim", onclick: () => show("map")}, "Map");

  return h("div", {class: "route-page"},
    h("div", {class: "route-card"},
      h("div", {class: "route-head"},
        h("button", {class: "icon-button", "aria-label": "Back", onclick: () => history.length > 1 ? history.back() : go(`/${dongleId}`)}, icon("back")),
        h("div", {class: "route-title"}, h("span", {class: "weekday"}, start.toLocaleDateString(undefined, {weekday: "long"}), " "), title),
        h("a", {class: "icon-button", "aria-label": "Close", href: `/${dongleId}`, onclick: link(`/${dongleId}`)}, icon("close"))),
      timeline(route, {video, ruler: true, lazy: false}),
      h("div", {class: "route-body"},
        h("div", {class: "options"},
          h("div", {class: "pills"}, videoPill, mapPill),
          h("div", {class: "anchor"},
            h("div", {class: "pills"},
              h("div", {class: "pill-item", onclick: (e) => { e.stopPropagation(); openPanel(files); }}, "Files"),
              h("div", {class: "pill-item", onclick: (e) => { e.stopPropagation(); openPanel(info); }}, "More info")),
            panel)),
        media,
        controls(route, video))));
}
