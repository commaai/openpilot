// live data over WebRTC from a device's webrtcd, or one on this PC (SPEC.md Live): cereal services as capnp Events on
// the "cereal" channel, camera tracks, and, from devices that predate "cereal", carState and deviceState as JSON on
// "data". The peer connection is shared with file reads (peer.js).
import {listeners} from "../../dom.js";
import {peer} from "./peer.js";

// what the drive, plot, CAN and logs views read: about 300 kB/s on the demo route. modelV2 alone is another 150 kB/s,
// so it's left to the page to ask for
export const SERVICES = ["carState", "carControl", "carOutput", "carParams", "controlsState", "selfdriveState", "onroadEvents",
  "longitudinalPlan", "radarState", "drivingModelData", "can", "sendcan", "pandaStates", "peripheralState", "deviceState",
  "managerState", "gpsLocationExternal", "lateralTorqueParameters", "vehicleParameters", "logMessage", "errorLogMessage"];
export const CAMERAS = ["road"];
const FILE = {road: "fcamera", wideRoad: "ecamera", driver: "dcamera"};  // a camera by the name of its log file
const RETRY_MS = [1000, 2000, 5000, 10000, 30000];

export function create({dongle, local = false, services = SERVICES, cameras = CAMERAS} = {}) {
  const target = local ? "local" : dongle, conn = peer(target), subs = listeners();
  const user = {};  // this source's place among the connection's users
  let sink = null, offs = [], cereal = false, retries = 0, retry = 0, attempt = 0, wasLive = false;
  const shown = new Map();  // camera → the track the sink has

  const set = (state, detail = "") => {
    if (source.state === state && source.detail === detail) return;
    [source.state, source.detail] = [state, detail];
    subs.emit(state, detail);
  };

  const video = (camera, track) => {
    if (shown.get(camera) === track) return;
    shown.set(camera, track);
    sink.video(FILE[camera], track);
  };

  const connect = async () => {
    clearTimeout(retry);
    const mine = ++attempt;
    set("connecting");
    try {
      await conn.use(user, {services: source.services, cameras: source.cameras, files: true});
    } catch (e) {
      if (sink && mine === attempt) wasLive ? again(e.message) : set("error", e.message);
      return;
    }
    if (!sink || mine !== attempt) return;
    retries = 0;
    wasLive = true;
    for (const [camera, track] of conn.tracks) video(camera, track);
    set("live", local ? "this PC" : "peer to peer");
  };

  // a session that ends on its own (webrtcd's 5 minute limit, a dropped network) reconnects; one that never came up, or
  // that another viewer took over, doesn't
  const again = (why) => {
    if (/another device has connected/i.test(why)) return set("closed", why);
    const ms = RETRY_MS[Math.min(retries++, RETRY_MS.length - 1)];
    set("error", `${why.replace(/\.$/, "")}; trying again in ${ms / 1000} s`);
    retry = setTimeout(connect, ms);
  };

  const source = {
    kind: "webrtc",
    label: local ? "webrtcd on this PC" : `device ${dongle}`,
    state: "closed",
    detail: "",
    services,
    cameras,
    on: (event, fn) => event === "state" ? subs.on(fn) : () => {},

    start(s) {
      sink = s;
      cereal = wasLive = false;
      offs = [
        // each message is one Event or more, as logged; the sink batches them for its worker
        conn.on("cereal", (bytes) => {
          cereal = true;
          sink.events(bytes);
        }),
        // a device without "cereal" sends JSON; one with it may still send the stock carState and deviceState
        conn.on("json", (m) => cereal || sink.json(m.type, m.logMonoTime, m.valid, m.data)),
        conn.on("track", video),
        conn.on("state", (state, detail) => {
          if (state === "connecting" && source.state === "connecting") set("connecting", detail);
          if ((state === "closed" || state === "error") && source.state === "live") again(detail || "the connection closed");
        }),
      ];
      connect();
    },

    stop() {
      clearTimeout(retry);
      attempt++;
      for (const off of offs) off();
      offs = [];
      sink = null;
      shown.clear();
      conn.release(user);
      set("closed");
    },

    // the services and cameras to stream; a live source reconnects with them
    configure(opts) {
      Object.assign(source, opts);
      if (sink) connect();
    },
  };
  return source;
}
