// one WebRTC peer connection per device (or per webrtcd on this PC), shared by the live source and file reads, and
// closed once nobody uses it. The cloud only signals (athena startStream, or the daemon's forward to this PC's
// webrtcd); data flows peer to peer. SPEC.md Live has the protocol.
import {api} from "../../api.js";
import {listeners} from "../../dom.js";
import {FilesChannel} from "./files.js";

const daemon = await import("/daemon.js").then((m) => m.daemon, () => null);

const STUN = [{urls: "stun:stun.l.google.com:19302"}];  // as the device's side (teleoprtc) uses
const GATHER_MS = 3000, SRFLX_MS = 100, CONNECT_MS = 20000, DISCONNECTED_MS = 5000;
export const CAMERAS = ["road", "wideRoad", "driver"];

const text = new TextDecoder();
const peers = new Map();  // "local" or a dongle id → Peer

export const peer = (target) => peers.get(target) || peers.set(target, new Peer(target)).get(target);

// what users need from the connection: {services: [name], cameras: [road|wideRoad|driver], files: bool}
const union = (needs) => ({
  services: [...new Set(needs.flatMap((n) => n.services || []))].sort(),
  cameras: CAMERAS.filter((c) => needs.some((n) => n.cameras?.includes(c))),
  files: needs.some((n) => n.files),
});
const covers = (have, want) => want.services.every((s) => have.services.includes(s)) &&
  want.cameras.every((c) => have.cameras.includes(c)) && (!want.files || have.files);

class Peer {
  constructor(target) {
    this.target = target;
    this.users = new Map();  // user → needs
    this.state = "closed";
    this.detail = "";
    this.events = {state: listeners(), cereal: listeners(), json: listeners(), track: listeners()};
    this.tracks = new Map();  // camera → MediaStreamTrack
    this.gen = 0;
  }

  on(event, fn) { return this.events[event].on(fn); }

  // registers user's needs and resolves once a connection serving them is up; one that serves less is replaced
  use(user, needs) {
    this.users.set(user, needs);
    const want = union([...this.users.values()]);
    if (!this.ready || !covers(this.want, want) || this.state === "closed" || this.state === "error") this.ready = this.connect(want);
    return this.ready;
  }

  // the last user leaving closes the connection
  release(user) {
    if (this.users.delete(user) && !this.users.size) this.close("closed", "");
  }

  set(state, detail = "") {
    if (this.state === state && this.detail === detail) return;
    [this.state, this.detail] = [state, detail];
    this.events.state.emit(state, detail);
  }

  close(state = "closed", detail = "", gen = this.gen) {
    if (gen !== this.gen) return;
    this.gen++;
    this.pc?.close();
    this.pc = this.files = this.ready = null;
    this.filesReady?.reject(new Error(detail || "the connection closed"));
    this.tracks.clear();
    this.set(state, detail);
  }

  async connect(want) {
    this.close("connecting", "");
    const gen = this.gen, pc = this.pc = new RTCPeerConnection(this.target === "local" ? {} : {iceServers: STUN});
    this.want = want;
    this.set("connecting", this.target === "local" ? "asking this PC's webrtcd" : "asking the device through athena");
    const live = () => gen === this.gen;
    // video m-lines answer in the order the cameras were asked for, unless the answer names the camera (devices that
    // pick their own: stock ones send wideRoad)
    const transceivers = want.cameras.map(() => pc.addTransceiver("video", {direction: "recvonly"}));
    pc.ontrack = ({track, transceiver}) => {
      if (!live()) return;
      const camera = cameraOf(pc.remoteDescription?.sdp, transceiver.mid) || want.cameras[transceivers.indexOf(transceiver)];
      if (!camera) return;
      this.tracks.set(camera, track);
      this.events.track.emit(camera, track);
    };
    // the browser opens every channel ("data" is what teleoprtc listens for); a device may also open its own
    const channel = (ch) => {
      if (ch.label === "data") {
        ch.binaryType = "arraybuffer";  // webrtcd sends its JSON as bytes
        ch.onmessage = (e) => live() && this.data(typeof e.data === "string" ? e.data : text.decode(e.data), gen);
      }
      if (ch.label === "cereal") {
        ch.binaryType = "arraybuffer";
        ch.onmessage = (e) => live() && typeof e.data !== "string" && this.events.cereal.emit(new Uint8Array(e.data));
      }
      if (ch.label === "files") {
        const open = () => live() && this.filesReady.resolve(this.files = new FilesChannel(ch));
        ch.readyState === "open" ? open() : ch.addEventListener("open", open, {once: true});
      }
    };
    this.filesReady = Promise.withResolvers();
    this.filesReady.promise.catch(() => {});
    channel(pc.createDataChannel("data"));
    if (want.services.length) channel(pc.createDataChannel("cereal"));
    if (want.files) channel(pc.createDataChannel("files"));
    pc.ondatachannel = (e) => channel(e.channel);
    pc.onconnectionstatechange = () => {
      if (!live()) return;
      if (pc.connectionState === "failed") this.close("error", "the peer-to-peer connection failed (ICE): no route between this browser and the device", gen);
      if (pc.connectionState === "disconnected") {
        setTimeout(() => live() && pc.connectionState === "disconnected" && this.close("error", "the device stopped answering", gen), DISCONNECTED_MS);
      }
      if (pc.connectionState === "closed") this.close("closed", "the device closed the connection", gen);
    };

    try {
      await pc.setLocalDescription(await pc.createOffer());
      await gathered(pc);
      if (!live()) throw new Error("replaced");
      const answer = await signal(this.target, pc.localDescription.sdp, want);
      if (!live()) throw new Error("replaced");
      await pc.setRemoteDescription({type: "answer", sdp: answer.sdp});
      await connected(pc);
      if (!live()) throw new Error("replaced");
      this.set("live", "");
      return this;
    } catch (e) {
      this.close("error", e.message, gen);
      throw e;
    }
  }

  // the "data" channel: JSON {type, logMonoTime, valid, data}, and the device's control messages
  data(text, gen) {
    let msg;
    try {
      msg = JSON.parse(text);
    } catch {
      return;
    }
    if (msg.type === "disconnect") return this.close("closed", typeof msg.data === "string" ? msg.data : "the device ended the stream", gen);
    if (msg.logMonoTime != null) this.events.json.emit(msg);
  }

  // the files client, once its channel is open
  async filesChannel() {
    await this.ready;
    return this.filesReady.promise;
  }
}

// the camera an answer's m-line carries: teleoprtc names its tracks "<camera>:<uuid>" in the msid, which Chrome doesn't
// take for the track's id
function cameraOf(sdp = "", mid) {
  const media = sdp.split(/\nm=/).find((m) => m.split(/\r?\n/).includes(`a=mid:${mid}`));
  const named = media?.match(/msid:\S+ (\w+):/)?.[1];
  return CAMERAS.includes(named) ? named : null;
}

// webrtcd takes no trickled candidates, so the offer goes once gathering is done, or soon after the first
// server-reflexive candidate (what a device on another network needs: Chrome can take seconds more to call gathering
// complete), or after GATHER_MS
function gathered(pc) {
  return new Promise((resolve) => {
    const done = () => pc.iceGatheringState === "complete" && resolve();
    pc.addEventListener("icegatheringstatechange", done);
    pc.addEventListener("icecandidate", (e) => e.candidate?.type === "srflx" && setTimeout(resolve, SRFLX_MS));
    setTimeout(resolve, GATHER_MS);
    done();
  });
}

function connected(pc) {
  return new Promise((resolve, reject) => {
    const check = () => {
      if (pc.connectionState === "connected") resolve();
      if (["failed", "closed"].includes(pc.connectionState)) reject(new Error("the peer-to-peer connection failed (ICE): no route between this browser and the device"));
    };
    pc.addEventListener("connectionstatechange", check);
    setTimeout(() => reject(new Error(`no peer-to-peer connection after ${CONNECT_MS / 1000} s`)), CONNECT_MS);
    check();
  });
}

// devices and webrtcds from before SPEC.md Live take no services, cameras or files (an unexpected keyword); asked
// again as they were, they send carState and deviceState as JSON on "data"
const OLD = /unexpected keyword|invalid params/i;

async function signal(target, sdp, {services, cameras, files}) {
  const enabled = !!(cameras.length || services.length);
  let answer;
  if (target === "local") {
    if (!daemon?.has("webrtc")) throw new Error("live data from this PC needs the local daemon (run tools/ycpe/serve.py)");
    // webrtcd's own errors are {error, message}: read here, as daemon.json would keep only the first
    const ask = (body) => fetch("/local/webrtc/stream", {method: "POST", headers: {...daemon.headers, "Content-Type": "application/json"},
      body: JSON.stringify(body)}).then((r) => r.json());
    answer = await ask({sdp, cameras, enabled, services, files});
    if (OLD.test(answer.message || "")) answer = await ask({sdp, cameras, enabled, bridge_services_out: services});
  } else {
    answer = await api.athena(target, "startStream", {sdp, enabled, services, cameras, files}).catch((e) => {
      if (OLD.test(e.message)) return api.athena(target, "startStream", {sdp, enabled});
      throw /not connected|offline|timed out/i.test(e.message) ? new Error(`the device is offline (${e.message})`) : e;
    });
  }
  if (answer?.error) throw new Error(answer.message || answer.error);
  if (!answer?.sdp) throw new Error("the device answered without an SDP");
  return answer;
}
