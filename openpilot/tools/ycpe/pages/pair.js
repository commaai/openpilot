// pairing, as connect does it: the add-device dialog scans the QR code on the device's screen (the camera and the
// platform's BarcodeDetector; where there's neither, the link that QR code holds can be pasted), a pair link
// (?pair=<token>, that QR code opened with any phone camera) pairs in a dialog over the page it opens, and noDevices()
// is connect's empty state. Loaded when needed: app.js imports it for a pair link or an account without devices
import {api, auth} from "../api.js";
import {h, css} from "../dom.js";
import {icon} from "../icons.js";
import {go, devicesChanged} from "../app.js";
import {loginPath} from "./login.js";

await css(new URL("./pair.css", import.meta.url));

// connect's words for the API's refusals (utils pairErrorToMessage)
const REFUSED = {
  400: "invalid request",
  401: "could not decode token - make sure your comma device is connected to the internet",
  403: "device paired with different owner - make sure you logged in with the correct account",
  404: "tried to pair invalid device",
  417: "pair token not true",
};
const CAMERA = {
  NotAllowedError: "Camera access denied. Please allow camera access in your browser settings and try again.",
  NotFoundError: "No camera found on this device.",
  NotReadableError: "Camera is in use by another application.",
};
// retry: the API refused it, so connect's pair link dialog says to try again
const failure = (message, retry = false) => Object.assign(new Error(message), {retry});

// the pair token in what a device's QR code holds: its pair link, or "<imei>--<serial>--<token>" from older openpilot
function readQr(text) {
  const fromUrl = text.startsWith("https://");
  const token = fromUrl ? URL.parse(text)?.searchParams.get("pair") : text.split("--")[2];
  if (!token) throw failure(fromUrl ? "could not parse pair token from detected url" : "invalid QR code detected");
  return {token, fromUrl};
}

// the dongle id the token pairs to this account. A token that doesn't name a device is caught here, as connect does
async function pair({token, fromUrl}) {
  let claims;
  try {
    claims = JSON.parse(atob(token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")));
  } catch {
    throw failure("invalid QR code, could not decode pair token");
  }
  if (!claims?.identity) throw failure("could not get identity from payload" + (fromUrl ? "" : ", make sure you are using openpilot 0.8.3 or newer"));
  const resp = await api.pilotPair(token).catch((e) => { throw failure(REFUSED[e.status] ?? "unable to pair", true); });
  if (!resp.dongle_id) throw failure("could not pair", true);
  devicesChanged();  // the next page asks for the list, with the new device in it
  return resp.dongle_id;
}

// the device's page, in the mode the pairing started from
const devicePath = (dongleId) => `${location.pathname.startsWith("/connect") ? "/connect" : ""}/${dongleId}`;
const paired = (dongleId) => h("p", {}, "Successfully paired device ", h("b", {class: "mono"}, dongleId));

// a modal in the theme; Escape, the x or a click outside closes it, and onclose runs once it has. It takes the focus
// itself, so the x isn't drawn focused
function modal(title, caption, onclose) {
  const body = h("div", {class: "pair-body"});
  const el = h("dialog", {class: "pair-dialog", "aria-label": title, autofocus: true, onclose: () => { el.remove(); onclose(); },
    onclick: (e) => { if (e.target === el) el.close(); }},  // the dialog has no padding: only the backdrop is itself
  h("div", {class: "pair-card"},
    h("div", {class: "pair-head"}, h("h2", {}, title), caption && h("span", {class: "small muted"}, caption),
      h("button", {class: "icon-btn", title: "Close", "aria-label": "Close", onclick: () => el.close()}, icon("close", 18))),
    body));
  document.body.append(el);
  el.showModal();
  return {el, body};
}

// QR codes are read by the platform where it can (Chrome on Android, macOS, ChromeOS), else by zxing in a worker
// (qr-worker.js), handed each frame as a bitmap; null when it can't load
async function qrDetector() {
  const formats = await globalThis.BarcodeDetector?.getSupportedFormats().catch(() => []);
  if (formats?.includes("qr_code")) return new BarcodeDetector({formats: ["qr_code"]});
  const worker = new Worker(new URL("./qr-worker.js", import.meta.url));
  const reply = () => new Promise((resolve) => worker.addEventListener("message", ({data}) => resolve(data), {once: true}));
  if ((await reply()).error) return worker.terminate(), null;
  return {
    async detect(video) {
      const frame = await createImageBitmap(video);
      worker.postMessage(frame, [frame]);
      const {codes, error} = await reply();
      if (error) throw new Error(error);
      return codes.map((rawValue) => ({rawValue}));
    },
    close: () => worker.terminate(),
  };
}

// connect's add-device dialog: the camera with a frame to hold the code in, and over it pairing, its error with
// "try again", or the paired device. Closing it after pairing opens the device
export async function openPair() {
  let stream = null, timer = 0, busy = false, dongleId = null, detector = null;
  const video = h("video", {muted: true, playsinline: true});
  const status = h("div", {class: "pair-status", hidden: true});
  const {el, body} = modal("Pair device", "scan QR code", () => {
    clearTimeout(timer);
    detector?.close?.();
    for (const track of stream?.getTracks() ?? []) track.stop();
    if (dongleId) go(devicePath(dongleId));
  });
  const say = (...children) => { status.hidden = !children.length; status.replaceChildren(...children); };
  body.append(h("div", {class: "pair-video"}, video, h("div", {class: "pair-frame"}, h("i"), h("i"), h("i"), h("i")), status));

  const scan = async () => {
    const [code] = await detector.detect(video).catch(() => []);
    if (!el.open || busy || !status.hidden) return;
    if (code?.rawValue) read(code.rawValue);
    else timer = setTimeout(scan, 100);  // not every frame: a 720p frame costs a few ms to hand over
  };
  const again = () => {
    say();
    if (stream) {
      video.play();
      scan();
    } else input.focus();
  };
  async function read(text) {
    if (busy || dongleId) return;
    busy = true;
    video.pause();
    say(h("span", {class: "spinner big"}));
    try {
      dongleId = await pair(readQr(text));
      say(paired(dongleId), h("button", {class: "button", onclick: () => el.close()}, "close"));
    } catch (e) {
      say(h("p", {}, `Error: ${e.message}`), h("button", {class: "button", onclick: again}, "try again"));
    }
    busy = false;
  }

  // without a camera or a reader: say why, and take the link pasted instead
  const input = h("input", {class: "field mono", placeholder: "https://connect.comma.ai/?pair=…", "aria-label": "Pair link", autocomplete: "off", spellcheck: false, required: true});
  const fallback = (why) => {
    body.replaceChildren(h("p", {}, why), h("p", {}, "You can also scan the QR code using any other QR code reader application."),
      h("form", {class: "pair-paste", onsubmit: (e) => { e.preventDefault(); read(input.value.trim()); }},
        h("label", {class: "small muted"}, "Or paste the link it opens:", input), h("button", {class: "button primary"}, "Pair")),
      status);
    status.classList.add("inline");
  };

  let camera;
  [detector, camera] = await Promise.all([qrDetector(),
    navigator.mediaDevices?.enumerateDevices().then((list) => list.some((d) => d.kind === "videoinput"), () => false)]);
  if (!el.open) return detector?.close?.();
  if (!detector) return fallback("Couldn't load the QR code reader. Scan the one on your device with your phone's camera: it opens a link that pairs it.");
  if (!camera) return fallback("Camera not found, please enable camera access.");
  try {
    stream = await navigator.mediaDevices.getUserMedia({video: {facingMode: "environment", width: {ideal: 1280}, height: {ideal: 720}}});
  } catch (e) {
    return el.open && fallback(CAMERA[e.name] ?? "Unable to access camera.");
  }
  if (!el.open) return stream.getTracks().forEach((track) => track.stop());
  video.srcObject = stream;
  await video.play().catch(() => {});
  scan();
}

// the add-device entry point: connect's "add new device", optionally with its icon (the sidebar's row)
export const pairButton = (text = "add new device", withIcon = false) =>
  h("button", {class: `button pair-add${withIcon ? " row" : ""}`, onclick: () => openPair()}, text, withIcon && icon("add_circle_outline", 20));

// signed in, without a device: connect's NoDeviceUpsell
export const noDevices = () => h("div", {class: "pair-empty"},
  h("div", {class: "pair-copy"},
    h("h2", {}, "Pair your device"),
    h("p", {}, "Scan the QR code on your device. If you cannot see a QR code, check the following:"),
    h("ul", {},
      h("li", {}, "Your device is connected to the internet"),
      h("li", {}, "You have installed the latest version of openpilot"),
      h("li", {}, "You may need to look for \"Pair Device\" in Settings")),
    h("p", {}, "If you still cannot see a QR code, your device may already be paired to another account. Make sure you have signed in with the same account you may have used previously."),
    pairButton()),
  h("img", {class: "pair-device", src: new URL("../icons/comma-device.webp", import.meta.url), alt: "comma four", width: 960, height: 741, decoding: "async"}));

// a device's pair link (?pair=<token>): signed out, sign in first (the token rides along in next=); signed in, it
// pairs in connect's "Pairing device" dialog over the page. The token leaves the URL, so neither a reload nor going
// back sends it again. True when q had one: the URL was rewritten and the caller renders it
export function handlePairLink(q) {
  const token = q.get("pair");
  if (!token) return false;
  if (!auth.isAuthenticated()) {
    history.replaceState(null, "", loginPath());
    return true;
  }
  q.delete("pair");
  history.replaceState(null, "", location.pathname + (q.size ? `?${q}` : "") + location.hash);
  let dongleId = null;
  const status = h("div", {class: "pair-status inline"}, h("span", {class: "spinner big"}));
  const {el, body} = modal("Pairing device", null, () => { if (dongleId) go(devicePath(dongleId)); });
  const close = h("button", {class: "button primary", onclick: () => el.close()}, "Close");
  body.append(status, h("div", {class: "pair-actions"}, close));
  close.focus();
  pair({token, fromUrl: true}).then((id) => {
    dongleId = id;
    status.replaceChildren(paired(id));
  }, (e) => status.replaceChildren(h("p", {}, `Error: ${e.message}${e.retry ? ", please try again" : ""}`)));
  return true;
}
