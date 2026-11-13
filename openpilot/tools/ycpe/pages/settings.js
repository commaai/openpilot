// device settings, as in connect: what the device is, its comma prime status, and (for the owner) unpairing. User mode
// has connect's own: its modal over the dashboard, from the device list's gear
import {api, auth} from "../api.js";
import {h, fmt, css, copyButton, keyValues} from "../dom.js";
import {icon} from "../icons.js";
import {cnIcon} from "./connect/icons.js";
import {onlineMark, link, go} from "../app.js";
import {renameDevice, shareDevice, unpairDevice} from "../devices.js";
import {deviceQueue, queueTable} from "./files.js";
import {cnModal, cnTitle, fill, ring} from "./cnui.js";

const DATE = new Intl.DateTimeFormat(undefined, {year: "numeric", month: "short", day: "numeric"});
const day = (unixSeconds) => unixSeconds ? DATE.format(unixSeconds * 1000) : null;

const rows = (pairs) => keyValues(pairs, "list settings");

// the device says whether it has prime; for its owner, the subscription says the plan and billing, and pages/prime.js
// manages it
function primeSection(device) {
  const to = `/connect/${device.dongle_id}/prime`;
  const manage = device.is_owner && h("a", {class: "button small", href: to, onclick: link(to)}, device.prime ? "Prime settings" : "Sign up for comma prime");
  const el = h("div", {}, rows([["status", device.prime ? "active" : "no subscription"]]), manage);
  if (!device.is_owner) return el;
  api.primeSubscription(device.dongle_id).then((sub) => {
    if (!sub) return;
    el.replaceChildren(rows([
      ["status", sub.cancel_at ? `active until ${day(sub.cancel_at)}` : "active"],
      ["plan", sub.plan],
      ["price", sub.amount ? `$${(sub.amount / 100).toFixed(2)} / month` : null],
      ["since", day(sub.subscribed_at)],
      ["trial ends", sub.trial_end && sub.trial_end * 1000 > Date.now() ? day(sub.trial_end) : null],
      ["next payment", !sub.cancel_at ? day(sub.next_charge_at) : null],
    ]), manage);
  }).catch((err) => el.append(h("p", {class: "small error"}, err.message)));
  return el;
}

// the key the device signs its requests with: the PEM's body on one line, the whole PEM copied
function publicKey(pem) {
  return h("span", {class: "pubkey"}, h("span", {class: "mono"}, pem.replace(/-----[^-]+-----|\s/g, "")), copyButton(() => pem, "Copy the public key", 14));
}

// what unpairing a prime device ends too
const unpairEnds = (device) => device.commacare
  ? ["Unpairing will also cancel comma prime and ", h("b", {}, "permanently end your commacare extended warranty."),
    " Your standard 1-year warranty still applies for any remaining time. ", h("a", {href: COMMACARE, target: "_blank", rel: "noreferrer"}, "What is commacare?")]
  : "Unpairing will also cancel the comma prime subscription for this device.";

// unpairing is for good, so it takes the dongle id typed out
function unpairSection(device, onUnpaired) {
  const dongleId = device.dongle_id;
  const status = h("p", {class: "small"});
  const input = h("input", {class: "field mono", placeholder: "type the dongle id", autocomplete: "off", spellcheck: "false", "aria-label": "Dongle id to confirm",
    oninput: () => { button.disabled = input.value.trim().toLowerCase() !== dongleId; }});
  const button = h("button", {class: "button danger", disabled: true}, "Unpair");
  const form = h("form", {class: "inline-form", onsubmit: async (e) => {
    e.preventDefault();
    if (button.disabled) return;
    button.disabled = input.disabled = true;
    status.className = "small muted";
    status.textContent = "Unpairing…";
    try {
      await unpairDevice(device);
      onUnpaired();
    } catch (err) {
      status.className = "small error";
      status.textContent = err.message;
      button.disabled = input.disabled = false;
    }
  }}, input, button);
  return h("div", {},
    h("p", {class: "muted"}, "Unpairing removes this device from your account, and from everyone it's shared with. Its drives stay ",
      "uploaded, and it can be paired again from its setup screen. Type the dongle id to confirm."),
    device.prime && h("p", {class: "warn"}, icon("warning", 16), h("span", {}, unpairEnds(device))),
    form, status);
}

export function settingsSection(device, onUnpaired) {
  const access = h("span");
  auth.user().then((me) => { access.textContent = device.is_owner ? "owner" : me?.superuser ? "superuser" : "read"; });
  return h("div", {},
    h("h3", {}, "Device"),
    rows([
      ["dongle id", h("span", {class: "mono"}, device.dongle_id)],
      ["alias", device.alias],
      ["type", device.device_type && fmt.deviceType(device.device_type)],
      ["serial", device.serial && h("span", {class: "mono"}, device.serial)],
      ["IMEI", device.imei && h("span", {class: "mono"}, device.imei)],  // the API shows it to the owner only
      ["openpilot", device.openpilot_version],
      ["SIM", device.sim_id && h("span", {class: "mono"}, device.sim_id)],
      ["last seen", onlineMark(device, "ago")],
      ["uploads", device.ignore_uploads ? h("span", {class: "error"}, "ignored by the comma API") : "accepted"],
      ["public key", device.public_key && publicKey(device.public_key)],
      ["your access", access],
    ]),
    h("h3", {}, "comma prime"),
    primeSection(device),
    device.is_owner && h("h3", {}, "Unpair"),
    device.is_owner && unpairSection(device, onUnpaired));
}

// ---- user mode: connect's DeviceSettingsModal ----

const COMMACARE = "https://comma.ai/connect#what-is-commacare";
const styled = () => Promise.all([css(new URL("./cnui.css", import.meta.url)), css(new URL("./settings.css", import.meta.url))]);

const outlined = (label, onclick) => h("button", {class: "cnui-btn outlined", onclick}, label);
const contained = (label, onclick) => h("button", {class: "cnui-btn grey", onclick}, label);

// MUI's TextField as connect themes it: a rounded box with its label inside until there's text or focus
function field(id, label, value, onInput, onEnter, help) {
  const input = h("input", {id, type: "text", placeholder: " ", autocomplete: "off", "aria-describedby": help ? `${id}-help` : null,
    oninput: () => onInput(input.value), onkeydown: (e) => { if (e.key === "Enter") onEnter(); }});
  input.value = value ?? "";
  return [h("div", {class: "cn-field"}, h("label", {for: id}, label), h("div", {class: "cn-input"}, input),
    help && h("p", {id: `${id}-help`}, help)), input];
}

// the icon button beside a field (save, share): a check once done, a ring around it while sending
function fieldAction(name, label, run) {
  const busy = ring(48, label);
  const btn = h("button", {class: "cn-iconbtn", "aria-label": label, onclick: run});
  const el = h("div", {class: "cn-fab"}, btn, busy);
  return {el, set(shown, state) {
    el.hidden = !shown;
    btn.replaceChildren(cnIcon(state === "done" ? "check" : name));
    busy.hidden = state !== "busy";
  }};
}

export async function settingsModal(device) {
  await styled();
  const dongleId = device.dongle_id;
  const error = h("div", {class: "cn-form-error", hidden: true});
  const fail = (text) => { error.hidden = !text; error.replaceChildren(h("p", {}, text)); };

  // the name, saved with its button or Enter; the button shows while it differs from the device's or was just saved
  let alias = device.alias, saved = false, saving = false;
  const save = fieldAction("save", "Save the name", () => setAlias());
  const showSave = () => save.set(device.alias !== alias || saved, saving ? "busy" : saved ? "done" : "");
  const [aliasRow] = field("device_alias", "Device name", alias, (v) => {
    if (v !== dongleId) saved = false;
    alias = v;
    showSave();
  }, () => setAlias());
  async function setAlias() {
    if (saving) return;
    saving = true;
    saved = false;
    showSave();
    try {
      await renameDevice(device, alias ?? "");
      saved = true;
    } catch (err) {
      fail(err.message);
    }
    saving = false;
    showSave();
  }

  // sharing: read access for an email or user id
  let email = "", shared = false, sharing = false;
  const share = fieldAction("share", "Share", () => addUser());
  const showShare = () => share.set(email.length > 0 || shared, sharing ? "busy" : shared ? "done" : "");
  const [shareRow, shareInput] = field("device_share", "Share by email or user id", "", (v) => {
    email = v;
    shared = false;
    fail(null);
    showShare();
  }, () => addUser(), "give another user read access to this device");
  async function addUser() {
    if (sharing) return;
    sharing = true;
    shared = false;
    showShare();
    try {
      await shareDevice(device, email);
      email = shareInput.value = "";
      shared = true;
      fail(null);
    } catch (err) {
      fail(err.status === 404 ? "could not find user" : "unable to share");
    }
    sharing = false;
    showShare();
  }
  showSave();
  showShare();

  const {el, paper} = cnModal("Device settings", "cn-paper cn-settings");
  fill(paper, cnTitle("Device settings", dongleId),
    h("div", {}, outlined("Prime settings", () => { el.close(); go(`/connect/${dongleId}/prime`); }),
      outlined("Unpair", () => unpairModal(device))),
    h("div", {class: "cn-modal-row"}, outlined("Uploads", () => {
      let draw = null;
      const q = deviceQueue(device, () => draw?.());
      uploadsModal(q, dongleId, (d) => { draw = d; q.poll(); return q.dispose; });
    }),
      device.commacare && h("a", {class: "cnui-care", href: COMMACARE, target: "_blank", rel: "noreferrer"},
        h("img", {src: "/icons/commacare.webp", alt: ""}), "commacare")),
    h("div", {class: "cn-form"}, error, h("div", {class: "cn-form-row"}, aliasRow, save.el), h("div", {class: "cn-form-row"}, shareRow, share.el)),
    h("div", {class: "cn-modal-buttons"}, contained("Close", () => el.close())));
  return el;
}

// unpairing, confirmed, and what it ends for a prime device. Once unpaired, closing goes home with the list asked again
function unpairModal(device) {
  let unpaired = false;
  const error = h("div", {class: "cn-unpair-error", hidden: true});
  const cancel = contained("Cancel", () => el.close());
  const confirm = outlined("Confirm", async () => {
    confirm.disabled = true;
    confirm.textContent = "Unpairing...";
    error.hidden = true;
    let msg = null;
    try {
      await unpairDevice(device);
    } catch (err) {
      msg = err.refused ? err.message : "Unable to unpair";
    }
    if (msg) {
      error.hidden = false;
      error.replaceChildren(cnIcon("error", 24), h("p", {}, msg));
      confirm.disabled = false;
      confirm.textContent = "Confirm";
      return;
    }
    unpaired = true;
    cancel.textContent = "Close";
    confirm.replaceWith(h("span", {class: "cn-unpaired"}, "Unpaired"));
  });
  const {el, paper} = cnModal("Unpair device", "cn-paper cn-unpair", () => {
    if (!unpaired) return;
    document.querySelector(".cn-settings")?.parentElement.close();
    go("/connect");
  });
  fill(paper, cnTitle("Unpair device", device.dongle_id), error,
    device.prime && h("div", {class: "cn-unpair-warning"}, icon("warning", 24), h("p", {}, unpairEnds(device))),
    h("div", {class: "cn-unpair-buttons"}, cancel, confirm));
}

// connect's upload queue (Files/UploadQueue.jsx), from here and a drive's Files menu: q is the device's queue
// (pages/files.js deviceQueue) or a drive's uploads; watch(draw) calls draw as q changes, and returns what stops that
export async function uploadsModal(q, dongleId, watch) {
  await styled();
  const body = h("div", {class: "cm-queue-body"});
  const all = contained("Cancel All", () => q.cancel(q.waiting()));
  const draw = () => {
    all.disabled = !q.cancel || !q.waiting().length;
    body.replaceChildren(queueTable(q));
  };
  draw();
  const {el, paper} = cnModal("Upload queue", "cn-paper cn-upload-queue", watch(draw));
  fill(paper, cnTitle("Upload queue", dongleId), body, h("div", {class: "cn-modal-buttons"}, all, contained("Close", () => el.close())));
}
