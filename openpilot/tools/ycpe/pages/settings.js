// device settings, as in connect: what the device is, its comma prime status, and (for the owner) unpairing. User mode
// has connect's own: its modal over the dashboard, from the device list's gear
import {api, auth, billingReachable} from "../api.js";
import {h, fmt, css, copyButton, openModal} from "../dom.js";
import {icon} from "../icons.js";
import {cnIcon} from "./connect/icons.js";
import {onlineMark, link, go, isOnline, devicesChanged} from "../app.js";

const DATE = new Intl.DateTimeFormat(undefined, {year: "numeric", month: "short", day: "numeric"});
const day = (unixSeconds) => unixSeconds ? DATE.format(unixSeconds * 1000) : null;

const rows = (pairs) => h("table", {class: "list settings"},
  h("tbody", {}, pairs.filter(([, v]) => v !== null && v !== undefined && v !== "").map(([k, v]) =>
    h("tr", {}, h("td", {class: "muted"}, k), h("td", {}, v)))));

// the device says whether it has prime; for its owner, the subscription says the plan and billing (where billing
// answers this page), and pages/prime.js manages it
function primeSection(device) {
  const to = `/${device.dongle_id}/prime`;
  const manage = device.is_owner && h("a", {class: "button small", href: to, onclick: link(to)}, device.prime ? "Prime settings" : "Sign up for comma prime");
  const el = h("div", {}, rows([["status", device.prime ? "active" : "no subscription"]]), manage);
  if (!device.is_owner || !billingReachable()) return el;
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
      await api.unpairDevice(dongleId);
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
const styled = () => css(new URL("./settings.css", import.meta.url));

// MUI's Modal: a backdrop that fades in and the paper over it; Escape or a click beside the paper closes it
function modal(cls, onClose, ...children) {
  return openModal({class: "cn-modal", onclose: onClose}, h("div", {class: `cn-paper ${cls}`, role: "document", tabindex: "-1", autofocus: true}, ...children));
}

const title = (text, dongleId) => [h("div", {class: "cn-modal-title"}, h("h2", {}, text), h("span", {}, dongleId)), h("hr")];
const outlined = (label, onclick) => h("button", {class: "cn-btn", onclick}, label);
const contained = (label, onclick) => h("button", {class: "cn-btn contained", onclick}, label);
const spinner = (size) => h("span", {class: "cn-progress", style: `--size: ${size}px`});

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
  const ring = spinner(48);
  const btn = h("button", {class: "cn-iconbtn", "aria-label": label, onclick: run});
  const el = h("div", {class: "cn-fab"}, btn, ring);
  return {el, set(shown, state) {
    el.hidden = !shown;
    btn.replaceChildren(cnIcon(state === "done" ? "check" : name));
    ring.hidden = state !== "busy";
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
      Object.assign(device, await api.setDeviceAlias(dongleId, (alias ?? "").trim()));
      saved = true;
      renamed(device);
    } catch (err) {
      fail(err.message);
    }
    saving = false;
    showSave();
  }

  // sharing: read access for an email or user id
  let email = "", shared = false, sharing = false;
  const share = fieldAction("share", "Share", () => shareDevice());
  const showShare = () => share.set(email.length > 0 || shared, sharing ? "busy" : shared ? "done" : "");
  const [shareRow, shareInput] = field("device_share", "Share by email or user id", "", (v) => {
    email = v;
    shared = false;
    fail(null);
    showShare();
  }, () => shareDevice(), "give another user read access to this device");
  async function shareDevice() {
    if (sharing) return;
    sharing = true;
    shared = false;
    showShare();
    try {
      await api.addDeviceUser(dongleId, email.trim());
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

  const el = modal("cn-settings", null,
    title("Device settings", dongleId),
    h("div", {}, outlined("Prime settings", () => { el.close(); go(`/connect/${dongleId}/prime`); }),
      outlined("Unpair", () => unpairModal(device))),
    h("div", {class: "cn-modal-row"}, outlined("Uploads", () => uploadsModal(device)),
      device.commacare && h("a", {class: "cn-commacare", href: COMMACARE, target: "_blank", rel: "noreferrer"},
        h("img", {src: "/icons/commacare.webp", alt: ""}), "commacare")),
    h("div", {class: "cn-form"}, error, h("div", {class: "cn-form-row"}, aliasRow, save.el), h("div", {class: "cn-form-row"}, shareRow, share.el)),
    h("div", {class: "cn-modal-buttons"}, contained("Close", () => el.close())));
  return el;
}

// the new name where the device list and the dashboard show it
function renamed(device) {
  for (const b of document.querySelectorAll(`.cn-dev a[href="/connect/${device.dongle_id}"] .cn-dev-name > b`)) b.textContent = fmt.deviceName(device);
  if (location.pathname === `/connect/${device.dongle_id}`) go(location.pathname + location.search, true);
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
      const resp = await api.unpairDevice(device.dongle_id);
      if (!resp?.success) msg = resp?.error || "Could not successfully unpair";
    } catch {
      msg = "Unable to unpair";
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
  const el = modal("cn-unpair", () => {
    if (!unpaired) return;
    document.querySelector("dialog.cn-modal")?.close();
    devicesChanged();
    go("/connect");
  },
  title("Unpair device", device.dongle_id),
  error,
  device.prime && h("div", {class: "cn-unpair-warning"}, icon("warning", 24), h("p", {}, unpairEnds(device))),
  h("div", {class: "cn-unpair-buttons"}, cancel, confirm));
}

// connect's upload queue: what the device is sending, asked every 2 s while it's open
function uploadsModal(device) {
  const dongleId = device.dongle_id;
  const body = h("div", {class: "cn-uploads"}, spinner(17));
  const cancelling = new Set();
  let queue = [], timer = 0;
  const cancel = (ids) => {
    ids = ids.filter((id) => queue.some((q) => q.id === id && !q.current));
    for (const id of ids) cancelling.add(id);
    show();
    api.athena(dongleId, "cancelUpload", {upload_id: ids}).catch(() => {}).finally(poll);
  };
  const all = contained("Cancel All", () => cancel(queue.map((q) => q.id)));
  all.disabled = true;
  function show() {
    const offline = !isOnline(device);
    all.disabled = offline || !queue.length;
    if (offline) return body.replaceChildren(h("p", {}, "device offline"));
    if (!queue.length) return body.replaceChildren(h("p", {}, "no uploads"));
    // the one uploading first
    const rows = [...queue.filter((q) => q.current), ...queue.filter((q) => !q.current)];
    body.replaceChildren(h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "segment"), h("th", {}, "type"), h("th", {}, "progress"), h("th"))),
      h("tbody", {}, rows.map((q) => {
        const [route, n, file] = new URL(q.url).pathname.split("/").slice(-3), seg = `${route}--${n}`;
        return h("tr", {}, h("td", {}, h("div", {class: "cn-seg"}, h("span", {}, seg.slice(0, 12)), h("span", {}, seg.slice(12)))),
          h("td", {}, file.split(".")[0]),
          q.current ? h("td", {}, h("div", {class: "cn-upload-progress"}, h("div", {style: `--p: ${q.progress}`})))
            : [h("td", {}, q.paused ? "paused" : "pending"), h("td", {class: "cn-cancel-cell"}, cancelling.has(q.id) ? spinner(15)
              : h("button", {class: "cn-iconbtn small", "aria-label": "Cancel upload", onclick: () => cancel([q.id])}, cnIcon("cancel", 18)))]);
      }))));
  }
  async function poll() {
    clearTimeout(timer);
    if (!el.isConnected) return;
    if (isOnline(device)) {
      queue = await api.athena(dongleId, "listUploadQueue").catch(() => queue);
      for (const id of cancelling) if (!queue.some((q) => q.id === id)) cancelling.delete(id);
    }
    show();
    if (isOnline(device)) timer = setTimeout(poll, 2000);
  }
  const el = modal("cn-upload-queue", () => clearTimeout(timer), title("Upload queue", dongleId), body,
    h("div", {class: "cn-modal-buttons"}, all, contained("Close", () => el.close())));
  poll();
}
