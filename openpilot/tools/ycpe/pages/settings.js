// device settings, as in connect: what the device is, its comma prime status, and (for the owner) unpairing
import {api, auth, billingReachable} from "../api.js";
import {h, fmt} from "../dom.js";
import {icon} from "../icons.js";
import {onlineMark, link} from "../app.js";

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
  const label = "Copy the public key";
  const copy = h("button", {class: "icon-btn", title: label, "aria-label": label, onclick: async () => {
    const ok = await navigator.clipboard.writeText(pem).then(() => true, () => false);
    copy.title = ok ? "Copied" : "Couldn't copy: the clipboard was denied";
    copy.replaceChildren(icon(ok ? "check" : "warning", 14));
    setTimeout(() => { copy.title = label; copy.replaceChildren(icon("copy", 14)); }, 1500);
  }}, icon("copy", 14));
  return h("span", {class: "pubkey"}, h("span", {class: "mono"}, pem.replace(/-----[^-]+-----|\s/g, "")), copy);
}

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
    device.prime && h("p", {class: "warn"}, icon("warning", 16), device.commacare
      ? h("span", {}, "Unpairing will also cancel comma prime and ", h("b", {}, "permanently end your commacare extended warranty."),
        " Your standard 1-year warranty still applies for any remaining time. ",
        h("a", {href: "https://comma.ai/connect#what-is-commacare", target: "_blank", rel: "noopener"}, "What is commacare?"))
      : h("span", {}, "Unpairing will also cancel the comma prime subscription for this device.")),
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
