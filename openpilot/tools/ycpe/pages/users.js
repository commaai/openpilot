// device users: owners list, add (read access) and remove; everyone else can only remove themselves
import {api} from "../api.js";
import {h, ask} from "../dom.js";
import {canManage} from "./home.js";
import {shareDevice} from "../devices.js";

// a button that takes access away, once asked (question)
function removing(label, question, fn) {
  const b = h("button", {class: "button small", onclick: async () => {
    if (!await ask(question, label)) return;
    b.disabled = true;
    fn().finally(() => { b.disabled = false; });
  }}, label);
  return b;
}

// left(): the signed-in user removed their own access
export function usersSection(device, left) {
  const dongleId = device.dongle_id;
  const el = h("div", {}, h("p", {class: "muted"}, "Loading…"));
  const status = h("p", {class: "small"});
  const say = (text, error = false) => { status.className = `small ${error ? "error" : "muted"}`; status.textContent = text; };
  const fail = (err) => say(err.status === 404 ? "Could not find that user." : err.message, true);

  async function load() {
    const users = await canManage(device) ? await api.deviceUsers(dongleId).catch(() => null) : null;
    if (!users) {
      el.replaceChildren(h("p", {class: "muted"}, "You have read access to this device."),
        removing("Remove my access", "Remove your access to this device? Its owner can share it with you again.", () => api.leaveDevice(dongleId).then(() => {
          left();
          el.replaceChildren(h("p", {class: "muted"}, "Your access was removed."));
        }, fail)),
        status);
      return;
    }
    const email = h("input", {type: "email", class: "field", placeholder: "email address", required: true});
    const add = h("button", {class: "button"}, "Share");
    const form = h("form", {class: "inline-form", onsubmit: async (e) => {
      e.preventDefault();
      add.disabled = true;
      try {
        await shareDevice(device, email.value);
        say(`Shared with ${email.value.trim()}.`);
        email.value = "";
        load();
      } catch (err) {
        fail(err);
      }
      add.disabled = false;
    }}, email, add);
    el.replaceChildren(
      h("table", {class: "list users"},
        h("thead", {}, h("tr", {}, h("th", {}, "email"), h("th", {}, "permission"), h("th"))),
        h("tbody", {}, users.map((u) => h("tr", {},
          h("td", {}, u.email || u.user_id),
          h("td", {class: "muted"}, u.permission.replace("_", " ")),
          h("td", {class: "right"}, u.permission === "owner" ? "" : !u.email
            ? h("button", {class: "button small", disabled: true, title: "The API removes users by email, and this one has none"}, "Remove")
            : removing("Remove", `Remove ${u.email}'s access to this device?`, () => api.removeDeviceUser(dongleId, u.email).then(load, fail))))))),
      h("h3", {}, "Share read access"), form, status);
  }
  load();
  return el;
}
