// device users: owners list, add (read access) and remove; everyone else can only remove themselves
import {api} from "../api.js";
import {h} from "../dom.js";
import {canManage} from "./home.js";

// destructive buttons ask once more inline: the first click arms them for 3 s
function armed(label, fn) {
  let timer = 0;
  const b = h("button", {class: "button small", onclick: () => {
    if (!b.classList.contains("danger")) {
      b.classList.add("danger");
      b.textContent = "Confirm";
      timer = setTimeout(() => { b.classList.remove("danger"); b.textContent = label; }, 3000);
      return;
    }
    clearTimeout(timer);
    b.disabled = true;
    fn().finally(() => { b.disabled = false; });
  }}, label);
  return b;
}

export function usersSection(device) {
  const dongleId = device.dongle_id;
  const el = h("div", {}, h("p", {class: "muted"}, "Loading…"));
  const status = h("p", {class: "small"});
  const say = (text, error = false) => { status.className = `small ${error ? "error" : "muted"}`; status.textContent = text; };
  const fail = (err) => say(err.status === 404 ? "Could not find that user." : err.message, true);

  async function load() {
    const users = await canManage(device) ? await api.deviceUsers(dongleId).catch(() => null) : null;
    if (!users) {
      el.replaceChildren(h("p", {class: "muted"}, "You have read access to this device."),
        armed("Remove my access", () => api.leaveDevice(dongleId).then(() => el.replaceChildren(h("p", {class: "muted"}, "Your access was removed.")), fail)),
        status);
      return;
    }
    const email = h("input", {type: "email", class: "field", placeholder: "email address", required: true});
    const add = h("button", {class: "button"}, "Share");
    const form = h("form", {class: "inline-form", onsubmit: async (e) => {
      e.preventDefault();
      add.disabled = true;
      try {
        await api.addDeviceUser(dongleId, email.value.trim());
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
            : armed("Remove", () => api.removeDeviceUser(dongleId, u.email).then(load, fail))))))),
      h("h3", {}, "Share read access"), form, status);
  }
  load();
  return el;
}
