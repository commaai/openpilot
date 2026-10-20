// device users: owners list, add (read access) and remove; everyone else can only remove themselves
import {api} from "../api.js";
import {h} from "../dom.js";
import {icon} from "../icons.js";
import {canManage} from "./devices.js";

export function usersSection(device) {
  const dongleId = device.dongle_id;
  const el = h("div", {class: "section"}, h("p", {class: "muted"}, "Loading…"));
  const status = h("p", {class: "small muted"});
  const fail = (err) => { status.className = "small error"; status.textContent = err.status === 404 ? "Could not find that user." : err.message; };

  async function removeSelf() {
    if (!confirm(`Remove your access to ${device.alias || dongleId}?`)) return;
    try {
      await api.removeDeviceUser(dongleId);
      el.replaceChildren(h("p", {class: "muted"}, "Your access was removed."));
    } catch (err) { fail(err); }
  }

  async function load() {
    const users = await canManage(device) ? await api.deviceUsers(dongleId).catch(() => null) : null;
    if (!users) {
      el.replaceChildren(h("p", {class: "muted"}, "You have read access to this device."),
        h("button", {class: "button danger", onclick: removeSelf}, "Remove my access"), status);
      return;
    }
    const email = h("input", {type: "email", class: "field", placeholder: "email address", required: true});
    const add = h("form", {class: "inline-form", onsubmit: async (e) => {
      e.preventDefault();
      add.querySelector("button").disabled = true;
      try {
        await api.addDeviceUser(dongleId, email.value.trim());
        status.className = "small muted";
        status.textContent = `Shared with ${email.value.trim()}.`;
        load();
      } catch (err) {
        fail(err);
        add.querySelector("button").disabled = false;
      }
    }}, email, h("button", {class: "button"}, "Add user"));
    el.replaceChildren(
      h("table", {class: "list"},
        h("tr", {}, h("th", {}, "email"), h("th", {}, "permission"), h("th")),
        users.map((u) => h("tr", {},
          h("td", {}, u.email || u.user_id),
          h("td", {class: "muted"}, u.permission.replace("_", " ")),
          h("td", {}, u.permission === "owner" ? "" : h("button", {class: "icon-button small", "aria-label": `Remove ${u.email}`,
            onclick: async () => {
              if (!confirm(`Remove ${u.email} from ${device.alias || dongleId}?`)) return;
              try { await api.removeDeviceUser(dongleId, u.email); load(); } catch (err) { fail(err); }
            }}, icon("close", 18)))))),
      h("h3", {}, "Share read access"), add, status);
  }
  load();
  return el;
}
