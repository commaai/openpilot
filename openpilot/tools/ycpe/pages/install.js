// iOS has no install prompt of its own: say how, as connect's IosPwaPopup does, to the signed in until they dismiss it
import {auth} from "../api.js";
import {h, css} from "../dom.js";
import {icon} from "../icons.js";

const HIDE_KEY = "ycpe_hide_install";

export async function installPrompt() {
  const ios = /iphone|ipad|ipod/i.test(navigator.userAgent);
  if (!ios || navigator.standalone || !auth.isAuthenticated() || localStorage.getItem(HIDE_KEY)) return null;
  await css(new URL("./account.css", import.meta.url));
  const el = h("aside", {class: "install", "aria-label": "Add to home screen"},
    h("div", {}, h("b", {}, "Add to home screen"),
      h("p", {class: "small muted"}, "Install ycpe as an app: tap ", icon("iosShare", 14), " and then ‘Add to Home Screen’.")),
    h("button", {class: "icon-btn", title: "Dismiss", "aria-label": "Dismiss", onclick: () => {
      localStorage.setItem(HIDE_KEY, "1");
      el.remove();
    }}, icon("close", 16)));
  return el;
}
