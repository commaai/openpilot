// light and dark themes: CSS variables on <html>, read once for canvas panes
import {h, listeners} from "/dom.js";

const KEY = "ycpe_theme";
const subs = listeners();
const vars = {};  // theme → (name → value): a computed style read forces a style recalc
const chosen = () => localStorage.getItem(KEY) || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
// user mode is connect's, which is dark only: shown in place of the setting, which is kept for developer mode. Held from
// the start on a page that opens in user mode (index.html guesses which), so it never paints light first
let held = document.documentElement.dataset.mode === "user" ? "dark" : null;

export const theme = {
  get name() { return document.documentElement.dataset.theme; },
  // a CSS variable's value, e.g. theme.color("--fg")
  color(name) {
    const cache = vars[theme.name];
    let v = cache.get(name);
    if (v === undefined) cache.set(name, v = getComputedStyle(document.documentElement).getPropertyValue(name).trim());
    return v;
  },
  on: subs.on,
  set(name) {
    localStorage.setItem(KEY, name);
    show(held || name);
  },
  hold(name) {
    held = name;
    show(held || chosen());
  },
  toggle() { theme.set(chosen() === "dark" ? "light" : "dark"); },
};

function show(name) {
  if (name === theme.name) return;
  document.documentElement.dataset.theme = name;
  paintBar();
  subs.emit(name);
}

document.documentElement.dataset.theme = held || chosen();
// every variable of both themes, before there's a workspace to restyle (a pane's first read would pay for all of it),
// from hidden elements carrying each: a switch then never reads a style it has just invalidated
for (const name of ["light", "dark"]) {
  const probe = document.body.appendChild(h("div", {"data-theme": name, hidden: true}));
  const style = getComputedStyle(probe);
  vars[name] = new Map();
  for (const v of style) if (v.startsWith("--")) vars[name].set(v, style.getPropertyValue(v).trim());
  probe.remove();
}

// the browser's own bar (a phone's status bar, an installed app's title bar) as the top bar, by the app's theme rather
// than the system's
const bar = document.head.appendChild(h("meta", {name: "theme-color"}));
function paintBar() { bar.content = theme.color("--panel"); }
paintBar();
