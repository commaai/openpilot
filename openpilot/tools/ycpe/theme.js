// light and dark themes: CSS variables on <html>, read once per switch for canvas panes
const KEY = "ycpe_theme";
const subs = new Set();
const cache = new Map();  // name → value for the current theme: a computed style read forces a style recalc

export const theme = {
  get name() { return document.documentElement.dataset.theme; },
  // a CSS variable's value, e.g. theme.color("--fg")
  color(name) {
    let v = cache.get(name);
    if (v === undefined) cache.set(name, v = getComputedStyle(document.documentElement).getPropertyValue(name).trim());
    return v;
  },
  on(fn) { subs.add(fn); return () => subs.delete(fn); },
  set(name) {
    document.documentElement.dataset.theme = name;
    localStorage.setItem(KEY, name);
    readAll();
    for (const fn of subs) fn(name);
  },
  toggle() { theme.set(theme.name === "dark" ? "light" : "dark"); },
};

// every variable in one read, before there's a workspace to restyle (a pane's first read would pay for all of it)
function readAll() {
  cache.clear();
  const style = getComputedStyle(document.documentElement);
  for (const name of style) if (name.startsWith("--")) cache.set(name, style.getPropertyValue(name).trim());
}

document.documentElement.dataset.theme = localStorage.getItem(KEY) || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
readAll();
