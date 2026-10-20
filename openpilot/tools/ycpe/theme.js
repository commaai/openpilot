// light and dark themes: CSS variables on <html>, read once per switch for canvas panes
const KEY = "ycpe_theme";
const subs = new Set();
let cache = null;

export const theme = {
  get name() { return document.documentElement.dataset.theme; },
  // a CSS variable's value, e.g. theme.color("--fg")
  color(name) {
    cache ||= getComputedStyle(document.documentElement);
    return cache.getPropertyValue(name).trim();
  },
  on(fn) { subs.add(fn); return () => subs.delete(fn); },
  set(name) {
    document.documentElement.dataset.theme = name;
    localStorage.setItem(KEY, name);
    cache = null;
    for (const fn of subs) fn(name);
  },
  toggle() { theme.set(theme.name === "dark" ? "light" : "dark"); },
};

document.documentElement.dataset.theme = localStorage.getItem(KEY) || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
