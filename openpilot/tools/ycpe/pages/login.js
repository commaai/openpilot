// signing in: the /login page, the /auth OAuth callback, and the inline card shown when a page needs an account.
// Where to land afterwards rides in ?next= and, across the OAuth round trip, in sessionStorage.
import {auth} from "../api.js";
import {daemon} from "../daemon.js";
import {h} from "../dom.js";
import {logo} from "../icons.js";
import {go, link} from "../app.js";

const RETURN_KEY = "ycpe_return_to";
const DEMO = "/5beb9b58bd12b691/0000010a--a51155e496/drive";
const PROVIDERS = {google: "Google", github: "GitHub", apple: "Apple"};

const here = () => location.pathname + location.search;
// only paths on this site, never "//host", "/\host" or a full URL: anything the browser would resolve elsewhere
function safe(path) {
  try {
    const u = new URL(path || "/", location.origin);
    return u.origin === location.origin ? u.pathname + u.search + u.hash : "/";
  } catch {
    return "/";
  }
}
export const loginPath = (next = here()) => `/login?${new URLSearchParams({next})}`;

// serve.py has ~/.comma/auth.json, turned off from the account menu: one click turns it back on
const daemonButton = (onclick) => daemon.hasJwt && !auth.isAuthenticated() &&
  h("button", {class: "button primary", onclick: () => { auth.useDaemon(); onclick?.(); }}, "Sign in with ~/.comma/auth.json");

function card(...children) {
  return h("div", {class: "login"}, h("div", {class: "login-card"},
    h("div", {class: "login-mark"}, logo(28), h("span", {}, "ycpe")), ...children));
}

function tokenForm(next) {
  const input = h("input", {class: "field mono", placeholder: "eyJ0eXAiOiJKV1Qi…", autocomplete: "off", spellcheck: false, required: true});
  const error = h("p", {class: "small error", hidden: true});
  const button = h("button", {class: "button"}, "Use token");
  const form = h("form", {class: "token-form", onsubmit: async (e) => {
    e.preventDefault();
    button.disabled = true;
    error.hidden = true;
    try {
      await auth.useToken(input.value.trim());
      go(next, true);
    } catch (err) {
      error.textContent = err.status === 401 ? "That token isn't valid." : `Couldn't check the token: ${err.message}`;
      error.hidden = false;
      button.disabled = false;
    }
  }}, input, button);
  return h("details", {class: "token"},
    h("summary", {}, "Developer: paste a JWT"),
    h("p", {class: "small muted"}, "From useradmin's developer page or ~/.comma/auth.json."), form, error);
}

export function loginPage(params) {
  const next = safe(params.get("next"));
  if (auth.isAuthenticated()) return card(h("p", {}, "You're signed in."), h("a", {class: "button primary", href: next, onclick: link(next)}, "Continue"));
  return card(
    h("p", {class: "muted"}, "Sign in to see your devices and drives."),
    h("div", {class: "providers"}, daemonButton(() => go(next, true)), Object.entries(PROVIDERS).map(([p, name]) =>
      h("a", {class: "button provider", href: auth.loginUrl(p), onclick: () => sessionStorage.setItem(RETURN_KEY, next)}, `Continue with ${name}`))),
    tokenForm(next),
    h("p", {class: "small muted login-foot"}, "Public routes open without an account. ",
      h("a", {href: DEMO, onclick: link(DEMO)}, "Open the demo route")));
}

// the provider sent us back with a one-time code: trade it for a token, then return to where sign-in started.
// Only if sign-in started in this tab, so a link with someone else's code can't sign you in to their account.
export function authPage(params) {
  const started = sessionStorage.getItem(RETURN_KEY);
  sessionStorage.removeItem(RETURN_KEY);
  const next = safe(started);
  const status = h("div", {class: "login-status"}, h("span", {class: "spinner"}), "Signing in…");
  const el = card(status);
  const code = params.get("code"), provider = params.get("provider");
  const exchange = !code ? Promise.reject(new Error(params.get("error") || "no sign-in code was returned"))
    : started === null ? Promise.reject(new Error("Sign-in wasn't started here, so this sign-in link was ignored."))
    : auth.exchangeCode(code, provider);
  exchange.then(() => go(next, true), (err) => status.replaceWith(
    h("p", {class: "error"}, "Sign-in failed."),
    h("p", {class: "small muted"}, err.status ? `The server said ${err.status}. The code may have expired; signing in again usually works.` : err.message),
    h("div", {class: "providers"},
      h("a", {class: "button primary", href: loginPath(next), onclick: link(loginPath(next))}, "Try again"),
      h("a", {class: "button", href: next, onclick: link(next)}, "Continue without signing in"))));
  return el;
}

// in place of a page the API refused: signed out, offer sign-in; signed in, say so and offer another account.
// A device page is just /<dongle id>; anything deeper is a route.
export function signInCard() {
  const signedIn = auth.isAuthenticated();
  const device = /^\/[0-9a-f]{16}\/?$/i.test(location.pathname);
  return card(
    h("p", {}, signedIn ? "This account doesn't have access to this." : device ? "Sign in to see this device's drives." : "Sign in to view this."),
    h("p", {class: "small muted"}, signedIn ? "Ask the device owner to share it with you, or sign in with another account."
      : device ? "Devices are only visible to their owner and people they've shared with." : "It isn't public, so it needs an account with access."),
    h("div", {class: "providers"}, daemonButton(),
      h("a", {class: daemon.hasJwt && !signedIn ? "button" : "button primary", href: loginPath(), onclick: (e) => { if (signedIn) auth.signOut(); link(loginPath())(e); }},
        signedIn ? "Use another account" : "Sign in")));
}
