// signing in: the /login page, the /auth OAuth callback, and the inline card shown when a page needs an account.
// Where to land afterwards rides in ?next= and, across the OAuth round trip, in sessionStorage.
import {auth} from "../api.js";
import {h} from "../dom.js";
import {logo} from "../icons.js";
import {go, link} from "../app.js";

const RETURN_KEY = "ycpe_return_to";
const DEMO = "/5beb9b58bd12b691/0000010a--a51155e496/drive";
const PROVIDERS = {google: "Google", github: "GitHub", apple: "Apple"};

const here = () => location.pathname + location.search;
// only same-site paths, never "//host" or a full URL
const safe = (path) => (path?.startsWith("/") && !path.startsWith("//") ? path : "/");
export const loginPath = (next = here()) => `/login?${new URLSearchParams({next})}`;

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
    h("div", {class: "providers"}, Object.entries(PROVIDERS).map(([p, name]) =>
      h("a", {class: "button provider", href: auth.loginUrl(p), onclick: () => sessionStorage.setItem(RETURN_KEY, next)}, `Continue with ${name}`))),
    tokenForm(next),
    h("p", {class: "small muted login-foot"}, "Public routes open without an account. ",
      h("a", {href: DEMO, onclick: link(DEMO)}, "Open the demo route")));
}

// the provider sent us back with a one-time code: trade it for a token, then return to where sign-in started
export function authPage(params) {
  const next = safe(sessionStorage.getItem(RETURN_KEY));
  const status = h("div", {class: "login-status"}, h("span", {class: "spinner"}), "Signing in…");
  const el = card(status);
  const code = params.get("code"), provider = params.get("provider");
  const exchange = code ? auth.exchangeCode(code, provider) : Promise.reject(new Error(params.get("error") || "no sign-in code was returned"));
  exchange.then(() => {
    sessionStorage.removeItem(RETURN_KEY);
    go(next, true);
  }, (err) => status.replaceWith(
    h("p", {class: "error"}, "Sign-in failed."),
    h("p", {class: "small muted"}, err.status ? `The server said ${err.status}. The code may have expired; signing in again usually works.` : err.message),
    h("div", {class: "providers"},
      h("a", {class: "button primary", href: loginPath(next), onclick: link(loginPath(next))}, "Try again"),
      h("a", {class: "button", href: next, onclick: link(next)}, "Continue without signing in"))));
  return el;
}

// in place of a page the API refused: signed out, offer sign-in; signed in, say so and offer another account
export function signInCard() {
  const signedIn = auth.isAuthenticated();
  return card(
    h("p", {}, signedIn ? "This account doesn't have access to this." : "Sign in to view this."),
    h("p", {class: "small muted"}, signedIn ? "Ask the device owner to share it with you, or sign in with another account." : "It isn't public, so it needs an account with access."),
    h("div", {class: "providers"},
      h("a", {class: "button primary", href: loginPath(), onclick: (e) => { if (signedIn) auth.signOut(); link(loginPath())(e); }},
        signedIn ? "Use another account" : "Sign in")));
}
