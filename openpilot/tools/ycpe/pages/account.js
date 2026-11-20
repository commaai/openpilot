// the account: who's signed in (the account menu's head and /account, useradmin's profile page), the developer token
// with its expiry and a fresh one minted by signing in again (useradmin's "Get JWT"), and the version of ycpe running
import {USERADMIN, auth, jwtClaims} from "../api.js";
import {commitLink} from "../daemon.js";
import {h, css, copyButton, keyValues} from "../dom.js";
import {icon} from "../icons.js";
import {PROVIDERS, startSignIn, takeNewToken, signInCard} from "./login.js";

// the account menu and the pages here wait for it
export const styled = css(new URL("./account.css", import.meta.url));

const DATE = new Intl.DateTimeFormat(undefined, {year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"});

// the commit serve.py runs from (its daemon says); a static server can't tell
export const version = () => h("span", {class: "small muted"}, "ycpe ", commitLink({rel: "noopener", class: "mono"}) ?? "dev");

// the account menu's head: email, username, user id to copy
export const accountHead = (me) => h("div", {class: "account-head"},
  h("b", {}, me.email || me.username || "signed in"),
  me.username && me.email && h("span", {class: "muted"}, me.username),
  h("span", {class: "account-id muted"}, h("span", {class: "mono"}, me.user_id), copyButton(() => me.user_id, "Copy user id")));

// how long it has left, as useradmin says it
function validity(exp) {
  if (!exp) return "It doesn't say when it expires.";
  const until = new Date(exp * 1000), min = Math.ceil((until - Date.now()) / 60000);
  if (min <= 0) return `Expired ${DATE.format(until)}.`;
  const unit = (n, name) => n && `${n} ${name}${n === 1 ? "" : "s"}`;
  const left = [unit(Math.floor(min / 1440), "day"), unit(Math.floor(min % 1440 / 60), "hour")].filter(Boolean);
  return `Valid for ${left.length ? left.join(", ") : unit(min, "minute")} (until ${DATE.format(until)}).`;
}

// masked until revealed, so a screen share doesn't give it away
function tokenField(token, label) {
  const input = h("input", {class: "field mono", type: "password", value: token, readonly: true, autocomplete: "off", "aria-label": label,
    onfocus: () => input.select()});
  const reveal = h("button", {class: "icon-btn", title: "Show", "aria-label": "Show the token", "aria-pressed": "false", onclick: () => {
    const shown = input.type === "password";
    input.type = shown ? "text" : "password";
    reveal.setAttribute("aria-pressed", String(shown));
    reveal.title = shown ? "Hide" : "Show";
    reveal.replaceChildren(icon(shown ? "visibilityOff" : "visibility", 16));
  }}, icon("visibility", 16));
  return h("div", {class: "jwt"}, input, reveal, copyButton(() => token, "Copy the token"));
}

function developerSection(me) {
  const token = auth.token(), fresh = takeNewToken();
  const provider = me?.user_id?.split("_")[0];
  const again = PROVIDERS[provider] && h("a", {class: "button", href: auth.loginUrl(provider), onclick: () => startSignIn(provider, "/account", true)},
    "Get a new token");
  return h("section", {class: "developer"},
    h("h3", {id: "token"}, "Developer token"),
    h("p", {class: "small muted"}, "The JWT this browser signs in with, for scripts and tools: send it as ", h("span", {class: "mono"}, "Authorization: JWT <token>"),
      " or keep it in ", h("span", {class: "mono"}, "~/.comma/auth.json"), ". Anyone who has it can act as you."),
    fresh && [h("p", {class: "fresh"}, "New token"), tokenField(fresh, "New token"),
      h("p", {class: "small muted"}, validity(jwtClaims(fresh)?.exp), " It isn't saved anywhere: copy it now. This browser stays signed in as it was.")],
    fresh && h("p", {class: "fresh"}, "This browser's token"),
    tokenField(token, "Your token"),
    h("p", {class: "small muted"}, validity(jwtClaims(token)?.exp), auth.viaDaemon() ? " It's ~/.comma/auth.json's, through the local daemon." : ""),
    again && h("div", {class: "again"}, again, h("span", {class: "small muted"},
      `Signs in with ${PROVIDERS[provider]} again for a fresh 90-day token; this one keeps working.`)));
}

export async function accountPage(me) {
  await styled;
  if (!auth.isAuthenticated()) return signInCard();
  return h("div", {class: "page account-page"},
    h("h2", {}, "Account"),
    me ? keyValues([
      ["email", me.email],
      ["username", me.username],
      ["user id", h("span", {class: "account-id"}, h("span", {class: "mono"}, me.user_id), copyButton(() => me.user_id, "Copy user id"))]], "list settings")
      : h("p", {class: "small error"}, "Couldn't load the account."),
    developerSection(me),
    h("p", {class: "account-foot"}, version(), " · ",
      h("a", {class: "small", href: USERADMIN, target: "_blank", rel: "noopener"}, "Manage account at useradmin.comma.ai")));
}
