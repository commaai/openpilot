// referrals, as connect has them: the share link, what's been earned, claiming by email; the top bar's gift and the
// device page's "refer a friend" card for owners
import {api, auth} from "../api.js";
import {h, css} from "../dom.js";
import {icon} from "../icons.js";
import {REFERRALS_SEEN, link} from "../app.js";
import {copyButton} from "./account.js";
import {signInCard} from "./login.js";
import {mode} from "./connect/mode.js";
import {promo} from "./cnui.js";

const REFER = "https://refer.comma.ai";
const TERMS = "https://comma.ai/terms#referral-terms";
const PROMO_KEY = "ycpe_referrals_promo_2026-09-02";  // connect's campaign: a new one shows the card again
const STEPS = [
  ["Share your link", "Send it to a friend who is interested in comma four."],
  ["They save $50", "Your link takes $50 off their comma four order."],
  ["You earn $50 cash", "Every successful referral puts $50 in your pocket."],
];

const dollars = (n) => `$${(n || 0).toFixed(0)}`;

// the email to support that claims what's available, as connect writes it
export function claimMailto(me, codes, referrals, available) {
  const n = referrals.length;
  const body = `Hi comma support,\n\nI would like to claim $${Number(available || 0).toFixed(2)} cash for ${n} referral${n === 1 ? "" : "s"}.\n\n` +
    `Account: ${me.email} (${me.user_id})\nReferral coupon: ${codes}\n\nThanks!`;
  return `mailto:support@comma.ai?subject=${encodeURIComponent("Referral Claim")}&body=${encodeURIComponent(body)}`;
}

function shareSection(summary) {
  if (summary.limit_reached) {
    return h("p", {class: "ref-note"}, "Referral limit reached. Read the ", h("a", {href: TERMS, target: "_blank", rel: "noopener"}, "terms"), " for more.");
  }
  const url = `${REFER}/${encodeURIComponent(summary.code)}`;
  const label = "Share your link";
  const button = h("button", {class: "button primary ref-share"}, label);
  const said = (text) => {
    button.textContent = text;
    setTimeout(() => { button.textContent = label; }, 2000);
  };
  // a phone's share sheet, else the clipboard; closing the sheet isn't a failure
  button.addEventListener("click", async () => {
    if (navigator.share && matchMedia("(pointer: coarse)").matches) {
      try {
        await navigator.share({title: "Give $50, Get $50 with comma", text: "Get $50 off comma four using this referral link.", url});
        return said("Shared");
      } catch (e) {
        if (e.name === "AbortError") return;
      }
    }
    navigator.clipboard.writeText(url).then(() => said("Copied"), () => said("Couldn't copy"));
  });
  return h("div", {}, button, h("div", {class: "ref-link"}, h("span", {class: "mono"}, url), copyButton(() => url, "Copy referral link")));
}

function rewardsSection(summary, me) {
  const cash = summary.cash || {}, available = (summary.referrals || []).filter((r) => r.status === "available");
  const claim = `Claim rewards (${dollars(cash.available)})`;
  return h("div", {},
    h("dl", {class: "ref-cash"},
      h("div", {}, h("dt", {}, "Already claimed"), h("dd", {}, dollars(cash.claimed))),
      h("div", {}, h("dt", {}, "Pending rewards"), h("dd", {}, dollars(cash.pending))),
      h("div", {}, h("dt", {}, "Available to claim"), h("dd", {class: cash.available > 0 ? "earned" : ""}, dollars(cash.available)))),
    cash.available > 0 && available.length && me
      ? h("a", {class: "button primary ref-claim", href: claimMailto(me, (summary.codes || [summary.code]).join(", "), available, cash.available)}, claim)
      : h("button", {class: "button ref-claim", disabled: true}, claim));
}

// ---- user mode: connect's page, its markup and copy ----

const COPY = "M180-81q-24 0-42-18t-18-42v-603h60v603h474v60H180Zm120-120q-24 0-42-18t-18-42v-560q0-24 18-42t42-18h440q24 0 42 18t18 42v560q0 24-18 42t-42 18H300Zm0-60h440v-560H300v560Zm0 0v-560 560Z";
const ring = (size, label) => h("span", {class: "cn-ref-ring", style: `--size: ${size}px`, role: "progressbar", "aria-label": label});
const terms = (text) => h("a", {href: TERMS, target: "_blank", rel: "noopener noreferrer"}, text);

function connectReferrals(me) {
  let summary = null, error = null, status = null, statusTimer = 0;
  const share = h("section", {class: "cn-ref-share"}), mine = h("div", {class: "cn-ref-mine"});
  const url = () => `${REFER}/${encodeURIComponent(summary.code)}`;
  const said = (text, ms) => {
    status = text;
    show();
    clearTimeout(statusTimer);
    statusTimer = setTimeout(() => { status = null; show(); }, ms);
  };
  const copy = () => navigator.clipboard.writeText(url()).then(() => said("copied", 2000), () => said("unable to share", 2000));
  // a phone's share sheet, else the clipboard; closing the sheet isn't a failure
  async function shareLink() {
    if (!navigator.share || !matchMedia("(pointer: coarse)").matches) return copy();
    try {
      await navigator.share({title: "Give $50, Get $50 with comma", text: "Get $50 off comma four using this referral link.", url: url()});
      said("shared", 1500);
    } catch (e) {
      if (e.name !== "AbortError") copy();
    }
  }
  function show() {
    // the share link: invisible (keeping its place) under a spinner until loaded
    const cash = summary?.cash || {available: 0, claimed: 0, pending: 0};
    const hide = (el) => { if (!summary) { el.classList.add("cn-ref-hidden"); el.ariaHidden = "true"; } return el; };
    share.replaceChildren(...[
      summary?.limit_reached ? h("div", {class: "cn-ref-box"}, h("p", {}, "Referral limit reached. Read ", terms("terms"), " for more details."))
        : !error && hide(h("div", {},
          h("button", {type: "button", class: "cn-ref-white", disabled: !summary, onclick: shareLink}, h("span", {}, status || "share your link")),
          h("div", {class: "cn-ref-link"}, h("span", {}, summary ? url() : ""),
            h("button", {type: "button", "aria-label": "Copy referral link", onclick: copy}, glyph(COPY))))),
      !summary && !error && h("div", {class: "cn-ref-loading"}, ring(28, "Loading referral link")),
      error && h("p", {class: "cn-ref-unavailable"}, "Your share link is unavailable.")].filter(Boolean));
    const available = (summary?.referrals || []).filter((r) => r.status === "available");
    const claim = `claim rewards ($${cash.available.toFixed(0)})`;
    mine.replaceChildren(...[
      error && h("div", {class: "cn-ref-box"}, h("p", {}, error.message),
        error.retryable && h("button", {type: "button", class: "cn-ref-retry", onclick: load}, "Retry")),
      !error && hide(h("div", {},
        h("div", {class: "cn-ref-cash"}, h("dl", {},
          h("div", {}, h("dt", {}, "Already claimed:"), h("dd", {}, `$${cash.claimed.toFixed(0)}`)),
          h("div", {}, h("dt", {}, "Pending rewards:"), h("dd", {}, `$${cash.pending.toFixed(0)}`)),
          h("div", {}, h("dt", {}, "Available to claim:"), h("dd", {class: cash.available > 0 ? "earned" : ""}, `$${cash.available.toFixed(0)}`)))),
        summary && cash.available > 0 && available.length && me
          ? claimLink(claimMailto(me, (summary.codes || [summary.code]).join(", "), available, cash.available), claim)
          : h("button", {type: "button", class: "cn-ref-claim off", disabled: true}, claim))),
      !summary && !error && h("div", {class: "cn-ref-loading"}, ring(32, "Loading your referrals"))].filter(Boolean));
  }
  function load() {
    error = null;
    show();
    api.referrals().then((s) => {
      if (!s) throw new Error("Referral request returned no data");
      summary = s;
    }).catch((e) => {
      error = e.status === 403 ? {message: "Referrals are only available for comma device owners", retryable: false}
        : {message: "Could not load referrals. Please try again.", retryable: true};
    }).then(show);
  }
  load();
  return h("main", {class: "cn-referrals"},
    h("h1", {}, "Refer a friend, ", h("br"), "Get $50."),
    h("section", {}, h("ol", {}, STEPS.map(([t, detail], i) => h("li", {},
      h("div", {class: "cn-ref-n"}, h("span", {}, i + 1)),
      h("div", {}, h("span", {class: "cn-sr"}, `Step ${i + 1}: `), h("h2", {}, t), h("p", {}, detail)))))),
    share,
    h("section", {class: "cn-ref-yours"}, h("h2", {}, "Your Referrals"), mine),
    h("p", {class: "cn-ref-foot"}, "Referral rewards become available to claim after each order's return period ends."),
    h("p", {class: "cn-ref-foot"}, "Referrals are subject to ", terms("terms and conditions"), "."));
}

// the claim: opens the mail app, and says so for a moment
function claimLink(href, label) {
  const a = h("a", {class: "cn-ref-claim", href, onclick: (e) => {
    if (a.ariaDisabled === "true") return e.preventDefault();
    a.ariaDisabled = "true";
    a.replaceChildren(ring(20), "Opening mail app…");
    setTimeout(() => { a.ariaDisabled = "false"; a.replaceChildren(label); }, 1500);
  }}, label);
  return a;
}

function glyph(d) {
  const svg = icon("copy", 14);
  svg.setAttribute("viewBox", "0 -960 960 960");
  svg.firstChild.setAttribute("d", d);
  return svg;
}

export async function referralsPage(me) {
  await css(new URL("./account.css", import.meta.url));
  localStorage.setItem(REFERRALS_SEEN, "1");
  if (!auth.isAuthenticated()) return signInCard();
  if (mode() === "user") return connectReferrals(me);
  const body = h("div", {class: "ref-body"}, h("p", {class: "muted"}, h("span", {class: "spinner"}), " Loading your referrals…"));
  const load = () => api.referrals().then((summary) => body.replaceChildren(
    h("section", {}, shareSection(summary)),
    h("h3", {}, "Your referrals"),
    rewardsSection(summary, me),
  ), (e) => body.replaceChildren(h("div", {class: "ref-note"},
    h("p", {}, e.status === 403 ? "Referrals are only for owners of a comma device." : "Couldn't load your referrals."),
    e.status !== 403 && h("button", {class: "button", onclick: load}, "Retry"))));
  load();
  return h("div", {class: "page referrals-page"},
    h("h2", {}, "Refer a friend, get $50"),
    h("ol", {class: "ref-steps"}, STEPS.map(([title, detail], i) => h("li", {}, h("span", {class: "ref-n"}, i + 1),
      h("div", {}, h("b", {}, title), h("p", {class: "small muted"}, detail))))),
    body,
    h("p", {class: "small muted center"}, "Rewards can be claimed once each order's return period ends. Referrals are subject to the ",
      h("a", {href: TERMS, target: "_blank", rel: "noopener"}, "terms and conditions"), "."));
}

// connect's promotion on its device page, for the owner, until dismissed or followed
export async function referralPromo(device) {
  if (!device?.is_owner || localStorage.getItem(PROMO_KEY)) return null;
  await css(new URL("./account.css", import.meta.url));
  let el = null;
  const dismiss = () => {
    localStorage.setItem(PROMO_KEY, "1");
    el.remove();
  };
  // user mode: connect's Notification
  if (mode() === "user") {
    return el = promo("Referral promotion", "Refer a friend. Get $50.",
      h("a", {href: "/connect/referrals", onclick: (e) => { dismiss(); link("/connect/referrals")(e); }}, "refer"),
      "Earn $50 for each comma four purchased with your referral link.", dismiss);
  }
  el = h("div", {class: "ref-promo-slot"}, h("aside", {class: "ref-promo", "aria-label": "Referral promotion"},
    h("div", {}, h("b", {}, "Refer a friend. Get $50."), h("p", {class: "small muted"}, "Earn $50 for each comma four bought with your referral link.")),
    h("a", {class: "button small primary", href: "/referrals", onclick: (e) => { dismiss(); link("/referrals")(e); }}, "Refer"),
    h("button", {class: "icon-btn", title: "Dismiss", "aria-label": "Dismiss referral promotion", onclick: dismiss}, icon("close", 16))));
  return el;
}
