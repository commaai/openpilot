// referrals, connect's page (its markup and copy): the share link, what's been earned, claiming by email; and the device
// page's "refer a friend" card for owners
import {api, auth} from "../api.js";
import {h, css} from "../dom.js";
import {cnIcon} from "./connect/icons.js";
import {REFERRALS_SEEN, link} from "../app.js";
import {signInCard} from "./login.js";
import {promo, ring} from "./cnui.js";

const REFER = "https://refer.comma.ai";
const TERMS = "https://comma.ai/terms#referral-terms";
const PROMO_KEY = "ycpe_referrals_promo_2026-09-02";  // connect's campaign: a new one shows the card again
const STEPS = [
  ["Share your link", "Send it to a friend who is interested in comma four."],
  ["They save $50", "Your link takes $50 off their comma four order."],
  ["You earn $50 cash", "Every successful referral puts $50 in your pocket."],
];

const dollars = (n) => `$${(n || 0).toFixed(0)}`;

const shareUrl = (summary) => `${REFER}/${encodeURIComponent(summary.code)}`;

// the link from a phone's share sheet, else (or with sheet false) copied: "shared", "copied" or "failed", or null when
// the sheet was closed, which isn't a failure
async function shareLink(url, sheet = true) {
  if (sheet && navigator.share && matchMedia("(pointer: coarse)").matches) {
    try {
      await navigator.share({title: "Give $50, Get $50 with comma", text: "Get $50 off comma four using this referral link.", url});
      return "shared";
    } catch (e) {
      if (e.name === "AbortError") return null;
    }
  }
  return navigator.clipboard.writeText(url).then(() => "copied", () => "failed");
}

// the email to support that claims what's available, as connect writes it; null when there's nothing to claim
function claimMailto(me, summary) {
  const available = (summary?.referrals || []).filter((r) => r.status === "available"), cash = summary?.cash?.available;
  if (!(cash > 0) || !available.length || !me) return null;
  const n = available.length, codes = (summary.codes || [summary.code]).join(", ");
  const body = `Hi comma support,\n\nI would like to claim $${cash.toFixed(2)} cash for ${n} referral${n === 1 ? "" : "s"}.\n\n` +
    `Account: ${me.email} (${me.user_id})\nReferral coupon: ${codes}\n\nThanks!`;
  return `mailto:support@comma.ai?subject=${encodeURIComponent("Referral Claim")}&body=${encodeURIComponent(body)}`;
}

// what's been claimed, what's pending and what can be claimed, as connect labels them
const cashRows = (cash) => [["Already claimed:", cash.claimed], ["Pending rewards:", cash.pending], ["Available to claim:", cash.available]]
  .map(([label, n], i) => h("div", {}, h("dt", {}, label), h("dd", {class: i === 2 && n > 0 ? "earned" : ""}, dollars(n))));

const terms = (text) => h("a", {href: TERMS, target: "_blank", rel: "noopener noreferrer"}, text);

function connectReferrals(me) {
  let summary = null, error = null, status = null, statusTimer = 0;
  const shareBox = h("section", {class: "cn-ref-share"}), mine = h("div", {class: "cn-ref-mine"});
  // what sharing did, said on the button for a moment
  const share = async (sheet) => {
    const done = await shareLink(shareUrl(summary), sheet);
    if (!done) return;
    status = {shared: "shared", copied: "copied", failed: "unable to share"}[done];
    show();
    clearTimeout(statusTimer);
    statusTimer = setTimeout(() => { status = null; show(); }, done === "shared" ? 1500 : 2000);
  };
  function show() {
    // the share link: invisible (keeping its place) under a spinner until loaded
    const cash = summary?.cash || {};
    const hide = (el) => { if (!summary) { el.classList.add("cn-ref-hidden"); el.ariaHidden = "true"; } return el; };
    shareBox.replaceChildren(...[
      summary?.limit_reached ? h("div", {class: "cn-ref-box"}, h("p", {}, "Referral limit reached. Read ", terms("terms"), " for more details."))
        : !error && hide(h("div", {},
          h("button", {type: "button", class: "cn-ref-white", disabled: !summary, onclick: () => share(true)}, h("span", {}, status || "share your link")),
          h("div", {class: "cn-ref-link"}, h("span", {}, summary ? shareUrl(summary) : ""),
            h("button", {type: "button", "aria-label": "Copy referral link", onclick: () => share(false)}, cnIcon("copy", 14))))),
      !summary && !error && h("div", {class: "cn-ref-loading"}, ring(28, "Loading referral link")),
      error && h("p", {class: "cn-ref-unavailable"}, "Your share link is unavailable.")].filter(Boolean));
    const claim = `claim rewards (${dollars(cash.available)})`, mailto = claimMailto(me, summary);
    mine.replaceChildren(...[
      error && h("div", {class: "cn-ref-box"}, h("p", {}, error.message),
        error.retryable && h("button", {type: "button", class: "cn-ref-retry", onclick: load}, "Retry")),
      !error && hide(h("div", {},
        h("div", {class: "cn-ref-cash"}, h("dl", {}, cashRows(cash))),
        mailto ? claimLink(mailto, claim) : h("button", {type: "button", class: "cn-ref-claim off", disabled: true}, claim))),
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
    shareBox,
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

export async function referralsPage(me) {
  await css(new URL("./account.css", import.meta.url));
  localStorage.setItem(REFERRALS_SEEN, "1");
  if (!auth.isAuthenticated()) return signInCard();
  return connectReferrals(me);
}

// connect's promotion on its device page, for the owner, until dismissed or followed
export function referralPromo(device) {
  if (!device?.is_owner || localStorage.getItem(PROMO_KEY)) return null;
  const dismiss = () => {
    localStorage.setItem(PROMO_KEY, "1");
    el.remove();
  };
  // connect's Notification
  const el = promo("Referral promotion", "Refer a friend. Get $50.",
    h("a", {href: "/connect/referrals", onclick: (e) => { dismiss(); link("/connect/referrals")(e); }}, "refer"),
    "Earn $50 for each comma four purchased with your referral link.", dismiss);
  return el;
}
