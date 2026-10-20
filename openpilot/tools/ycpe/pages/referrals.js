// referrals, as connect has them: the share link, what's been earned, claiming by email; the top bar's gift and the
// device page's "refer a friend" card for owners
import {api, auth} from "../api.js";
import {h, css} from "../dom.js";
import {icon} from "../icons.js";
import {REFERRALS_SEEN, link} from "../app.js";
import {copyButton} from "./account.js";
import {signInCard} from "./login.js";

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

export async function referralsPage(me) {
  await css(new URL("./account.css", import.meta.url));
  localStorage.setItem(REFERRALS_SEEN, "1");
  if (!auth.isAuthenticated()) return signInCard();
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
  const dismiss = () => {
    localStorage.setItem(PROMO_KEY, "1");
    el.remove();
  };
  const el = h("div", {class: "ref-promo-slot"}, h("aside", {class: "ref-promo", "aria-label": "Referral promotion"},
    h("div", {}, h("b", {}, "Refer a friend. Get $50."), h("p", {class: "small muted"}, "Earn $50 for each comma four bought with your referral link.")),
    h("a", {class: "button small primary", href: "/referrals", onclick: (e) => { dismiss(); link("/referrals")(e); }}, "Refer"),
    h("button", {class: "icon-btn", title: "Dismiss", "aria-label": "Dismiss referral promotion", onclick: dismiss}, icon("close", 16))));
  return el;
}
