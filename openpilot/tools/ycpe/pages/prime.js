// comma prime, as connect has it at /<dongle>/prime: checkout (plans, the trial, SIM checks) for a device without it,
// managing the subscription (Stripe's portal, switching plans, cancelling) for one with it, and the sign-up banner.
// The calls go to billing.comma.ai, which only answers pages on comma.ai origins (api.js billingReachable)
import {api, auth, billingReachable} from "../api.js";
import {h, css, fmt, ask} from "../dom.js";
import {icon} from "../icons.js";
import {link} from "../app.js";
import {signInCard} from "./login.js";

const COMMACARE = "https://comma.ai/connect#what-is-commacare";
const FAQ = "https://comma.ai/connect#faq";
const SIM_SHOP = "https://comma.ai/shop/comma-prime-sim";
const DATA_SIMS = ["blue", "magenta_new", "webbing"];
const PLANS = {nodata: {name: "Lite", price: 14}, data: {name: "Standard", price: 24}};
const other = (plan) => plan === "data" ? "nodata" : "data";

const LONG = new Intl.DateTimeFormat(undefined, {month: "long", day: "numeric", year: "numeric"});
const SHORT = new Intl.DateTimeFormat(undefined, {month: "long", day: "numeric"});
const date = (unixSeconds, f = LONG) => f.format((unixSeconds || 0) * 1000);

const ext = (href, text) => h("a", {href, target: "_blank", rel: "noopener"}, text);
// a status block: kind is info, notice, error, success or loading
const note = (kind, ...children) => h("div", {class: `prime-note prime-${kind}`},
  kind === "loading" ? h("span", {class: "spinner"}) : icon({notice: "warning", error: "warning", success: "check"}[kind] ?? "info", 18), h("div", {}, children));
const leaving = (url) => location.assign(url);  // to Stripe, which comes back to this page

// the page's head: back to the device, which it's for
function head(device) {
  const back = location.pathname.replace(/\/prime\/?$/, "");
  return h("div", {class: "prime-head"},
    h("a", {class: "icon-btn", href: back, title: "Go back", "aria-label": "Go back", onclick: link(back)}, icon("back", 20)),
    h("span", {}, h("b", {}, fmt.deviceName(device)), " ", h("span", {class: "prime-dongle"}, `(${device.dongle_id})`)));
}

export async function primePage(device) {
  await css(new URL("./prime.css", import.meta.url));
  if (!auth.isAuthenticated()) return signInCard();
  const q = new URLSearchParams(location.search);
  const me = await auth.user();
  return h("div", {class: "page prime-page"}, !device.is_owner && !me?.superuser ? [head(device), h("p", {}, "No access")]
    : !billingReachable() ? [head(device), h("h2", {}, "comma prime"), unreachable(device)]
    : device.prime || q.has("stripe_success") ? manage(device, q.get("stripe_success"))
    : checkout(device, q.has("stripe_cancelled")));
}

// billing won't answer this origin, and nothing relays it: say where it works
function unreachable(device) {
  const there = `https://connect.comma.ai/${device.dongle_id}/prime`;
  return h("div", {},
    note("info", h("p", {}, "comma prime is managed through billing.comma.ai, which only answers pages served from comma.ai ",
      `(or localhost:3000). This page is on ${location.host}.`)),
    h("a", {class: "button primary prime-go", href: there}, "Manage on connect.comma.ai"));
}

// ---- checkout ----

// why the Standard plan can't be chosen, in connect's words; null when it can
function noDataPlan(device, info) {
  const shop = (text) => [text, ext(SIM_SHOP, "shop")];
  if (!device.eligible_features?.prime_data) return "Standard plan is not available for your device.";
  if (!info.sim_id && info.device_online) return "Standard plan not available, no SIM was detected. Ensure SIM is securely inserted and try again.";
  if (!info.sim_id) return "Standard plan not available, device could not be reached. Connect device to the internet and try again.";
  if (!info.is_prime_sim) return "Standard plan not available, detected a third-party SIM.";
  if (!DATA_SIMS.includes(info.sim_type)) return shop("Standard plan not available, old SIM type detected, new SIM cards are available in the ");
  if (info.sim_usable === false && info.sim_type === "blue") {
    return shop("Standard plan not available, SIM has been canceled and is therefore no longer usable, new SIM cards are available in the ");
  }
  if (info.sim_usable === false) return shop("Standard plan not available, SIM is no longer usable, new SIM cards are available in the ");
  return null;
}

function checkout(device, cancelled) {
  const commacare = device.eligible_features?.commacare;
  let info = null, plan = null;
  const planBox = (key, sub) => h("button", {class: "prime-plan", "data-plan": key, disabled: true, "aria-pressed": "false",
    onclick: () => { plan = key; update(); }},
  h("b", {}, PLANS[key].name.toLowerCase()), h("span", {class: "prime-price"}, `$${PLANS[key].price}/month`), h("span", {class: "small"}, sub));
  const plans = [planBox("nodata", ["bring your own", h("br"), "sim card"]), planBox("data", ["including data plan", h("br"), "only offered in the U.S."])];
  const fetching = h("div", {class: "prime-fetching"}, h("span", {class: "spinner"}), "Fetching SIM data");
  const disabled = h("div"), error = h("div"), charge = h("p", {class: "small prime-charge"});
  const go = h("button", {class: "button primary prime-go", disabled: true, onclick: async () => {
    go.disabled = true;
    go.replaceChildren(h("span", {class: "spinner"}));
    try {
      const resp = await api.prime.checkout(device.dongle_id, plan === "data" ? info.sim_id : undefined, plan);
      if (!resp.url) throw new Error(resp.error || "Billing didn't answer with a checkout");
      return leaving(resp.url);  // the button stays busy until the page goes
    } catch (e) {
      error.replaceChildren(note("error", h("p", {}, e.message)));
    }
    update();
  }});
  const trialEnd = () => info?.[plan === "data" ? "trial_end_data" : "trial_end_nodata"];
  function update() {
    const why = info && noDataPlan(device, info);
    plans.forEach((el) => {
      el.disabled = !info || (el.dataset.plan === "data" && !!why);
      el.setAttribute("aria-pressed", String(el.dataset.plan === plan));
    });
    fetching.hidden = !!info;
    disabled.replaceChildren(why ? note("info", h("p", {}, why)) : "");
    go.disabled = !info || !plan;
    go.textContent = info && plan && trialEnd() ? "Claim trial" : "Go to checkout";
    charge.textContent = info && plan && trialEnd() ? `Your first charge will be on ${date(trialEnd(), SHORT)}, then monthly thereafter.` : "";
  }
  if (cancelled) error.append(note("error", h("p", {}, "Checkout cancelled")));
  api.prime.subscribeInfo(device.dongle_id).then((resp) => {
    if (resp.error) throw new Error(resp.error);  // billing's 200 when the SIM lookup failed
    info = resp;
    plan = noDataPlan(device, info) ? "nodata" : "data";
    update();
  }).catch((e) => {
    fetching.replaceChildren(h("span", {class: "error"}, `Couldn't fetch SIM data: ${e.message}`));
  });
  update();
  return [head(device), h("h2", {}, "comma prime"),
    h("ul", {class: "prime-checks"}, ["24/7 connectivity", "Take pictures remotely", "1 year storage of drive videos", "Simple SSH for developers",
      commacare && "commacare extended warranty"].filter(Boolean).map((t) => h("li", {}, icon("check", 20), t))),
    commacare && h("div", {class: "prime-commacare"}, h("b", {}, "INCLUDES COMMACARE"),
      h("p", {}, "comma four includes a standard 1-year limited warranty. Subscribe within 30 days of delivery to extend your coverage to 2 years ",
        "for the duration of your prime subscription. ", ext(COMMACARE, "Learn more"))),
    device.device_type === "four" && !commacare && h("div", {class: "prime-note"}, h("p", {class: "small"},
      "This device is past the delivery window. commacare extended warranty won't be included with this subscription. ", ext(COMMACARE, "What is commacare?"))),
    h("div", {class: "prime-plans"}, plans, fetching),
    disabled,
    h("p", {}, "Learn more about comma prime from our ", ext(FAQ, "FAQ")),
    error, go, charge];
}

// ---- manage ----

// what went wrong switching to plan, as connect says it
function switchError(e, plan) {
  const name = PLANS[plan].name;
  if (e.code === "unexpected_response") return `Billing returned an unexpected response. Your plan may not have changed. Please try switching to ${name} again.`;
  if (e.status === 400) {
    return plan === "data" ? "Standard could not be activated. Make sure this device is eligible and has a usable, supported comma SIM, then try again."
      : `This subscription could not be switched to ${name}. Please refresh the page and try again.`;
  }
  if (e.status === 401) return "Your session has expired. Sign in again, then try switching plans.";
  if (e.status === 403) return "You don't have permission to change this device's subscription.";
  if (e.status >= 500) return "The billing service is having trouble right now. Your plan was not changed. Please try again later.";
  return "Could not reach the billing service. Check your internet connection and try again.";
}

function manage(device, stripeSession) {
  const dongle = device.dongle_id;
  // back from checkout, a device without prime yet has no subscription to load until the payment is through
  const waiting = stripeSession && !device.prime;
  const status = h("div"), body = h("div", {}, !waiting && h("p", {class: "muted"}, h("span", {class: "spinner"}), " Loading your subscription…"));
  let sub = null, paid = null;

  // after checkout: Stripe's payment first, then billing's webhook making the subscription
  function showStatus() {
    status.replaceChildren(!stripeSession ? ""
      : paid !== "paid" ? note("loading", h("p", {}, "Waiting for confirmed payment"))
      : !sub?.user_id ? note("loading", h("p", {}, "Processing subscription"))
      : h("div", {class: "prime-note prime-success"}, h("div", {}, h("p", {}, h("b", {}, "comma prime activated")),
        sub.is_prime_sim && h("p", {}, "Connectivity will be enabled as soon as activation propagates to your local cell tower. Rebooting your device may help."))));
  }
  async function pollSession() {
    try {
      paid = (await api.prime.session(dongle, stripeSession)).payment_status;
    } catch (e) {
      return status.replaceChildren(note("error", h("p", {}, `Couldn't check the payment: ${e.message}`)));
    }
    showStatus();
    if (paid === "paid") fetchSub(true);
    else setTimeout(() => body.isConnected && pollSession(), 2000);
  }
  // the subscription; it 404s (or has no user) until billing has made it, which repeat waits out
  async function fetchSub(repeat = false) {
    let next;
    try {
      next = await api.primeSubscription(dongle);
    } catch (e) {
      if (!sub) body.replaceChildren(note("error", h("p", {}, `Couldn't load the subscription: ${e.message}`)));
      return;
    }
    if (next?.user_id) {
      sub = next;
      showStatus();
      return show();
    }
    if (next || repeat) setTimeout(() => body.isConnected && fetchSub(true), 2000);
    body.replaceChildren(stripeSession ? "" : h("p", {class: "muted"}, "This device has no comma prime subscription this account can manage."));
  }

  const message = h("div");
  const say = (kind, text) => message.replaceChildren(note(kind, h("p", {}, text)));
  async function portal(button) {
    button.disabled = true;
    try {
      leaving((await api.prime.portal(dongle)).url);
    } catch (e) {
      say("error", `Couldn't open the payment portal: ${e.message}`);
      button.disabled = false;
    }
  }
  async function switchPlan(button) {
    const plan = other(sub.plan), name = PLANS[plan].name;
    const about = plan === "data" ? ["The Standard plan costs $24/month, includes a data plan, and is ", h("b", {}, "only available in the U.S.")]
      : "The Lite plan costs $14/month and does not include a data plan.";
    if (!await ask(h("span", {class: "prime-ask"}, h("b", {}, `Switch to ${name} plan`), h("span", {}, about)), "Confirm switch")) return;
    button.disabled = true;
    button.replaceChildren(h("span", {class: "spinner"}));
    say("loading", "Switching your plan…");
    try {
      const simId = plan === "data" ? (await api.prime.subscribeInfo(dongle)).sim_id : undefined;
      const resp = await api.prime.switchPlan(dongle, plan, simId);
      if (!resp?.success) throw Object.assign(new Error("Unexpected billing response"), {code: "unexpected_response"});
      await fetchSub();
      say("success", `Your subscription has been switched to ${name} successfully. ` +
        (plan === "data" ? "Your plan now includes data for $24/month." : "Your plan no longer includes data and costs $14/month."));
    } catch (e) {
      say("error", switchError(e, plan));
      button.disabled = false;
      button.textContent = `Switch to ${name} plan`;
    }
  }
  async function cancel(button) {
    const ok = await ask(h("span", {class: "prime-ask"}, h("b", {}, "Cancel prime subscription"),
      h("span", {}, `Device: ${fmt.deviceName(device)} (${dongle})`), h("span", {}, "We're sorry to see you go."),
      h("span", {}, "Your subscription will be cancelled immediately and can be resumed at any time."),
      device.commacare && h("span", {class: "warn"}, h("span", {}, "Cancelling will ", h("b", {}, "permanently end your extended warranty coverage."),
        " Your standard 1-year warranty still applies for any remaining time. ", ext(COMMACARE, "What is commacare?")))),
    "Cancel subscription", "Close");
    if (!ok) return;
    button.disabled = true;
    say("loading", "Cancelling…");
    const unknown = "Could not cancel due to unknown error. Please try again.";
    const resp = await api.prime.cancel(dongle).catch(() => ({}));
    if (!resp.success) {
      say("error", resp.error && resp.description || unknown);
      button.disabled = false;
      return;
    }
    say("success", "Cancelled subscription.");
    fetchSub();
  }

  function show() {
    const cancelAt = !!sub.cancel_at && sub.cancel_at <= sub.next_charge_at;
    const discontinued = cancelAt && !device.eligible_features?.prime_data && sub.plan === "data";
    const item = (label, ...value) => h("div", {class: "prime-item"}, h("dt", {}, label), h("dd", {}, value));
    const pay = h("button", {class: "button primary prime-go", disabled: discontinued, onclick: () => portal(pay)},
      cancelAt ? "Renew subscription" : "Update payment method");
    const sw = h("button", {class: "button prime-go", onclick: () => switchPlan(sw)}, `Switch to ${PLANS[other(sub.plan)].name} plan`);
    const stop = h("button", {class: "button prime-go prime-cancel", onclick: () => cancel(stop)}, "Cancel subscription");
    body.replaceChildren(h("div", {},
      h("dl", {class: "prime-items"},
        item("Device", h("b", {}, fmt.deviceName(device)), " ", h("span", {class: "prime-dongle"}, `(${dongle})`)),
        item("Plan", PLANS[sub.plan]?.name ?? sub.plan, " ", h("span", {class: "muted"}, sub.plan === "nodata" ? "(without data plan)" : "(with data plan)")),
        item("Joined", date(sub.subscribed_at)),
        cancelAt ? item("Subscription end", date(sub.cancel_at)) : item("Next payment", date(sub.next_charge_at)),
        item("Amount", `$${((sub.amount || 0) / 100).toFixed(2)}`)),
      device.commacare && h("a", {class: "prime-pill", href: COMMACARE, target: "_blank", rel: "noopener"}, "commacare"),
      message,
      h("div", {class: "prime-buttons"}, pay, !cancelAt && sw, !cancelAt && stop),
      sub.requires_migration && note("notice", h("p", {}, "Your prime subscription will be canceled on May 15th unless you replace the SIM card in your ",
        "device. A new SIM card can be ordered from the ", ext(SIM_SHOP, "shop"), ". Use discount code SIMSWAP at checkout to receive a free SIM card.")),
      discontinued && note("info", h("p", {}, `Standard comma prime discontinued for ${fmt.deviceType(device.device_type)}`))));
  }

  if (stripeSession) {
    showStatus();
    pollSession();
  }
  if (!waiting) fetchSub();
  return [head(device), h("h2", {}, "comma prime"), status, body];
}

// ---- banner ----

const dismissed = new Set();  // as connect: until the page is reloaded

// connect's "sign up" card over the device page, for an owner's device without prime; null when there's none to show
export async function primeBanner(devices) {
  const device = [devices].flat().find((d) => d?.is_owner && !d.prime && !dismissed.has(d.dongle_id));
  if (!device) return null;
  await css(new URL("./prime.css", import.meta.url));
  const to = `/${device.dongle_id}/prime`;
  const el = h("div", {class: "prime-promo-slot"}, h("aside", {class: "prime-promo", "aria-label": "comma prime promotion"},
    h("button", {class: "icon-btn prime-dismiss", title: "Dismiss", "aria-label": "Dismiss prime promotion",
      onclick: () => { dismissed.add(device.dongle_id); el.remove(); }}, icon("close", 14)),
    h("div", {class: "prime-promo-head"}, h("b", {}, "comma prime"), h("a", {class: "button small primary", href: to, onclick: link(to)}, "sign up")),
    h("p", {}, device.eligible_features?.commacare
      ? "Put your car on the internet with comma prime and extend your 1-year limited warranty with commacare"
      : "Put your car on the internet with comma prime")));
  return el;
}
