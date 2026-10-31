// comma prime, as connect has it at /<dongle>/prime: checkout (plans, the trial, SIM checks) for a device without it,
// managing the subscription (Stripe's portal, switching plans, cancelling) for one with it, and the sign-up banner.
// The calls go to billing.comma.ai, which only answers pages on comma.ai origins (api.js billingReachable). User mode
// draws all of it as connect does (Prime/PrimeCheckout.jsx, PrimeManage.jsx, Notification): its column, its paper
// dialogs for switching and cancelling, its blocks; developer mode keeps the theme's
import {api, auth, billingReachable} from "../api.js";
import {h, css, fmt, ask} from "../dom.js";
import {icon} from "../icons.js";
import {link} from "../app.js";
import {signInCard} from "./login.js";
import {userMode, cnIcon, cnModal, progress, spinner, promo, fill} from "./cnui.js";

const COMMACARE = "https://comma.ai/connect#what-is-commacare";
const FAQ = "https://comma.ai/connect#faq";
const SIM_SHOP = "https://comma.ai/shop/comma-prime-sim";
const DATA_SIMS = ["blue", "magenta_new", "webbing"];
const PLANS = {nodata: {name: "Lite", price: 14}, data: {name: "Standard", price: 24}};
const other = (plan) => plan === "data" ? "nodata" : "data";
const CARE_ICON = new URL("../icons/commacare.webp", import.meta.url);

const LONG = new Intl.DateTimeFormat(undefined, {month: "long", day: "numeric", year: "numeric"});
const SHORT = new Intl.DateTimeFormat(undefined, {month: "long", day: "numeric"});
const date = (unixSeconds, f = LONG) => f.format((unixSeconds || 0) * 1000);

const ext = (href, text) => h("a", {href, target: "_blank", rel: "noopener"}, text);
// a status block: kind is info, notice, error, success or loading. In user mode connect's: info and notice on its
// grey, an error on red, loading on a shade (success has its own, show's)
const note = (kind, ...children) => h("div", {class: `prime-note prime-${kind}`},
  kind === "loading" ? spinner(19, "#fff")
    : userMode() ? cnIcon({notice: "priority", error: "error"}[kind] ?? "info", 24) : icon({notice: "warning", error: "warning", success: "check"}[kind] ?? "info", 18),
  h("div", {}, children));
const leaving = (url) => location.assign(url);  // to Stripe, which comes back to this page
const backPath = () => location.pathname.replace(/\/prime\/?$/, "");
const backButton = () => {
  const back = backPath();
  return userMode() ? h("a", {class: "cnui-ibtn", href: back, "aria-label": "Go Back", onclick: link(back)}, cnIcon("back", 24))
    : h("a", {class: "icon-btn", href: back, title: "Go back", "aria-label": "Go back", onclick: link(back)}, icon("back", 20));
};
// the device, as connect's head names it: its name, then its dongle id
const deviceName = (device) => [h(userMode() ? "aside" : "b", {}, fmt.deviceName(device)), " ", h("span", {class: "prime-dongle"}, `(${device.dongle_id})`)];

// the page's head: back to the device, which it's for
const head = (device) => h("div", {class: "prime-head"}, backButton(), h("span", {class: "prime-head-device"}, deviceName(device)));

export async function primePage(device) {
  await css(new URL("./prime.css", import.meta.url));
  if (!auth.isAuthenticated()) return signInCard();
  const q = new URLSearchParams(location.search);
  const me = await auth.user();
  const um = userMode();
  return h("div", {class: `page prime-page${um ? " um" : ""}`}, !device.is_owner && !me?.superuser ? (um ? h("p", {class: "prime-body1"}, "No access") : [head(device), h("p", {}, "No access")])
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
    h("a", {class: userMode() ? "cnui-btn white prime-go" : "button primary prime-go", href: there}, "Manage on connect.comma.ai"));
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
  const um = userMode();
  const commacare = device.eligible_features?.commacare;
  let info = null, plan = null;
  const planBox = (key, sub) => h("button", {class: "prime-plan", "data-plan": key, disabled: true, "aria-pressed": "false",
    onclick: () => { plan = key; update(); }},
  h("b", {}, PLANS[key].name.toLowerCase()), h("span", {class: "prime-price"}, `$${PLANS[key].price}/month`), h("span", {class: "small"}, sub));
  const plans = [planBox("nodata", ["bring your own", h("br"), "sim card"]), planBox("data", ["including data plan", h("br"), "only offered in the U.S."])];
  const fetching = h("div", {class: "prime-fetching"}, spinner(38, "#fff"), um ? h("p", {}, "Fetching SIM data") : "Fetching SIM data");
  const disabled = h("div"), error = h("div"), charge = h("p", {class: "small prime-charge"});
  const go = h("button", {class: um ? "cnui-btn white prime-go" : "button primary prime-go", disabled: true, onclick: async () => {
    go.disabled = true;
    go.replaceChildren(spinner(19));
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
    disabled.replaceChildren(...why ? [note("info", h("p", {}, why))] : []);
    go.disabled = !info || !plan;
    go.textContent = info && plan && trialEnd() ? "Claim trial" : "Go to checkout";
    charge.textContent = info && plan && trialEnd() ? `Your first charge will be on ${date(trialEnd(), SHORT)}, then monthly thereafter.` : "";
    charge.hidden = !charge.textContent;
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
  const checks = ["24/7 connectivity", "Take pictures remotely", "1 year storage of drive videos", "Simple SSH for developers", commacare && "commacare extended warranty"].filter(Boolean);
  // the warranty a subscription extends, or that a comma four past the delivery window doesn't get it
  const care = commacare && h("p", {}, "comma four includes a standard 1-year limited warranty. Subscribe within 30 days of delivery to extend your ",
    "coverage to 2 years for the duration of your prime subscription. ", ext(COMMACARE, "Learn more"));
  const past = device.device_type === "four" && !commacare &&
    ["This device is past the delivery window. commacare extended warranty won't be included with this subscription. ", ext(COMMACARE, "What is commacare?")];
  if (um) {
    // connect's column: each block 24px below the last (16px on a phone)
    return [head(device), h("h2", {class: "prime-title"}, "comma prime"),
      h("div", {class: "prime-block prime-checks"}, checks.map((t) => h("div", {}, cnIcon("check", 21), h("p", {}, t)))),
      care && h("div", {class: "prime-block prime-commacare"}, h("div", {}, h("img", {src: CARE_ICON, alt: ""}), h("strong", {}, "INCLUDES COMMACARE")), care),
      past && h("div", {class: "prime-block prime-past"}, h("p", {}, past)),
      h("div", {class: "prime-block prime-plans"}, h("div", {}, plans), fetching),
      h("div", {class: "prime-block-slot"}, disabled),
      h("div", {class: "prime-block"}, h("p", {class: "prime-body1"}, "Learn more about comma prime from our ", ext(FAQ, "FAQ"))),
      error, h("div", {class: "prime-block"}, go), h("div", {class: "prime-block-slot"}, charge)];
  }
  return [head(device), h("h2", {}, "comma prime"),
    h("ul", {class: "prime-checks"}, checks.map((t) => h("li", {}, icon("check", 20), t))),
    care && h("div", {class: "prime-commacare"}, h("b", {}, "INCLUDES COMMACARE"), care),
    past && h("div", {class: "prime-note"}, h("p", {class: "small"}, past)),
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
const switchAbout = (plan) => plan === "data" ? ["The Standard plan costs $24/month, includes a data plan, and is ", h("b", {}, "only available in the U.S.")]
  : "The Lite plan costs $14/month and does not include a data plan.";
const switchDone = (plan) => plan === "data" ? "Your plan now includes data for $24/month." : "Your plan no longer includes data and costs $14/month.";
const careWarning = () => h("span", {}, "Cancelling will ", h("b", {}, "permanently end your extended warranty coverage."),
  " Your standard 1-year warranty still applies for any remaining time. ", ext(COMMACARE, "What is commacare?"));

// switching to plan, the call itself; throws what switchError words
async function switchTo(dongle, plan) {
  const simId = plan === "data" ? (await api.prime.subscribeInfo(dongle)).sim_id : undefined;
  const resp = await api.prime.switchPlan(dongle, plan, simId);
  if (!resp?.success) throw Object.assign(new Error("Unexpected billing response"), {code: "unexpected_response"});
}

function manage(device, stripeSession) {
  const um = userMode();
  const dongle = device.dongle_id;
  // back from checkout, a device without prime yet has no subscription to load until the payment is through
  const waiting = stripeSession && !device.prime;
  // (connect shows nothing until there's a subscription or a checkout to wait for)
  const status = h("div"), body = h("div", {}, !waiting && !um && h("p", {class: "muted"}, h("span", {class: "spinner"}), " Loading your subscription…"));
  let sub = null, paid = null, manageEl = null;  // manageEl: user mode's page, shown once there's a subscription or a checkout

  // after checkout: Stripe's payment first, then billing's webhook making the subscription
  function showStatus() {
    status.replaceChildren(!stripeSession ? ""
      : paid !== "paid" ? note("loading", h("p", {}, "Waiting for confirmed payment"))
      : !sub?.user_id ? note("loading", h("p", {}, "Processing subscription"))
      : um ? h("div", {class: "prime-note prime-success"}, h("p", {}, "comma prime activated"),
        sub.is_prime_sim && h("p", {}, "Connectivity will be enabled as soon as activation propagates to your local cell tower. Rebooting your device may help."))
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
      if (!sub && !um) body.replaceChildren(note("error", h("p", {}, `Couldn't load the subscription: ${e.message}`)));
      return;
    }
    if (next?.user_id) {
      sub = next;
      showStatus();
      return show();
    }
    if (next || repeat) setTimeout(() => body.isConnected && fetchSub(true), 2000);
    body.replaceChildren(stripeSession || um ? "" : h("p", {class: "muted"}, "This device has no comma prime subscription this account can manage."));
  }

  const message = h("div");
  const say = (kind, text) => message.replaceChildren(note(kind, h("p", {}, text)));
  async function portal(button) {
    button.disabled = true;
    try {
      leaving((await api.prime.portal(dongle)).url);
    } catch (e) {
      if (!um) say("error", `Couldn't open the payment portal: ${e.message}`);  // connect says nothing
      button.disabled = false;
    }
  }
  async function switchPlan(button) {
    const plan = other(sub.plan), name = PLANS[plan].name;
    if (!await ask(h("span", {class: "prime-ask"}, h("b", {}, `Switch to ${name} plan`), h("span", {}, switchAbout(plan))), "Confirm switch")) return;
    button.disabled = true;
    button.replaceChildren(h("span", {class: "spinner"}));
    say("loading", "Switching your plan…");
    try {
      await switchTo(dongle, plan);
      await fetchSub();
      say("success", `Your subscription has been switched to ${name} successfully. ${switchDone(plan)}`);
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
      device.commacare && h("span", {class: "warn"}, careWarning())),
    "Cancel subscription", "Close");
    if (!ok) return;
    button.disabled = true;
    say("loading", "Cancelling…");
    const resp = await cancelCall();
    if (!resp.ok) {
      say("error", resp.message);
      button.disabled = false;
      return;
    }
    say("success", "Cancelled subscription.");
    fetchSub();
  }
  async function cancelCall() {
    const unknown = "Could not cancel due to unknown error. Please try again.";
    const resp = await api.prime.cancel(dongle).catch(() => ({}));
    return resp.success ? {ok: true} : {ok: false, message: resp.error && resp.description || unknown};
  }

  // connect's switch dialog (PrimeManage.jsx): confirm, then switching, then its welcome or the error with "Try again"
  function switchModal() {
    const plan = other(sub.plan), name = PLANS[plan].name;
    let busy = false;
    const {el, paper} = cnModal(`Switch to ${name} plan`, "prime-paper", () => {}, () => busy);  // not while it's switching, as connect
    const close = () => !busy && el.close();
    const draw = (state, text = "") => {
      const done = state === "success";
      fill(paper, h("h2", {class: "cnui-title"}, done ? `Welcome to ${name}` : `Switch to ${name} plan`),
        done ? [h("div", {class: "prime-modal-ok"}, h("p", {}, text)), h("p", {class: "prime-modal-text"}, switchDone(plan))]
          : h("p", {class: "prime-modal-text"}, switchAbout(plan)),
        state === "loading" && h("div", {class: "prime-modal-busy"}, progress(19, "#fff"), h("p", {}, "Switching your plan…")),
        state === "error" && h("div", {class: "prime-modal-error"}, h("p", {}, text)),
        h("div", {class: "prime-modal-buttons"}, done ? h("button", {class: "cnui-btn grey prime-modal-close", onclick: close}, "Done")
          : [h("button", {class: "cnui-btn grey prime-modal-confirm", disabled: busy, onclick: go}, state === "error" ? "Try again" : "Confirm switch"),
            h("button", {class: "cnui-btn grey prime-modal-close", disabled: busy, onclick: close}, "Cancel")]));
    };
    async function go() {
      busy = true;
      draw("loading");
      try {
        await switchTo(dongle, plan);
        await fetchSub();
        busy = false;
        draw("success", `Your subscription has been switched to ${name} successfully.`);
      } catch (e) {
        busy = false;
        draw("error", switchError(e, plan));
      }
    }
    draw("confirm");
  }
  // connect's cancel dialog: what cancelling means, its outcome above that, and it stays open after
  function cancelModal() {
    const {el, paper} = cnModal("Cancel prime subscription", "prime-paper cancel");
    const outcome = h("div");
    const confirm = h("button", {class: "cnui-btn grey prime-modal-confirm", onclick: async () => {
      confirm.disabled = true;
      confirm.replaceChildren(progress(19, "#fff"));
      const resp = await cancelCall();
      confirm.textContent = "Cancel subscription";
      outcome.replaceChildren(h("div", {class: resp.ok ? "prime-modal-ok" : "prime-modal-error"}, h("p", {}, resp.ok ? "Cancelled subscription." : resp.message)));
      confirm.disabled = resp.ok;
      if (resp.ok) fetchSub();
    }}, "Cancel subscription");
    paper.append(h("h2", {class: "cnui-title"}, "Cancel prime subscription"), outcome,
      h("p", {}, `Device: ${fmt.deviceName(device)} (${dongle})`), h("p", {}, "We're sorry to see you go."),
      h("p", {}, "Your subscription will be cancelled immediately and can be resumed at any time."),
      device.commacare && h("div", {class: "prime-modal-care"}, h("p", {}, careWarning())),
      h("div", {class: "prime-modal-buttons"}, confirm, h("button", {class: "cnui-btn grey prime-modal-close", onclick: () => el.close()}, "Close")));
  }

  function show() {
    const cancelAt = !!sub.cancel_at && sub.cancel_at <= sub.next_charge_at;
    const discontinued = cancelAt && !device.eligible_features?.prime_data && sub.plan === "data";
    const migrate = sub.requires_migration && note("notice", h("p", {}, "Your prime subscription will be canceled on May 15th unless you replace the SIM card in your ",
      "device. A new SIM card can be ordered from the ", ext(SIM_SHOP, "shop"), ". Use discount code SIMSWAP at checkout to receive a free SIM card."));
    const ended = discontinued && note("info", h("p", {}, `Standard comma prime discontinued for ${fmt.deviceType(device.device_type)}`));
    const planText = sub.plan === "nodata" ? "(without data plan)" : "(with data plan)";
    // user mode's items are connect's headings, and switching and cancelling its dialogs; developer mode's are a list,
    // and the theme's ask()
    const item = (label, ...value) => h("div", {class: "prime-item"}, h(um ? "h3" : "dt", {}, label), h(um ? "p" : "dd", {}, value));
    const items = [
      item("Plan", PLANS[sub.plan]?.name ?? sub.plan, " ", h("span", {class: "muted"}, planText)),
      item("Joined", date(sub.subscribed_at)),
      cancelAt ? item("Subscription end", date(sub.cancel_at)) : item("Next payment", date(sub.next_charge_at)),
      item("Amount", `$${((sub.amount || 0) / 100).toFixed(2)}`)];
    const look = um ? "cnui-btn white" : "button";
    const pay = h("button", {class: `${look} primary prime-go`, disabled: discontinued, onclick: () => portal(pay)},
      cancelAt ? "Renew subscription" : "Update payment method");
    const sw = h("button", {class: `${look} prime-go`, onclick: () => um ? switchModal() : switchPlan(sw)}, `Switch to ${PLANS[other(sub.plan)].name} plan`);
    const stop = h("button", {class: `${look} prime-go prime-cancel`, onclick: () => um ? cancelModal() : cancel(stop)}, "Cancel subscription");
    const rest = [
      device.commacare && h("a", {class: um ? "cnui-care" : "prime-pill", href: COMMACARE, target: "_blank", rel: "noopener"},
        um && h("img", {src: CARE_ICON, alt: ""}), "commacare"),
      message,
      h("div", {class: "prime-buttons"}, pay, !cancelAt && sw, !cancelAt && stop),
      migrate, ended];
    if (um) {
      manageEl.hidden = false;
      return body.replaceChildren(...items, ...rest.filter(Boolean));
    }
    body.replaceChildren(h("div", {}, h("dl", {class: "prime-items"}, item("Device", deviceName(device)), items), rest));
  }

  // connect's: a bar with the back button, then the title, checkout's status and the device above the subscription
  const deviceItem = h("div", {class: "prime-item"},
    h("h3", {}, "Device"), h("div", {class: "prime-item-device"}, h("aside", {}, fmt.deviceName(device)), h("span", {}, `(${dongle})`)));
  if (stripeSession) {
    showStatus();
    pollSession();
  }
  if (!waiting) fetchSub();
  if (um) return manageEl = h("div", {class: "prime-manage", hidden: !stripeSession}, h("div", {class: "prime-bar"}, backButton()),
    h("div", {class: "prime-bar prime-main"}, h("h2", {}, "comma prime"), status, deviceItem, body));
  return [head(device), h("h2", {}, "comma prime"), status, body];
}

// ---- banner ----

const dismissed = new Set();  // as connect: until the page is reloaded

// connect's "sign up" card over the device page, for an owner's device without prime; null when there's none to show
export async function primeBanner(devices) {
  const device = [devices].flat().find((d) => d?.is_owner && !d.prime && !dismissed.has(d.dongle_id));
  if (!device) return null;
  await css(new URL("./prime.css", import.meta.url));
  const to = `${userMode() ? "/connect" : ""}/${device.dongle_id}/prime`;
  const um = userMode();
  const dismiss = () => { dismissed.add(device.dongle_id); el.remove(); };
  const signup = h("a", {class: um ? "" : "button small primary", href: to, onclick: link(to)}, "sign up");
  const text = device.eligible_features?.commacare
    ? "Put your car on the internet with comma prime and extend your 1-year limited warranty with commacare"
    : "Put your car on the internet with comma prime";
  const el = um ? promo("comma prime promotion", "comma prime", signup, text, dismiss)
    : h("div", {class: "prime-promo-slot"}, h("aside", {class: "prime-promo", "aria-label": "comma prime promotion"},
      h("button", {class: "icon-btn prime-dismiss", title: "Dismiss", "aria-label": "Dismiss prime promotion", onclick: dismiss}, icon("close", 14)),
      h("div", {class: "prime-promo-head"}, h("b", {}, "comma prime"), signup), h("p", {}, text)));
  return el;
}
