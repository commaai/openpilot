// a route's name shown part by part, so comma's naming can be read off the name itself. A segment's full name is
// "<dongle id>/<log id>/<segment>" (formerly "<dongle id>|<log id>--<segment>"): the dongle id names the device;
// loggerd's log id is an 8 hex digit counter of the device's routes, "--", then 10 random hex digits that keep the name
// unique should the counter start over (older openpilot named routes by their start time on the device's clock
// instead); segments are the route's minutes, from 0.
import {h, css} from "/dom.js";

css(new URL("./routeid.css", import.meta.url));

const COUNTED = /^([0-9a-f]{8})--([0-9a-f]{10})$/, TIMED = /^(\d{4}-\d\d-\d\d)--(\d\d-\d\d-\d\d)$/;

// the log id's parts: [{kind, text, label, detail}], kind being counter, random, time or other
export function logIdParts(logId) {
  const counted = COUNTED.exec(logId), timed = TIMED.exec(logId);
  if (counted) {
    const n = parseInt(counted[1], 16);
    return [
      {kind: "counter", text: counted[1], label: `route ${n}`, detail: `counts this device's routes, in hex: 0x${n.toString(16)} = ${n}`},
      {kind: "random", text: counted[2], label: "random", detail: "keeps the name unique, should the counter start over"},
    ];
  }
  if (timed) return [{kind: "time", text: logId, label: "start time", detail: "when the route started by the device's clock, which may be off (older openpilot)"}];
  return [{kind: "other", text: logId, label: "log id", detail: "not a name openpilot gives its routes"}];
}

// the counter's leading zeros, dimmed: the digits that matter stand out
function partText(p) {
  if (p.kind !== "counter") return p.text;
  const lead = p.text.match(/^0*(?=.)/)[0];
  return [h("i", {}, lead), p.text.slice(lead.length)];
}

// the log id (and segment) as the top bar shows it: each part its own color, its meaning in its tooltip. segment: a
// span whose text the caller keeps current, or null
export function routeId(logId, segment = null) {
  const parts = logIdParts(logId);
  return h("span", {class: "rid"}, parts.map((p, i) => [
    i > 0 && h("span", {class: "rid-sep"}, "--"),
    h("span", {class: `rid-${p.kind}`, title: `${p.label}: ${p.detail}`}, partText(p)),
  ]), segment && [h("span", {class: "rid-sep"}, "/"), segment]);
}

// the whole name with each part labelled under it, then what each means: the crumb's hover card and its menu.
// route: an API route record; segment: the segment at the playhead (null: none); facts: more lines under it
export function routeAnatomy(route, segment, facts = []) {
  const [dongle, logId] = route.fullname.split("|");
  const parts = [{kind: "dongle", text: dongle, label: "device", detail: "the dongle id: one per comma device"}, ...logIdParts(logId)];
  if (segment !== null) parts.push({kind: "segment", text: String(segment), label: "segment", detail: "routes are logged a minute at a time, numbered from 0; this one is at the playhead"});
  // the log id's own parts are joined by "--", the name's by "/"
  const seps = parts.slice(1).map((p, i) => i === 0 || p.kind === "segment" ? "/" : "--");
  return h("div", {class: "rid-anatomy"},
    h("div", {class: "rid-diagram", style: `grid-template-columns: repeat(${2 * parts.length - 1}, auto)`},
      parts.map((p, i) => [i > 0 && h("span", {class: "rid-sep"}, seps[i - 1]), h("span", {class: `rid-${p.kind}`}, partText(p))]),
      parts.map((p, i) => [i > 0 && h("span"), h("span", {class: `rid-label rid-${p.kind}`}, p.label)])),
    h("dl", {}, parts.map((p) => [h("dt", {class: `rid-${p.kind}`}, p.label), h("dd", {}, p.detail)])),
    facts.filter(Boolean).length > 0 && h("p", {class: "muted small"}, facts.filter(Boolean).join(" · ")));
}
