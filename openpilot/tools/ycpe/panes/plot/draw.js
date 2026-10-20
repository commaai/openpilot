// canvas plumbing shared by the plot panes and the series browser: crisp canvases, ticks, number formats,
// curve marks, and their stylesheet
import {css} from "/dom.js";

css(new URL("./plot.css", import.meta.url));

// categorical slots validated for the dark surface, assigned in order
export const PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
export const STYLES = ["line", "steps", "dots", "line+dots", "sticks"];

// canvas colors from the shell's CSS variables; re-read on theme change
export function colors(theme) {
  const fg = theme.color("--fg-rgb"), a = (x) => `rgba(${fg},${x})`;
  return {
    bg: theme.color("--panel"), grid: a(0.08), zero: a(0.2), text: theme.color("--muted"), fg: theme.color("--fg"),
    playhead: a(0.85), tracker: a(0.4), box: "rgba(252,190,40,.14)", boxEdge: "rgba(232,170,20,.8)",
    invalid: "rgba(201,34,49,.45)", mixed: a(0.12), unloaded: a(0.035), hatch: a(0.07), font: `11px ${theme.color("--mono")}`,
    // state blocks with a meaning of their own (selfdriveState/state); [color, fill alpha]
    named: {enabled: [theme.color("--engaged"), 0.3], preEnabled: [theme.color("--engaged"), 0.12], overriding: [theme.color("--override"), 0.3],
      softDisabling: [theme.color("--alert"), 0.3], disabled: [theme.color("--muted"), 0.14]},
  };
}

// muted state block hues: light fills under --fg labels read in both themes
const STATE = ["#6f8faf", "#5f9e80", "#b39a5a", "#a8647a", "#8a7fc0", "#4f9a9a", "#b07a5a", "#7f8c99"];

export function nextColor(used) {
  return PALETTE.find((c) => !used.includes(c)) || PALETTE[used.length % PALETTE.length];
}

// [color, fill alpha] of a state block: by meaning when its label has one, else by value (or a hash of text)
function stateStyle(v, label, col) {
  if (col.named[label]) return col.named[label];
  if (typeof v === "string") {
    let x = 0;
    for (let i = 0; i < v.length; i++) x = (x * 31 + v.charCodeAt(i)) | 0;
    v = x;
  }
  return [STATE[Math.abs(v) % STATE.length], 0.26];
}

// sizes the backing store to the css size at devicePixelRatio; drawing stays in css px
export function fitCanvas(cv, w, hgt) {
  const r = devicePixelRatio || 1;
  cv.width = Math.max(1, Math.round(w * r));
  cv.height = Math.max(1, Math.round(hgt * r));
  cv.style.width = `${w}px`;
  cv.style.height = `${hgt}px`;
  const g = cv.getContext("2d");
  g.setTransform(r, 0, 0, r, 0, 0);
  return g;
}

// 1-2-5 steps with about `count` ticks over [lo, hi]
export function ticks(lo, hi, count) {
  const raw = (hi - lo) / Math.max(1, count);
  if (!(raw > 0) || !isFinite(raw)) return {step: 1, values: []};
  const mag = 10 ** Math.floor(Math.log10(raw)), f = raw / mag;
  const step = (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * mag;
  const values = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) values.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return {step, values};
}

const decimals = (step) => Math.max(0, Math.min(9, -Math.floor(Math.log10(step) + 1e-9)));

// route seconds: plain seconds, or m:ss / h:mm:ss on a clock
export function fmtTime(t, step = 0.001, clock = Math.abs(t) >= 120) {
  const d = decimals(step);
  if (!clock) return t.toFixed(d);
  const sign = t < 0 ? "-" : "", a = Math.round(Math.abs(t) * 10 ** d) / 10 ** d;  // 119.997 is 2:00.00, not 1:60.00
  const hh = Math.floor(a / 3600), mm = Math.floor(a / 60) % 60, ss = (a % 60).toFixed(d).padStart(d ? d + 3 : 2, "0");
  return hh ? `${sign}${hh}:${String(mm).padStart(2, "0")}:${ss}` : `${sign}${mm}:${ss}`;
}

// s cut with an ellipsis to fit w px in g's font
export function ellipsize(g, s, w) {
  if (g.measureText(s).width <= w) return s;
  let n = s.length;
  while (n > 1 && g.measureText(`${s.slice(0, n)}…`).width > w) n--;
  return `${s.slice(0, n)}…`;
}

export function fmtTick(v, step) {
  const a = Math.abs(step);
  return a >= 1e7 || (Math.abs(v) < 1e-4 && v !== 0) ? v.toExponential(1) : v.toFixed(decimals(a));
}

// a sample for display: enum names, text, integers as is, other numbers to 5 significant figures
export function fmtValue(v, names) {
  if (v === undefined || v === null) return "";
  if (typeof v === "string") return v || "(empty)";
  if (names) return names[v] ?? String(v);
  if (Number.isInteger(v) || !isFinite(v)) return String(v);
  const a = Math.abs(v);
  return a >= 1e5 ? v.toFixed(0) : a < 1e-4 ? v.toExponential(3) : v.toPrecision(5);
}

// one decimated series ({t, min, max}) in a style; X and Y map time and value to css px, and the line
// breaks between consecutive points ta, tb where gap(ta, tb)
export function strokeSeries(g, d, X, Y, style, width, gap = () => false) {
  const {t, min, max} = d, n = t.length;
  if (style !== "dots" && style !== "sticks") {
    const steps = style === "steps";
    g.lineWidth = width;
    g.beginPath();
    let py = NaN, pi = -1;
    for (let i = 0; i < n; i++) {
      const lo = min[i];
      if (lo !== lo) continue;
      const x = X(t[i]), a = Y(lo), b = Y(max[i]);
      // a bucket is a vertical span: enter it at the end nearer the previous point
      const [p, q] = Math.abs(a - py) <= Math.abs(b - py) ? [a, b] : [b, a];
      if (py !== py || gap(t[pi], t[i])) g.moveTo(x, p);
      else {
        if (steps) g.lineTo(x, py);
        g.lineTo(x, p);
      }
      if (q !== p) g.lineTo(x, q);
      py = q;
      pi = i;
    }
    g.stroke();
  }
  if (style === "dots" || style === "line+dots") {
    const r = width + 0.75;
    g.beginPath();
    for (let i = 0; i < n; i++) {
      if (min[i] !== min[i]) continue;
      const x = X(t[i]), a = Y(min[i]), b = Y(max[i]);
      g.rect(x - r, Math.min(a, b) - r, 2 * r, Math.abs(a - b) + 2 * r);
    }
    g.fill();
  }
  if (style === "sticks") {
    const y0 = Y(0);
    g.lineWidth = width;
    g.beginPath();
    for (let i = 0; i < n; i++) {
      if (min[i] !== min[i]) continue;
      const x = X(t[i]), a = Y(min[i]), b = Y(max[i]);
      g.moveTo(x, Math.max(y0, a, b));
      g.lineTo(x, Math.min(y0, a, b));
    }
    g.stroke();
  }
}

// time with no data loaded: a faint band with diagonal hatching, inside the box r
export function drawUnloaded(g, spans, X, r, c) {
  const xs = spans.map(([a, b]) => [Math.max(r.x, Math.round(X(a))), Math.min(r.x + r.w, Math.round(X(b)))]).filter(([a, b]) => b > a);
  if (!xs.length) return;
  g.save();
  g.fillStyle = c.unloaded;
  g.strokeStyle = c.hatch;
  g.lineWidth = 1;
  g.beginPath();
  for (const [x0, x1] of xs) g.rect(x0, r.y, x1 - x0, r.h);
  g.fill();
  g.clip();
  g.beginPath();
  for (let x = r.x - r.h; x < r.x + r.w; x += 8) {
    g.moveTo(x, r.y + r.h);
    g.lineTo(x + r.h, r.y);
  }
  g.stroke();
  g.restore();
}

// labeled state blocks in a row; mixed spans (several states inside a pixel) are gray
export function drawBlocks(g, blocks, X, y0, y1, names, c) {
  g.textAlign = "left";
  g.textBaseline = "middle";
  for (const b of blocks) {
    const x0 = X(b.a), x1 = Math.max(X(b.b), x0 + 1);
    if (b.mixed) {
      g.fillStyle = c.mixed;
      g.fillRect(x0, y0, x1 - x0, y1 - y0);
      continue;
    }
    const label = fmtValue(b.v, names), [color, alpha] = stateStyle(b.v, label, c);
    g.globalAlpha = alpha;
    g.fillStyle = color;
    g.fillRect(x0, y0, x1 - x0, y1 - y0);
    g.globalAlpha = 1;
    g.fillRect(x0, y0, 2, y1 - y0);
    const text = x1 - x0 > 24 && y1 - y0 > 9 ? ellipsize(g, label, x1 - x0 - 10) : "";
    if (text.length > 2) {
      g.fillStyle = c.fg;
      g.fillText(text, x0 + 6, (y0 + y1) / 2);
    }
  }
}

// grid lines, the zero line and tick labels around the plot rect r = {x, y, w, h}
export function axes(g, col, r, xt, X, fmtX, yt, Y) {
  g.lineWidth = 1;
  g.strokeStyle = col.grid;
  g.beginPath();
  for (const v of xt.values) { const x = Math.round(X(v)) + 0.5; g.moveTo(x, r.y); g.lineTo(x, r.y + r.h); }
  for (const v of yt.values) { const y = Math.round(Y(v)) + 0.5; g.moveTo(r.x, y); g.lineTo(r.x + r.w, y); }
  g.stroke();
  const zero = Math.round(Y(0)) + 0.5;
  if (yt.values.length && zero > r.y + 1 && zero < r.y + r.h - 1) {
    g.strokeStyle = col.zero;
    g.beginPath();
    g.moveTo(r.x, zero);
    g.lineTo(r.x + r.w, zero);
    g.stroke();
  }
  g.font = col.font;
  g.fillStyle = col.text;
  g.textAlign = "center";
  g.textBaseline = "top";
  // labels kept inside the pane; one that would touch the previous one is skipped
  let end = -Infinity;
  for (const v of xt.values) {
    const s = fmtX(v), w = g.measureText(s).width / 2, x = Math.min(X(v), r.x + r.w + 6 - w);
    if (x - w < end + 8) continue;
    g.fillText(s, x, r.y + r.h + 5);
    end = x + w;
  }
  g.textAlign = "right";
  g.textBaseline = "middle";
  for (const v of yt.values) g.fillText(fmtTick(v, yt.step), r.x - 6, Y(v));
}
