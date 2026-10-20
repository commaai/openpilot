// canvas plumbing shared by the plot panes and the series browser: crisp canvases, ticks, number formats,
// curve marks, and their stylesheet
document.head.append(Object.assign(document.createElement("link"), {rel: "stylesheet", href: new URL("./plot.css", import.meta.url).href}));

// categorical slots validated for the dark surface, assigned in order
export const PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
export const STYLES = ["line", "steps", "dots", "line+dots", "sticks"];

// canvas colors from the shell's CSS variables; re-read on theme change
export function colors(theme) {
  const fg = theme.color("--fg-rgb"), a = (x) => `rgba(${fg},${x})`;
  return {
    bg: theme.color("--panel"), grid: a(0.08), zero: a(0.2), text: theme.color("--muted"), fg: theme.color("--fg"),
    playhead: a(0.85), tracker: a(0.4), box: "rgba(252,190,40,.14)", boxEdge: "rgba(232,170,20,.8)",
    invalid: "rgba(201,34,49,.45)", mixed: a(0.12), font: `11px ${theme.color("--mono")}`,
  };
}

// jotpluggler's state block colors
const STATE = ["#6f8faf", "#00a36c", "#ffc300", "#c70039", "#7b61ff", "#009688", "#d63031", "#8395a7"];

export function nextColor(used) {
  return PALETTE.find((c) => !used.includes(c)) || PALETTE[used.length % PALETTE.length];
}

export function stateColor(v) {
  if (typeof v === "string") {
    let x = 0;
    for (let i = 0; i < v.length; i++) x = (x * 31 + v.charCodeAt(i)) | 0;
    v = x;
  }
  return STATE[Math.abs(v) % STATE.length];
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
  const sign = t < 0 ? "-" : "", a = Math.abs(t);
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
  return a >= 1e6 || (a < 1e-4 && v !== 0) ? v.toExponential(1) : v.toFixed(decimals(a));
}

export function fmtValue(v, names) {
  if (v === undefined || v === null) return "";
  if (typeof v === "string") return v || "(empty)";
  if (names) return names[v] ?? String(v);
  return Number.isInteger(v) ? String(v) : String(+v.toPrecision(6));
}

// one decimated series ({t, min, max}) in a style; X and Y map time and value to css px
export function strokeSeries(g, d, X, Y, style, width) {
  const {t, min, max} = d, n = t.length;
  if (style !== "dots" && style !== "sticks") {
    const steps = style === "steps";
    g.lineWidth = width;
    g.beginPath();
    let py = NaN;
    for (let i = 0; i < n; i++) {
      const lo = min[i];
      if (lo !== lo) continue;
      const x = X(t[i]), a = Y(lo), b = Y(max[i]);
      // a bucket is a vertical span: enter it at the end nearer the previous point
      const [p, q] = Math.abs(a - py) <= Math.abs(b - py) ? [a, b] : [b, a];
      if (py !== py) g.moveTo(x, p);
      else {
        if (steps) g.lineTo(x, py);
        g.lineTo(x, p);
      }
      if (q !== p) g.lineTo(x, q);
      py = q;
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

// labeled state blocks in a row; mixed spans (several states inside a pixel) are gray
export function drawBlocks(g, blocks, X, y0, y1, names, c) {
  g.textBaseline = "middle";
  for (const b of blocks) {
    const x0 = X(b.a), x1 = Math.max(X(b.b), x0 + 1);
    if (b.mixed) {
      g.fillStyle = c.mixed;
      g.fillRect(x0, y0, x1 - x0, y1 - y0);
      continue;
    }
    const color = stateColor(b.v);
    g.globalAlpha = 0.22;
    g.fillStyle = color;
    g.fillRect(x0, y0, x1 - x0, y1 - y0);
    g.globalAlpha = 1;
    g.fillRect(x0, y0, 2, y1 - y0);
    if (x1 - x0 > 16 && y1 - y0 > 9) {
      g.save();
      g.beginPath();
      g.rect(x0 + 5, y0, x1 - x0 - 9, y1 - y0);
      g.clip();
      g.fillStyle = c.fg;
      g.fillText(fmtValue(b.v, names), x0 + 6, (y0 + y1) / 2);
      g.restore();
    }
  }
}
