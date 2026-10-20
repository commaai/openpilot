// series math off the main thread: transforms, statistics, CSV, XY pairs and spectra. This module is both
// the main-side API and, loaded as a module worker, the worker itself. The worker keeps the chunks of each
// series it has seen and their concatenation, so a chunk crosses threads once.

// transform kinds: fn(t, v, param, inputs) → {t, v}; `inputs` are extra series resampled to t
export const TRANSFORMS = {
  derivative: {label: "Derivative", param: {label: "dt (0 = sample)", value: 0}, fn: derivative},
  integral: {label: "Integral", fn: integral},
  diff: {label: "First difference", fn: (t, v) => ({t: t.subarray(1), v: v.subarray(1).map((x, i) => x - v[i])})},
  abs: {label: "Absolute value", fn: (t, v) => ({t, v: v.map(Math.abs)})},
  mean: {label: "Moving average", param: {label: "window (samples)", value: 10}, fn: (t, v, n) => moving(t, v, n, (s, q, k) => s / k)},
  rms: {label: "Moving RMS", param: {label: "window (samples)", value: 10}, fn: (t, v, n) => moving(t, v, n, (s, q, k) => Math.sqrt(q / k))},
  variance: {label: "Moving variance", param: {label: "window (samples)", value: 10}, fn: (t, v, n) => moving(t, v, n, (s, q, k) => Math.max(0, q / k - (s / k) ** 2))},
  outliers: {label: "Outlier removal", param: {label: "threshold (σ)", value: 4}, fn: outliers},
  roll: {label: "Quaternion → roll", quat: true, fn: (t, v, p, q) => ({t, v: rpy(v, q, 0)})},
  pitch: {label: "Quaternion → pitch", quat: true, fn: (t, v, p, q) => ({t, v: rpy(v, q, 1)})},
  yaw: {label: "Quaternion → yaw", quat: true, fn: (t, v, p, q) => ({t, v: rpy(v, q, 2)})},
};

// the store path of a transformed curve
export function derivedPath(path, {kind, param}) {
  return `${path}[${kind}${param !== undefined && param !== null ? ` ${param}` : ""}]`;
}

// quaternion transforms take the w, x, y, z siblings of a list path (…/0 … …/3)
export const transformInputs = (path, {kind}) => TRANSFORMS[kind].quat ? [0, 1, 2, 3].map((k) => path.replace(/\/\d+$/, `/${k}`)) : [path];

function derivative(t, v, dt) {
  const ot = [], ov = [];
  for (let i = 1; i < t.length; i++) {
    const d = dt > 0 ? dt : t[i] - t[i - 1];
    if (d > 0) { ot.push(t[i]); ov.push((v[i] - v[i - 1]) / d); }
  }
  return {t: Float64Array.from(ot), v: Float64Array.from(ov)};
}

function integral(t, v) {
  const out = new Float64Array(v.length);
  for (let i = 1; i < v.length; i++) out[i] = out[i - 1] + (v[i] + v[i - 1]) / 2 * (t[i] - t[i - 1]);
  return {t, v: out};
}

// trailing window of n samples; f(sum, sum of squares, count)
function moving(t, v, n, f) {
  n = Math.max(1, Math.round(n));
  const out = new Float64Array(v.length);
  let s = 0, q = 0;
  for (let i = 0; i < v.length; i++) {
    s += v[i];
    q += v[i] * v[i];
    if (i >= n) { s -= v[i - n]; q -= v[i - n] * v[i - n]; }
    out[i] = f(s, q, Math.min(i + 1, n));
  }
  return {t, v: out};
}

// drops samples further than k standard deviations from the mean of the 25 samples around them
function outliers(t, v, k) {
  const W = 12, ot = [], ov = [];
  let s = 0, q = 0, lo = 0, hi = 0;
  for (let i = 0; i < v.length; i++) {
    for (; hi < Math.min(v.length, i + W + 1); hi++) { s += v[hi]; q += v[hi] * v[hi]; }
    for (; lo < i - W; lo++) { s -= v[lo]; q -= v[lo] * v[lo]; }
    // compare against the window without the sample itself
    const m = hi - lo - 1, mean = (s - v[i]) / m, sd = Math.sqrt(Math.max(0, (q - v[i] * v[i]) / m - mean * mean));
    if (!(m > 0) || Math.abs(v[i] - mean) <= k * sd + 1e-12) { ot.push(t[i]); ov.push(v[i]); }
  }
  return {t: Float64Array.from(ot), v: Float64Array.from(ov)};
}

function rpy(_, [w, x, y, z], axis) {
  const out = new Float64Array(w.length);
  for (let i = 0; i < w.length; i++) {
    const a = w[i], b = x[i], c = y[i], d = z[i];
    out[i] = axis === 0 ? Math.atan2(2 * (a * b + c * d), 1 - 2 * (b * b + c * c))
      : axis === 1 ? Math.asin(Math.max(-1, Math.min(1, 2 * (a * c - d * b))))
        : Math.atan2(2 * (a * d + b * c), 1 - 2 * (c * c + d * d));
  }
  return out;
}

// ---- worker side

// the worker's copy of store series, per store id: id → Map(path → {version, chunks, t, v})
const stores = new Map();
let series = null;  // the current request's store's

// applies a message from Feed.feed() to the worker's copy; → its store's series, or null for a drop message
export function receive({store, data, drop}) {
  if (drop !== undefined) {
    stores.delete(drop);
    return null;
  }
  if (!stores.has(store)) stores.set(store, new Map());
  const held = stores.get(store);
  for (const [path, {version, chunks: sent}] of Object.entries(data)) {
    const old = held.get(path)?.chunks, chunks = sent.map((c) => typeof c === "number" ? old[c] : c);
    held.set(path, {version, chunks, ...concat(chunks)});
  }
  return held;
}

// one {t, v} from store chunks ({tt, v}), without missing values
export function concat(chunks) {
  let n = 0, text = false;
  for (const c of chunks) {
    n += c.tt.length;
    text ||= !ArrayBuffer.isView(c.v) && c.v.some((x) => typeof x === "string");
  }
  const t = new Float64Array(n), v = text ? new Array(n) : new Float64Array(n);
  let k = 0;
  for (const {tt, v: cv} of chunks) {
    for (let j = 0; j < tt.length; j++) {
      const x = cv[j];
      if (x === null || x !== x) continue;
      t[k] = tt[j];
      v[k++] = x;
    }
  }
  return {t: t.subarray(0, k), v: text ? v.slice(0, k) : v.subarray(0, k)};
}

// first k with t[k] > x
function upper(t, x) {
  let lo = 0, hi = t.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (t[mid] <= x) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// v resampled onto ref with sample-and-hold (the first value before the series starts)
export function resample(ref, t, v) {
  const out = new Float64Array(ref.length);
  let k = 0;
  for (let i = 0; i < ref.length; i++) {
    while (k < t.length && t[k] <= ref[i]) k++;
    out[i] = v.length ? v[Math.max(0, k - 1)] : NaN;
  }
  return out;
}

const MAX_FFT = 65536;  // samples a spectrum is computed from at most

// sample index range [a, b) of s over [t0, t1]
function range(s, t0, t1) {
  let a = upper(s.t, t0);
  while (a > 0 && s.t[a - 1] === t0) a--;
  return [a, upper(s.t, t1)];
}

const ops = {
  derive({path, transform: {kind, param}, inputs}) {
    const [first, ...rest] = inputs.map((p) => series.get(p));
    const extra = [first.v, ...rest.map((s) => resample(first.t, s.t, s.v))];
    const r = TRANSFORMS[kind].fn(first.t, first.v, param, extra);
    return {t: Float64Array.from(r.t), v: Float64Array.from(r.v)};
  },

  stats({curves, t0, t1}) {
    return curves.map(({path, shift = 0, scale = 1, offset = 0}) => {
      const s = series.get(path);
      if (!s || !ArrayBuffer.isView(s.v)) return null;
      const [a, b] = range(s, t0 - shift, t1 - shift);
      let min = Infinity, max = -Infinity, sum = 0, sq = 0;
      for (let i = a; i < b; i++) {
        const x = s.v[i] * scale + offset;
        if (x < min) min = x;
        if (x > max) max = x;
        sum += x;
        sq += x * x;
      }
      const count = b - a, mean = sum / count;
      return {min, max, mean, std: Math.sqrt(Math.max(0, sq / count - mean * mean)), count, rate: count > 1 ? (count - 1) / (s.t[b - 1] - s.t[a]) : 0};
    });
  },

  // one row per distinct sample time over the range; blank where a column has no sample at that time
  csv({curves, t0, t1}) {
    const cols = curves.map((c) => {
      const s = series.get(c.path), [a, b] = range(s, t0 - (c.shift || 0), t1 - (c.shift || 0));
      return {...c, s, k: a, b};
    });
    const esc = (x) => /[",\n]/.test(x) ? `"${x.replaceAll('"', '""')}"` : x;
    const rows = [["time", ...cols.map((c) => esc(c.label))].join(",")];
    for (;;) {
      let t = Infinity;
      for (const c of cols) if (c.k < c.b) t = Math.min(t, c.s.t[c.k] + (c.shift || 0));
      if (t === Infinity) break;
      rows.push([t.toFixed(6), ...cols.map((c) => {
        if (c.k >= c.b || c.s.t[c.k] + (c.shift || 0) !== t) return "";
        const x = c.s.v[c.k++];
        return typeof x === "string" ? esc(x) : String(x * (c.scale ?? 1) + (c.offset ?? 0));
      })].join(","));
    }
    return rows.join("\n") + "\n";
  },

  // y sampled at x's times over the range, thinned to points that move at least 1/2000 of the extent
  xy({x, y, t0, t1}) {
    const sx = series.get(x), sy = series.get(y), [a, b] = range(sx, t0, t1);
    const tx = sx.t.subarray(a, b), vx = sx.v.subarray(a, b), vy = resample(tx, sy.t, sy.v);
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < tx.length; i++) {
      if (vx[i] < x0) x0 = vx[i];
      if (vx[i] > x1) x1 = vx[i];
      if (vy[i] < y0) y0 = vy[i];
      if (vy[i] > y1) y1 = vy[i];
    }
    const ex = (x1 - x0) / 2000 || 1, ey = (y1 - y0) / 2000 || 1;
    const ox = [], oy = [];
    for (let i = 0, px = NaN, py = NaN; i < tx.length; i++) {
      if (i === tx.length - 1 || !(Math.abs(vx[i] - px) < ex && Math.abs(vy[i] - py) < ey)) {
        ox.push(px = vx[i]);
        oy.push(py = vy[i]);
      }
    }
    return {x: Float64Array.from(ox), y: Float64Array.from(oy), bounds: [x0, x1, y0, y1], count: tx.length};
  },

  // single-sided amplitude spectrum of the range: interpolated onto a uniform grid at the median sample interval
  // (so it ends at the series' own Nyquist frequency), Hann windowed and zero-padded to a power of two
  fft({path, t0, t1}) {
    const s = series.get(path), [a, b] = range(s, t0, t1), count = b - a;
    if (count < 4) return {f: new Float64Array(0), mag: new Float64Array(0), bounds: [0, 1, 0, 1]};
    const gaps = s.t.subarray(a + 1, b).map((x, i) => x - s.t[a + i]).sort(), span = s.t[b - 1] - s.t[a];
    const dt = Math.max(gaps[gaps.length >> 1], span / (MAX_FFT - 1)), m = Math.floor(span / dt) + 1;
    const n = 2 ** Math.ceil(Math.log2(m)), re = new Float64Array(n), im = new Float64Array(n);
    let mean = 0;
    for (let i = 0, k = a; i < m; i++) {
      const t = s.t[a] + i * dt;
      while (k < b - 2 && s.t[k + 1] <= t) k++;
      const w = Math.min(1, (t - s.t[k]) / (s.t[k + 1] - s.t[k]) || 0);
      re[i] = s.v[k] + (s.v[k + 1] - s.v[k]) * w;
      mean += re[i] / m;
    }
    for (let i = 0; i < m; i++) re[i] = (re[i] - mean) * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (m - 1)));
    fftInPlace(re, im);
    const half = n / 2 + 1, f = new Float64Array(half), mag = new Float64Array(half);
    for (let k = 0; k < half; k++) {
      f[k] = k / (n * dt);
      mag[k] = 4 * Math.hypot(re[k], im[k]) / m;  // ×2 single-sided, ×2 Hann gain
    }
    let top = 0;
    for (let k = 1; k < half; k++) top = Math.max(top, mag[k]);
    return {f, mag, bounds: [f[1], f[half - 1], 0, top]};
  },
};

function fftInPlace(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let j = 0; j < len / 2; j++) {
        const a = i + j, b = a + len / 2;
        const xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr;
        im[b] = im[a] - xi;
        re[a] += xr;
        im[a] += xi;
        [cr, ci] = [cr * wr - ci * wi, cr * wi + ci * wr];
      }
    }
  }
}

if (globalThis.WorkerGlobalScope && self.location.href === import.meta.url) {
  onmessage = ({data: msg}) => {
    const {id, op, args} = msg;
    if (!(series = receive(msg))) return;
    try {
      const result = ops[op](args);
      postMessage({id, result}, Object.values(result || {}).filter(ArrayBuffer.isView).map((a) => a.buffer));
    } catch (e) {
      postMessage({id, error: String(e)});
    }
  };
}

// ---- main side

let worker = null, feed = null, nextId = 0;
const pending = new Map();

// exact copies of chunks' samples: a view would clone its whole pack buffer. Their buffers go in transfer
export function copy({tt, v}, transfer) {
  const c = {tt: tt.slice(), v: ArrayBuffer.isView(v) ? v.slice() : v};
  transfer.push(c.tt.buffer, ...ArrayBuffer.isView(c.v) ? [c.v.buffer] : []);
  return c;
}

// what a worker holds of the stores' series, so a chunk crosses threads once. Each store has an id there, and
// a store that's gone is dropped from the worker's copy (receive() takes the messages)
let lastStore = 0;
const storeIds = new WeakMap();
export class Feed {
  constructor(worker) {
    this.worker = worker;
    this.sent = new WeakMap();  // store → Map(path → {version, at: WeakMap(chunk → its index in the worker's copy)})
    this.gone = new FinalizationRegistry((id) => this.worker.postMessage({drop: id}));
  }

  // the fields of a message for a worker that reads `paths` of store: of a changed series only the chunks the
  // worker doesn't hold go, the others as their index in its copy. Their buffers go in transfer
  feed(store, paths, transfer) {
    const all = [...new Set(paths)].map((path) => [path, store.series(path)]), none = all.find(([, s]) => !s);
    if (none) throw new Error(`no series ${none[0]}`);
    if (!storeIds.has(store)) storeIds.set(store, ++lastStore);
    if (!this.sent.has(store)) {
      this.sent.set(store, new Map());
      this.gone.register(store, storeIds.get(store));
    }
    const have = this.sent.get(store), data = {};
    for (const [path, s] of all) {
      const old = have.get(path);
      if (old?.version === s.version) continue;
      const at = new WeakMap();
      data[path] = {version: s.version, chunks: s.chunks.map((c, k) => {
        const held = old?.at.get(c);
        at.set(c, k);
        return held ?? copy(c, transfer);
      })};
      have.set(path, {version: s.version, at});
    }
    return {store: storeIds.get(store), data};
  }
}

// runs op in the worker; `paths` are the store series it reads
export function compute(store, op, args, paths) {
  if (!worker) {
    worker = new Worker(import.meta.url, {type: "module"});
    worker.onmessage = ({data: {id, result, error}}) => {
      const p = pending.get(id);
      pending.delete(id);
      error ? p.reject(new Error(error)) : p.resolve(result);
    };
    feed = new Feed(worker);
  }
  const id = nextId++, transfer = [];
  try {
    worker.postMessage({id, op, args, ...feed.feed(store, paths, transfer)}, transfer);
  } catch (e) {
    return Promise.reject(e);
  }
  return new Promise((resolve, reject) => pending.set(id, {resolve, reject}));
}

// computes a transformed curve into the store under derivedPath(); resolves with that path
export async function derive(store, path, transform) {
  const out = derivedPath(path, transform), inputs = transformInputs(path, transform);
  const {t, v} = await compute(store, "derive", {path, transform, inputs}, inputs);
  store.put(out, {type: "f64", t, v});
  return out;
}
