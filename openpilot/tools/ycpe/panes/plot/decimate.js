// store.decimate on a fixed grid of buckets [k·w, (k + 1)·w), cached per path. w is quantized to 1/64 octave
// steps, so a pan keeps the grid: the buckets still in view are reused and only the newly exposed edge is
// decimated. Exact samples (a sparse series) are binned onto the same grid. A zoom keeps the cached grid while
// it's within half an octave of the right one, so a wheel tick re-decimates only what's newly exposed; the right
// grid is decimated SETTLE_MS after the zoom stops, and settled() is called to draw it.
const SETTLE_MS = 150;

export class Decimator {
  constructor(store, settled) {
    this.store = store;
    this.settled = settled;
    this.cache = new Map();  // path → {version, w, k0, k1, t, min, max}
    this.used = new Set();
    this.zooming = false;  // a cached grid stood in for the right one since the last sweep
    this.exact = false;  // the zoom has settled: decimate on the right grid until it's drawn
    this.timer = 0;
  }

  // ~n buckets covering [a, b] → {t, min, max, w}, or null for text and unknown paths (no version)
  get(path, version, a, b, n) {
    if (version === undefined) return null;
    this.used.add(path);
    const c = this.cache.get(path);
    let w = 2 ** (Math.round(Math.log2((b - a) / n) * 64) / 64);
    if (!this.exact && c?.version === version && c.w !== w && Math.abs(Math.log2(c.w / w)) <= 0.5 && c.k0 * c.w < b && a < c.k1 * c.w) {
      w = c.w;
      this.zooming = true;
    }
    const k0 = Math.floor(a / w), k1 = Math.ceil(b / w), m = k1 - k0;
    const keep = c && c.version === version && c.w === w && c.k0 < k1 && k0 < c.k1;
    if (keep && c.k0 === k0 && c.k1 === k1) return c;
    const out = {version, w, k0, k1, t: new Float64Array(m), min: new Float64Array(m).fill(NaN), max: new Float64Array(m).fill(NaN)};
    for (let i = 0; i < m; i++) out.t[i] = (k0 + i + 0.5) * w;
    if (keep) {
      const lo = Math.max(k0, c.k0), hi = Math.min(k1, c.k1);
      out.min.set(c.min.subarray(lo - c.k0, hi - c.k0), lo - k0);
      out.max.set(c.max.subarray(lo - c.k0, hi - c.k0), lo - k0);
      if (k0 < lo && !this.fill(out, path, k0, lo)) return null;
      if (hi < k1 && !this.fill(out, path, hi, k1)) return null;
    } else if (!this.fill(out, path, k0, k1)) return null;
    this.cache.set(path, out);
    return out;
  }

  // decimates buckets [lo, hi) into out
  fill(out, path, lo, hi) {
    const {w, k0} = out, d = this.store.decimate(path, lo * w, hi * w, hi - lo);
    if (!d) return false;
    if (d.min !== d.max) {
      out.min.set(d.min, lo - k0);
      out.max.set(d.max, lo - k0);
      return true;
    }
    for (let i = 0; i < d.t.length; i++) {
      const k = Math.max(lo, Math.min(hi - 1, Math.floor(d.t[i] / w))) - k0, v = d.min[i];
      if (!(out.min[k] <= v)) out.min[k] = v;
      if (!(out.max[k] >= v)) out.max[k] = v;
    }
    return true;
  }

  // after a draw: drops the paths not asked for since the last sweep
  sweep() {
    for (const path of this.cache.keys()) if (!this.used.has(path)) this.cache.delete(path);
    this.used.clear();
    this.exact = false;
    if (!this.zooming) return;
    this.zooming = false;
    clearTimeout(this.timer);
    this.timer = setTimeout(() => { this.exact = true; this.settled(); }, SETTLE_MS);
  }

  dispose() { clearTimeout(this.timer); }
}
