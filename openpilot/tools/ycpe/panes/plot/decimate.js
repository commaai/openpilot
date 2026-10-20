// store.decimate on a fixed grid of buckets [k·w, (k + 1)·w), cached per path. w is quantized to 1/64 octave
// steps, so a pan keeps the grid: the buckets still in view are reused and only the newly exposed edge is
// decimated. Exact samples (a sparse series) are binned onto the same grid.

export class Decimator {
  constructor(store) {
    this.store = store;
    this.cache = new Map();  // path → {version, w, k0, k1, t, min, max}
    this.used = new Set();
  }

  // ~n buckets covering [a, b] → {t, min, max, w}, or null for text and unknown paths (no version)
  get(path, version, a, b, n) {
    if (version === undefined) return null;
    this.used.add(path);
    const w = 2 ** (Math.round(Math.log2((b - a) / n) * 64) / 64);
    const k0 = Math.floor(a / w), k1 = Math.ceil(b / w), m = k1 - k0;
    const c = this.cache.get(path);
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

  // drops the paths not asked for since the last sweep
  sweep() {
    for (const path of this.cache.keys()) if (!this.used.has(path)) this.cache.delete(path);
    this.used.clear();
  }
}
