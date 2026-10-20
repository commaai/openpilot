// state blocks for enum and text series: runs of equal values per chunk (built once per chunk, then cached),
// cut to the visible range so that runs narrower than a pixel merge into one mixed block: O(pixels · log n)

const cache = new WeakMap();  // chunk values → {t: run starts, v: run values, end}

function runs(c) {
  let r = cache.get(c.v);
  if (r) return r;
  const len = c.i ? c.i.length : c.t.length, at = (k) => c.t[c.i ? c.i[k] : k];
  const t = [], v = [];
  for (let k = 0; k < len; k++) {
    const x = c.v[k];
    if (x === null || x !== x) continue;
    if (!v.length || x !== v[v.length - 1]) { t.push(at(k)); v.push(x); }
  }
  r = {t: Float64Array.from(t), v, end: len ? at(len - 1) : -Infinity};
  cache.set(c.v, r);
  return r;
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

// [{a, b, v} | {a, b, mixed: true}] over [t0, t1] at px pixels
export function blocks(series, t0, t1, px) {
  const out = [], dt = (t1 - t0) / Math.max(1, px);
  const push = (a, b, v, mixed) => {
    a = Math.max(a, t0);
    b = Math.min(b, t1);
    if (b < a) return;
    const last = out[out.length - 1];
    if (last && a - last.b < dt && (mixed ? last.mixed : !last.mixed && last.v === v)) last.b = b;
    else out.push(mixed ? {a, b, mixed} : {a, b, v});
  };
  for (const c of series.chunks) {
    const r = runs(c), n = r.t.length;
    if (!n || r.end < t0 || r.t[0] > t1) continue;
    let k = Math.max(0, upper(r.t, t0) - 1);
    while (k < n && r.t[k] <= t1) {
      const a = r.t[k], b = k + 1 < n ? r.t[k + 1] : r.end;
      if (b - a >= dt || k + 1 === n) {
        push(a, b, r.v[k], false);
        k++;
      } else {
        const next = upper(r.t, a + dt) - 1;  // > k, since run k is narrower than a pixel
        push(a, r.t[next], null, true);
        k = next;
      }
    }
  }
  return out;
}
