// binary searches in [lo, hi)

// the first k with pred(k) false, where pred holds up to some k and fails from there on
export function partition(lo, hi, pred) {
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (pred(mid)) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// a rising array's (typed or not), written out: through partition these hot ones run up to 3x slower

// the first k with a[k] >= x
export function lowerBound(a, x, lo = 0, hi = a.length) {
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] < x) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// the first k with a[k] > x
export function upperBound(a, x, lo = 0, hi = a.length) {
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] <= x) lo = mid + 1; else hi = mid;
  }
  return lo;
}
