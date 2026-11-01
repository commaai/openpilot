// binary searches of a rising array (typed or not) in [lo, hi)

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
