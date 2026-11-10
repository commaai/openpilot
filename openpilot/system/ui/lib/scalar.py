from bisect import bisect_right


def clip(value, low, high):
  return min(high, max(low, value))


def interp(value, xs, ys):
  if value != value:  # NaN, like np.interp
    return value
  if value <= xs[0]:
    return ys[0]
  if value >= xs[-1]:
    return ys[-1]
  i = bisect_right(xs, value) - 1
  t = (value - xs[i]) / (xs[i + 1] - xs[i])
  return ys[i] + t * (ys[i + 1] - ys[i])
