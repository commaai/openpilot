def clip(x, lo, hi):
  return max(lo, min(hi, x))


def interp(x, xp, fp):
  N = len(xp)
  if not hasattr(x, '__iter__'):
    hi = 0
    while hi < N and x > xp[hi]:
      hi += 1
    low = hi - 1
    return fp[-1] if hi == N and x > xp[low] else (
      fp[0] if hi == 0 else
      (x - xp[low]) * (fp[hi] - fp[low]) / (xp[hi] - xp[low]) + fp[low])

  result = []
  for v in x:
    hi = 0
    while hi < N and v > xp[hi]:
      hi += 1
    low = hi - 1
    result.append(fp[-1] if hi == N and v > xp[low] else (fp[
      0] if hi == 0 else (v - xp[low]) * (fp[hi] - fp[low]) / (xp[hi] - xp[low]
                                                               ) + fp[low]))
  return result
