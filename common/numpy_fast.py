import bisect

def clip(x, lo, hi):
  return max(lo, min(hi, x))

def interp(x, xp, fp):
  def get_interp(xv):
    hi = bisect.bisect_left(xp, xv)
    low = hi - 1
    if hi == len(xp):
        return fp[-1]
    if hi == 0:
        return fp[0]
    return (xv - xp[low]) * (fp[hi] - fp[low]) / (xp[hi] - xp[low]) + fp[low]

  return [get_interp(v) for v in x] if hasattr(x, '__iter__') else get_interp(x)

def mean(x):
  return sum(x) / len(x)
