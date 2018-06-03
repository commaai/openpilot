def int_rnd(x):
  return int(round(x))

def clip(x, lo, hi):
  return max(lo, min(hi, x))

def interp(x, xp, fp):
  return [_get_interp(v, xp, fp) for v in x] if hasattr(
    x, '__iter__') else _get_interp(x, xp, fp)
  
def _get_interp(xv, xp, fp):
  N = len(xp)
  hi = 0
  while hi < N and xv > xp[hi]:
    hi += 1
  low = hi - 1
  return fp[-1] if hi == N and xv > xp[low] else (
    fp[0] if hi == 0 else 
	  (xv - xp[low]) * (fp[hi] - fp[low]) / (xp[hi] - xp[low]) + fp[low])
