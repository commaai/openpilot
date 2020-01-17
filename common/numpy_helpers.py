import bisect
import numpy as np
from scipy.interpolate import interp1d


def deep_interp_0_fast(dx, x, y):
  FIX = False
  if len(y.shape) == 1:
    y = y[:, None]
    FIX = True
  ret = np.zeros((dx.shape[0], y.shape[1]))
  index = list(x)
  for i in range(dx.shape[0]):
    idx = bisect.bisect_left(index, dx[i])
    if idx == x.shape[0]:
      idx = x.shape[0] - 1
    ret[i] = y[idx]

  if FIX:
    return ret[:, 0]
  else:
    return ret


def running_mean(x, N):
  cumsum = np.cumsum(np.insert(x, [0]*(int(N/2)) + [-1]*(N-int(N/2)), [x[0]]*int(N/2) + [x[-1]]*(N-int(N/2))))
  return (cumsum[N:] - cumsum[:-N]) / N


def deep_interp_np(x, xp, fp):
  x = np.atleast_1d(x)
  xp = np.array(xp)
  if len(xp) < 2:
    return np.repeat(fp, len(x), axis=0)
  if min(np.diff(xp)) < 0:
    raise RuntimeError('Bad x array for interpolation')
  j = np.searchsorted(xp, x) - 1
  j = np.clip(j, 0, len(xp)-2)
  d = np.divide(x - xp[j], xp[j + 1] - xp[j], out=np.ones_like(x, dtype=np.float64), where=xp[j + 1] - xp[j] != 0)
  vals_interp = (fp[j].T*(1 - d)).T + (fp[j + 1].T*d).T
  if len(vals_interp) == 1:
    return vals_interp[0]
  else:
    return vals_interp


def clipping_deep_interp(x, xp, fp):
  if len(xp) < 2:
    return deep_interp_np(x, xp, fp)
  bad_idx = np.where(np.diff(xp) < 0)[0]
  if len(bad_idx) > 0:
    if bad_idx[0] ==1:
      return np.zeros([] + list(fp.shape[1:]))
    return deep_interp_np(x, xp[:bad_idx[0]], fp[:bad_idx[0]])
  else:
    return deep_interp_np(x, xp, fp)


def deep_interp(dx, x, y, kind="slinear"):
  return interp1d(
    x, y,
    axis=0,
    kind=kind,
    bounds_error=False,
    fill_value="extrapolate",
    assume_sorted=True)(dx)
