import numpy as np


def deep_interp_np(x, xp, fp, axis=None):
  if axis is not None:
    fp = fp.swapaxes(0,axis)
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
  if axis is not None:
    vals_interp = vals_interp.swapaxes(0,axis)
  if len(vals_interp) == 1:
    return vals_interp[0]
  else:
    return vals_interp
