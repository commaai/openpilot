import sys

def int_rnd(x):
  return int(round(x))

def clip(x, lo, hi):
  return max(lo, min(hi, x))

def interp(x, xp, fp):
  N = len(xp)
  T = len(fp)

  def get_interp(xv):
    hi = 0
    while hi < N and xv > xp[hi]:
      hi += 1
    low = hi - 1
    return fp[-1] if hi >= T or low >= T or (hi >= N and xv >= xp[low]) or (N < T and xv == xp[-1]) else (
      fp[0] if hi == 0 else
      (xv - xp[low]) * (fp[hi] - fp[low]) / (xp[hi] - xp[low]) + fp[low])

  return [get_interp(v) for v in x] if hasattr(x, '__iter__') else get_interp(x)

def mean(x):
  return sum(x) / len(x)

def incremental_avg(cur_avg, new_val, new_size):
  return cur_avg + ((new_val - cur_avg) / new_size)

def find_nearest_index(x, y):
  '''
  Finds nearest index of x based on the value(s) of y.
  '''
  def get_nearest(yv):
    dist = sys.maxsize
    i = 0
    xi = 0

    for xv in x:
      d = abs(abs(xv) - abs(yv))
      # print(f'd({d}) = xv({xv}) - yv({yv})')
      if d <= dist:
        dist = d
        i = xi
      xi += 1

    return i

  idxs = [get_nearest(yv) for yv in y] if hasattr(y, '__iter__') else get_nearest(y)

  # if hasattr(y, '__iter__'):
  #   for yv in y:
  #     idxs.append(get_nearest(yv))
  # else:
  #   idxs.append(get_nearest(y))

  return idxs

def is_multi_iter(x):
  multi_iter = False

  if hasattr(x, '__iter__'):
    for i in x:
      if hasattr(i, '__iter__'):
        multi_iter = True
        break

  return multi_iter