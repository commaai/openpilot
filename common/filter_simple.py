class FirstOrderFilter:
  def __init__(self, x0, rc, dt, initialized=True):
    self.x = x0
    self.dt = dt
    self.update_alpha(rc)
    self.initialized = initialized

  def update_alpha(self, rc):
    self.alpha = self.dt / (rc + self.dt)

  def update(self, x):
    if self.initialized:
      self.x = (1. - self.alpha) * self.x + self.alpha * x
    else:
      self.initialized = True
      self.x = x
    return self.x


class JerkEstimator1:
  def __init__(self, dt):
    self.dt = dt
    self.prev_x = 0.0
    self.initialized = False
    self.filter = FirstOrderFilter(0.0, 0.2, dt, initialized=False)

  @property
  def x(self):
    return self.filter.x

  def update(self, x):
    if not self.initialized:
      self.prev_x = x
      self.initialized = True

    self.filter.update((x - self.prev_x) / self.dt)
    self.prev_x = x
    return self.filter.x


class JerkEstimator2:
  def __init__(self, dt):
    self.dt = dt
    self.initialized = False
    from opendbc.car.common.simple_kalman import KF1D, get_kalman_gain
    import numpy as np

    DT_CTRL = 0.01

    Q = [[0.0, 0.0], [0.0, 100.0]]
    R = 0.3
    A = [[1.0, DT_CTRL], [0.0, 1.0]]
    C = [[1.0, 0.0]]
    x0 = [[0.0], [0.0]]
    K = get_kalman_gain(DT_CTRL, np.array(A), np.array(C), np.array(Q), R)
    self.kf = KF1D(x0=x0, A=A, C=C[0], K=K)

  @property
  def x(self):
    return self.kf.x[1][0] if self.initialized else 0.0

  def update(self, x):
    if not self.initialized:
      self.kf.set_x([[x], [0.0]])
      self.initialized = True
    self.kf.update(x)
    return self.kf.x[1][0]


class JerkEstimator3:
  def __init__(self, dt):
    self.dt = dt
    self.prev_x = 0.0
    self.initialized = False
    self.filter = FirstOrderFilter(0.0, 0.2, dt, initialized=False)
    self._x = 0.0

  @property
  def x(self):
    return self._x

  def update(self, x):
    filtered_x = self.filter.update(x)

    if not self.initialized:
      self.prev_x = filtered_x
      self.initialized = True

    self._x = (filtered_x - self.prev_x) / self.dt

    self.prev_x = filtered_x
    return self._x


class JerkEstimator4:
  def __init__(self, dt):
    from collections import deque
    self.dt = dt
    self.xs = deque(maxlen=int(0.25 / dt))
    self._x = 0

  @property
  def x(self):
    return self._x

  def update(self, x):
    self.xs.append(x)
    if len(self.xs) < 2:
      return 0.0

    self._x = (self.xs[-1] - self.xs[0]) / ((len(self.xs) - 1) * self.dt)
    return self._x
