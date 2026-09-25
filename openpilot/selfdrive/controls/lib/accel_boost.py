import numpy as np
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL

ACCEL_BOOST_MAX = 0.5
ACCEL_BOOST_RATE = 0.1
ACCEL_BOOST_MIN_SPEED = 10 * CV.MPH_TO_MS


class AccelBoost:
  def __init__(self, dt=DT_MDL):
    self.dt = dt
    self.value = 0.0

  def update(self, enabled, gas_pressed, v_ego, model_limited):
    if not enabled:
      self.value = 0.0
    elif v_ego < ACCEL_BOOST_MIN_SPEED:
      self.value = max(0.0, self.value - ACCEL_BOOST_RATE * self.dt)
    elif gas_pressed and model_limited:
      self.value = min(ACCEL_BOOST_MAX, self.value + ACCEL_BOOST_RATE * self.dt)

  def apply(self, accel):
    return accel + np.interp(accel, [-1.0, -0.5, 5.0], [0.0, self.value, self.value], right=0.0)
