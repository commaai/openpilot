import math
import numpy as np

from cereal import log
from common.filter_simple import FirstOrderFilter
from common.numpy_fast import clip, interp
from common.realtime import DT_CTRL
from selfdrive.controls.lib.latcontrol import LatControl


class LatControlINDI(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)
    self.angle_steers_des = 0.

    A = np.array([[1.0, DT_CTRL, 0.0],
                  [0.0, 1.0, DT_CTRL],
                  [0.0, 0.0, 1.0]])
    C = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]])

    # Q = np.matrix([[1e-2, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 10.0]])
    # R = np.matrix([[1e-2, 0.0], [0.0, 1e3]])

    # (x, l, K) = control.dare(np.transpose(A), np.transpose(C), Q, R)
    # K = np.transpose(K)
    K = np.array([[7.30262179e-01, 2.07003658e-04],
                  [7.29394177e+00, 1.39159419e-02],
                  [1.71022442e+01, 3.38495381e-02]])

    self.speed = 0.

    self.K = K
    self.A_K = A - np.dot(K, C)
    self.x = np.array([[0.], [0.], [0.]])

    self._RC = (CP.lateralTuning.indi.timeConstantBP, CP.lateralTuning.indi.timeConstantV)
    self._G = (CP.lateralTuning.indi.actuatorEffectivenessBP, CP.lateralTuning.indi.actuatorEffectivenessV)
    self._outer_loop_gain = (CP.lateralTuning.indi.outerLoopGainBP, CP.lateralTuning.indi.outerLoopGainV)
    self._inner_loop_gain = (CP.lateralTuning.indi.innerLoopGainBP, CP.lateralTuning.indi.innerLoopGainV)

    self.steer_filter = FirstOrderFilter(0., self.RC, DT_CTRL)
    self.reset()

  @property
  def RC(self):
    return interp(self.speed, self._RC[0], self._RC[1])

  @property
  def G(self):
    return interp(self.speed, self._G[0], self._G[1])

  @property
  def outer_loop_gain(self):
    return interp(self.speed, self._outer_loop_gain[0], self._outer_loop_gain[1])

  @property
  def inner_loop_gain(self):
    return interp(self.speed, self._inner_loop_gain[0], self._inner_loop_gain[1])

  def reset(self):
    super().reset()
    self.steer_filter.x = 0.
    self.speed = 0.

  def update(self, active, CS, VM, params, last_actuators, steer_limited, desired_curvature, desired_curvature_rate, llk):
    self.speed = CS.vEgo
    # Update Kalman filter
    y = np.array([[math.radians(CS.steeringAngleDeg)], [math.radians(CS.steeringRateDeg)]])
    self.x = np.dot(self.A_K, self.x) + np.dot(self.K, y)

    indi_log = log.ControlsState.LateralINDIState.new_message()
    indi_log.steeringAngleDeg = math.degrees(self.x[0])
    indi_log.steeringRateDeg = math.degrees(self.x[1])
    indi_log.steeringAccelDeg = math.degrees(self.x[2])

    steers_des = VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll)
    steers_des += math.radians(params.angleOffsetDeg)
    indi_log.steeringAngleDesiredDeg = math.degrees(steers_des)

    # desired rate is the desired rate of change in the setpoint, not the absolute desired curvature
    rate_des = VM.get_steer_from_curvature(-desired_curvature_rate, CS.vEgo, 0)
    indi_log.steeringRateDesiredDeg = math.degrees(rate_des)

    if not active:
      indi_log.active = False
      self.steer_filter.x = 0.0
      output_steer = 0
    else:
      # Expected actuator value
      self.steer_filter.update_alpha(self.RC)
      self.steer_filter.update(last_actuators.steer)

      # Compute acceleration error
      rate_sp = self.outer_loop_gain * (steers_des - self.x[0]) + rate_des
      accel_sp = self.inner_loop_gain * (rate_sp - self.x[1])
      accel_error = accel_sp - self.x[2]

      # Compute change in actuator
      g_inv = 1. / self.G
      delta_u = g_inv * accel_error

      # If steering pressed, only allow wind down
      if CS.steeringPressed and (delta_u * last_actuators.steer > 0):
        delta_u = 0

      output_steer = self.steer_filter.x + delta_u

      output_steer = clip(output_steer, -self.steer_max, self.steer_max)

      indi_log.active = True
      indi_log.rateSetPoint = float(rate_sp)
      indi_log.accelSetPoint = float(accel_sp)
      indi_log.accelError = float(accel_error)
      indi_log.delayedOutput = float(self.steer_filter.x)
      indi_log.delta = float(delta_u)
      indi_log.output = float(output_steer)
      indi_log.saturated = self._check_saturation(self.steer_max - abs(output_steer) < 1e-3, CS, steer_limited)

    return float(output_steer), float(steers_des), indi_log
