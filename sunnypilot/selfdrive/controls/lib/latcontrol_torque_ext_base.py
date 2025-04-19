"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
import numpy as np

from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.modeld.constants import ModelConstants

LAT_PLAN_MIN_IDX = 5


def get_predicted_lateral_jerk(lat_accels, t_diffs):
  # compute finite difference between subsequent model_v2.acceleration.y values
  # this is just two calls of np.diff followed by an element-wise division
  lat_accel_diffs = np.diff(lat_accels)
  lat_jerk = lat_accel_diffs / t_diffs
  # return as python list
  return lat_jerk.tolist()


def sign(x):
  return 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)


def get_lookahead_value(future_vals, current_val):
  if len(future_vals) == 0:
    return current_val

  same_sign_vals = [v for v in future_vals if sign(v) == sign(current_val)]

  # if any future val has opposite sign of current val, return 0
  if len(same_sign_vals) < len(future_vals):
    return 0.0

  # otherwise return the value with minimum absolute value
  min_val = min(same_sign_vals + [current_val], key=lambda x: abs(x))
  return min_val


class LatControlTorqueExtBase:
  def __init__(self, lac_torque, CP, CP_SP):
    self.model_v2 = None
    self.model_valid = False
    self.use_steering_angle = lac_torque.use_steering_angle

    self.actual_lateral_jerk: float = 0.0
    self.lateral_jerk_setpoint: float = 0.0
    self.lateral_jerk_measurement: float = 0.0
    self.lookahead_lateral_jerk: float = 0.0

    self.torque_from_lateral_accel = lac_torque.torque_from_lateral_accel
    self.torque_params = lac_torque.torque_params

    self._ff = 0.0
    self._pid_log = None
    self._setpoint = 0.0
    self._measurement = 0.0
    self._lateral_accel_deadzone = 0.0
    self._desired_lateral_accel = 0.0
    self._actual_lateral_accel = 0.0
    self._desired_curvature = 0.0
    self._actual_curvature = 0.0

    # twilsonco's Lateral Neural Network Feedforward
    # Instantaneous lateral jerk changes very rapidly, making it not useful on its own,
    # however, we can "look ahead" to the future planned lateral jerk in order to gauge
    # whether the current desired lateral jerk will persist into the future, i.e.
    # whether it's "deliberate" or not. This allows us to simply ignore short-lived jerk.
    # Note that LAT_PLAN_MIN_IDX is defined above and is used in order to prevent
    # using a "future" value that is actually planned to occur before the "current" desired
    # value, which is offset by the steerActuatorDelay.
    # TODO-SP: Reevaluate lookahead v values that determines how low a desired lateral jerk signal needs to
    #          persist in order to be used.
    self.friction_look_ahead_v = [1.4, 2.0]  # how many seconds in the future to look ahead in [0, ~2.1] in 0.1 increments
    self.friction_look_ahead_bp = [9.0, 30.0]  # corresponding speeds in m/s in [0, ~40] in 1.0 increments

    # Scaling the lateral acceleration "friction response" could be helpful for some.
    # Increase for a stronger response, decrease for a weaker response.
    self.lat_jerk_friction_factor = 0.4
    self.lat_accel_friction_factor = 0.7  # in [0, 3], in 0.05 increments. 3 is arbitrary safety limit

    # precompute time differences between ModelConstants.T_IDXS
    self.t_diffs = np.diff(ModelConstants.T_IDXS)
    self.desired_lat_jerk_time = CP.steerActuatorDelay + 0.3

  def update_model_v2(self, model_v2):
    self.model_v2 = model_v2
    self.model_valid = self.model_v2 is not None and len(self.model_v2.orientation.x) >= CONTROL_N

  def update_friction_input(self, val_1, val_2):
    _error = val_1 - val_2
    _value = self.lat_accel_friction_factor * _error + self.lat_jerk_friction_factor * self.lookahead_lateral_jerk

    return _value

  def update_calculations(self, CS, VM, desired_lateral_accel):
    self.actual_lateral_jerk = 0.0
    self.lateral_jerk_setpoint = 0.0
    self.lateral_jerk_measurement = 0.0
    self.lookahead_lateral_jerk = 0.0

    if self.use_steering_angle:
      actual_curvature_rate = -VM.calc_curvature(math.radians(CS.steeringRateDeg), CS.vEgo, 0.0)
      self.actual_lateral_jerk = actual_curvature_rate * CS.vEgo ** 2

    if self.model_valid:
      # prepare "look-ahead" desired lateral jerk
      lookahead = np.interp(CS.vEgo, self.friction_look_ahead_bp, self.friction_look_ahead_v)
      friction_upper_idx = next((i for i, val in enumerate(ModelConstants.T_IDXS) if val > lookahead), 16)
      predicted_lateral_jerk = get_predicted_lateral_jerk(self.model_v2.acceleration.y, self.t_diffs)
      desired_lateral_jerk = (np.interp(self.desired_lat_jerk_time, ModelConstants.T_IDXS,
                              self.model_v2.acceleration.y) - desired_lateral_accel) / self.desired_lat_jerk_time
      self.lookahead_lateral_jerk = get_lookahead_value(predicted_lateral_jerk[LAT_PLAN_MIN_IDX:friction_upper_idx], desired_lateral_jerk)
      if not self.use_steering_angle or self.lookahead_lateral_jerk == 0.0:
        self.lookahead_lateral_jerk = 0.0
        self.actual_lateral_jerk = 0.0
        self.lat_accel_friction_factor = 1.0
      self.lateral_jerk_setpoint = self.lat_jerk_friction_factor * self.lookahead_lateral_jerk
      self.lateral_jerk_measurement = self.lat_jerk_friction_factor * self.actual_lateral_jerk
