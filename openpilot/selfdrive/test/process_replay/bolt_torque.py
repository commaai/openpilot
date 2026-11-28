import json

import numpy as np

from opendbc.car.gm.values import CarControllerParams
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque


class BoltAssistMap:
  def __init__(self, calibration):
    self.speed = np.array(calibration['speed_kph'], dtype=float) / 3.6
    self.handwheel = np.array(calibration['handwheel_nm'], dtype=float)
    self.motor = np.array(calibration['motor_nm'], dtype=float)
    self.scale = float(calibration['motor_nm_per_normalized_torque'])
    if self.speed.ndim != 1 or len(self.speed) < 2 or not np.all(np.diff(self.speed) > 0):
      raise ValueError("expected increasing speed breakpoints")
    if self.handwheel.ndim != 2 or self.handwheel.shape != self.motor.shape or self.handwheel.shape[0] != len(self.speed) or self.handwheel.shape[1] < 2:
      raise ValueError("expected matching handwheel and motor torque rows for each speed")
    if not all(np.isfinite(x).all() for x in (self.speed, self.handwheel, self.motor, self.scale)) or self.scale <= 0:
      raise ValueError("expected finite calibration values and a positive motor torque scale")
    if (np.any(self.handwheel[:, 0] != 0) or np.any(self.motor[:, 0] != 0) or
        np.any(np.diff(self.handwheel) <= 0) or np.any(np.diff(self.motor) < 0)):
      raise ValueError("expected monotone assist curves starting at zero")

    self.max_counts = CarControllerParams.STEER_MAX
    self.commands = np.arange(-self.max_counts, self.max_counts + 1) / self.max_counts
    self.update(0., 0.)

  def update(self, speed, driver_torque):
    i = int(np.clip(np.searchsorted(self.speed, speed) - 1, 0, len(self.speed) - 2))
    weight = np.interp(speed, self.speed[i:i + 2], [0., 1.])
    # In controller polarity, one Bolt command count adds 0.01 handwheel N·m.
    handwheel = driver_torque + self.commands * (self.max_counts * 0.01)
    magnitude = np.abs(handwheel)
    lo = np.interp(magnitude, self.handwheel[i], self.motor[i])
    hi = np.interp(magnitude, self.handwheel[i + 1], self.motor[i + 1])
    assist = np.sign(handwheel) * (lo + weight * (hi - lo))
    self.delta = assist - assist[self.max_counts]

  def command(self, normalized_motor_torque):
    target = normalized_motor_torque * self.scale
    # Search from zero so flat parts of the map choose the smallest overlay.
    response = self.delta[self.max_counts:] if target >= 0. else self.delta[self.max_counts::-1]
    counts = np.argmin(np.abs(response - target))
    return float(np.sign(target) * counts / self.max_counts)

  def normalized_motor_torque(self, command):
    return float(np.interp(command, self.commands, self.delta) / self.scale)


class LatControlBoltReplay(LatControlTorque):
  def __init__(self, CP, CI, dt, calibration_path):
    super().__init__(CP, CI, dt)
    with open(calibration_path) as f:
      self.assist = BoltAssistMap(json.load(f))

    def command_from_accel(accel, params):
      return self.assist.command(CI.torque_from_lateral_accel_linear(accel, params))

    def accel_from_command(command, params):
      return CI.lateral_accel_from_torque_linear(self.assist.normalized_motor_torque(command), params)

    self.torque_from_lateral_accel = command_from_accel
    self.lateral_accel_from_torque = accel_from_command
    self.update_limits()

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, curvature_limited, lat_delay):
    # LatControlTorque negates its output to convert to CAN polarity.
    self.assist.update(CS.vEgo, -CS.steeringTorque)
    self.update_limits()
    return super().update(active, CS, VM, params, steer_limited_by_safety, desired_curvature, curvature_limited, lat_delay)
