"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.nnlc import NeuralNetworkLateralControl


class LatControlTorqueExt(NeuralNetworkLateralControl):
  def __init__(self, lac_torque, CP, CP_SP):
    super().__init__(lac_torque, CP, CP_SP)

  def update(self, CS, VM, params, ff, pid_log, setpoint, measurement, calibrated_pose, roll_compensation,
             desired_lateral_accel, actual_lateral_accel, lateral_accel_deadzone, gravity_adjusted_lateral_accel):
    self._ff = ff
    self._pid_log = pid_log
    self._setpoint = setpoint
    self._measurement = measurement
    self._lateral_accel_deadzone = lateral_accel_deadzone
    self._desired_lateral_accel = desired_lateral_accel
    self._actual_lateral_accel = actual_lateral_accel

    self.update_calculations(CS, VM, desired_lateral_accel)
    self.update_neural_network_feedforward(CS, params, calibrated_pose)

    return self._ff, self._pid_log
