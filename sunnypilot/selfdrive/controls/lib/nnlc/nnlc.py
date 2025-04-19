"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from collections import deque
import math
import numpy as np

from opendbc.car.interfaces import LatControlInputs
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_base import LatControlTorqueExtBase
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.helpers import MOCK_MODEL_PATH
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.model import NNTorqueModel

LOW_SPEED_X = [0, 10, 20, 30]
LOW_SPEED_Y = [12, 3, 1, 0]


# At a given roll, if pitch magnitude increases, the
# gravitational acceleration component starts pointing
# in the longitudinal direction, decreasing the lateral
# acceleration component. Here we do the same thing
# to the roll value itself, then passed to nnff.
def roll_pitch_adjust(roll, pitch):
  return roll * math.cos(pitch)


class NeuralNetworkLateralControl(LatControlTorqueExtBase):
  def __init__(self, lac_torque, CP, CP_SP):
    super().__init__(lac_torque, CP, CP_SP)
    self.params = Params()
    self.enabled = self.params.get_bool("NeuralNetworkLateralControl")
    self.has_nn_model = CP_SP.neuralNetworkLateralControl.model.path != MOCK_MODEL_PATH

    # NN model takes current v_ego, lateral_accel, lat accel/jerk error, roll, and past/future/planned data
    # of lat accel and roll
    # Past value is computed using previous desired lat accel and observed roll
    self.model = NNTorqueModel(CP_SP.neuralNetworkLateralControl.model.path)

    self.pitch = FirstOrderFilter(0.0, 0.5, 0.01)
    self.pitch_last = 0.0

    # setup future time offsets
    self.nn_time_offset = CP.steerActuatorDelay + 0.2
    future_times = [0.3, 0.6, 1.0, 1.5] # seconds in the future
    self.nn_future_times = [i + self.nn_time_offset for i in future_times]

    # setup past time offsets
    self.past_times = [-0.3, -0.2, -0.1]
    history_check_frames = [int(abs(i)*100) for i in self.past_times]
    self.history_frame_offsets = [history_check_frames[0] - i for i in history_check_frames]
    self.lateral_accel_desired_deque = deque(maxlen=history_check_frames[0])
    self.roll_deque = deque(maxlen=history_check_frames[0])
    self.error_deque = deque(maxlen=history_check_frames[0])
    self.past_future_len = len(self.past_times) + len(self.nn_future_times)

  def update_neural_network_feedforward(self, CS, params, calibrated_pose) -> None:
    if not self.enabled or not self.model_valid or not self.has_nn_model:
      return

    low_speed_factor = float(np.interp(CS.vEgo, LOW_SPEED_X, LOW_SPEED_Y)) ** 2
    self._setpoint = self._desired_lateral_accel + low_speed_factor * self._desired_curvature
    self._measurement = self._actual_lateral_accel + low_speed_factor * self._actual_curvature

    # update past data
    roll = params.roll
    if calibrated_pose is not None:
      pitch = self.pitch.update(calibrated_pose.orientation.pitch)
      roll = roll_pitch_adjust(roll, pitch)
      self.pitch_last = pitch
    self.roll_deque.append(roll)
    self.lateral_accel_desired_deque.append(self._desired_lateral_accel)

    # prepare past and future values
    # adjust future times to account for longitudinal acceleration
    adjusted_future_times = [t + 0.5 * CS.aEgo * (t / max(CS.vEgo, 1.0)) for t in self.nn_future_times]
    past_rolls = [self.roll_deque[min(len(self.roll_deque) - 1, i)] for i in self.history_frame_offsets]
    future_rolls = [roll_pitch_adjust(np.interp(t, ModelConstants.T_IDXS, self.model_v2.orientation.x) + roll,
                                      np.interp(t, ModelConstants.T_IDXS, self.model_v2.orientation.y) + self.pitch_last) for t in
                    adjusted_future_times]
    past_lateral_accels_desired = [self.lateral_accel_desired_deque[min(len(self.lateral_accel_desired_deque) - 1, i)]
                                   for i in self.history_frame_offsets]
    future_planned_lateral_accels = [np.interp(t, ModelConstants.T_IDXS, self.model_v2.acceleration.y) for t in
                                     adjusted_future_times]

    # compute NNFF error response
    nnff_setpoint_input = [CS.vEgo, self._setpoint, self.lateral_jerk_setpoint, roll] \
                          + [self._setpoint] * self.past_future_len \
                          + past_rolls + future_rolls
    # past lateral accel error shouldn't count, so use past desired like the setpoint input
    nnff_measurement_input = [CS.vEgo, self._measurement, self.lateral_jerk_measurement, roll] \
                             + [self._measurement] * self.past_future_len \
                             + past_rolls + future_rolls
    torque_from_setpoint = self.model.evaluate(nnff_setpoint_input)
    torque_from_measurement = self.model.evaluate(nnff_measurement_input)
    self._pid_log.error = torque_from_setpoint - torque_from_measurement

    # compute feedforward (same as nn setpoint output)
    friction_input = self.update_friction_input(self._setpoint, self._measurement)
    nn_input = [CS.vEgo, self._desired_lateral_accel, friction_input, roll] \
               + past_lateral_accels_desired + future_planned_lateral_accels \
               + past_rolls + future_rolls
    self._ff = self.model.evaluate(nn_input)

    # apply friction override for cars with low NN friction response
    if self.model.friction_override:
      self._pid_log.error += self.torque_from_lateral_accel(LatControlInputs(0.0, 0.0, CS.vEgo, CS.aEgo), self.torque_params,
                                                            friction_input,
                                                            self._lateral_accel_deadzone, friction_compensation=True,
                                                            gravity_adjusted=False)
