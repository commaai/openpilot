import math
import numpy as np

from selfdrive.controls.lib.drive_helpers import get_steer_max
from cereal import log
from selfdrive.controls.lib.latcontrol import LatControl, MIN_STEER_SPEED


controls_list = ['can_cmds_squared', 'cmds_by_vsquared', 'cmds_squared_by_vsquared']
NU = len(controls_list)

live_param_list = ['roll', 'roll_squared', 'roll_by_speed', 'roll_by_vsquared']
N_live_param = len(live_param_list)


class LatControlSteerModel(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)

    model_param = np.asarray(list(CP.lateralTuning.steerModel.modelparam))
    self.B = model_param[0:NU]
    self.R = model_param[NU:NU+N_live_param]
    assert(NU+N_live_param == len(model_param))

  def update(self, active, CS, CP, VM, params, last_actuators, desired_curvature, desired_curvature_rate):
    model_log = log.ControlsState.LateralSteerModelState.new_message()

    steers_max = get_steer_max(CP, CS.vEgo)

    # offset does not contribute to resistive torque
    angle_steers_des_no_offset = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll))
    angle_steers_des = angle_steers_des_no_offset + params.angleOffsetDeg

    model_log.steeringAngleDesiredDeg = angle_steers_des

    if CS.vEgo < MIN_STEER_SPEED or not active:
      output_steer = 0.0
      model_log.active = False
      self.reset()
    else:
      # NOTE: live_param_list dependent.
      # live_param_list = ['roll', 'roll_squared', 'roll_by_speed', 'roll_by_vsquared']
      torque_prev = last_actuators.steer
      roll_deg = np.degrees(params.roll)
      live_param = np.array([roll_deg, roll_deg * abs(roll_deg), roll_deg / CS.vEgo, roll_deg/(CS.vEgo**2)])
      Rp = self.R @ live_param

      # NOTE: controls_list dependent.
      # controls_list = ['can_cmds_squared', 'cmds_by_vsquared', 'cmds_squared_by_vsquared']
      du_dtorque = np.array([2*abs(torque_prev), 1/(CS.vEgo**2), 2*abs(torque_prev)/(CS.vEgo**2) ])
      B_tilde = self.B @ du_dtorque
      output_steer = float( (1/B_tilde) * (angle_steers_des_no_offset - Rp) )

      model_log.active = True

    # model_log.steeringAngleDeg = float(self.xcurrent[0])
    model_log.output = output_steer
    model_log.saturated = self._check_saturation(steers_max - abs(output_steer) < 1e-3, CS)

    return output_steer, angle_steers_des, model_log
