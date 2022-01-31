import math
import numpy as np

from selfdrive.controls.lib.drive_helpers import get_steer_max
from cereal import log
from selfdrive.controls.lib.latcontrol import LatControl, MIN_STEER_SPEED
from common.realtime import DT_CTRL

NX = 2
NU = 1
N_live_param = 4

# TODO:
# - update model structure?
# - get angle measurements sometimes?


class LatControlSteerModel(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)

    self.xcurrent = np.zeros((NX, ))

    model_param = np.asarray(list(CP.lateralTuning.steerModel.modelparam))

    self.A = model_param[0:NX*NX].reshape((NX, NX), order='F')
    self.B = model_param[NX*NX:NX*(NX+NU)].reshape((NX, NU), order='F')
    self.R = model_param[NX*(NX+NU):].reshape((NX, N_live_param), order='F')
    self.M = - np.linalg.solve(self.B.T @ self.B, self.B.T)
    self.B = self.B.reshape((NX,))
    # CP.lateralTuning.steerModel.?
    self.get_steer_feedforward = CI.get_steer_feedforward_function()


  def reset(self):
    # when is this called? only in if below?
    super().reset()
    self.xcurrent = np.zeros((NX, ))

  def update(self, active, CS, CP, VM, params, last_actuators, desired_curvature, desired_curvature_rate):
    model_log = log.ControlsState.LateralSteerModelState.new_message()

    steers_max = get_steer_max(CP, CS.vEgo)

    angle_steers_des_no_offset = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll))
    angle_steers_des = angle_steers_des_no_offset + params.angleOffsetDeg

    # live_param_list = ['roll', 'speeds', 'speed_squared', 'speed_times_roll']
    live_param = np.array([params.roll, CS.vEgo, CS.vEgo ** 2, CS.vEgo * params.roll])

    model_log.steeringAngleDesiredDeg = angle_steers_des

    if CS.vEgo < MIN_STEER_SPEED or not active:
      output_steer = 0.0
      model_log.active = False
      self.reset()
    else:
      # self.pid.pos_limit = steers_max
      # self.pid.neg_limit = -steers_max

      # torque = argmin norm(xcurrent + DT_CTRL * (A*x + R*live_param + B*u)
      # analytical solution similar to steady state.
      # offset does not contribute to resistive torque
      torque_np = self.M @ (self.A @ self.xcurrent + self.R @ live_param +
                (self.xcurrent - np.array([angle_steers_des_no_offset, self.xcurrent[1,]]))/DT_CTRL )

      # TODO: needed?
      # steer_feedforward = self.get_steer_feedforward(angle_steers_des_no_offset, CS.vEgo)

      # update state estimate with forward simulation
      self.xcurrent = self.xcurrent + DT_CTRL * ((self.A @ self.xcurrent) + (self.B * torque_np) + (self.R @ live_param))
      output_steer = float(torque_np[0])

      model_log.active = True

    model_log.steeringAngleDeg = float(self.xcurrent[0])
    model_log.output = output_steer
    model_log.saturated = self._check_saturation(steers_max - abs(output_steer) < 1e-3, CS)

    return output_steer, angle_steers_des, model_log
