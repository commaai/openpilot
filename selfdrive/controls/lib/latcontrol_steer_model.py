import math
import numpy as np

from selfdrive.controls.lib.drive_helpers import get_steer_max
from cereal import log
from selfdrive.controls.lib.latcontrol import LatControl, MIN_STEER_SPEED
from common.realtime import DT_CTRL
from common.numpy_fast import clip

NX = 2
NU = 1
N_live_param = 1

# TODO:
# - update model structure?
# - get angle measurements sometimes?


class LatControlSteerModel(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)

    self.xcurrent = np.zeros((NX, ))

    model_param = np.asarray(list(CP.lateralTuning.steerModel.modelparam))

    self.A = model_param[0:NX*NX].reshape((NX, NX), order='F')
    B = model_param[NX*NX:NX*(NX+NU)].reshape((NX, NU), order='F')
    self.R = model_param[NX*(NX+NU):].reshape((NX, N_live_param), order='F')
    self.M = - np.linalg.solve(B.T @ B, B.T)
    self.B = B.reshape((NX,))

    self.W = np.diag([1e0, 1e-1])
    self.PHI = (DT_CTRL * self.B).reshape((2,1))
    self.M_tilde = - 1/(self.PHI.T @ self.W @ self.PHI) * (self.PHI.T@self.W)
    self.get_steer_feedforward = CI.get_steer_feedforward_function()
    self.torque = 0.0

  def reset(self):
    # when is this called? only in if below?
    super().reset()
    self.xcurrent = np.zeros((NX, ))

  def update(self, active, CS, CP, VM, params, last_actuators, desired_curvature, desired_curvature_rate):
    model_log = log.ControlsState.LateralSteerModelState.new_message()

    steers_max = get_steer_max(CP, CS.vEgo)

    # offset does not contribute to resistive torque
    angle_steers_des_no_offset = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll))
    angle_steers_des = angle_steers_des_no_offset + params.angleOffsetDeg

    # live_param_list = ['roll', 'speeds', 'speed_squared', 'speed_times_roll']
    # live_param = np.array([params.roll, CS.vEgo, CS.vEgo ** 2, CS.vEgo * params.roll])
    live_param = np.array([params.roll])

    model_log.steeringAngleDesiredDeg = angle_steers_des

    if CS.vEgo < MIN_STEER_SPEED or not active:
      output_steer = 0.0
      model_log.active = False
      self.reset()
    else:

      # analytical solution similar to steady state.
      # steady_state_torque = self.M * (self.A @ self.xcurrent + self.R @ live_param) # solve for xdot = 0

      # torque = argmin norm([desired_angle, xcurrent_1] + DT_CTRL * (A*xcurrent + R*live_param + B*u))
      # torque_np = self.M @ (self.A @ self.xcurrent + self.R @ live_param +
                # (self.xcurrent - np.array([angle_steers_des_no_offset, self.xcurrent[1,]]))/DT_CTRL )
      # hacky but works well..
      # torque_np = self.M @ (self.A @ self.xcurrent + self.R @ live_param +
      #           (self.xcurrent - np.array([angle_steers_des_no_offset, .95*self.xcurrent[1,]]))/DT_CTRL )

      #
      AxplusRp = (self.A @ self.xcurrent + self.R @ live_param)
      desired_angle_rate = math.degrees(VM.get_steer_from_curvature(-desired_curvature_rate, CS.vEgo, params.roll))

      # torque = argmin norm(xcurrent + DT_CTRL * (A*xcurrent + R*live_param + B*u) - [desired_angle, desired_angle_rate])_W
      y = (self.xcurrent - np.array([angle_steers_des_no_offset, desired_angle_rate]) + DT_CTRL * AxplusRp).reshape((2,1))
      torque_np = self.M_tilde @ y

      # TODO: remove clipping later, good for prototype testing!
      # When removing, use last_actuators instead of self.torque to update xcurrent
      STEER_DELTA_UP = 10/1500       # 1.5s time to peak torque
      STEER_DELTA_DOWN = 25/1500
      if self.torque > 0:
          torque_np = clip(torque_np, max(self.torque - STEER_DELTA_DOWN, -STEER_DELTA_UP),
                          self.torque + STEER_DELTA_UP)
      else:
          torque_np = clip(torque_np, self.torque - STEER_DELTA_UP,
                  min(self.torque + STEER_DELTA_DOWN, STEER_DELTA_UP))
      output_steer = float(clip(torque_np, -1.0, 1.0))

      # update state estimate with forward simulation
      self.xcurrent = self.xcurrent + DT_CTRL * (AxplusRp + (self.B * output_steer))

      self.torque = output_steer
      model_log.active = True

    model_log.steeringAngleDeg = float(self.xcurrent[0])
    model_log.output = output_steer
    model_log.saturated = self._check_saturation(steers_max - abs(output_steer) < 1e-3, CS)

    return output_steer, angle_steers_des, model_log
