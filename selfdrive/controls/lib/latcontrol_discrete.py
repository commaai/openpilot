import math

from selfdrive.controls.lib.latcontrol import LatControl, MIN_STEER_SPEED
from selfdrive.controls.lib.discrete import DiscreteController
from common.numpy_fast import clip
from common.realtime import DT_CTRL
from cereal import log

def linearize_error(error, speed):
  return error * (1 + 1*((speed/40)**2))

class LatControlDiscrete(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)
    gains = [g for g in CP.lateralTuning.discrete.gains]
    self.discrete = DiscreteController(gains, rate=(1 / DT_CTRL))

  def reset(self):
    super().reset()
    self.discrete.reset()

  def update(self, active, CS, CP, VM, params, last_actuators, desired_curvature, desired_curvature_rate, llk):
    angle_steers_des = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll))
    angle_steers = CS.steeringAngleDeg-params.angleOffsetAverageDeg
    error = linearize_error(angle_steers_des - angle_steers, CS.vEgo)
    
    discrete_log = log.ControlsState.LateralDiscreteState.new_message()
    discrete_log.steeringAngleDesiredDeg = angle_steers_des
    discrete_log.steeringAngleDeg = float(angle_steers)
    discrete_log.angleError = error
    
    if CS.vEgo < MIN_STEER_SPEED or not active:
      self.discrete.reset()
      output_steer = 0.0
    else:
      output_steer = self.discrete.update(error, last_actuators.steer)
      output_steer = clip(output_steer, -self.steer_max, self.steer_max)

      discrete_log.active = True
      discrete_log.output = float(self.discrete.u[1])
      discrete_log.saturated = self._check_saturation(self.steer_max - abs(output_steer) < 1e-3, CS)
      discrete_log.gains = [float(g*u[1]) for g, u in zip(self.discrete.gains, self.discrete.d)]
    return output_steer, angle_steers_des, discrete_log
