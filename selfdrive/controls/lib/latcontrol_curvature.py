import math

from cereal import log
from selfdrive.controls.lib.latcontrol import STEER_ANGLE_SATURATION_THRESHOLD
from selfdrive.controls.lib.latcontrol_pid import LatControlPID


class LatControlCurvature(LatControlPID):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)
    self.pid.pos_limit = 1e308
    self.pid.neg_limit = -1e308

  def update(self, active, CS, VM, params, actuators, last_actuators, steer_limited, desired_curvature,
             desired_curvature_rate, llk):
    curvature_log = log.ControlsState.LateralCurvatureState.new_message()

    curvature_log.actualCurvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    curvature_log.desiredCurvature = desired_curvature
    curvature_log.error = desired_curvature - curvature_log.actualCurvature

    angle_steer_des = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll)) + params.angleOffsetDeg
    angle_control_saturated = abs(angle_steer_des - CS.steeringAngleDeg) > STEER_ANGLE_SATURATION_THRESHOLD
    curvature_log.saturated = self._check_saturation(angle_control_saturated, CS, steer_limited)

    if not active:
      output = 0.0
      curvature_log.active = False
    else:
      freeze_integrator = steer_limited or CS.steeringPressed or CS.vEgo < 5
      output = self.pid.update(curvature_log.error,
                               feedforward=desired_curvature,
                               speed=CS.vEgo,
                               freeze_integrator=freeze_integrator)

      curvature_log.active = True
      curvature_log.p = self.pid.p
      curvature_log.i = self.pid.i
      curvature_log.f = self.pid.f
      curvature_log.output = output

    actuators.curvature = output
    return curvature_log
