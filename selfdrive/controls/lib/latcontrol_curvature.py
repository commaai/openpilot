import math
from cereal import log
from selfdrive.controls.lib.latcontrol import LatControl, MIN_STEER_SPEED


class LatControlCurvature(LatControl):
  def update(self, active, CS, VM, params, last_actuators, desired_curvature, desired_curvature_rate, llk):
    curvature_log = log.ControlsState.LateralCurvatureState.new_message()

    steer_angle_without_offset = math.radians(CS.steeringAngleDeg - params.angleOffsetDeg)
    curvature = -VM.calc_curvature(steer_angle_without_offset, CS.vEgo, params.roll)

    curvature_log.steeringAngleDeg = float(CS.steeringAngleDeg)
    curvature_log.curvature = float(curvature)

    if CS.vEgo < MIN_STEER_SPEED or not active:
      curvature_log.active = False
      desired_curvature = 0
      desired_curvature_rate = 0
      angle_steers_des = 0
    else:
      curvature_log.active = True
      angle_steers_des = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll))
      angle_steers_des += params.angleOffsetDeg

    # TODO: calculate saturated, like latcontrol_angle
    curvature_log.saturated = False
    curvature_log.curvature = curvature
    curvature_log.desiredCurvature = float(desired_curvature)
    curvature_log.desiredCurvatureRate = float(desired_curvature_rate)
    curvature_log.steeringAngleDesiredDeg = angle_steers_des
    return 0, 0, desired_curvature, desired_curvature_rate, curvature_log
