import math

from cereal import log
from selfdrive.controls.lib.latcontrol import LatControl, MIN_STEER_SPEED


class LatControlCurvature(LatControl):
  def update(self, active, CS, VM, params, last_actuators, lat_plan, model_v2, desired_curvature, desired_curvature_rate, llk):
    curvature_log = log.ControlsState.LateralCurvatureState.new_message()

    if CS.vEgo < MIN_STEER_SPEED or not active:
      curvature_log.active = False
      curvature = 0.0
      curvature_rate = 0.0
      path_angle = 0.0
      path_deviation = 0.0
    else:
      curvature_log.active = True

      curvatures = lat_plan.curvatures
      path_points = lat_plan.dPathPoints

      position = model_v2.position

      # "road/lane curvature" is different from the immediate manoeuvre/path curvature
      # later curvature values are probably closer to this "road curvature" value
      curvature = -curvatures[6]
      path_deviation = -path_points[0] if len(path_points) > 0 else 0

      # calculate the angle of the path (t = approx 1s)
      path_angle = math.atan(-position.y[10] / position.x[10])

      # TODO
      curvature_rate = 0

    # TODO: calculate saturated, like latcontrol_angle
    curvature_log.saturated = False
    curvature_log.curvature = curvature
    curvature_log.curvatureRate = curvature_rate
    curvature_log.pathAngle = path_angle
    curvature_log.pathDeviation = path_deviation
    return 0, 0, curvature, curvature_rate, path_angle, path_deviation, curvature_log
