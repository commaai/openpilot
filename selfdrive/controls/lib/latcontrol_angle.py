import math

from cereal import log


class LatControlAngle():
  def __init__(self, CP):
    pass

  def reset(self):
    pass

  def update(self, active, CS, CP, VM, params, last_actuators, desired_curvature, desired_curvature_rate):
    angle_log = log.ControlsState.LateralAngleState.new_message()

    if CS.vEgo < 0.3 or not active:
      angle_log.active = False
      angle_steers_des = float(CS.steeringAngleDeg)
    else:
      angle_log.active = True
      angle_steers_des = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll))
      angle_steers_des += params.angleOffsetDeg

    angle_log.saturated = False
    angle_log.steeringAngleDeg = float(CS.steeringAngleDeg)
    angle_log.steeringAngleDesiredDeg = angle_steers_des

    return 0, float(angle_steers_des), angle_log
