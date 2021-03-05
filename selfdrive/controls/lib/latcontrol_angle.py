from cereal import log

class LatControlAngle():
  def __init__(self, CP):
    self.angle_steers_des = 0.

  def reset(self):
    pass

  def update(self, active, CS, CP, lat_plan):
    output_steer = 0.0
    angle_log = log.ControlsState.LateralAngleState.new_message()
    angle_log.steeringAngleDeg = float(CS.steeringAngleDeg)

    if CS.vEgo < 0.3 or not active:
      angle_log.active = False
      self.angle_steers_des = float(CS.steeringAngleDeg)
    else:
      angle_log.active = True
      self.angle_steers_des = lat_plan.steeringAngleDeg

    angle_log.output = self.angle_steers_des
    angle_log.saturated = False
    return output_steer, float(self.angle_steers_des), angle_log
