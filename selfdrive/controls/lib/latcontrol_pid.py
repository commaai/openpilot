from selfdrive.controls.lib.pid import PIController
from selfdrive.controls.lib.drive_helpers import get_steer_max
from cereal import car
from cereal import log


class LatControlPID():
  def __init__(self, CP):
    self.pid = PIController((CP.lateralTuning.pid.kpBP, CP.lateralTuning.pid.kpV),
                            (CP.lateralTuning.pid.kiBP, CP.lateralTuning.pid.kiV),
                            k_f=CP.lateralTuning.pid.kf, pos_limit=1.0, sat_limit=CP.steerLimitTimer)
    self.angle_steers_des = 0.

  def reset(self):
    self.pid.reset()

  def update(self, active, CS, CP, path_plan):
    pid_log = log.ControlsState.LateralPIDState.new_message()
    pid_log.steerAngle = float(CS.steeringAngle)
    pid_log.steerRate = float(CS.steeringRate)

    if CS.vEgo < 0.3 or not active:
      output_steer = 0.0
      pid_log.active = False
      self.pid.reset()
    else:
      self.angle_steers_des = path_plan.angleSteers  # get from MPC/PathPlanner

      steers_max = get_steer_max(CP, CS.vEgo)
      self.pid.pos_limit = steers_max
      self.pid.neg_limit = -steers_max
      steer_feedforward = self.angle_steers_des   # feedforward desired angle
      if CP.steerControlType == car.CarParams.SteerControlType.torque:
        # TODO: feedforward something based on path_plan.rateSteers
        steer_feedforward -= path_plan.angleOffset   # subtract the offset, since it does not contribute to resistive torque
        steer_feedforward *= CS.vEgo**2  # proportional to realigning tire momentum (~ lateral accel)
      deadzone = 0.0

      check_saturation = (CS.vEgo > 10) and not CS.steeringRateLimited and not CS.steeringPressed
      output_steer = self.pid.update(self.angle_steers_des, CS.steeringAngle, check_saturation=check_saturation, override=CS.steeringPressed,
                                     feedforward=steer_feedforward, speed=CS.vEgo, deadzone=deadzone)
      pid_log.active = True
      pid_log.p = self.pid.p
      pid_log.i = self.pid.i
      pid_log.f = self.pid.f
      pid_log.output = output_steer
      pid_log.saturated = bool(self.pid.saturated)

    return output_steer, float(self.angle_steers_des), pid_log
