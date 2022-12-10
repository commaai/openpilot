import math

from cereal import log
from selfdrive.controls.lib.latcontrol import LatControl, MIN_STEER_SPEED
from selfdrive.controls.lib.pid import PIDController

STEER_ANGLE_SATURATION_THRESHOLD = 2.5  # Degrees


class LatControlAngle(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)
    self.pid = PIDController((CP.lateralTuning.pid.kpBP, CP.lateralTuning.pid.kpV),
                             (CP.lateralTuning.pid.kiBP, CP.lateralTuning.pid.kiV),
                             k_f=CP.lateralTuning.pid.kf)

  def reset(self):
    super().reset()
    self.pid.reset()

  def update(self, active, CS, VM, params, last_actuators, steer_limited, desired_curvature, desired_curvature_rate, llk):
    angle_log = log.ControlsState.LateralAngleState.new_message()

    if CS.vEgo < MIN_STEER_SPEED or not active:
      angle_log.active = False
      angle_steers_des = float(CS.steeringAngleDeg)
      output_angle = float(CS.steeringAngleDeg)
    else:
      angle_log.active = True
      angle_steers_des = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll))
      angle_steers_des += params.angleOffsetDeg

      angle_log.error = angle_steers_des - CS.steeringAngleDeg

    freeze_integrator = steer_limited or CS.steeringPressed or CS.vEgo < 5
    output_angle = self.pid.update(error,
                                   feedforward=angle_steers_des,
                                   speed=CS.vEgo,
                                   freeze_integrator=freeze_integrator)

    # if CS.vEgo < MIN_STEER_SPEED or not active:
    #   angle_log.active = False
    #   angle_steers_des = float(CS.steeringAngleDeg)
    # else:
    #   angle_log.active = True
    #   angle_steers_des = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll))
    #   angle_steers_des += params.angleOffsetDeg

    angle_control_saturated = abs(angle_steers_des - CS.steeringAngleDeg) > STEER_ANGLE_SATURATION_THRESHOLD
    angle_log.saturated = self._check_saturation(angle_control_saturated, CS, steer_limited)
    # angle_log.steeringAngleDeg = float(CS.steeringAngleDeg)
    angle_log.steeringAngleDesiredDeg = angle_steers_des
    return 0, float(angle_steers_des), angle_log
