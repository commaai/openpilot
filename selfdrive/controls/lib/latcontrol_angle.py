import math

from cereal import log
from selfdrive.controls.lib.latcontrol import MIN_STEER_SPEED
from selfdrive.controls.lib.latcontrol_pid import LatControlPID

STEER_ANGLE_SATURATION_THRESHOLD = 2.5  # Degrees


class LatControlAngle(LatControlPID):
  steer_max = float('inf')  # output is angle and should be unrestricted

  def update(self, active, CS, VM, params, last_actuators, steer_limited, desired_curvature, desired_curvature_rate, llk):
    angle_log = log.ControlsState.LateralAngleState.new_message()

    angle_log.steeringAngleDeg = float(CS.steeringAngleDeg)
    angle_log.steeringAngleDesiredDeg = float(CS.steeringAngleDeg)
    angle_log.output = float(CS.steeringAngleDeg)

    if CS.vEgo >= MIN_STEER_SPEED and active:
      angle_log.steeringAngleDesiredDeg = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll))
      angle_log.steeringAngleDesiredDeg += params.angleOffsetDeg

      angle_log.error = angle_log.steeringAngleDesiredDeg - CS.steeringAngleDeg

      freeze_integrator = steer_limited or CS.steeringPressed or CS.vEgo < 5  # TODO check steer_limited
      angle_log.output = self.pid.update(angle_log.error,
                                         feedforward=angle_log.steeringAngleDesiredDeg,
                                         speed=CS.vEgo,
                                         freeze_integrator=freeze_integrator)

      angle_log.active = True
      angle_log.p = self.pid.p
      angle_log.i = self.pid.i
      angle_control_saturated = abs(angle_log.steeringAngleDesiredDeg - CS.steeringAngleDeg) > STEER_ANGLE_SATURATION_THRESHOLD
      angle_log.saturated = self._check_saturation(angle_control_saturated, CS, steer_limited)

    return 0, float(angle_log.output), angle_log
