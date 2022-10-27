import math

from cereal import log
from selfdrive.controls.lib.latcontrol import LatControl, MIN_STEER_SPEED
from selfdrive.controls.lib.pid import PIDController

STEER_ANGLE_SATURATION_THRESHOLD = 2.5  # Degrees


class LatControlAngle(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)
    self.pid = PIDController(([0.], [0.01]), # kP
                             ([0.], [0.0]), # kI
                             k_f=1., pos_limit=0.02094, neg_limit=-0.02)

  def reset(self):
    super().reset()
    self.pid.reset()

  def update(self, active, CS, VM, params, last_actuators, steer_limited, desired_curvature, desired_curvature_rate, llk):
    angle_log = log.ControlsState.LateralAngleState.new_message()
    angle_log.steeringAngleDeg = float(CS.steeringAngleDeg)
    angle_log.steeringAngleDesiredDeg = desired_curvature

    actual_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    error = desired_curvature - actual_curvature

    if CS.vEgo < MIN_STEER_SPEED or not active:
      output = 0.0
      angle_log.active = False
    else:
      freeze_integrator = steer_limited or CS.steeringPressed or CS.vEgo < 5
      output = -self.pid.update(error,
                                feedforward=desired_curvature,
                                speed=CS.vEgo,
                                freeze_integrator=freeze_integrator)
      angle_log.active = True

    return 0, output, angle_log
