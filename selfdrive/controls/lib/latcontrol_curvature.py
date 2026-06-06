import math

from cereal import log
from openpilot.common.pid import PIDController
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.controls.lib.drive_helpers import MAX_CURVATURE

CURVATURE_SATURATION_THRESHOLD = 1e-3


class LatControlCurvature(LatControl):
  def __init__(self, CP, CI, dt):
    super().__init__(CP, CI, dt)
    self.sat_check_min_speed = 5.
    self.pid = PIDController(([10., 40.], [0., 1.45]), ([10., 40.], [0., 0.12]),
                             pos_limit=MAX_CURVATURE, neg_limit=-MAX_CURVATURE, rate=1 / dt)

  def reset(self):
    super().reset()
    self.pid.reset()

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, curvature_limited, lat_delay):
    pid_log = log.ControlsState.LateralCurvatureState.new_message()
    actual_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    error = desired_curvature - actual_curvature

    pid_log.error = float(error)
    pid_log.actualCurvature = float(actual_curvature)
    pid_log.desiredCurvature = float(desired_curvature)

    if not active:
      output_curvature = 0.0
      pid_log.active = False
      self.pid.reset()
    else:
      if CS.steeringPressed:
        # while overriding, only command feedforward so we don't fight the driver
        self.pid.reset()
        output_curvature = desired_curvature
      else:
        output_curvature = self.pid.update(error, speed=CS.vEgo, feedforward=desired_curvature,
                                           freeze_integrator=steer_limited_by_safety or CS.vEgo < 5)
      pid_log.active = True
      pid_log.p = float(self.pid.p)
      pid_log.i = float(self.pid.i)
      pid_log.f = float(self.pid.f)

    pid_log.output = float(output_curvature)
    pid_log.saturated = bool(self._check_saturation(abs(error) > CURVATURE_SATURATION_THRESHOLD, CS,
                                                    steer_limited_by_safety, curvature_limited))

    return 0.0, float(output_curvature), pid_log
