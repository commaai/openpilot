from cereal import log
from openpilot.common.pid import PIDController
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.controls.lib.drive_helpers import MAX_CURVATURE


class LatControlCurvature(LatControl):
  def __init__(self, CP, CI, dt):
    super().__init__(CP, CI, dt)
    self.pid = PIDController(([10., 40.], [0., 1.45]), ([10., 40.], [0., 0.12]), k_f=1.,
                             pos_limit=MAX_CURVATURE, neg_limit=-MAX_CURVATURE, rate=1 / dt)

  def reset(self):
    super().reset()
    self.pid.reset()

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, curvature_limited, lat_delay):
    curvature_log = log.ControlsState.LateralCurvatureState.new_message()
    # curvature the car is actually following
    actual_curvature = CS.yawRate / max(CS.vEgo, 0.1)

    if not active:
      output_curvature = 0.0
      curvature_log.active = False
      self.pid.reset()
    else:
      error = desired_curvature - actual_curvature
      freeze_integrator = steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 5
      output_curvature = self.pid.update(error, speed=CS.vEgo, feedforward=desired_curvature, freeze_integrator=freeze_integrator)

      curvature_log.active = True
      curvature_log.p = float(self.pid.p)
      curvature_log.i = float(self.pid.i)
      curvature_log.f = float(self.pid.f)
      curvature_log.error = float(error)
      curvature_log.output = float(output_curvature)
      curvature_log.actualCurvature = float(actual_curvature)
      curvature_log.desiredCurvature = float(desired_curvature)
      curvature_log.saturated = bool(self._check_saturation(MAX_CURVATURE - abs(output_curvature) < 1e-3, CS, steer_limited_by_safety, curvature_limited))

    return 0.0, float(output_curvature), curvature_log
