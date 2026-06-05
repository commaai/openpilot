import math
import numpy as np

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
    pid_log = log.ControlsState.LateralCurvatureState.new_message()
    if not active:
      output_curvature = 0.0
      pid_log.active = False
      self.pid.reset()
    else:
      roll_compensation = -VM.roll_compensation(params.roll, CS.vEgo)
      actual_curvature_vm_no_roll = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, 0.)
      actual_curvature_vm = actual_curvature_vm_no_roll - roll_compensation

      actual_curvature = actual_curvature_vm
      if CS.vEgo > 5.0:
        actual_curvature_pose = CS.yawRate / max(CS.vEgo, 0.1)
        actual_curvature = np.interp(CS.vEgo, [2.0, 5.0], [actual_curvature_vm, actual_curvature_pose])

      feedforward = desired_curvature - roll_compensation
      pid_log.error = float(desired_curvature - actual_curvature)
      freeze_integrator = steer_limited_by_safety or CS.vEgo < 5 or CS.steeringPressed

      output_curvature = self.pid.update(pid_log.error, speed=CS.vEgo,
                                         feedforward=feedforward,
                                         freeze_integrator=freeze_integrator)

      saturated = abs(output_curvature) >= MAX_CURVATURE

      pid_log.active = True
      pid_log.p = float(self.pid.p)
      pid_log.i = float(self.pid.i)
      pid_log.f = float(self.pid.f)
      pid_log.output = float(output_curvature)
      pid_log.actualCurvature = float(actual_curvature)
      pid_log.desiredCurvature = float(desired_curvature)
      pid_log.saturated = bool(self._check_saturation(saturated, CS, steer_limited_by_safety, curvature_limited))

    return 0.0, float(output_curvature), pid_log
