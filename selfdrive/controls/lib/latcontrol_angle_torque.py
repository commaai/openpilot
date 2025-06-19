from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle


class LatControlAngleTorque(LatControlTorque, LatControlAngle):
  def __init__(self, CP, CP_SP, CI):
    LatControlTorque.__init__(self, CP, CP_SP, CI)
    LatControlAngle.__init__(self, CP, CP_SP, CI)

  def update(self, active, CS, VM, params, steer_limited_by_controls, desired_curvature, calibrated_pose, curvature_limited):
    torque, _, _ = LatControlTorque.update(self, active, CS, VM, params, steer_limited_by_controls, desired_curvature, calibrated_pose, curvature_limited)
    _, angle, angle_log = LatControlAngle.update(self, active, CS, VM, params, steer_limited_by_controls, desired_curvature, calibrated_pose, curvature_limited)
    return torque, angle, angle_log
