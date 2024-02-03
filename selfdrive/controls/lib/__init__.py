from openpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from openpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from cereal import car


def get_lateral_controller(CP, CI):
  if CP.steerControlType == car.CarParams.SteerControlType.angle:
    return LatControlAngle(CP, CI)
  else:
    if CP.lateralTuning.which() == 'torque':
      return LatControlPID(CP, CI)
    else:
      return LatControlTorque(CP, CI)
