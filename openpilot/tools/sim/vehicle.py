import math
import numpy as np

from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_CTRL

STEER_RC = 0.1
ACCEL_RC = 0.3


class Vehicle:
  """Civic 2022 kinematics. pose is (x fwd, y right, heading right-positive) of the road camera."""
  def __init__(self):
    self.VM = VehicleModel(CarInterface.get_non_essential_params(CAR.HONDA_CIVIC_2022))
    self.steer = FirstOrderFilter(0., STEER_RC, DT_CTRL)
    self.accel = FirstOrderFilter(0., ACCEL_RC, DT_CTRL)
    self.pose = np.zeros(3)
    self.v = 0.
    self.yaw_rate = 0.

  def set_state(self, pose: np.ndarray, v: float, yaw_rate: float) -> None:
    self.pose[:] = pose
    self.v = float(v)
    self.yaw_rate = float(yaw_rate)
    self.steer.x = math.degrees(self.VM.get_steer_from_curvature(-yaw_rate / max(v, 1.), v, 0.))
    self.accel.x = 0.

  def step(self, steer_angle_deg: float, accel: float) -> None:
    self.steer.update(float(steer_angle_deg))
    self.accel.update(float(accel))
    self.v = max(self.v + self.accel.x * DT_CTRL, 0.)
    # VehicleModel curvature is left-positive
    self.yaw_rate = -self.VM.calc_curvature(math.radians(self.steer.x), self.v, 0.) * self.v
    heading = self.pose[2] + self.yaw_rate * DT_CTRL / 2
    self.pose += [self.v * math.cos(heading) * DT_CTRL, self.v * math.sin(heading) * DT_CTRL, self.yaw_rate * DT_CTRL]
