from dataclasses import dataclass
from collections import namedtuple

# from selfdrive.car.interfaces import get_interface_attr
from selfdrive.car.interfaces import CarInterfaceBase, DT_CTRL
from openpilot.selfdrive.car.car_helpers import get_car, get_one_can


class Car:
  """All encompassing class for interfacing with a vehicle."""

  def __init__(self, logcan, sendcan, dt_ctrl=DT_CTRL):
    """
    Inputs:
    - logcan: function that returns a list of the latest CAN messages
    - sendcan: function that sends a CAN message
    - dt_ctrl: control loop time step
    """
    self.logcan = logcan
    self.sendcan = sendcan
    self.dt_ctrl = dt_ctrl

    # TODO: 100 Hz is currently the only supported rate
    assert self.dt_ctrl == DT_CTRL

    self.CI: CarInterfaceBase | None = None

  def fingerprint(self, experimental_long_allowed: bool, num_pandas: int) -> CarInterfaceBase:
    """Fingerprint the vehicle and return a car interface."""

    if self.CI is None:
      # TODO: return a car interface object all filled out, with no cereal/capnp
      self.CI = get_car(self.logcan, self.sendcan, experimental_long_allowed, num_pandas)
    return self.CI
