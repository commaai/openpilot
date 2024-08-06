from dataclasses import dataclass
from collections import namedtuple

# from selfdrive.car.interfaces import get_interface_attr
from openpilot.selfdrive.car import carlog
from openpilot.selfdrive.car.car_helpers import get_car, get_one_can
from openpilot.selfdrive.car.interfaces import CarInterfaceBase, DT_CTRL


@dataclass(frozen=True)
class CanData:
  address: int
  dat: bytes
  src: int


class OpenCar:
  """All encompassing class for interfacing with a vehicle."""

  def __init__(self, logcan, sendcan, dt_ctrl=DT_CTRL, CI=None):
    """
    Inputs:
    - logcan: function that returns a list of the latest CAN messages
    - sendcan: function that sends a CAN message
    - dt_ctrl: control loop time step
    """
    self.logcan = logcan
    self.sendcan = sendcan
    self.dt_ctrl = dt_ctrl
    self.CI: CarInterfaceBase | None = CI

    # TODO: 100 Hz is currently the only supported rate
    assert self.dt_ctrl == DT_CTRL

  def get_car(self, experimental_long_allowed: bool, num_pandas: int) -> CarInterfaceBase:
    candidate, fingerprints, vin, car_fw, source, exact_match = fingerprint(logcan, sendcan, num_pandas)

    if candidate is None:
      carlog.error({"event": "car doesn't match any fingerprints", "fingerprints": repr(fingerprints)})
      candidate = "MOCK"

    CarInterface, _, _ = interfaces[candidate]
    CP = CarInterface.get_params(candidate, fingerprints, car_fw, experimental_long_allowed, docs=False)
    CP.carVin = vin
    CP.carFw = car_fw
    CP.fingerprintSource = source
    CP.fuzzyFingerprint = not exact_match

    return get_car_interface(CP)

  def fingerprint(self, experimental_long_allowed: bool, num_pandas: int) -> CarInterfaceBase:
    """Fingerprint the vehicle and return a car interface."""

    if self.CI is None:
      # TODO: return a car interface object all filled out, with no cereal/capnp
      self.CI = get_car(self.logcan, self.sendcan, experimental_long_allowed, num_pandas)
    return self.CI
