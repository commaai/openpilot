import time
from contextlib import AbstractContextManager

from panda import Panda
from opendbc.car.car_helpers import get_car
from opendbc.car.can_definitions import CanData
from opendbc.car.structs import CarParams, CarControl


class PandaRunner(AbstractContextManager):
  def __enter__(self):
    self.p = Panda()
    self.p.reset()

    # setup + fingerprinting
    self.p.set_safety_mode(CarParams.SafetyModel.elm327, 1)
    self.CI = get_car(self._can_recv, self.p.can_send_many, self.p.set_obd, True, False)
    assert self.CI.CP.carFingerprint.lower() != "mock", "Unable to identify car. Check connections and ensure car is supported."

    safety_model = self.CI.CP.safetyConfigs[0].safetyModel
    self.p.set_safety_mode(CarParams.SafetyModel.elm327, 1)
    self.CI.init(self.CI.CP, self._can_recv, self.p.can_send_many)
    self.p.set_safety_mode(safety_model, self.CI.CP.safetyConfigs[0].safetyParam)

    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.p.set_safety_mode(CarParams.SafetyModel.noOutput)
    self.p.reset()  # avoid siren
    return super().__exit__(exc_type, exc_value, traceback)

  @property
  def panda(self) -> Panda:
    return self.p

  def _can_recv(self, wait_for_one: bool = False) -> list[list[CanData]]:
    recv = self.p.can_recv()
    while len(recv) == 0 and wait_for_one:
      recv = self.p.can_recv()
    return [[CanData(addr, dat, bus) for addr, dat, bus in recv], ]

  def read(self, strict: bool = True):
    cs = self.CI.update([int(time.monotonic()*1e9), self._can_recv()[0]])
    if strict:
      assert cs.canValid, "CAN went invalid, check connections"
    return cs

  def write(self, cc: CarControl) -> None:
    if cc.enabled and not self.p.health()['controls_allowed']:
      # prevent the car from faulting. print a warning?
      cc = CarControl(enabled=False)
    _, can_sends = self.CI.apply(cc)
    self.p.can_send_many(can_sends, timeout=25)
    self.p.send_heartbeat()


if __name__ == "__main__":
  with PandaRunner() as p:
    print(p.read())
