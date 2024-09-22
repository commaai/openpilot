from cereal import car

from opendbc.car import gen_empty_fingerprint
from opendbc.car.car_helpers import interfaces
from opendbc.car.honda.values import CAR

from openpilot.selfdrive.car.card import Car
from openpilot.selfdrive.car.car_specific import CarSpecificEvents


BelowSteerSpeed = car.OnroadEvent.EventName.belowSteerSpeed


class TestHondaLowSpeedAlert:
  def setUp(self):
    # test other models later
    test_car = CAR.HONDA_ODYSSEY_BOSCH

    # setup
    CarInterface, CarController, CarState, RadarInterface = interfaces[test_car]

    self.CP = CarInterface.get_params(test_car, gen_empty_fingerprint(), {}, experimental_long=False, docs=False)

    self.CI = CarInterface(self.CP, CarController, CarState)
    self.RI = RadarInterface(self.CP) # unused
    self.Card = Car(self.CI, RadarInterface)

    self.CC = self.CI.CC
    self.CS = self.CI.CS

    self.CC_prev = self.Card.CC_prev
    self.CS_prev = self.Card.CS_prev

    self.car_events = CarSpecificEvents(self.CP)

  def run(self):
    self.CS.events = self.car_events.update(self.CS, self.CS_prev, self.CC, self.CC_prev)

  def set_speeds(self, cs, cs_prev):
    self.CS.out.vEgo = cs
    self.CP.minSteerSpeed = cs_prev

  def test_no_alert(self):
    self.setUp()
    self.set_speeds(self.CP.minSteerSpeed , max(self.CP.minSteerSpeed - 1, 0))
    self.run()
    if self.CS.out.vEgo < self.CP.minSteerSpeed:
      assert self.car_events.low_speed_alert
    if self.CS.out.vEgo >= self.CP.minSteerSpeed:
      assert self.car_events.low_speed_pre_alert == self.car_events.low_speed_alert
    assert BelowSteerSpeed not in self.CS.events.names

  def test_under_speed(self):
    self.setUp()
    for speed in range(100):
      self.set_speeds(speed, self.CP.minSteerSpeed)
      self.run()
      assert self.car_events.low_speed_alert == bool(speed < self.CP.minSteerSpeed)

  def test_transitions(self):
    #     not enabled > speed 36 > enabled > alert > speed 48 > no alert > speed 55 > speed 50 > pre\

    # test 1
    pass
