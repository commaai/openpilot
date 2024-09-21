#!/usr/bin/env python3

from cereal import car
from opendbc.car.honda.values import CAR
from opendbc.car import gen_empty_fingerprint
from openpilot.selfdrive.car.car_specific import CarSpecificEvents
from opendbc.car.car_helpers import interfaces
from openpilot.selfdrive.car.card import Car


BelowSteerSpeed = car.OnroadEvent.EventName.belowSteerSpeed


class TestHondaLowSpeedAlert:
  def __init__(self):
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

  def test_low_speed_alert(self):
#     not enabled > speed 36 > enabled > alert > speed 48 > no alert > speed 55 > speed 50 > pre\

    # test 1

    self.CS.out.vEgo, self.CS_prev.vEgo = self.CP.minSteerSpeed , self.CP.minSteerSpeed - 1

    if self.CS.out.vEgo < self.CP.minSteerSpeed:
      assert self.car_events.low_speed_alert
    elif self.CS.out.vEgo >= self.CP.minSteerSpeed:
      assert self.car_events.low_speed_pre_alert == self.car_events.low_speed_alert

    self.CS.events = self.car_events.update(self.CS, self.CS_prev, self.CC, self.CC_prev)
    assert BelowSteerSpeed not in self.CS.events.names



if __name__ == "__main__":
  _ = TestHondaLowSpeedAlert()
  _.test_low_speed_alert()
