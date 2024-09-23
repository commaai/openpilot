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

    # no

    self.CS.steer_on = True

  def _enabled(self, _):
    self.CC_prev.enabled = _

  def _speed(self, _):
    self.CS.out.vEgo = _
    # print(_)

  def _steer_on(self, _):
    self.CS.steer_on = _

  def _update_events(self):
    self.CS.events = self.car_events.update(self.CS, self.CS_prev, self.CC, self.CC_prev)


  def test(self):
    for enabled in (True, False):
      self.setUp()
      self._enabled(enabled)

      # same
      self._speed(self.CP.minSteerSpeed)
      self._update_events()
      assert not self.car_events.low_speed_pre_alert
      assert self.car_events.low_speed_alert
      assert BelowSteerSpeed in self.CS.events.names

      # faster first time
      self._speed(self.CP.minSteerSpeed + 3.6)
      self._update_events()
      assert self.car_events.low_speed_pre_alert == enabled
      assert not self.car_events.low_speed_alert
      assert BelowSteerSpeed not in self.CS.events.names

      # self._speed(self.CP.minSteerSpeed + 1.)
      # self._update_events()


      # self._speed(self.CP.minSteerSpeed + 1)
      # self._update_events()
      # assert not self.car_events.low_speed_pre_alert
      # assert not self.car_events.low_speed_alert
      # assert BelowSteerSpeed in self.CS.events.names

      # vEgo same or lesser
      for i in range(-1, 1, 1):
        print(i)
        self._speed(self.CP.minSteerSpeed+i)
        self._enabled(enabled)
        self._update_events()
        assert BelowSteerSpeed in self.CS.events.names


  # def test_under_speed(self):
  #   self.setUp()
  #   for spd in range(100):
  #     self._speed(spd)
  #     self._update_events()
  #     assert self.car_events.low_speed_alert == bool(spd < self.CP.minSteerSpeed)


  # def test_transitions(self):
    #     not enabled > speed 36 > enabled > alert > speed 48 > no alert > speed 55 > speed 50 > pre\

    # test 1
