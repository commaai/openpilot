from cereal import car

from opendbc.car import gen_empty_fingerprint
from opendbc.car.car_helpers import interfaces
from opendbc.car.honda.values import CAR

from openpilot.selfdrive.car.card import Car
from openpilot.selfdrive.car.car_specific import CarSpecificEvents


BelowSteerSpeed = car.OnroadEvent.EventName.belowSteerSpeed


class TestHondaLowSpeedAlert:
  def setUp(self, op_long):
    test_car = CAR.HONDA_ODYSSEY_BOSCH

    # setup
    CarInterface, CarController, CarState, RadarInterface = interfaces[test_car]

    CP = CarInterface.get_params(test_car, gen_empty_fingerprint(), {}, experimental_long=op_long, docs=False)

    self.CI = CarInterface(CP, CarController, CarState)
    self.RI = RadarInterface(CP) # unused
    self.Card = Car(self.CI, RadarInterface)

    self.CP = self.CI.CP

    self.CC = self.CI.CC
    self.CS = self.CI.CS

    self.CC_prev = self.Card.CC_prev
    self.CS_prev = self.Card.CS_prev

    self.car_events = CarSpecificEvents(self.CP)

    # cause not real
    # self.CS.steer_on = True
    self.car_events.min_steer_alert_speed = 0.

  def _enabled(self, _):
    self.CC_prev.enabled = _

  def _speed(self, _):
    self.CS.out.vEgo = _

  def _steer_on(self, _):
    self.CS.steer_on = _

  def _update_events(self):
    self.CS.events = self.car_events.update(self.CS, self.CS_prev, self.CC, self.CC_prev)


  def test_alert_transitions(self):
    for e in (True, False):
      # check stock and op long
      # op long minsteer is 0

      self.setUp(op_long=e)
      self._update_events()

      if self.CP.minSteerSpeed < 6.:
        assert BelowSteerSpeed not in self.CS.events.names
      else:
        self._enabled(e)

        # never at standstill
        self.CS.out.standstill = True
        self._update_events()
        assert BelowSteerSpeed not in self.CS.events.names
        # reset
        self.CS.out.standstill = False
        self._update_events()

        # a little slower, show alert
        # same, show alert
        for _ in (self._speed(self.CP.minSteerSpeed-1),
                  self._speed(self.CP.minSteerSpeed)):
          self._update_events()
          assert BelowSteerSpeed in self.CS.events.names

        # just over min steer
        # faster first time, below threshold
        # slower again
        for _ in (self._speed(self.CP.minSteerSpeed+0.001),
                  self._speed(self.CP.minSteerSpeed + 3.4),
                  self._speed(self.CP.minSteerSpeed + 1)):
          self._update_events()
          assert BelowSteerSpeed not in self.CS.events.names

        # slow back below, show alert
        self._speed(self.CP.minSteerSpeed - 1)
        self._update_events()
        assert BelowSteerSpeed in self.CS.events.names

        # faster second time
        self._speed(self.CP.minSteerSpeed + 1)
        self._update_events()
        assert BelowSteerSpeed not in self.CS.events.names

        # faster again, but over the pre_alert threshold
        # now slow just under the pre_alert threshold
        # even slower still
        for _ in (self._speed(self.CP.minSteerSpeed + 3.6),
                  self._speed(self.CP.minSteerSpeed + 3.4),
                  self._speed(self.CP.minSteerSpeed + 1.6)):
          self._update_events()
          assert BelowSteerSpeed not in self.CS.events.names

      # slow down to the lower threshold, show alert
      # slow down below the lower threshold, show alert if pre_alert
        for _ in (self._speed(self.CP.minSteerSpeed + 1.5),
                  self._speed(self.CP.minSteerSpeed + 1.4)):
          self._update_events()
          if self.car_events.min_steer_alert_speed is not self.CP.minSteerSpeed:
            assert BelowSteerSpeed in self.CS.events.names
          else:
            assert BelowSteerSpeed not in self.CS.events.names

        # speed back up slightly, no alert
        self._speed(self.CP.minSteerSpeed+1.6)
        self._update_events()
        assert BelowSteerSpeed not in self.CS.events.names

        # slow down to same, show alert
        # slow down again for the last time, show alert
        for _ in (self._speed(self.CP.minSteerSpeed),
                  self._speed(self.CP.minSteerSpeed-1)):
          self._update_events()
          assert BelowSteerSpeed in self.CS.events.names

