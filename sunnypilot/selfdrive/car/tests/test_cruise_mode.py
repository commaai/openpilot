from parameterized import parameterized_class
from cereal import car
from openpilot.selfdrive.selfdrived.events import Events
from openpilot.sunnypilot.selfdrive.car.cruise_helpers import CruiseHelper, DISTANCE_LONG_PRESS

ButtonEvent = car.CarState.ButtonEvent
ButtonType = car.CarState.ButtonEvent.Type


@parameterized_class(('openpilot_longitudinal',), [(True,)])
class TestCruiseHelper:
  def setup_method(self):
    self.CP = car.CarParams(openpilotLongitudinalControl=self.openpilot_longitudinal)
    self.cruise_helper = CruiseHelper(self.CP)
    self.cruise_helper.experimental_mode_switched = False
    self.events = Events()

  def reset(self):
    for _ in range(2):
      CS = car.CarState(cruiseState={"available": False})
      CS.buttonEvents = [ButtonEvent(type=ButtonType.gapAdjustCruise, pressed=False)]
      self.cruise_helper._experimental_mode = False
      self.cruise_helper.experimental_mode_switched = False
      self.cruise_helper.update(CS, self.events, False)


  def test_gap_adjust_cruise_long_press_toggle_mode(self) -> None:
    for pressed in (True, False):
      for experimental_mode in (True, False):
        self.reset()
        self.cruise_helper._experimental_mode = experimental_mode
        toggled_mode = not experimental_mode if pressed else experimental_mode

        for i in range(DISTANCE_LONG_PRESS):
          CS = car.CarState(cruiseState={"available": True})
          CS.buttonEvents = [ButtonEvent(type=ButtonType.gapAdjustCruise, pressed=pressed)] if i == 0 else []
          self.cruise_helper.update(CS, self.events, experimental_mode)

        # mode should be toggled
        assert self.cruise_helper._experimental_mode == toggled_mode
        assert self.cruise_helper.experimental_mode_switched is pressed

        # keep holding button after switching mode
        for _ in range(DISTANCE_LONG_PRESS):
          CS = car.CarState(cruiseState={"available": True})
          CS.buttonEvents = [ButtonEvent(type=ButtonType.gapAdjustCruise, pressed=pressed)]
          self.cruise_helper.update(CS, self.events, toggled_mode)

        # mode should not be toggled
        assert self.cruise_helper._experimental_mode == toggled_mode
        assert self.cruise_helper.experimental_mode_switched is pressed

  def test_gap_adjust_cruise_short_press_toggle_mode(self) -> None:
    for pressed in (True, False):
      for experimental_mode in (True, False):
        self.reset()
        self.cruise_helper._experimental_mode = experimental_mode

        for i in range(DISTANCE_LONG_PRESS - 1):
          CS = car.CarState(cruiseState={"available": True})
          CS.buttonEvents = [ButtonEvent(type=ButtonType.gapAdjustCruise, pressed=pressed)] if i == 0 else []
          self.cruise_helper.update(CS, self.events, experimental_mode)

        # mode should not be toggled
        assert self.cruise_helper._experimental_mode == experimental_mode
        assert self.cruise_helper.experimental_mode_switched is False
