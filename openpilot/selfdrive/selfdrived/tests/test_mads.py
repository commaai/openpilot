from opendbc.car.structs import car

from openpilot.selfdrive.selfdrived.selfdrived import SelfdriveD


class TestMadsPedals:
  def setup_method(self):
    self.selfdrive = SelfdriveD.__new__(SelfdriveD)
    self.selfdrive.CS_prev = car.CarState.new_message()
    self.selfdrive.disengage_on_accelerator = True
    self.selfdrive.mads_available = True
    self.selfdrive.mads_main_requested = False
    self.selfdrive.mads_lateral_only = False

  def test_pedals_do_not_disengage_with_main_on(self):
    self.selfdrive.mads_available = True
    for pedal in ("gasPressed", "brakePressed", "regenBraking"):
      cs = car.CarState.new_message()
      cs.cruiseState.available = True
      setattr(cs, pedal, True)
      assert not self.selfdrive.should_disengage_on_pedal(cs)

  def test_pedals_disengage_when_mads_or_main_is_off(self):
    for mads_available, main_on in ((False, True), (True, False)):
      self.selfdrive.mads_available = mads_available
      cs = car.CarState.new_message()
      cs.cruiseState.available = main_on
      cs.brakePressed = True
      assert self.selfdrive.should_disengage_on_pedal(cs)

  def test_invalid_can_does_not_clear_main_state(self):
    cs = car.CarState.new_message()
    cs.canValid = True
    cs.cruiseState.available = True
    cs.cruiseState.enabled = False
    self.selfdrive.update_mads_cruise_state(cs)
    assert self.selfdrive.mads_main_requested
    assert self.selfdrive.mads_lateral_only

    cs.canValid = False
    cs.cruiseState.available = False
    self.selfdrive.update_mads_cruise_state(cs)
    assert self.selfdrive.mads_main_requested
    assert self.selfdrive.mads_lateral_only

    cs.canValid = True
    self.selfdrive.update_mads_cruise_state(cs)
    assert not self.selfdrive.mads_main_requested
    assert not self.selfdrive.mads_lateral_only
