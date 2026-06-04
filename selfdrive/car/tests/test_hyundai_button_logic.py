"""Tests for Hyundai stock longitudinal button behavior (#30950):
  - Resume blocked until Set is pressed first
  - CAN FD: cancel/button-4 is a pause/resume toggle
"""
import pytest
from cereal import car, log
from opendbc.car import structs
from opendbc.car import gen_empty_fingerprint
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR
from openpilot.selfdrive.car.car_specific import CarSpecificEvents

ButtonType = structs.CarState.ButtonEvent.Type
EventName = log.OnroadEvent.EventName


def _make_btn(btn_type, pressed=False):
  b = structs.CarState.ButtonEvent()
  b.type = btn_type
  b.pressed = pressed
  return b


def _make_cs(button_events, button_enable=False):
  cs = car.CarState.new_message()
  cs.buttonEnable = button_enable
  if button_events:
    btns = cs.init('buttonEvents', len(button_events))
    for i, b in enumerate(button_events):
      btns[i].type = b.type
      btns[i].pressed = b.pressed
  return cs


def _make_spec(car_model, pcm_cruise=False):
  CP = CarInterface.get_params(car_model, gen_empty_fingerprint(), [], False, False, False)
  CP.pcmCruise = pcm_cruise
  return CarSpecificEvents(CP)


class TestHyundaiButtonLogicCAN:
  """CAN (non-FD) Hyundai openpilot long: resume blocked until Set pressed."""

  @pytest.fixture(autouse=True)
  def setup(self):
    self.spec = _make_spec(CAR.HYUNDAI_SONATA)
    self.cc = car.CarControl.new_message()
    self.cs_prev = _make_cs([])

  def _run(self, buttons, button_enable=False):
    cs = _make_cs(buttons, button_enable)
    events = self.spec.update(cs, self.cs_prev, self.cc)
    return EventName.buttonEnable in events.events

  def test_resume_blocked_before_set(self):
    assert not self._run([_make_btn(ButtonType.accelCruise)], button_enable=True)

  def test_set_always_enables(self):
    assert self._run([_make_btn(ButtonType.decelCruise)], button_enable=True)

  def test_resume_after_set(self):
    self._run([_make_btn(ButtonType.decelCruise)], button_enable=True)
    assert self._run([_make_btn(ButtonType.accelCruise)], button_enable=True)

  def test_cancel_resets_and_blocks_resume(self):
    self._run([_make_btn(ButtonType.decelCruise)], button_enable=True)
    self._run([_make_btn(ButtonType.cancel)])
    assert not self._run([_make_btn(ButtonType.accelCruise)], button_enable=True)

  def test_main_resets_and_blocks_resume(self):
    self._run([_make_btn(ButtonType.decelCruise)], button_enable=True)
    self._run([_make_btn(ButtonType.mainCruise, pressed=True)])
    assert not self._run([_make_btn(ButtonType.accelCruise)], button_enable=True)

  def test_set_after_cancel_reenables(self):
    self._run([_make_btn(ButtonType.decelCruise)], button_enable=True)
    self._run([_make_btn(ButtonType.cancel)])
    assert self._run([_make_btn(ButtonType.decelCruise)], button_enable=True)


class TestHyundaiButtonLogicCANFD:
  """CAN FD Hyundai openpilot long: button-4 is pause/resume toggle."""

  @pytest.fixture(autouse=True)
  def setup(self):
    self.spec = _make_spec(CAR.KIA_EV6)
    self.cc = car.CarControl.new_message()
    self.cs_prev = _make_cs([])

  def _run(self, buttons, button_enable=False):
    cs = _make_cs(buttons, button_enable)
    events = self.spec.update(cs, self.cs_prev, self.cc)
    return EventName.buttonEnable in events.events

  def test_resume_blocked_before_set(self):
    assert not self._run([_make_btn(ButtonType.accelCruise)], button_enable=True)

  def test_set_enables(self):
    assert self._run([_make_btn(ButtonType.decelCruise)], button_enable=True)

  def test_cancel_pauses_after_set(self):
    self._run([_make_btn(ButtonType.decelCruise)], button_enable=True)
    assert not self._run([_make_btn(ButtonType.cancel)])

  def test_cancel_resumes_after_pause(self):
    self._run([_make_btn(ButtonType.decelCruise)], button_enable=True)
    self._run([_make_btn(ButtonType.cancel)])
    assert self._run([_make_btn(ButtonType.cancel)])

  def test_cancel_without_speed_does_not_enable(self):
    assert not self._run([_make_btn(ButtonType.cancel)])

  def test_main_clears_pause_state(self):
    self._run([_make_btn(ButtonType.decelCruise)], button_enable=True)
    self._run([_make_btn(ButtonType.cancel)])
    self._run([_make_btn(ButtonType.mainCruise, pressed=True)])
    assert not self._run([_make_btn(ButtonType.cancel)])

  def test_pause_resume_multiple_cycles(self):
    self._run([_make_btn(ButtonType.decelCruise)], button_enable=True)
    assert not self._run([_make_btn(ButtonType.cancel)])
    assert self._run([_make_btn(ButtonType.cancel)])
    assert not self._run([_make_btn(ButtonType.cancel)])
    assert self._run([_make_btn(ButtonType.cancel)])


class TestHyundaiButtonLogicPCMCruise:
  """PCM cruise (non-alpha-long) should not be affected."""

  def test_pcm_cruise_not_affected(self):
    spec = _make_spec(CAR.HYUNDAI_SONATA, pcm_cruise=True)
    cc = car.CarControl.new_message()
    cs = _make_cs([_make_btn(ButtonType.accelCruise)], button_enable=False)
    cs_prev = _make_cs([])
    events = spec.update(cs, cs_prev, cc)
    assert EventName.buttonEnable not in events.events
