import pytest
import itertools
import numpy as np

from parameterized import parameterized_class
from cereal import log
from openpilot.selfdrive.controls.lib.drive_helpers import VCruiseHelper, V_CRUISE_MIN, V_CRUISE_MAX, V_CRUISE_INITIAL, CRUISE_LONG_PRESS
from cereal import car
from openpilot.common.conversions import Conversions as CV
from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver

ButtonEvent = car.CarState.ButtonEvent
ButtonType = car.CarState.ButtonEvent.Type

MPH_INCR_BY_1_VALUES = range(round(V_CRUISE_MIN * CV.KPH_TO_MPH), round(V_CRUISE_MAX * CV.KPH_TO_MPH)+1)
MPH_INCR_BY_5_VALUES = [MPH_INCR_BY_1_VALUES[i] for i, v in enumerate(MPH_INCR_BY_1_VALUES) if i == 0 or v % 5 == 0]
KPH_INCR_BY_1_VALUES = range(V_CRUISE_MIN, V_CRUISE_MAX+1)
KPH_INCR_BY_5_VALUES = [KPH_INCR_BY_1_VALUES[i] for i, v in enumerate(KPH_INCR_BY_1_VALUES) if i == 0 or v % 5 == 0]

def run_cruise_simulation(cruise, e2e, personality, t_end=20.):
  man = Maneuver(
    '',
    duration=t_end,
    initial_speed=max(cruise - 1., 0.0),
    lead_relevancy=True,
    initial_distance_lead=100,
    cruise_values=[cruise],
    prob_lead_values=[0.0],
    breakpoints=[0.],
    e2e=e2e,
    personality=personality,
  )
  valid, output = man.evaluate()
  assert valid
  return output[-1, 3]


@parameterized_class(("e2e", "personality", "speed"), itertools.product(
                      [True, False], # e2e
                      log.LongitudinalPersonality.schema.enumerants, # personality
                      [5,35])) # speed
class TestCruiseSpeed:
  def test_cruise_speed(self):
    print(f'Testing {self.speed} m/s')
    cruise_speed = float(self.speed)

    simulation_steady_state = run_cruise_simulation(cruise_speed, self.e2e, self.personality)
    assert simulation_steady_state == pytest.approx(cruise_speed, abs=.01), f'Did not reach {self.speed} m/s'


# TODO: test pcmCruise
@parameterized_class(('pcm_cruise',), [(False,)])
class TestVCruiseHelper:
  def setup_method(self):
    self.CP = car.CarParams(pcmCruise=self.pcm_cruise)
    self.v_cruise_helper = VCruiseHelper(self.CP)
    self.reset_cruise_speed_state()

  def reset_cruise_speed_state(self):
    # Two resets previous cruise speed
    for _ in range(2):
      self.v_cruise_helper.update_v_cruise(car.CarState(cruiseState={"available": False}), enabled=False, is_metric=False)

  def enable(self, v_ego, experimental_mode):
    # Simulates user pressing set with a current speed
    self.v_cruise_helper.initialize_v_cruise(car.CarState(vEgo=v_ego), experimental_mode)

  def press_button(self, button, enabled=True, is_metric=True):
    CS = car.CarState(cruiseState={"available": True})
    CS.buttonEvents=[ButtonEvent(type=button, pressed=True)]
    self.v_cruise_helper.update_v_cruise(CS, enabled=enabled, is_metric=is_metric)
    CS.buttonEvents=[ButtonEvent(type=button, pressed=False)]
    self.v_cruise_helper.update_v_cruise(CS, enabled=enabled, is_metric=is_metric)

  def hold_button(self, button, enabled=True, is_metric=True):
    CS = car.CarState(cruiseState={"available": True}, buttonEvents=[ButtonEvent(type=button, pressed=True)])
    self.v_cruise_helper.update_v_cruise(CS, enabled=enabled, is_metric=is_metric)
    CS = car.CarState(cruiseState={"available": True})
    for _ in range(CRUISE_LONG_PRESS):
      self.v_cruise_helper.update_v_cruise(CS, enabled=enabled, is_metric=is_metric)

  def simulate_cruise_speed_range(self, cruise_button, expected_values, start_value, is_metric, hold_button):
    self.enable(0, False)
    self.v_cruise_helper.v_cruise_kph = start_value

    # each button press should step through expected_values
    for expected_value in expected_values:
      if hold_button:
        self.hold_button(cruise_button, is_metric=is_metric)
      else:
        self.press_button(cruise_button, is_metric=is_metric)

      if is_metric:
        assert self.v_cruise_helper.v_cruise_kph == expected_value
      else:
        assert abs(self.v_cruise_helper.v_cruise_kph * CV.KPH_TO_MPH - expected_value) < 0.1

    # verify additional button presses do nothing
    for _ in range(3):
      if hold_button:
        self.hold_button(cruise_button, is_metric=is_metric)
      else:
        self.press_button(cruise_button, is_metric=is_metric)

      if is_metric:
        assert self.v_cruise_helper.v_cruise_kph == expected_values[-1]
      else:
        assert abs(self.v_cruise_helper.v_cruise_kph * CV.KPH_TO_MPH - expected_values[-1]) < 0.1

  def test_adjust_speed(self):
    """
    Asserts speed changes on falling edges of buttons.
    """

    self.enable(V_CRUISE_INITIAL * CV.KPH_TO_MS, False)

    for btn in (ButtonType.accelCruise, ButtonType.decelCruise):
      for pressed in (True, False):
        CS = car.CarState(cruiseState={"available": True})
        CS.buttonEvents = [ButtonEvent(type=btn, pressed=pressed)]

        self.v_cruise_helper.update_v_cruise(CS, enabled=True, is_metric=False)
        assert pressed == (self.v_cruise_helper.v_cruise_kph == self.v_cruise_helper.v_cruise_kph_last)

  def test_rising_edge_enable(self):
    """
    Some car interfaces may enable on rising edge of a button,
    ensure we don't adjust speed if enabled changes mid-press.
    """

    # NOTE: enabled is always one frame behind the result from button press in controlsd
    for enabled, pressed in ((False, False),
                             (False, True),
                             (True, False)):
      CS = car.CarState(cruiseState={"available": True})
      CS.buttonEvents = [ButtonEvent(type=ButtonType.decelCruise, pressed=pressed)]
      self.v_cruise_helper.update_v_cruise(CS, enabled=enabled, is_metric=False)
      if pressed:
        self.enable(V_CRUISE_INITIAL * CV.KPH_TO_MS, False)

      # Expected diff on enabling. Speed should not change on falling edge of pressed
      assert not pressed == self.v_cruise_helper.v_cruise_kph == self.v_cruise_helper.v_cruise_kph_last

  def test_resume_in_standstill(self):
    """
    Asserts we don't increment set speed if user presses resume/accel to exit cruise standstill.
    """

    self.enable(0, False)

    for standstill in (True, False):
      for pressed in (True, False):
        CS = car.CarState(cruiseState={"available": True, "standstill": standstill})
        CS.buttonEvents = [ButtonEvent(type=ButtonType.accelCruise, pressed=pressed)]
        self.v_cruise_helper.update_v_cruise(CS, enabled=True, is_metric=False)

        # speed should only update if not at standstill and button falling edge
        should_equal = standstill or pressed
        assert should_equal == (self.v_cruise_helper.v_cruise_kph == self.v_cruise_helper.v_cruise_kph_last)

  def test_set_gas_pressed(self):
    """
    Asserts pressing set while enabled with gas pressed sets
    the speed to the maximum of vEgo and current cruise speed.
    """

    for v_ego in np.linspace(0, 100, 101):
      self.reset_cruise_speed_state()
      self.enable(V_CRUISE_INITIAL * CV.KPH_TO_MS, False)

      # first decrement speed by 1 mph, then perform gas pressed logic
      expected_v_cruise_kph = round(round(self.v_cruise_helper.v_cruise_kph * CV.KPH_TO_MPH - 1) * CV.MPH_TO_KPH, 1)
      expected_v_cruise_kph = max(expected_v_cruise_kph, v_ego * CV.MS_TO_KPH)  # clip to min of vEgo
      expected_v_cruise_kph = float(np.clip(round(expected_v_cruise_kph, 1), V_CRUISE_MIN, V_CRUISE_MAX))

      CS = car.CarState(vEgo=float(v_ego), gasPressed=True, cruiseState={"available": True})
      CS.buttonEvents = [ButtonEvent(type=ButtonType.decelCruise, pressed=False)]
      self.v_cruise_helper.update_v_cruise(CS, enabled=True, is_metric=False)

      # TODO: fix skipping first run due to enabled on rising edge exception
      if v_ego == 0.0:
        continue
      assert expected_v_cruise_kph == self.v_cruise_helper.v_cruise_kph

  def test_initialize_v_cruise(self):
    """
    Asserts allowed cruise speeds on enabling with SET.
    """

    for experimental_mode in (True, False):
      for v_ego in np.linspace(0, 100, 101):
        self.reset_cruise_speed_state()
        assert not self.v_cruise_helper.v_cruise_initialized

        self.enable(float(v_ego), experimental_mode)
        assert V_CRUISE_INITIAL <= self.v_cruise_helper.v_cruise_kph <= V_CRUISE_MAX
        assert self.v_cruise_helper.v_cruise_initialized

  def test_increment_by_1_mph(self):
    """
    Asserts that cruise speed increments by 1 mph when pressing ACCEL button.
    """

    self.simulate_cruise_speed_range(ButtonType.accelCruise, list(MPH_INCR_BY_1_VALUES[1:]), V_CRUISE_MIN, False, False)

  def test_decrement_by_1_mph(self):
    """
    Asserts that cruise speed decrements by 1 mph when pressing DECEL button.
    """

    self.simulate_cruise_speed_range(ButtonType.decelCruise, list(reversed(MPH_INCR_BY_1_VALUES[:-1])), V_CRUISE_MAX, False, False)

  def test_increment_by_1_kph(self):
    """
    Asserts that cruise speed increments by 1 kph when pressing ACCEL button.
    """

    self.simulate_cruise_speed_range(ButtonType.accelCruise, list(KPH_INCR_BY_1_VALUES[1:]), V_CRUISE_MIN, True, False)

  def test_decrement_by_1_kph(self):
    """
    Asserts that cruise speed decrement by 1 kph when pressing DECEL button.
    """

    self.simulate_cruise_speed_range(ButtonType.decelCruise, list(reversed(KPH_INCR_BY_1_VALUES[:-1])), V_CRUISE_MAX, True, False)

  def test_increment_by_5_mph(self):
    """
    Asserts that speed increments by 5 mph when holding ACCEL button.
    """

    self.simulate_cruise_speed_range(ButtonType.accelCruise, list(MPH_INCR_BY_5_VALUES[1:]), V_CRUISE_MIN, False, True)

  def test_decrement_by_5_mph(self):
    """
    Asserts that speed decrements by 5 mph when holding DECEL button.
    """

    self.simulate_cruise_speed_range(ButtonType.decelCruise, list(reversed(MPH_INCR_BY_5_VALUES[:-1])), V_CRUISE_MAX, False, True)

  def test_increment_by_5_kph(self):
    """
    Asserts that speed increments by 5 kph when holding ACCEL button.
    """

    self.simulate_cruise_speed_range(ButtonType.accelCruise, list(KPH_INCR_BY_5_VALUES[1:]), V_CRUISE_MIN, True, True)

  def test_decrement_by_5_kph(self):
    """
    Asserts that speed decrements by 5 kph when holding DECEL button.
    """

    self.simulate_cruise_speed_range(ButtonType.decelCruise, list(reversed(KPH_INCR_BY_5_VALUES[:-1])), V_CRUISE_MAX, True, True)
