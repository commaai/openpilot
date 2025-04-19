"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from parameterized import parameterized

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper, LaneChangeState, LaneChangeDirection
from openpilot.sunnypilot.selfdrive.controls.lib.auto_lane_change import AutoLaneChangeController, AutoLaneChangeMode, \
  AUTO_LANE_CHANGE_TIMER, ONE_SECOND_DELAY

AUTO_LANE_CHANGE_TIMER_COMBOS = [
  (AutoLaneChangeMode.NUDGELESS, AUTO_LANE_CHANGE_TIMER[AutoLaneChangeMode.NUDGELESS]),
  (AutoLaneChangeMode.HALF_SECOND, AUTO_LANE_CHANGE_TIMER[AutoLaneChangeMode.HALF_SECOND]),
  (AutoLaneChangeMode.ONE_SECOND, AUTO_LANE_CHANGE_TIMER[AutoLaneChangeMode.ONE_SECOND]),
  (AutoLaneChangeMode.TWO_SECONDS, AUTO_LANE_CHANGE_TIMER[AutoLaneChangeMode.TWO_SECONDS]),
  (AutoLaneChangeMode.THREE_SECONDS, AUTO_LANE_CHANGE_TIMER[AutoLaneChangeMode.THREE_SECONDS])
]


class TestAutoLaneChangeController:
  def setup_method(self):
    self.DH = DesireHelper()
    self.alc = AutoLaneChangeController(self.DH)

  def _reset_states(self):
    self.alc.lane_change_bsm_delay = False
    self.alc.lane_change_set_timer = AutoLaneChangeMode.NUDGE
    self.lane_change_wait_timer = 0.0
    self.prev_brake_pressed = False
    self.prev_lane_change = False

  def test_reset(self):
    """Test that reset correctly sets timers back to default."""
    # Set some non-default values
    self.alc.lane_change_wait_timer = 2.0
    self.alc.prev_brake_pressed = True

    # Set the DesireHelper to trigger a reset
    self.DH.lane_change_state = LaneChangeState.off
    self.DH.lane_change_direction = LaneChangeDirection.none

    # Call reset
    self.alc.reset()

    # Check values were reset
    assert self.alc.lane_change_wait_timer == 0.0
    assert not self.alc.prev_brake_pressed

  @parameterized.expand([(AutoLaneChangeMode.OFF, ), (AutoLaneChangeMode.NUDGE, )])

  def test_off_and_nudge_mode(self, timer_state):
    """Test the default OFF and NUDGE mode behavior."""
    self._reset_states()
    # Setup mode
    self.alc.lane_change_bsm_delay = False  # BSM delay off
    self.alc.lane_change_set_timer = timer_state

    # Update controller
    num_updates = int(5.0 / DT_MDL)
    for _ in range(num_updates):  # Run for 5 seconds
      self.alc.update_lane_change(blindspot_detected=False, brake_pressed=False)

    # Mode should not allow lane change immediately
    assert not self.alc.auto_lane_change_allowed

  def test_nudgeless_mode(self):
    """Test the NUDGELESS mode behavior."""
    self._reset_states()
    # Setup NUDGELESS mode
    self.alc.lane_change_bsm_delay = False  # BSM delay off
    self.alc.lane_change_set_timer = AutoLaneChangeMode.NUDGELESS

    # Update controller once to read params
    self.alc.update_lane_change(blindspot_detected=False, brake_pressed=False)

    # Update multiple times to exceed the timer threshold
    for _ in range(1):  # Should exceed 0.1s with multiple DT_MDL updates
      self.alc.update_lane_change(blindspot_detected=False, brake_pressed=False)

    # Now lane change should be allowed
    assert self.alc.lane_change_wait_timer > self.alc.lane_change_delay
    assert self.alc.auto_lane_change_allowed

  @parameterized.expand(AUTO_LANE_CHANGE_TIMER_COMBOS)
  def test_timers(self, timer_state, timer_delay):
      self._reset_states()
      self.alc.lane_change_bsm_delay = False  # BSM delay off
      self.alc.lane_change_set_timer = timer_state

      # Update controller once
      self.alc.update_lane_change(blindspot_detected=False, brake_pressed=False)

      # The timer should still be below the threshold after one update
      assert not self.alc.auto_lane_change_allowed

      # Update enough times to exceed the threshold (seconds / DT_MDL)
      num_updates = int(timer_delay / DT_MDL) + 1  # Add one extra updates to ensure we exceed the threshold
      for _ in range(num_updates):
        self.alc.update_lane_change(blindspot_detected=False, brake_pressed=False)

      # Now lane change should be allowed
      assert self.alc.lane_change_wait_timer > self.alc.lane_change_delay
      assert self.alc.auto_lane_change_allowed

  @parameterized.expand(AUTO_LANE_CHANGE_TIMER_COMBOS)
  def test_brake_pressed_disables_auto_lane_change(self, timer_state, timer_delay):
    """Test that pressing the brake disables auto lane change."""
    self._reset_states()
    # Setup auto lane change mode
    self.alc.lane_change_bsm_delay = False
    self.alc.lane_change_set_timer = timer_state
    num_updates = int(timer_delay / DT_MDL) + 1  # Add one extra updates to ensure we exceed the threshold

    # Update with brake pressed for 1 second
    for _ in range(num_updates):
      self.alc.update_lane_change(blindspot_detected=False, brake_pressed=True)

    # Even though it is an auto lane change mode, lane change should be disallowed due to brake pressed prior initiating lane change
    assert not self.alc.auto_lane_change_allowed

    # Check that prev_brake_pressed is saved
    assert self.alc.prev_brake_pressed

    # Even releasing brake shouldn't allow auto lane change
    for _ in range(num_updates):
      self.alc.update_lane_change(blindspot_detected=False, brake_pressed=False)

    assert not self.alc.auto_lane_change_allowed

  @parameterized.expand(AUTO_LANE_CHANGE_TIMER_COMBOS)
  def test_blindspot_detected_with_bsm_delay(self, timer_state, timer_delay):
    """Test behavior when blindspot is detected with BSM delay enabled."""
    # Blindspot detected - should prevent auto lane change
    self._reset_states()
    self.alc.lane_change_bsm_delay = True  # BSM delay on
    self.alc.lane_change_set_timer = timer_state

    # Update with blindspot detected - this should prevent auto lane change
    self.alc.update_lane_change(blindspot_detected=True, brake_pressed=False)
    assert not self.alc.auto_lane_change_allowed

    # Keep updating with blindspot detected - should still prevent auto lane change
    num_updates = int(timer_delay / DT_MDL) + 1  # Add one extra updates to ensure we exceed the threshold
    for _ in range(num_updates):
      self.alc.update_lane_change(blindspot_detected=True, brake_pressed=False)
    assert not self.alc.auto_lane_change_allowed

  @parameterized.expand(AUTO_LANE_CHANGE_TIMER_COMBOS)
  def test_blindspot_detected_then_undetected_with_bsm_delay(self, timer_state, timer_delay):
    """Test behavior when blindspot is detected then undetected with BSM delay enabled."""
    # Blindspot clears - should allow auto lane change after sufficient time
    self._reset_states()
    self.alc.lane_change_bsm_delay = True
    self.alc.lane_change_set_timer = timer_state

    # First update with blindspot detected to set the negative timer
    self.alc.update_lane_change(blindspot_detected=True, brake_pressed=False)
    assert not self.alc.auto_lane_change_allowed

    # Now update with blindspot cleared - should start incrementing timer from negative value
    num_updates = int((timer_delay + abs(ONE_SECOND_DELAY)) / DT_MDL) + 1
    for _ in range(num_updates):
      self.alc.update_lane_change(blindspot_detected=False, brake_pressed=False)

    # After sufficient updates with no blindspot, auto lane change should be allowed
    assert self.alc.auto_lane_change_allowed

  @parameterized.expand(AUTO_LANE_CHANGE_TIMER_COMBOS)
  def test_disallow_continuous_auto_lane_change(self, timer_state, timer_delay):
      self._reset_states()
      self.alc.lane_change_bsm_delay = False  # BSM delay off
      self.alc.lane_change_set_timer = timer_state
      num_updates = int(timer_delay / DT_MDL) + 1  # Add one extra updates to ensure we exceed the threshold

      # Update enough times to exceed the threshold (seconds / DT_MDL)
      for _ in range(num_updates):
        self.alc.update_lane_change(blindspot_detected=False, brake_pressed=False)

      # Now lane change should be allowed
      assert self.alc.lane_change_wait_timer > self.alc.lane_change_delay
      assert self.alc.auto_lane_change_allowed

      # Simulate lane change is initiated
      self.DH.lane_change_state = LaneChangeState.laneChangeStarting
      self.alc.update_state()

      # Simulate lane change is completed, and one_blinker stays on
      self.DH.lane_change_state = LaneChangeState.preLaneChange
      self.alc.update_state()

      # Update enough times to exceed the threshold (seconds / DT_MDL)
      for _ in range(num_updates):
        self.alc.update_lane_change(blindspot_detected=False, brake_pressed=False)

      assert not self.alc.auto_lane_change_allowed

  def test_auto_lane_change_mode_off_disallows_lane_change(self):
    """Test that OFF mode never allows auto lane change."""
    self._reset_states()
    self.alc.lane_change_bsm_delay = False
    self.alc.lane_change_set_timer = AutoLaneChangeMode.OFF

    # Simulate updates for a long period of time (e.g., 10 seconds)
    num_updates = int(10.0 / DT_MDL)
    for _ in range(num_updates):
      self.alc.update_lane_change(blindspot_detected=False, brake_pressed=False)

    # Lane change should never be allowed
    assert not self.alc.auto_lane_change_allowed
