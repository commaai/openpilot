# flake8: noqa

import unittest
import numpy as np
from cereal import car, log
from common.realtime import DT_DMON
from selfdrive.controls.lib.events import Events
from selfdrive.monitoring.hands_on_wheel_monitor import HandsOnWheelStatus, _PRE_ALERT_THRESHOLD, \
                                  _PROMPT_ALERT_THRESHOLD, _TERMINAL_ALERT_THRESHOLD, \
                                  _MIN_MONITORING_SPEED

EventName = car.CarEvent.EventName
HandsOnWheelState = log.DriverMonitoringState.HandsOnWheelState

_TEST_TIMESPAN = 120  # seconds

# some common state vectors
test_samples = int(_TEST_TIMESPAN / DT_DMON)
half_test_samples = int(test_samples / 2.)
always_speed_over_threshold = [_MIN_MONITORING_SPEED + 1.] * test_samples
always_speed_under_threshold = [_MIN_MONITORING_SPEED - 1.] * test_samples
always_true = [True] * test_samples
always_false = [False] * test_samples
true_then_false = [True] * half_test_samples + [False] * (test_samples - half_test_samples)


def run_HOWState_seq(steering_wheel_interaction, openpilot_status, speed_status):
  # inputs are all 10Hz
  HOWS = HandsOnWheelStatus()
  events_from_HOWM = []
  hands_on_wheel_state_from_HOWM = []

  for idx in range(len(steering_wheel_interaction)):
    e = Events()
    # evaluate events at 10Hz for tests
    HOWS.update(e, steering_wheel_interaction[idx], openpilot_status[idx], speed_status[idx])
    events_from_HOWM.append(e)
    hands_on_wheel_state_from_HOWM.append(HOWS.hands_on_wheel_state)

  assert len(events_from_HOWM) == len(steering_wheel_interaction), 'somethings wrong'
  assert len(hands_on_wheel_state_from_HOWM) == len(steering_wheel_interaction), 'somethings wrong'
  return events_from_HOWM, hands_on_wheel_state_from_HOWM


class TestHandsMonitoring(unittest.TestCase):
  # 0. op engaged over monitoring speed, driver has hands on wheel all the time
  def test_hands_on_all_the_time(self):
    events_output, state_output = run_HOWState_seq(always_true, always_true, always_speed_over_threshold)
    self.assertTrue(np.sum([len(event) for event in events_output]) == 0)
    self.assertEqual(state_output, [HandsOnWheelState.ok for x in range(len(state_output))])

  # 1. op engaged under monitoring speed, steering wheel interaction is irrelevant
  def test_monitoring_under_threshold_speed(self):
    events_output, state_output = run_HOWState_seq(true_then_false, always_true, always_speed_under_threshold)
    self.assertTrue(np.sum([len(event) for event in events_output]) == 0)
    self.assertEqual(state_output, [HandsOnWheelState.none for x in range(len(state_output))])

  # 2. op engaged over monitoring speed, driver has no hands on wheel all the time
  def test_hands_off_all_the_time(self):
    events_output, state_output = run_HOWState_seq(always_false, always_true, always_speed_over_threshold)
    # Assert correctness before _PRE_ALERT_THRESHOLD
    self.assertTrue(np.sum([len(event) for event in events_output[:_PRE_ALERT_THRESHOLD - 1]]) == 0)
    self.assertEqual(state_output[:_PRE_ALERT_THRESHOLD - 1],
                     [HandsOnWheelState.minor for x in range(_PRE_ALERT_THRESHOLD - 1)])
    # Assert correctness before _PROMPT_ALERT_THRESHOLD
    self.assertEqual([event.names[0] for event in events_output[_PRE_ALERT_THRESHOLD:_PROMPT_ALERT_THRESHOLD - 1]],
                     [EventName.preKeepHandsOnWheel for x in range(_PROMPT_ALERT_THRESHOLD - 1 - _PRE_ALERT_THRESHOLD)])
    self.assertEqual(state_output[_PRE_ALERT_THRESHOLD:_PROMPT_ALERT_THRESHOLD - 1],
                     [HandsOnWheelState.warning for x in range(_PROMPT_ALERT_THRESHOLD - 1 - _PRE_ALERT_THRESHOLD)])
    # Assert correctness before _TERMINAL_ALERT_THRESHOLD
    self.assertEqual(
        [event.names[0] for event in events_output[_PROMPT_ALERT_THRESHOLD:_TERMINAL_ALERT_THRESHOLD - 1]],
        [EventName.promptKeepHandsOnWheel for x in range(_TERMINAL_ALERT_THRESHOLD - 1 - _PROMPT_ALERT_THRESHOLD)])
    self.assertEqual(
        state_output[_PROMPT_ALERT_THRESHOLD:_TERMINAL_ALERT_THRESHOLD - 1],
        [HandsOnWheelState.critical for x in range(_TERMINAL_ALERT_THRESHOLD - 1 - _PROMPT_ALERT_THRESHOLD)])
    # Assert correctness after _TERMINAL_ALERT_THRESHOLD
    self.assertEqual([event.names[0] for event in events_output[_TERMINAL_ALERT_THRESHOLD:]],
                     [EventName.keepHandsOnWheel for x in range(test_samples - _TERMINAL_ALERT_THRESHOLD)])
    self.assertEqual(state_output[_TERMINAL_ALERT_THRESHOLD:],
                     [HandsOnWheelState.terminal for x in range(test_samples - _TERMINAL_ALERT_THRESHOLD)])

  # 3. op engaged over monitoring speed, alert status resets to none when going under monitoring speed
  def test_status_none_when_speeds_goes_down(self):
    speed_vector = always_speed_over_threshold[:-1] + [_MIN_MONITORING_SPEED - 1.]
    events_output, state_output = run_HOWState_seq(always_false, always_true, speed_vector)
    # Assert correctness after _TERMINAL_ALERT_THRESHOLD
    self.assertEqual([event.names[0] for event in events_output[_TERMINAL_ALERT_THRESHOLD:test_samples - 1]],
                     [EventName.keepHandsOnWheel for x in range(test_samples - 1 - _TERMINAL_ALERT_THRESHOLD)])
    self.assertEqual(state_output[_TERMINAL_ALERT_THRESHOLD:test_samples - 1],
                     [HandsOnWheelState.terminal for x in range(test_samples - 1 - _TERMINAL_ALERT_THRESHOLD)])
    # Assert correctes on last sample where speed went under monitoring threshold
    self.assertEqual(len(events_output[-1]), 0)
    self.assertEqual(state_output[-1], HandsOnWheelState.none)

  # 4. op engaged over monitoring speed, alert status resets to ok when user interacts with steering wheel,
  # process repeats once hands are off wheel.
  def test_status_ok_after_interaction_with_wheel(self):
    interaction_vector = always_false[:_TERMINAL_ALERT_THRESHOLD] + [True
                                                                     ] + always_false[_TERMINAL_ALERT_THRESHOLD + 1:]
    events_output, state_output = run_HOWState_seq(interaction_vector, always_true, always_speed_over_threshold)
    # Assert correctness after _TERMINAL_ALERT_THRESHOLD
    self.assertEqual(events_output[_TERMINAL_ALERT_THRESHOLD - 1].names[0], EventName.keepHandsOnWheel)
    self.assertEqual(state_output[_TERMINAL_ALERT_THRESHOLD - 1], HandsOnWheelState.terminal)
    # Assert correctness for one sample when user interacts with steering wheel
    self.assertEqual(len(events_output[_TERMINAL_ALERT_THRESHOLD]), 0)
    self.assertEqual(state_output[_TERMINAL_ALERT_THRESHOLD], HandsOnWheelState.ok)
    # Assert process correctness on second run
    offset = _TERMINAL_ALERT_THRESHOLD + 1
    self.assertTrue(np.sum([len(event) for event in events_output[offset:offset + _PRE_ALERT_THRESHOLD - 1]]) == 0)
    self.assertEqual(state_output[offset:offset + _PRE_ALERT_THRESHOLD - 1],
                     [HandsOnWheelState.minor for x in range(_PRE_ALERT_THRESHOLD - 1)])
    self.assertEqual(
        [event.names[0] for event in events_output[offset + _PRE_ALERT_THRESHOLD:offset + _PROMPT_ALERT_THRESHOLD - 1]],
        [EventName.preKeepHandsOnWheel for x in range(_PROMPT_ALERT_THRESHOLD - 1 - _PRE_ALERT_THRESHOLD)])
    self.assertEqual(state_output[offset + _PRE_ALERT_THRESHOLD:offset + _PROMPT_ALERT_THRESHOLD - 1],
                     [HandsOnWheelState.warning for x in range(_PROMPT_ALERT_THRESHOLD - 1 - _PRE_ALERT_THRESHOLD)])
    self.assertEqual([
        event.names[0]
        for event in events_output[offset + _PROMPT_ALERT_THRESHOLD:offset + _TERMINAL_ALERT_THRESHOLD - 1]
    ], [EventName.promptKeepHandsOnWheel for x in range(_TERMINAL_ALERT_THRESHOLD - 1 - _PROMPT_ALERT_THRESHOLD)])
    self.assertEqual(
        state_output[offset + _PROMPT_ALERT_THRESHOLD:offset + _TERMINAL_ALERT_THRESHOLD - 1],
        [HandsOnWheelState.critical for x in range(_TERMINAL_ALERT_THRESHOLD - 1 - _PROMPT_ALERT_THRESHOLD)])
    self.assertEqual([event.names[0] for event in events_output[offset + _TERMINAL_ALERT_THRESHOLD:]],
                     [EventName.keepHandsOnWheel for x in range(test_samples - offset - _TERMINAL_ALERT_THRESHOLD)])
    self.assertEqual(state_output[offset + _TERMINAL_ALERT_THRESHOLD:],
                     [HandsOnWheelState.terminal for x in range(test_samples - offset - _TERMINAL_ALERT_THRESHOLD)])

  # 5. op not engaged, always hands off wheel
  #  - monitor should stay quiet when not engaged
  def test_pure_dashcam_user(self):
    events_output, state_output = run_HOWState_seq(always_false, always_false, always_speed_over_threshold)
    self.assertTrue(np.sum([len(event) for event in events_output]) == 0)
    self.assertEqual(state_output, [HandsOnWheelState.none for x in range(len(state_output))])


if __name__ == "__main__":
  unittest.main()
