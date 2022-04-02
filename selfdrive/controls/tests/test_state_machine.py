#!/usr/bin/env python3
import unittest

from cereal import car, log
from selfdrive.car.car_helpers import interfaces
from selfdrive.controls.controlsd import Controls
from selfdrive.controls.lib.events import ET

State = log.ControlsState.OpenpilotState

# The event types that maintain the current state
MAINTAIN_STATES = {State.enabled: None, State.disabled: None, State.softDisabling: ET.SOFT_DISABLE,
                   State.preEnabled: ET.PRE_ENABLE}
ALL_STATES = tuple(State.schema.enumerants.values())
# The event types checked in DISABLED section of state machine
ENABLE_EVENT_TYPES = (ET.ENABLE, ET.PRE_ENABLE)


class Events:
  # Provides identical API for state_transition
  def __init__(self):
    self.et = []

  def any(self, event_type):
    return event_type in self.et


class TestStateMachine(unittest.TestCase):

  def setUp(self):
    CarInterface, CarController, CarState = interfaces["mock"]
    CP = CarInterface.get_params("mock")
    CI = CarInterface(CP, CarController, CarState)

    self.controlsd = Controls(CI=CI)
    self.controlsd.events = Events()
    self.controlsd.soft_disable_timer = 200  # make sure timer never causes state to change
    self.CS = car.CarState()

  def test_immediate_disable(self):
    for state in ALL_STATES:
      self.controlsd.events.et = [MAINTAIN_STATES[state], ET.IMMEDIATE_DISABLE]
      self.controlsd.state = state
      self.controlsd.state_transition(self.CS)
      self.assertEqual(State.disabled, self.controlsd.state)

  def test_user_disable(self):
    for state in ALL_STATES:
      self.controlsd.events.et = [MAINTAIN_STATES[state], ET.USER_DISABLE]
      self.controlsd.state = state
      self.controlsd.state_transition(self.CS)
      self.assertEqual(State.disabled, self.controlsd.state)

  def test_no_entry(self):
    # Disabled
    for et in ENABLE_EVENT_TYPES:
      self.controlsd.events.et = [ET.NO_ENTRY, et]
      self.controlsd.state_transition(self.CS)
      self.assertEqual(self.controlsd.state, State.disabled)

    # preEnabled also should consider NO_ENTRY
    self.setUp()
    self.controlsd.state = State.preEnabled
    self.controlsd.events.et = [ET.NO_ENTRY]
    self.controlsd.state_transition(self.CS)
    self.assertEqual(self.controlsd.state, State.disabled)

  def test_maintain_states(self):
    for state in ALL_STATES:
      self.controlsd.state = state
      self.controlsd.events.et = [MAINTAIN_STATES[state]]
      self.controlsd.state_transition(self.CS)
      self.assertEqual(self.controlsd.state, state)


if __name__ == "__main__":
  unittest.main()
