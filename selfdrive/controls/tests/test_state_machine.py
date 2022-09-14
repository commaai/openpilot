#!/usr/bin/env python3
import unittest

from cereal import car, log
from common.realtime import DT_CTRL
from selfdrive.car.car_helpers import interfaces
from selfdrive.controls.controlsd import Controls, SOFT_DISABLE_TIME
from selfdrive.controls.lib.events import Events, ET, Alert, Priority, AlertSize, AlertStatus, VisualAlert, \
                                          AudibleAlert, EVENTS

State = log.ControlsState.OpenpilotState

# The event types that maintain the current state
MAINTAIN_STATES = {State.enabled: (None,), State.disabled: (None,), State.softDisabling: (ET.SOFT_DISABLE,),
                   State.preEnabled: (ET.PRE_ENABLE,), State.overriding: (ET.OVERRIDE_LATERAL, ET.OVERRIDE_LONGITUDINAL)}
ALL_STATES = tuple(State.schema.enumerants.values())
# The event types checked in DISABLED section of state machine
ENABLE_EVENT_TYPES = (ET.ENABLE, ET.PRE_ENABLE, ET.OVERRIDE_LATERAL, ET.OVERRIDE_LONGITUDINAL)


def make_event(event_types):
  event = {}
  for ev in event_types:
    event[ev] = Alert("", "", AlertStatus.normal, AlertSize.small, Priority.LOW,
                      VisualAlert.none, AudibleAlert.none, 1.)
  EVENTS[0] = event
  return 0


class TestStateMachine(unittest.TestCase):

  def setUp(self):
    CarInterface, CarController, CarState = interfaces["mock"]
    CP = CarInterface.get_params("mock")
    CI = CarInterface(CP, CarController, CarState)

    self.controlsd = Controls(CI=CI)
    self.controlsd.events = Events()
    self.controlsd.soft_disable_timer = int(SOFT_DISABLE_TIME / DT_CTRL)
    self.CS = car.CarState()

  def test_immediate_disable(self):
    for state in ALL_STATES:
      for et in MAINTAIN_STATES[state]:
        self.controlsd.events.add(make_event([et, ET.IMMEDIATE_DISABLE]))
        self.controlsd.state = state
        self.controlsd.state_transition(self.CS)
        self.assertEqual(State.disabled, self.controlsd.state)
        self.controlsd.events.clear()

  def test_user_disable(self):
    for state in ALL_STATES:
      for et in MAINTAIN_STATES[state]:
        self.controlsd.events.add(make_event([et, ET.USER_DISABLE]))
        self.controlsd.state = state
        self.controlsd.state_transition(self.CS)
        self.assertEqual(State.disabled, self.controlsd.state)
        self.controlsd.events.clear()

  def test_soft_disable(self):
    for state in ALL_STATES:
      if state == State.preEnabled:  # preEnabled considers NO_ENTRY instead
        continue
      for et in MAINTAIN_STATES[state]:
        self.controlsd.events.add(make_event([et, ET.SOFT_DISABLE]))
        self.controlsd.state = state
        self.controlsd.state_transition(self.CS)
        self.assertEqual(self.controlsd.state, State.disabled if state == State.disabled else State.softDisabling)
        self.controlsd.events.clear()

  def test_soft_disable_timer(self):
    self.controlsd.state = State.enabled
    self.controlsd.events.add(make_event([ET.SOFT_DISABLE]))
    self.controlsd.state_transition(self.CS)
    for _ in range(int(SOFT_DISABLE_TIME / DT_CTRL)):
      self.assertEqual(self.controlsd.state, State.softDisabling)
      self.controlsd.state_transition(self.CS)

    self.assertEqual(self.controlsd.state, State.disabled)

  def test_no_entry(self):
    # disabled with enable events
    for et in ENABLE_EVENT_TYPES:
      self.controlsd.events.add(make_event([ET.NO_ENTRY, et]))
      self.controlsd.state_transition(self.CS)
      self.assertEqual(self.controlsd.state, State.disabled)
      self.controlsd.events.clear()

  def test_no_entry_pre_enable(self):
    # preEnabled with preEnabled event
    self.controlsd.state = State.preEnabled
    self.controlsd.events.add(make_event([ET.NO_ENTRY, ET.PRE_ENABLE]))
    self.controlsd.state_transition(self.CS)
    self.assertEqual(self.controlsd.state, State.disabled)

  def test_maintain_states(self):
    # Given current state's event type, we should maintain state
    for state in ALL_STATES:
      for et in MAINTAIN_STATES[state]:
        self.controlsd.state = state
        self.controlsd.events.add(make_event([et]))
        self.controlsd.state_transition(self.CS)
        self.assertEqual(self.controlsd.state, state)
        self.controlsd.events.clear()


if __name__ == "__main__":
  unittest.main()
