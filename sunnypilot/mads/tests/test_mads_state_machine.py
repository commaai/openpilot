"""
The MIT License

Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Last updated: July 29, 2024
"""

import pytest
from pytest_mock import MockerFixture

from cereal import custom
from openpilot.common.realtime import DT_CTRL
from openpilot.sunnypilot.mads.state import StateMachine, SOFT_DISABLE_TIME, GEARS_ALLOW_PAUSED
from openpilot.selfdrive.selfdrived.events import ET, NormalPermanentAlert
from openpilot.sunnypilot.selfdrive.selfdrived.events import EVENTS_SP

State = custom.ModularAssistiveDrivingSystem.ModularAssistiveDrivingSystemState
EventNameSP = custom.OnroadEventSP.EventName

# The event types that maintain the current state
MAINTAIN_STATES = {State.enabled: (None,), State.disabled: (None,), State.softDisabling: (ET.SOFT_DISABLE,),
                   State.paused: (None,), State.overriding: (ET.OVERRIDE_LATERAL,)}
ALL_STATES = (State.schema.enumerants.values())
# The event types checked in DISABLED section of state machine
ENABLE_EVENT_TYPES = (ET.ENABLE, ET.OVERRIDE_LATERAL)


def make_event(event_types):
  event = {}
  for ev in event_types:
    event[ev] = NormalPermanentAlert("alert")
  EVENTS_SP[0] = event
  return 0


class MockMADS:
  def __init__(self, mocker: MockerFixture):
    self.selfdrive = mocker.MagicMock()
    self.selfdrive.state_machine = mocker.MagicMock()
    self.selfdrive.active = False


class TestMADSStateMachine:
  @pytest.fixture(autouse=True)
  def setup_method(self, mocker: MockerFixture):
    self.mads = MockMADS(mocker)
    self.state_machine = StateMachine(self.mads)
    self.events = self.state_machine._events
    self.events_sp = self.state_machine._events_sp
    self.mads.selfdrive.state_machine.soft_disable_timer = int(SOFT_DISABLE_TIME / DT_CTRL)

  def reset(self):
    self.events.clear()
    self.events_sp.clear()
    self.state_machine.state = State.disabled

  def test_immediate_disable(self):
    for state in ALL_STATES:
      for et in MAINTAIN_STATES[state]:
        self.events_sp.add(make_event([et, ET.IMMEDIATE_DISABLE]))
        self.state_machine.state = state
        self.state_machine.update(self.events, self.events_sp)
        assert State.disabled == self.state_machine.state
        self.reset()

  def test_user_disable(self):
    for state in ALL_STATES:
      for et in MAINTAIN_STATES[state]:
        self.events_sp.add(make_event([et, ET.USER_DISABLE]))
        self.state_machine.state = state
        self.state_machine.update(self.events, self.events_sp)
        assert State.disabled == self.state_machine.state
        self.reset()

  def test_user_disable_to_paused(self):
    paused_events = (EventNameSP.silentLkasDisable, EventNameSP.silentBrakeHold)
    for state in ALL_STATES:
      for et in MAINTAIN_STATES[state]:
        self.events_sp.add(make_event([et, ET.USER_DISABLE]))
        for en in paused_events:
          self.events_sp.add(en)
          self.state_machine.state = state
          self.state_machine.update(self.events, self.events_sp)
          final_state = State.paused if self.events_sp.has(en) and state != State.disabled else State.disabled
          assert self.state_machine.state == final_state
          self.reset()

  def test_soft_disable(self):
    for state in ALL_STATES:
      if state == State.paused:  # paused considers USER_DISABLE instead
        continue
      for et in MAINTAIN_STATES[state]:
        self.events_sp.add(make_event([et, ET.SOFT_DISABLE]))
        self.state_machine.state = state
        self.state_machine.update(self.events, self.events_sp)
        assert self.state_machine.state == State.disabled if state == State.disabled else State.softDisabling
        self.reset()

  def test_soft_disable_timer(self):
    self.state_machine.state = State.enabled
    self.events_sp.add(make_event([ET.SOFT_DISABLE]))
    self.state_machine.update(self.events, self.events_sp)
    for _ in range(int(SOFT_DISABLE_TIME / DT_CTRL)):
      assert self.state_machine.state == State.softDisabling
      self.mads.selfdrive.state_machine.soft_disable_timer -= 1
      self.state_machine.update(self.events, self.events_sp)

    assert self.state_machine.state == State.disabled

  def test_no_entry(self):
    for et in ENABLE_EVENT_TYPES:
      self.events_sp.add(make_event([ET.NO_ENTRY, et]))
      if not self.state_machine.check_contains_in_list(GEARS_ALLOW_PAUSED):
        self.state_machine.update(self.events, self.events_sp)
        assert self.state_machine.state == State.disabled
        self.reset()

  def test_no_entry_paused(self):
    self.state_machine.state = State.paused
    self.events_sp.add(make_event([ET.NO_ENTRY]))
    self.state_machine.update(self.events, self.events_sp)
    assert self.state_machine.state == State.paused

  def test_override_lateral(self):
    self.state_machine.state = State.enabled
    self.events_sp.add(make_event([ET.OVERRIDE_LATERAL]))
    self.state_machine.update(self.events, self.events_sp)
    assert self.state_machine.state == State.overriding

  def test_paused_to_enabled(self):
    self.state_machine.state = State.paused
    self.events_sp.add(make_event([ET.ENABLE]))
    self.state_machine.update(self.events, self.events_sp)
    assert self.state_machine.state == State.enabled

  def test_maintain_states(self):
    for state in ALL_STATES:
      for et in MAINTAIN_STATES[state]:
        self.state_machine.state = state
        if et is not None:
          self.events_sp.add(make_event([et]))
        self.state_machine.update(self.events, self.events_sp)
        assert self.state_machine.state == state
        self.reset()
