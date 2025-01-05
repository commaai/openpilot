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

from cereal import messaging, car, log, custom

from opendbc.car.hyundai.values import HyundaiFlags
from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP

from openpilot.sunnypilot.mads.helpers import MadsParams
from openpilot.sunnypilot.mads.state import StateMachine, GEARS_ALLOW_PAUSED_SILENT

State = custom.SelfdriveStateSP.ModularAssistiveDrivingSystem.ModularAssistiveDrivingSystemState
ButtonType = car.CarState.ButtonEvent.Type
EventName = log.OnroadEvent.EventName
SafetyModel = car.CarParams.SafetyModel

SET_SPEED_BUTTONS = (ButtonType.accelCruise, ButtonType.resumeCruise, ButtonType.decelCruise, ButtonType.setCruise)
IGNORED_SAFETY_MODES = (SafetyModel.silent, SafetyModel.noOutput)


class ModularAssistiveDrivingSystem:
  def __init__(self, selfdrive):
    self.mads_params = MadsParams()

    self.enabled = False
    self.active = False
    self.available = False
    self.mismatch_counter = 0
    self.allow_always = False
    self.selfdrive = selfdrive
    self.selfdrive.enabled_prev = False
    self.state_machine = StateMachine(self)
    self.events = self.selfdrive.events

    if self.selfdrive.CP.carName == "hyundai":
      if (self.selfdrive.CP.sunnypilotFlags & HyundaiFlagsSP.HAS_LFA_BUTTON) or \
            (self.selfdrive.CP.flags & HyundaiFlags.CANFD):
        self.allow_always = True

    # read params on init
    self.enabled_toggle = self.mads_params.read_param("Mads")
    self.main_enabled_toggle = self.mads_params.read_param("MadsMainCruiseAllowed")
    self.pause_lateral_on_brake_toggle = self.mads_params.read_param("MadsPauseLateralOnBrake")
    self.unified_engagement_mode = self.mads_params.read_param("MadsUnifiedEngagementMode")

  def read_params(self):
    self.main_enabled_toggle = self.mads_params.read_param("MadsMainCruiseAllowed")
    self.unified_engagement_mode = self.mads_params.read_param("MadsUnifiedEngagementMode")

  def update_controls_mismatch(self, sm: messaging.SubMaster):
    heartbeat_engaged = self.active if self.pause_lateral_on_brake_toggle else self.enabled

    if not heartbeat_engaged:
      self.mismatch_counter = 0

    if heartbeat_engaged and any(not ps.controlsAllowedLat for ps in sm['pandaStates']
                                 if ps.safetyModel not in IGNORED_SAFETY_MODES):
      self.mismatch_counter += 1

    if self.mismatch_counter >= 200:
      self.events.add(EventName.controlsMismatchLateral)

  def update_events(self, CS: car.CarState, sm: messaging.SubMaster):
    def update_unified_engagement_mode():
      uem_blocked = self.enabled or (self.selfdrive.enabled and self.selfdrive.enabled_prev)
      if (self.unified_engagement_mode and uem_blocked) or not self.unified_engagement_mode:
        self.events.remove(EventName.pcmEnable)
        self.events.remove(EventName.buttonEnable)

    def transition_paused_state():
      if self.state_machine.state != State.paused:
        self.events.add(EventName.silentLkasDisable)

    if not self.selfdrive.enabled and self.enabled:
      if self.events.has(EventName.doorOpen):
        self.events.replace(EventName.doorOpen, EventName.silentDoorOpen)
        transition_paused_state()
      if self.events.has(EventName.seatbeltNotLatched):
        self.events.replace(EventName.seatbeltNotLatched, EventName.silentSeatbeltNotLatched)
        transition_paused_state()
      if self.events.has(EventName.wrongGear):
        self.events.replace(EventName.wrongGear, EventName.silentWrongGear)
        transition_paused_state()
      if self.events.has(EventName.reverseGear):
        self.events.replace(EventName.reverseGear, EventName.silentReverseGear)
        transition_paused_state()
      if self.events.has(EventName.brakeHold):
        self.events.replace(EventName.brakeHold, EventName.silentBrakeHold)
        transition_paused_state()
      if self.events.has(EventName.parkBrake):
        self.events.replace(EventName.parkBrake, EventName.silentParkBrake)
        transition_paused_state()

      if self.pause_lateral_on_brake_toggle:
        if CS.brakePressed:
          transition_paused_state()

      if not (self.pause_lateral_on_brake_toggle and CS.brakePressed) and \
         not self.events.contains_in_list(GEARS_ALLOW_PAUSED_SILENT):
        if self.state_machine.state == State.paused:
          self.events.add(EventName.silentLkasEnable)

      self.events.remove(EventName.preEnableStandstill)

    if self.events.has(EventName.pcmEnable) or self.events.has(EventName.buttonEnable):
      update_unified_engagement_mode()
    else:
      if self.main_enabled_toggle:
        if CS.cruiseState.available and not self.selfdrive.CS_prev.cruiseState.available:
          self.events.add(EventName.lkasEnable)

    for be in CS.buttonEvents:
      if be.type == ButtonType.cancel:
        if not self.selfdrive.enabled and self.selfdrive.enabled_prev:
          self.events.add(EventName.manualLongitudinalRequired)
      if be.type == ButtonType.lkas and be.pressed and (CS.cruiseState.available or self.allow_always):
        if self.enabled:
          if self.selfdrive.enabled:
            self.events.add(EventName.manualSteeringRequired)
          else:
            self.events.add(EventName.lkasDisable)
        else:
          self.events.add(EventName.lkasEnable)

    if not CS.cruiseState.available:
      self.events.remove(EventName.buttonEnable)
      if self.selfdrive.CS_prev.cruiseState.available:
        self.events.add(EventName.lkasDisable)

    self.events.remove(EventName.pcmDisable)
    self.events.remove(EventName.buttonCancel)
    self.events.remove(EventName.pedalPressed)
    self.events.remove(EventName.wrongCruiseMode)
    if not any(be.type in SET_SPEED_BUTTONS for be in CS.buttonEvents):
      self.events.remove(EventName.wrongCarMode)

    self.update_controls_mismatch(sm)

  def update(self, CS: car.CarState, sm: messaging.SubMaster):
    if not self.enabled_toggle:
      return

    self.update_events(CS, sm)

    if not self.selfdrive.CP.passive and self.selfdrive.initialized:
      self.enabled, self.active = self.state_machine.update(self.events)

    # Copy of previous SelfdriveD states for MADS events handling
    self.selfdrive.enabled_prev = self.selfdrive.enabled
