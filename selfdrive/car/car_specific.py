import json
import os
import numpy as np
import tomllib
from abc import abstractmethod, ABC
from enum import StrEnum
from typing import Any, NamedTuple
from collections.abc import Callable
from functools import cache
from dataclasses import dataclass

from cereal import car
from openpilot.common.basedir import BASEDIR
from openpilot.common.simple_kalman import KF1D, get_kalman_gain
from openpilot.selfdrive.car import DT_CTRL, apply_hysteresis, gen_empty_fingerprint, scale_rot_inertia, scale_tire_stiffness, get_friction, STD_CARGO_KG
from openpilot.selfdrive.car.interfaces import MAX_CTRL_SPEED
from openpilot.selfdrive.car.can_definitions import CanData, CanRecvCallable, CanSendCallable
from openpilot.selfdrive.car.conversions import Conversions as CV
from openpilot.selfdrive.car.helpers import clip
from openpilot.selfdrive.car.values import PLATFORMS
from openpilot.selfdrive.controls.lib.events import Events

ButtonType = car.CarState.ButtonEvent.Type
GearShifter = car.CarState.GearShifter
EventName = car.CarEvent.EventName


class CarSpecificEvents:
  def __init__(self, CP: car.CarParams):
    self.CP = CP

    self.steering_unpressed = 0
    self.low_speed_alert = False
    self.no_steer_warning = False
    self.silent_steer_warning = True

  def update(self, CS, CS_prev):
    if self.CP.carName == 'chrysler':
      events = self.create_common_events(CS, CS_prev, extra_gears=[car.CarState.GearShifter.low])

      # Low speed steer alert hysteresis logic
      if self.CP.minSteerSpeed > 0. and CS.vEgo < (self.CP.minSteerSpeed + 0.5):
        self.low_speed_alert = True
      elif CS.vEgo > (self.CP.minSteerSpeed + 1.):
        self.low_speed_alert = False
      if self.low_speed_alert:
        events.add(car.CarEvent.EventName.belowSteerSpeed)
    else:
      raise ValueError(f"Unsupported car: {self.CP.carName}")

    return events

  def create_common_events(self, CS, CS_prev, extra_gears=None, pcm_enable=True, allow_enable=True,
                           enable_buttons=(ButtonType.accelCruise, ButtonType.decelCruise)):
    events = Events()

    if CS.doorOpen:
      events.add(EventName.doorOpen)
    if CS.seatbeltUnlatched:
      events.add(EventName.seatbeltNotLatched)
    if CS.gearShifter != GearShifter.drive and (extra_gears is None or
       CS.gearShifter not in extra_gears):
      events.add(EventName.wrongGear)
    if CS.gearShifter == GearShifter.reverse:
      events.add(EventName.reverseGear)
    if not CS.cruiseState.available:
      events.add(EventName.wrongCarMode)
    if CS.espDisabled:
      events.add(EventName.espDisabled)
    if CS.espActive:
      events.add(EventName.espActive)
    if CS.stockFcw:
      events.add(EventName.stockFcw)
    if CS.stockAeb:
      events.add(EventName.stockAeb)
    if CS.vEgo > MAX_CTRL_SPEED:
      events.add(EventName.speedTooHigh)
    if CS.cruiseState.nonAdaptive:
      events.add(EventName.wrongCruiseMode)
    if CS.brakeHoldActive and self.CP.openpilotLongitudinalControl:
      events.add(EventName.brakeHold)
    if CS.parkingBrake:
      events.add(EventName.parkBrake)
    if CS.accFaulted:
      events.add(EventName.accFaulted)
    if CS.steeringPressed:
      events.add(EventName.steerOverride)
    if CS.brakePressed and CS.standstill:
      events.add(EventName.preEnableStandstill)
    if CS.gasPressed:
      events.add(EventName.gasPressedOverride)
    if CS.vehicleSensorsInvalid:
      events.add(EventName.vehicleSensorsInvalid)

    # Handle button presses
    for b in CS.buttonEvents:
      # Enable OP long on falling edge of enable buttons (defaults to accelCruise and decelCruise, overridable per-port)
      if not self.CP.pcmCruise and (b.type in enable_buttons and not b.pressed):
        events.add(EventName.buttonEnable)
      # Disable on rising and falling edge of cancel for both stock and OP long
      if b.type == ButtonType.cancel:
        events.add(EventName.buttonCancel)

    # Handle permanent and temporary steering faults
    self.steering_unpressed = 0 if CS.steeringPressed else self.steering_unpressed + 1
    if CS.steerFaultTemporary:
      if CS.steeringPressed and (not CS_prev.steerFaultTemporary or self.no_steer_warning):
        self.no_steer_warning = True
      else:
        self.no_steer_warning = False

        # if the user overrode recently, show a less harsh alert
        if self.silent_steer_warning or CS.standstill or self.steering_unpressed < int(1.5 / DT_CTRL):
          self.silent_steer_warning = True
          events.add(EventName.steerTempUnavailableSilent)
        else:
          events.add(EventName.steerTempUnavailable)
    else:
      self.no_steer_warning = False
      self.silent_steer_warning = False
    if CS.steerFaultPermanent:
      events.add(EventName.steerUnavailable)

    # we engage when pcm is active (rising edge)
    # enabling can optionally be blocked by the car interface
    if pcm_enable:
      if CS.cruiseState.enabled and not CS_prev.cruiseState.enabled and allow_enable:
        events.add(EventName.pcmEnable)
      elif not CS.cruiseState.enabled:
        events.add(EventName.pcmDisable)

    return events
