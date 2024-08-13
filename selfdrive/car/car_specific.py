import copy
import json
import os
import numpy as np
import tomllib
from abc import abstractmethod, ABC
from enum import StrEnum
from typing import Any, NamedTuple
from collections.abc import Callable
from functools import cache

from cereal import car
from openpilot.common.basedir import BASEDIR
from openpilot.common.simple_kalman import KF1D, get_kalman_gain
from openpilot.selfdrive.car import DT_CTRL, apply_hysteresis, gen_empty_fingerprint, scale_rot_inertia, scale_tire_stiffness, get_friction, STD_CARGO_KG
from openpilot.selfdrive.car import structs
from openpilot.selfdrive.car.interfaces import MAX_CTRL_SPEED
from openpilot.selfdrive.car.can_definitions import CanData, CanRecvCallable, CanSendCallable
from openpilot.selfdrive.car.conversions import Conversions as CV
from openpilot.selfdrive.car.helpers import clip
from openpilot.selfdrive.car.values import PLATFORMS
from openpilot.selfdrive.controls.lib.events import Events

ButtonType = structs.CarState.ButtonEvent.Type
GearShifter = structs.CarState.GearShifter
EventName = car.CarEvent.EventName


def create_common_events(CP: structs.CarParams, cs_out, extra_gears=None, pcm_enable=True, allow_enable=True,
                         enable_buttons=(ButtonType.accelCruise, ButtonType.decelCruise)):
  events = Events()

  if cs_out.doorOpen:
    events.add(EventName.doorOpen)
  if cs_out.seatbeltUnlatched:
    events.add(EventName.seatbeltNotLatched)
  if cs_out.gearShifter != GearShifter.drive and (extra_gears is None or
     cs_out.gearShifter not in extra_gears):
    events.add(EventName.wrongGear)
  if cs_out.gearShifter == GearShifter.reverse:
    events.add(EventName.reverseGear)
  if not cs_out.cruiseState.available:
    events.add(EventName.wrongCarMode)
  if cs_out.espDisabled:
    events.add(EventName.espDisabled)
  if cs_out.espActive:
    events.add(EventName.espActive)
  if cs_out.stockFcw:
    events.add(EventName.stockFcw)
  if cs_out.stockAeb:
    events.add(EventName.stockAeb)
  if cs_out.vEgo > MAX_CTRL_SPEED:
    events.add(EventName.speedTooHigh)
  if cs_out.cruiseState.nonAdaptive:
    events.add(EventName.wrongCruiseMode)
  if cs_out.brakeHoldActive and CP.openpilotLongitudinalControl:
    events.add(EventName.brakeHold)
  if cs_out.parkingBrake:
    events.add(EventName.parkBrake)
  if cs_out.accFaulted:
    events.add(EventName.accFaulted)
  if cs_out.steeringPressed:
    events.add(EventName.steerOverride)
  if cs_out.brakePressed and cs_out.standstill:
    events.add(EventName.preEnableStandstill)
  if cs_out.gasPressed:
    events.add(EventName.gasPressedOverride)
  if cs_out.vehicleSensorsInvalid:
    events.add(EventName.vehicleSensorsInvalid)

  # Handle button presses
  for b in cs_out.buttonEvents:
    # Enable OP long on falling edge of enable buttons (defaults to accelCruise and decelCruise, overridable per-port)
    if not CP.pcmCruise and (b.type in enable_buttons and not b.pressed):
      events.add(EventName.buttonEnable)
    # Disable on rising and falling edge of cancel for both stock and OP long
    if b.type == ButtonType.cancel:
      events.add(EventName.buttonCancel)

  # Handle permanent and temporary steering faults
  self.steering_unpressed = 0 if cs_out.steeringPressed else self.steering_unpressed + 1
  if cs_out.steerFaultTemporary:
    if cs_out.steeringPressed and (not self.CS.out.steerFaultTemporary or self.no_steer_warning):
      self.no_steer_warning = True
    else:
      self.no_steer_warning = False

      # if the user overrode recently, show a less harsh alert
      if self.silent_steer_warning or cs_out.standstill or self.steering_unpressed < int(1.5 / DT_CTRL):
        self.silent_steer_warning = True
        events.add(EventName.steerTempUnavailableSilent)
      else:
        events.add(EventName.steerTempUnavailable)
  else:
    self.no_steer_warning = False
    self.silent_steer_warning = False
  if cs_out.steerFaultPermanent:
    events.add(EventName.steerUnavailable)

  # we engage when pcm is active (rising edge)
  # enabling can optionally be blocked by the car interface
  if pcm_enable:
    if cs_out.cruiseState.enabled and not self.CS.out.cruiseState.enabled and allow_enable:
      events.add(EventName.pcmEnable)
    elif not cs_out.cruiseState.enabled:
      events.add(EventName.pcmDisable)

  return events


def create_car_events(CP: structs.CarParams, cs_out: car.CarState):
  events = create_common_events(CP, cs_out)
