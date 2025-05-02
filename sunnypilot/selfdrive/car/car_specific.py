"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from cereal import log, custom
from opendbc.car import structs

from opendbc.car.chrysler.values import RAM_DT
from openpilot.selfdrive.selfdrived.events import Events
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP

EventName = log.OnroadEvent.EventName
EventNameSP = custom.OnroadEventSP.EventName
GearShifter = structs.CarState.GearShifter


class CarSpecificEventsSP:
  def __init__(self, CP: structs.CarParams, params):
    self.CP = CP
    self.params = params

    self.low_speed_alert = False
    self.hyundai_radar_tracks = self.params.get_bool("HyundaiRadarTracks")
    self.hyundai_radar_tracks_confirmed = self.params.get_bool("HyundaiRadarTracksConfirmed")

  def read_params(self):
    self.hyundai_radar_tracks = self.params.get_bool("HyundaiRadarTracks")
    self.hyundai_radar_tracks_confirmed = self.params.get_bool("HyundaiRadarTracksConfirmed")

  def update(self, CS: structs.CarState, events: Events):
    events_sp = EventsSP()

    if self.CP.brand == 'chrysler':
      if self.CP.carFingerprint in RAM_DT:
        # remove belowSteerSpeed event from CarSpecificEvents as RAM_DT uses a different logic
        if events.has(EventName.belowSteerSpeed):
          events.remove(EventName.belowSteerSpeed)

        # TODO-SP: use if/elif to have the gear shifter condition takes precedence over the speed condition
        # TODO-SP: add 1 m/s hysteresis
        if CS.vEgo >= self.CP.minEnableSpeed:
          self.low_speed_alert = False
        if CS.gearShifter != GearShifter.drive:
          self.low_speed_alert = True
      if self.low_speed_alert:
        events.add(EventName.belowSteerSpeed)

    if self.CP.brand == 'hyundai':
      if self.hyundai_radar_tracks and not self.hyundai_radar_tracks_confirmed:
        events_sp.add(EventNameSP.hyundaiRadarTracksConfirmed)

    return events_sp
