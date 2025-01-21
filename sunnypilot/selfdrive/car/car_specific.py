"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from cereal import log
from opendbc.car import structs

from openpilot.selfdrive.selfdrived.events import Events

EventName = log.OnroadEvent.EventName


class CarSpecificEventsSP:
  def __init__(self, CP: structs.CarParams, params):
    self.CP = CP
    self.params = params

    self.hyundai_radar_tracks = self.params.get_bool("HyundaiRadarTracks")
    self.hyundai_radar_tracks_confirmed = self.params.get_bool("HyundaiRadarTracksConfirmed")

  def read_params(self):
    self.hyundai_radar_tracks = self.params.get_bool("HyundaiRadarTracks")
    self.hyundai_radar_tracks_confirmed = self.params.get_bool("HyundaiRadarTracksConfirmed")

  def update(self):
    events = Events()
    if self.CP.carName == 'hyundai':
      if self.hyundai_radar_tracks and not self.hyundai_radar_tracks_confirmed:
        events.add(EventName.hyundaiRadarTracksConfirmed)

    return events
