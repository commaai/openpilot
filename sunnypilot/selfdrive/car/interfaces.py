"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from opendbc.car import Bus, structs
from opendbc.car.can_definitions import CanRecvCallable, CanSendCallable
from opendbc.car.car_helpers import can_fingerprint
from opendbc.car.hyundai.radar_interface import RADAR_START_ADDR
from opendbc.car.hyundai.values import HyundaiFlags, DBC as HYUNDAI_DBC
from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP

import openpilot.system.sentry as sentry


def log_fingerprint(CP: structs.CarParams) -> None:
  if CP.carFingerprint == "MOCK":
    sentry.capture_fingerprint_mock()
  else:
    sentry.capture_fingerprint(CP.carFingerprint, CP.carName)


def setup_car_interface_sp(CP: structs.CarParams, params):
  if CP.carName == 'hyundai':
    if CP.flags & HyundaiFlags.MANDO_RADAR and CP.radarUnavailable:
      # Having this automatic without a toggle causes a weird process replay diff because
      # somehow it sees fewer logs than intended
      if params.get_bool("HyundaiRadarTracksToggle"):
        CP.sunnypilotFlags |= HyundaiFlagsSP.ENABLE_RADAR_TRACKS.value
        if params.get_bool("HyundaiRadarTracks"):
          CP.radarUnavailable = False


def initialize_car_interface_sp(CP: structs.CarParams, params, can_recv: CanRecvCallable, can_send: CanSendCallable):
  if CP.carName == 'hyundai':
    if CP.sunnypilotFlags & HyundaiFlagsSP.ENABLE_RADAR_TRACKS:
      can_recv()
      _, fingerprint = can_fingerprint(can_recv)
      radar_unavailable = RADAR_START_ADDR not in fingerprint[1] or Bus.radar not in HYUNDAI_DBC[CP.carFingerprint]

      radar_tracks = params.get_bool("HyundaiRadarTracks")
      radar_tracks_persistent = params.get_bool("HyundaiRadarTracksPersistent")

      params.put_bool_nonblocking("HyundaiRadarTracksConfirmed", radar_tracks)

      if not radar_tracks_persistent:
        params.put_bool_nonblocking("HyundaiRadarTracks", not radar_unavailable)
        params.put_bool_nonblocking("HyundaiRadarTracksPersistent", True)
