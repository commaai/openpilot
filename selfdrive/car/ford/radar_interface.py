#!/usr/bin/env python3
from math import cos, sin
from cereal import car
from opendbc.can.parser import CANParser
from selfdrive.car.ford.values import CANBUS, DBC
from selfdrive.car.interfaces import RadarInterfaceBase

RADAR_START_ADDR = 0x120
RADAR_MSG_COUNT = 64


def _create_radar_can_parser(CP):
  if DBC[CP.carFingerprint]['radar'] is None:
    return None

  signals = []
  checks = []

  for i in range(1, RADAR_MSG_COUNT + 1):
    msg = f"MRR_Detection_{i:03d}"
    signals += [
      (f"CAN_DET_VALID_LEVEL_{i:02d}", msg),
      (f"CAN_DET_AZIMUTH_{i:02d}", msg),
      (f"CAN_DET_RANGE_{i:02d}", msg),
      (f"CAN_DET_RANGE_RATE_{i:02d}", msg),
      (f"CAN_DET_AMPLITUDE_{i:02d}", msg),
      (f"CAN_SCAN_INDEX_2LSB_{i:02d}", msg),
    ]
    checks += [(msg, 20)]

  return CANParser(DBC[CP.carFingerprint]['radar'], signals, checks, CANBUS.radar)

class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.updated_messages = set()
    self.trigger_msg = RADAR_START_ADDR + RADAR_MSG_COUNT - 1
    self.track_id = 0

    self.rcp = _create_radar_can_parser(CP)

  def update(self, can_strings):
    if self.rcp is None:
      return super().update(None)

    vls = self.rcp.update_strings(can_strings)
    self.updated_messages.update(vls)

    if self.trigger_msg not in self.updated_messages:
      return None

    rr = self._update(self.updated_messages)
    self.updated_messages.clear()

    return rr

  def _update(self, updated_messages):
    ret = car.RadarData.new_message()
    if self.rcp is None:
      return ret

    errors = []

    if not self.rcp.can_valid:
      errors.append("canError")
    ret.errors = errors

    for ii in range(1, RADAR_MSG_COUNT + 1):
      msg = self.rcp.vl[f"MRR_Detection_{ii:03d}"]

      # SCAN_INDEX rotates through 0..3 on each message
      # treat these as separate points
      scanIndex = msg[f"CAN_SCAN_INDEX_2LSB_{ii:02d}"]
      i = (ii - 1) * 4 + scanIndex

      if i not in self.pts:
        self.pts[i] = car.RadarData.RadarPoint.new_message()
        self.pts[i].trackId = self.track_id
        self.pts[i].aRel = float('nan')
        self.pts[i].yvRel = float('nan')
        self.track_id += 1

      valid = bool(msg[f"CAN_DET_VALID_LEVEL_{ii:02d}"])
      amplitude = msg[f"CAN_DET_AMPLITUDE_{ii:02d}"]            # dBsm [-64|63]

      if valid and 0 < amplitude <= 15:
        azimuth = msg[f"CAN_DET_AZIMUTH_{ii:02d}"]              # rad [-3.1416|3.13964]
        dist = msg[f"CAN_DET_RANGE_{ii:02d}"]                  # m [0|255.984]
        distRate = msg[f"CAN_DET_RANGE_RATE_{ii:02d}"]         # m/s [-128|127.984]

        # *** openpilot radar point ***
        self.pts[i].dRel = cos(azimuth) * dist                  # m from front of car
        self.pts[i].yRel = -sin(azimuth) * dist                 # in car frame's y axis, left is positive
        self.pts[i].vRel = distRate                             # m/s

        self.pts[i].measured = True

      else:
        del self.pts[i]

    ret.points = list(self.pts.values())
    return ret
