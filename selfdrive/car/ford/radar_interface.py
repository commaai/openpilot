#!/usr/bin/env python3
from math import sin
from cereal import car
from opendbc.can.parser import CANParser
from selfdrive.car.ford.values import CANBUS, DBC
from selfdrive.car.interfaces import RadarInterfaceBase

RADAR_MSGS = list(range(0x120, 0x15F + 1))  # 64 points
LAST_MSG = max(RADAR_MSGS)
NUM_MSGS = len(RADAR_MSGS)


def _create_radar_can_parser(CP):
  if DBC[CP.carFingerprint]['radar'] is None:
    return None

  signals = []
  checks = []

  for ii in range(1, NUM_MSGS + 1):
    msg = f"MRR_Detection_{ii:03d}"
    signals += [
      (f"CAN_DET_VALID_LEVEL_{ii:02d}", msg),
      (f"CAN_DET_RANGE_{ii:02d}", msg),
      (f"CAN_DET_AZIMUTH_{ii:02d}", msg),
      (f"CAN_DET_RANGE_RATE_{ii:02d}", msg),
      (f"CAN_DET_AMPLITUDE_{ii:02d}", msg),
    ]
    checks += [(msg, 20)]

  return CANParser(DBC[CP.carFingerprint]['radar'], signals, checks, CANBUS.radar)

class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.updated_messages = set()
    self.trigger_msg = LAST_MSG
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

    for ii in range(1, NUM_MSGS + 1):
      msg = self.rcp.vl[f"MRR_Detection_{ii:03d}"]

      if ii not in self.pts:
        self.pts[ii] = car.RadarData.RadarPoint.new_message()
        self.pts[ii].trackId = self.track_id
        self.track_id += 1

      # radar point only valid if valid signal asserted
      valid = msg[f"CAN_DET_VALID_LEVEL_{ii:02d}"] > 0
      if valid:
        rel_distance = msg[f"CAN_DET_RANGE_{ii:02d}"]  # m
        azimuth = msg[f"CAN_DET_AZIMUTH_{ii:02d}"]  # rad

        self.pts[ii].dRel = rel_distance  # m from front of car
        self.pts[ii].yRel = sin(-azimuth) * rel_distance  # in car frame's y axis, left is positive
        self.pts[ii].vRel = msg[f"CAN_DET_RANGE_RATE_{ii:02d}"]  # m/s relative velocity

        # use aRel for debugging AMPLITUDE (reflection size)
        self.pts[ii].aRel = msg[f"CAN_DET_AMPLITUDE_{ii:02d}"]  # dBsm

        self.pts[ii].yvRel = float('nan')
        self.pts[ii].measured = True

      else:
        del self.pts[ii]

    ret.points = list(self.pts.values())
    return ret
