#!/usr/bin/env python3
import math

from cereal import car
from opendbc.can.parser import CANParser
from selfdrive.car.interfaces import RadarInterfaceBase
from selfdrive.car.hyundai.values import DBC


def get_radar_can_parser(CP):
  signals = []
  checks = []

  for addr in range(0x500, 0x500 + 32):
    msg = f"R_{hex(addr)}"
    signals += [
      ("NEW_SIGNAL_2", msg, 0),
      ("NEW_SIGNAL_3", msg, 0),
      ("NEW_SIGNAL_4", msg, 0),
      ("NEW_SIGNAL_5", msg, 0),
      ("NEW_SIGNAL_9", msg, 0),
    ]
    checks += [(msg, 50)]
  return CANParser(DBC[CP.carFingerprint]['radar'], signals, checks, 1)


class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.rcp = get_radar_can_parser(CP)
    self.updated_messages = set()
    self.trigger_msg = 0x500 + 31
    self.track_id = 0

     # TODO: set based of fingerprint
    self.radar_off_can = False 

  def update(self, can_strings):
    if self.radar_off_can:
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
    cpt = self.rcp.vl

    errors = []
    if not self.rcp.can_valid:
      errors.append("canError")
    ret.errors = errors

    for addr in range(0x500, 0x500 + 32):
      msg = f"R_{hex(addr)}"

      if addr not in self.pts:
        self.pts[addr] = car.RadarData.RadarPoint.new_message()
        self.pts[addr].trackId = self.track_id
        self.track_id += 1

      valid = cpt[msg]['NEW_SIGNAL_3'] in [3, 3]
      if valid:
        self.pts[addr].measured = True
        self.pts[addr].dRel = cpt[msg]['NEW_SIGNAL_4']
        self.pts[addr].yRel = 0.5 * -math.sin(math.radians(cpt[msg]['NEW_SIGNAL_2'])) * cpt[msg]['NEW_SIGNAL_4']
        self.pts[addr].vRel = cpt[msg]['NEW_SIGNAL_9'] / 3.0

        self.pts[addr].aRel = float('nan')
        self.pts[addr].yvRel = float('nan')

        # sign?
        # self.pts[addr].aRel = -cpt[msg]['NEW_SIGNAL_5'] 
      else:
        del self.pts[addr]

    ret.points = list(self.pts.values())
    return ret
