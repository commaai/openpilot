#!/usr/bin/env python3
from collections import defaultdict

from cereal import car, messaging
from opendbc.can.parser import CANParser
from selfdrive.car.interfaces import RadarInterfaceBase
from selfdrive.car.hyundai.values import DBC

import math

RADAR_START_ADDR = 0x500
RADAR_MSG_COUNT = 32

BLINDSPOT_BUS = 9

BLINDSPOT_ADDRS = []
for i in range(4):
  BLINDSPOT_ADDRS += list(range(0x300 + i * 0x100, 0x308 + i * 0x100))

# 0400(1024)( 23500) a3805e0511624c8069810a06a16050807b809e0a1c614c80977f4a0ccca6dc807f82f20cea555c807d7f4a0f69a84c806f7ed60f7381e080677faa110f684880 
# 0401(1025)( 23500) 737f0611b5680080777fe2192b6428807d7e42193d6b6880737e0618e66cd4806b7e7e19cba84c809177b62a0b7be880917cda2bf7a6bc808d779e2fa368fc80 
# 0402(1026)( 23500) 8177623707a74c807f775e2e33a7dc8083779e19cbb240806d788e09bd8dd48087777221a39e308085778215d49aa480717a26070d822c80777d1e0599723c80 
# 0403(1027)( 23500) 6d798207d986708077828e058558e8807177da0e7a95148079786e0af18ef8809578160d389288806d7a5a077580c0808b775e51cfa19c807178d20a3d8b2080 
# 0404(1028)( 23500) 6b793a0965886c80697ac607d17dc48087775a27d99dc4807d77821cc9adfc809577b21749b08c808b77ce12e395848093780611c8b5a8807d777624e79c5880 
# 0405(1029)( 23500) 65786e0ec18f40807f77662f43ac4880ad77564b2d9f9c8093777a45539f30807f777a40efab4880877b460d8d7b34808183420cbd5580807377e2228795a880 
# 0406(1030)( 23500) 917be60e0177c8808577ee251fb3f4807d83e2111d52cc807b808e111d6204807583c213e952cc807778062bb592d08080800200008008808080020000800880 
# 0407(1031)( 23500) 80800200008008808080020000800880808002000080088080800200008008808080020000800880808002000080088080800200008008808080020000800880 

def get_radar_can_parser(CP):
  if DBC[CP.carFingerprint]['radar'] is None:
    return None

  signals = []
  checks = []

  for addr in range(RADAR_START_ADDR, RADAR_START_ADDR + RADAR_MSG_COUNT):
    msg = f"RADAR_TRACK_{addr:x}"
    signals += [
      ("STATE", msg),
      ("AZIMUTH", msg),
      ("LONG_DIST", msg),
      ("REL_ACCEL", msg),
      ("REL_SPEED", msg),
    ]
    checks += [(msg, 50)]
  return CANParser(DBC[CP.carFingerprint]['radar'], signals, checks, 1)


class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.updated_messages = set()
    self.track_id = 0

    self.radar_off_can = False
    self.actives = defaultdict(int)

  def update(self, can_strings):
    for s in can_strings:
      can = messaging.log_from_bytes(s)
      for c in can.can:
        if c.src != BLINDSPOT_BUS:
          continue

        if c.address in BLINDSPOT_ADDRS:
          radar = (c.address // 0x100) - 3

          for i in range(8):
            ii = (c.address, i)

            point_dat = c.dat[i * 8: (i+1) * 8]
            active = not point_dat.startswith(b'\x80\x80')

            if active:
              angle = ((point_dat[5] << 8) + point_dat[6]) / 256
              d = ((point_dat[3] << 8) + point_dat[4]) / 256

              if radar in [0, 1]:
                angle = 165 - angle
              else:
                angle = 175 - angle

              if ii not in self.pts:
                self.pts[ii] = car.RadarData.RadarPoint.new_message()
                self.pts[ii].trackId = self.track_id
                self.track_id += 1

              lr = 1 if radar in [0, 2] else -1
              fr = 1 if radar in [0, 1] else -1
              self.pts[ii].dRel = fr * math.cos(math.radians(angle)) * d
              self.pts[ii].yRel = lr * math.sin(math.radians(angle)) * d
              self.pts[ii].measured = True

            else:
              if ii in self.pts:
                del self.pts[ii]


    # if self.trigger_msg not in self.updated_messages:
    #   return None
    ret = car.RadarData.new_message()
    ret.points = list(self.pts.values())
    # for p in self.pts.values():
    #   print(p)
    # rr = self._update(self.updated_messages)

    return ret
