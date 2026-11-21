import math

from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.interfaces import RadarInterfaceBase
from opendbc.car.rivian.values import DBC

RADAR_START_ADDR = 0x500
RADAR_MSG_COUNT = 32


def get_radar_can_parser(CP):
  messages = [(f"RADAR_TRACK_{addr:x}", 20) for addr in range(RADAR_START_ADDR, RADAR_START_ADDR + RADAR_MSG_COUNT)]
  return CANParser(DBC[CP.carFingerprint][Bus.radar], messages, 1)


class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.rcp = None if CP.radarUnavailable else get_radar_can_parser(CP)
    self.trigger_msg = RADAR_START_ADDR + RADAR_MSG_COUNT - 1

  def _update(self, rcp, updated_messages):
    ret = structs.RadarData()
    if not rcp.can_valid:
      ret.errors.canError = True

    for addr in range(RADAR_START_ADDR, RADAR_START_ADDR + RADAR_MSG_COUNT):
      msg = rcp.vl[f"RADAR_TRACK_{addr:x}"]

      # STATE: 1=New, 2=New_updated, 3=Updated, 4=Coasting, 7=New_coasting
      valid = msg['STATE'] in (1, 2, 3, 4, 7)

      # Rivian's Short Range Radar (SSR) detects close stationary objects like guardrails, which cause phantom braking.
      # MODE: 1=SRR, 2=LRR, 3=SRR_and_LRR
      valid = valid and msg['MODE'] in (2, 3)

      if valid:
        if addr not in self.pts or msg['STATE'] in (1, 2, 7):
          self.pts[addr] = structs.RadarData.RadarPoint()
          self.pts[addr].trackId = self.track_id
          self.track_id += 1

        azimuth = math.radians(msg['AZIMUTH'])
        self.pts[addr].dRel = math.cos(azimuth) * msg['LONG_DIST']
        self.pts[addr].yRel = -math.sin(azimuth) * msg['LONG_DIST']
        self.pts[addr].vRel = msg['REL_SPEED']
      elif addr in self.pts:
        del self.pts[addr]

    ret.points = list(self.pts.values())
    return ret
