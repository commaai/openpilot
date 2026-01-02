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
    self.updated_messages = set()
    self.trigger_msg = RADAR_START_ADDR + RADAR_MSG_COUNT - 1
    self.track_id = 0

    self.radar_off_can = CP.radarUnavailable
    self.rcp = get_radar_can_parser(CP)

  def update(self, can_strings):
    if self.radar_off_can or (self.rcp is None):
      return super().update(None)

    vls = self.rcp.update(can_strings)
    self.updated_messages.update(vls)

    if self.trigger_msg not in self.updated_messages:
      return None

    rr = self._update(self.updated_messages)
    self.updated_messages.clear()

    return rr

  def _update(self, updated_messages):
    ret = structs.RadarData()
    if self.rcp is None:
      return ret

    if not self.rcp.can_valid:
      ret.errors.canError = True

    for addr in range(RADAR_START_ADDR, RADAR_START_ADDR + RADAR_MSG_COUNT):
      msg = self.rcp.vl[f"RADAR_TRACK_{addr:x}"]

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

        self.pts[addr].measured = msg['STATE'] in (2, 3)
        azimuth = math.radians(msg['AZIMUTH'])
        self.pts[addr].dRel = math.cos(azimuth) * msg['LONG_DIST']
        self.pts[addr].yRel = 0.5 * -math.sin(azimuth) * msg['LONG_DIST']
        self.pts[addr].vRel = msg['REL_SPEED']
        self.pts[addr].aRel = float('nan')
        self.pts[addr].yvRel = float('nan')
      elif addr in self.pts:
        del self.pts[addr]

    ret.points = list(self.pts.values())
    return ret
