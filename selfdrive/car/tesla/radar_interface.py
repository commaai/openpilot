#!/usr/bin/env python3
from cereal import car
from opendbc.can.parser import CANParser
from openpilot.selfdrive.car.tesla.values import DBC, CANBUS
from openpilot.selfdrive.car.interfaces import RadarInterfaceBase

RADAR_MSGS_A = list(range(0x310, 0x36E, 3))
RADAR_MSGS_B = list(range(0x311, 0x36F, 3))
NUM_POINTS = len(RADAR_MSGS_A)

def get_radar_can_parser(CP):
  # Status messages
  messages = [
    ('TeslaRadarSguInfo', 10),
  ]

  # Radar tracks. There are also raw point clouds available,
  # we don't use those.
  for i in range(NUM_POINTS):
    msg_id_a = RADAR_MSGS_A[i]
    msg_id_b = RADAR_MSGS_B[i]
    messages.extend([
      (msg_id_a, 8),
      (msg_id_b, 8),
    ])

  return CANParser(DBC[CP.carFingerprint]['radar'], messages, CANBUS.radar)

class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.rcp = get_radar_can_parser(CP)
    self.updated_messages = set()
    self.track_id = 0
    self.trigger_msg = RADAR_MSGS_B[-1]

  def update(self, can_strings):
    if self.rcp is None:
      return super().update(None)

    values = self.rcp.update_strings(can_strings)
    self.updated_messages.update(values)

    if self.trigger_msg not in self.updated_messages:
      return None

    ret = car.RadarData.new_message()

    # Errors
    errors = []
    sgu_info = self.rcp.vl['TeslaRadarSguInfo']
    if not self.rcp.can_valid:
      errors.append('canError')
    if sgu_info['RADC_HWFail'] or sgu_info['RADC_SGUFail'] or sgu_info['RADC_SensorDirty']:
      errors.append('fault')
    ret.errors = errors

    # Radar tracks
    for i in range(NUM_POINTS):
      msg_a = self.rcp.vl[RADAR_MSGS_A[i]]
      msg_b = self.rcp.vl[RADAR_MSGS_B[i]]

      # Make sure msg A and B are together
      if msg_a['Index'] != msg_b['Index2']:
        continue

      # Check if it's a valid track
      if not msg_a['Tracked']:
        if i in self.pts:
          del self.pts[i]
        continue

      # New track!
      if i not in self.pts:
        self.pts[i] = car.RadarData.RadarPoint.new_message()
        self.pts[i].trackId = self.track_id
        self.track_id += 1

      # Parse track data
      self.pts[i].dRel = msg_a['LongDist']
      self.pts[i].yRel = msg_a['LatDist']
      self.pts[i].vRel = msg_a['LongSpeed']
      self.pts[i].aRel = msg_a['LongAccel']
      self.pts[i].yvRel = msg_b['LatSpeed']
      self.pts[i].measured = bool(msg_a['Meas'])

    ret.points = list(self.pts.values())
    self.updated_messages.clear()
    return ret
