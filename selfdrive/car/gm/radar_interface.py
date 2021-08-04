#!/usr/bin/env python3
from __future__ import print_function
import math
from cereal import car
from opendbc.can.parser import CANParser
from selfdrive.car.gm.values import DBC, CAR, CanBus
from selfdrive.config import Conversions as CV
from selfdrive.car.interfaces import RadarInterfaceBase

RADAR_HEADER_MSG = 1120
SLOT_1_MSG = RADAR_HEADER_MSG + 1
NUM_SLOTS = 20

# Actually it's 0x47f, but can parser only reports
# messages that are present in DBC
LAST_RADAR_MSG = RADAR_HEADER_MSG + NUM_SLOTS

def create_radar_can_parser(car_fingerprint):
  if car_fingerprint not in (CAR.VOLT, CAR.MALIBU, CAR.HOLDEN_ASTRA, CAR.ACADIA, CAR.CADILLAC_ATS):
    return None

  # C1A-ARS3-A by Continental
  radar_targets = list(range(SLOT_1_MSG, SLOT_1_MSG + NUM_SLOTS))
  signals = list(zip(['FLRRNumValidTargets',
                 'FLRRSnsrBlckd', 'FLRRYawRtPlsblityFlt',
                 'FLRRHWFltPrsntInt', 'FLRRAntTngFltPrsnt',
                 'FLRRAlgnFltPrsnt', 'FLRRSnstvFltPrsntInt'] +
                ['TrkRange'] * NUM_SLOTS + ['TrkRangeRate'] * NUM_SLOTS +
                ['TrkRangeAccel'] * NUM_SLOTS + ['TrkAzimuth'] * NUM_SLOTS +
                ['TrkWidth'] * NUM_SLOTS + ['TrkObjectID'] * NUM_SLOTS,
                [RADAR_HEADER_MSG] * 7 + radar_targets * 6,
                [0] * 7 +
                [0.0] * NUM_SLOTS + [0.0] * NUM_SLOTS +
                [0.0] * NUM_SLOTS + [0.0] * NUM_SLOTS +
                [0.0] * NUM_SLOTS + [0] * NUM_SLOTS))

  checks = list({(s[1], 14) for s in signals})

  return CANParser(DBC[car_fingerprint]['radar'], signals, checks, CanBus.OBSTACLE)

class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP):
    super().__init__(CP)

    self.rcp = create_radar_can_parser(CP.carFingerprint)

    self.trigger_msg = LAST_RADAR_MSG
    self.updated_messages = set()
    self.radar_ts = CP.radarTimeStep

  def update(self, can_strings):
    if self.rcp is None:
      return super().update(None)

    vls = self.rcp.update_strings(can_strings)
    self.updated_messages.update(vls)

    if self.trigger_msg not in self.updated_messages:
      return None

    ret = car.RadarData.new_message()
    header = self.rcp.vl[RADAR_HEADER_MSG]
    fault = header['FLRRSnsrBlckd'] or header['FLRRSnstvFltPrsntInt'] or \
      header['FLRRYawRtPlsblityFlt'] or header['FLRRHWFltPrsntInt'] or \
      header['FLRRAntTngFltPrsnt'] or header['FLRRAlgnFltPrsnt']
    errors = []
    if not self.rcp.can_valid:
      errors.append("canError")
    if fault:
      errors.append("fault")
    ret.errors = errors

    currentTargets = set()
    num_targets = header['FLRRNumValidTargets']

    # Not all radar messages describe targets,
    # no need to monitor all of the self.rcp.msgs_upd
    for ii in self.updated_messages:
      if ii == RADAR_HEADER_MSG:
        continue

      if num_targets == 0:
        break

      cpt = self.rcp.vl[ii]
      # Zero distance means it's an empty target slot
      if cpt['TrkRange'] > 0.0:
        targetId = cpt['TrkObjectID']
        currentTargets.add(targetId)
        if targetId not in self.pts:
          self.pts[targetId] = car.RadarData.RadarPoint.new_message()
          self.pts[targetId].trackId = targetId
        distance = cpt['TrkRange']
        self.pts[targetId].dRel = distance  # from front of car
        # From driver's pov, left is positive
        self.pts[targetId].yRel = math.sin(cpt['TrkAzimuth'] * CV.DEG_TO_RAD) * distance
        self.pts[targetId].vRel = cpt['TrkRangeRate']
        self.pts[targetId].aRel = float('nan')
        self.pts[targetId].yvRel = float('nan')

    for oldTarget in list(self.pts.keys()):
      if oldTarget not in currentTargets:
        del self.pts[oldTarget]

    ret.points = list(self.pts.values())
    self.updated_messages.clear()
    return ret
