#!/usr/bin/env python
import math
import time
import numpy as np
from cereal import car
from selfdrive.can.parser import CANParser
from selfdrive.car.gm.interface import CanBus
from selfdrive.car.gm.values import DBC, CAR
from common.realtime import sec_since_boot

RADAR_HEADER_MSG = 1120
SLOT_1_MSG = RADAR_HEADER_MSG + 1
NUM_SLOTS = 20

# Actually it's 0x47f, but can parser only reports
# messages that are present in DBC
LAST_RADAR_MSG = RADAR_HEADER_MSG + NUM_SLOTS

def create_radar_can_parser(canbus, car_fingerprint):

  dbc_f = DBC[car_fingerprint]['radar']
  if car_fingerprint in (CAR.VOLT, CAR.MALIBU, CAR.HOLDEN_ASTRA, CAR.ACADIA, CAR.CADILLAC_ATS):
    # C1A-ARS3-A by Continental
    radar_targets = range(SLOT_1_MSG, SLOT_1_MSG + NUM_SLOTS)
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

    checks = []

    return CANParser(dbc_f, signals, checks, canbus.obstacle)
  else:
    return None

class RadarInterface(object):
  def __init__(self, CP):
    # radar
    self.pts = {}

    self.delay = 0.0  # Delay of radar

    canbus = CanBus()
    print "Using %d as obstacle CAN bus ID" % canbus.obstacle
    self.rcp = create_radar_can_parser(canbus, CP.carFingerprint)

  def update(self):
    updated_messages = set()
    ret = car.RadarData.new_message()
    while 1:

      if self.rcp is None:
        time.sleep(0.05)   # nothing to do
        return ret

      tm = int(sec_since_boot() * 1e9)
      _, vls = self.rcp.update(tm, True)
      updated_messages.update(vls)
      if LAST_RADAR_MSG in updated_messages:
        break

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
    for ii in updated_messages:
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
        self.pts[targetId].dRel = distance # from front of car
        # From driver's pov, left is positive
        deg_to_rad = np.pi/180.
        self.pts[targetId].yRel = math.sin(deg_to_rad * cpt['TrkAzimuth']) * distance
        self.pts[targetId].vRel = cpt['TrkRangeRate']
        self.pts[targetId].aRel = float('nan')
        self.pts[targetId].yvRel = float('nan')

    for oldTarget in self.pts.keys():
      if not oldTarget in currentTargets:
        del self.pts[oldTarget]

    ret.points = self.pts.values()
    return ret

if __name__ == "__main__":
  RI = RadarInterface(None)
  while 1:
    ret = RI.update()
    print(chr(27) + "[2J")
    print(ret)
