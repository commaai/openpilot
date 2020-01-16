#!/usr/bin/env python3
import os
import time
from opendbc.can.parser import CANParser
from cereal import car
from selfdrive.car.toyota.values import NO_DSU_CAR, DBC, TSS2_CAR
from selfdrive.car.interfaces import RadarInterfaceBase

def _create_radar_can_parser(car_fingerprint):
  dbc_f = DBC[car_fingerprint]['radar']

  if car_fingerprint in TSS2_CAR:
    RADAR_A_MSGS = list(range(0x180, 0x190))
    RADAR_B_MSGS = list(range(0x190, 0x1a0))
  else:
    RADAR_A_MSGS = list(range(0x210, 0x220))
    RADAR_B_MSGS = list(range(0x220, 0x230))

  msg_a_n = len(RADAR_A_MSGS)
  msg_b_n = len(RADAR_B_MSGS)

  signals = list(zip(['LONG_DIST'] * msg_a_n + ['NEW_TRACK'] * msg_a_n + ['LAT_DIST'] * msg_a_n +
                ['REL_SPEED'] * msg_a_n + ['VALID'] * msg_a_n + ['SCORE'] * msg_b_n,
                RADAR_A_MSGS * 5 + RADAR_B_MSGS,
                [255] * msg_a_n + [1] * msg_a_n + [0] * msg_a_n + [0] * msg_a_n + [0] * msg_a_n + [0] * msg_b_n))

  checks = list(zip(RADAR_A_MSGS + RADAR_B_MSGS, [20]*(msg_a_n + msg_b_n)))

  return CANParser(os.path.splitext(dbc_f)[0], signals, checks, 1)

class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP):
    # radar
    self.pts = {}
    self.track_id = 0

    self.delay = 0  # Delay of radar
    self.radar_ts = CP.radarTimeStep

    if CP.carFingerprint in TSS2_CAR:
      self.RADAR_A_MSGS = list(range(0x180, 0x190))
      self.RADAR_B_MSGS = list(range(0x190, 0x1a0))
    else:
      self.RADAR_A_MSGS = list(range(0x210, 0x220))
      self.RADAR_B_MSGS = list(range(0x220, 0x230))

    self.valid_cnt = {key: 0 for key in self.RADAR_A_MSGS}

    self.rcp = _create_radar_can_parser(CP.carFingerprint)
    self.trigger_msg = self.RADAR_B_MSGS[-1]
    self.updated_messages = set()

    # No radar dbc for cars without DSU which are not TSS 2.0
    # TODO: make a adas dbc file for dsu-less models
    self.no_radar = CP.carFingerprint in NO_DSU_CAR and CP.carFingerprint not in TSS2_CAR

  def update(self, can_strings):
    if self.no_radar:
      time.sleep(self.radar_ts)
      return car.RadarData.new_message()

    vls = self.rcp.update_strings(can_strings)
    self.updated_messages.update(vls)

    if self.trigger_msg not in self.updated_messages:
      return None

    rr =  self._update(self.updated_messages)
    self.updated_messages.clear()

    return rr

  def _update(self, updated_messages):
    ret = car.RadarData.new_message()
    errors = []
    if not self.rcp.can_valid:
      errors.append("canError")
    ret.errors = errors

    for ii in sorted(updated_messages):
      if ii in self.RADAR_A_MSGS:
        cpt = self.rcp.vl[ii]

        if cpt['LONG_DIST'] >=255 or cpt['NEW_TRACK']:
          self.valid_cnt[ii] = 0    # reset counter
        if cpt['VALID'] and cpt['LONG_DIST'] < 255:
          self.valid_cnt[ii] += 1
        else:
          self.valid_cnt[ii] = max(self.valid_cnt[ii] -1, 0)

        score = self.rcp.vl[ii+16]['SCORE']
        # print ii, self.valid_cnt[ii], score, cpt['VALID'], cpt['LONG_DIST'], cpt['LAT_DIST']

        # radar point only valid if it's a valid measurement and score is above 50
        if cpt['VALID'] or (score > 50 and cpt['LONG_DIST'] < 255 and self.valid_cnt[ii] > 0):
          if ii not in self.pts or cpt['NEW_TRACK']:
            self.pts[ii] = car.RadarData.RadarPoint.new_message()
            self.pts[ii].trackId = self.track_id
            self.track_id += 1
          self.pts[ii].dRel = cpt['LONG_DIST']  # from front of car
          self.pts[ii].yRel = -cpt['LAT_DIST']  # in car frame's y axis, left is positive
          self.pts[ii].vRel = cpt['REL_SPEED']
          self.pts[ii].aRel = float('nan')
          self.pts[ii].yvRel = float('nan')
          self.pts[ii].measured = bool(cpt['VALID'])
        else:
          if ii in self.pts:
            del self.pts[ii]

    ret.points = list(self.pts.values())
    return ret
