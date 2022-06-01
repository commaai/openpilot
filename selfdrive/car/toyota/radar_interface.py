#!/usr/bin/env python3
from opendbc.can.parser import CANParser
from cereal import car
from selfdrive.car.toyota.values import RADAR_ACC_CAR_TSS1, DBC, TSS2_CAR
from selfdrive.car.interfaces import RadarInterfaceBase

def _create_radar_acc_tss1_can_parser(car_fingerprint):
  if DBC[car_fingerprint]['radar'] is None:
    return None

  # object 0 to 11
  RADAR_A_MSGS = list(range(0x301, 0x318, 2))
  msg_n = len(RADAR_A_MSGS)
  signals = list(zip(
        ['ID'] * msg_n + ['LONG_DIST'] * msg_n + ['LAT_DIST'] * msg_n + ['SPEED'] * msg_n +
        ['LAT_SPEED'] * msg_n,
        RADAR_A_MSGS * 5))

  checks = list(zip(RADAR_A_MSGS, [15] * msg_n))
  return CANParser(DBC[car_fingerprint]['radar'], signals, checks, 1)

def _create_radar_can_parser(car_fingerprint):
  if DBC[car_fingerprint]['radar'] is None:
    return None

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
                     RADAR_A_MSGS * 5 + RADAR_B_MSGS))

  checks = list(zip(RADAR_A_MSGS + RADAR_B_MSGS, [20] * (msg_a_n + msg_b_n)))

  return CANParser(DBC[car_fingerprint]['radar'], signals, checks, 1)

class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.track_id = 0
    self.radar_ts = CP.radarTimeStep
    self.radar_acc_tss1 = CP.carFingerprint in RADAR_ACC_CAR_TSS1

    if self.radar_acc_tss1:
      self.RADAR_A_MSGS = self.RADAR_B_MSGS = list(range(0x301, 0x318, 2))
      self.valid_cnt = {key: 0 for key in range(0x3f)}
      self.rcp = _create_radar_acc_tss1_can_parser(CP.carFingerprint)
    else:
      if CP.carFingerprint in TSS2_CAR:
        self.RADAR_A_MSGS = list(range(0x180, 0x190))
        self.RADAR_B_MSGS = list(range(0x190, 0x1a0))
      else:
        self.RADAR_A_MSGS = list(range(0x210, 0x220))
        self.RADAR_B_MSGS = list(range(0x220, 0x230))

      self.rcp = _create_radar_can_parser(CP.carFingerprint)
      self.valid_cnt = {key: 0 for key in self.RADAR_A_MSGS}

    self.trigger_msg = self.RADAR_B_MSGS[-1]
    self.updated_messages = set()

  def update(self, can_strings):
    if self.rcp is None:
      return super().update(None)

    vls = self.rcp.update_strings(can_strings)
    self.updated_messages.update(vls)

    if self.trigger_msg not in self.updated_messages:
      return None

    if self.radar_acc_tss1:
      rr = self._update_radar_acc_tss1(self.updated_messages)
    else:
      rr = self._update(self.updated_messages)
    self.updated_messages.clear()

    return rr

  def _update_radar_acc_tss1(self, updated_messages):
    ret = car.RadarData.new_message()
    errors = []
    if not self.rcp.can_valid:
      errors.append("canError")
    ret.errors = errors

    updated_ids = set()
    for ii in sorted(updated_messages):
      if ii in self.RADAR_A_MSGS:
        cpt = self.rcp.vl[ii]
        track_id = int(cpt['ID'])
        if track_id != 0x3f and cpt['LONG_DIST'] > 0:
          updated_ids.add(track_id)
          self.valid_cnt[track_id] = min(self.valid_cnt[track_id] + 1, int(3.0 / self.radar_ts))

          # new track or staled track
          if track_id not in self.pts or (not self.pts[track_id].measured):
            self.pts[track_id] = car.RadarData.RadarPoint.new_message()
            self.pts[track_id].trackId = self.track_id
            self.track_id += 1

          self.pts[track_id].dRel = cpt['LONG_DIST']  # from front of car
          self.pts[track_id].yRel = cpt['LAT_DIST']  # in car frame's y axis, left is positive
          self.pts[track_id].vRel = cpt['SPEED'] # it's absolute speed
          self.pts[track_id].aRel = float('nan')
          self.pts[track_id].yvRel = cpt['LAT_SPEED']
          self.pts[track_id].measured = True

    for track_id in list(self.pts):
      if track_id not in updated_ids:
        self.valid_cnt[track_id] = 0
        del self.pts[track_id]

    ret.points = list(self.pts.values())
    return ret

  def _update(self, updated_messages):
    ret = car.RadarData.new_message()
    errors = []
    if not self.rcp.can_valid:
      errors.append("canError")
    ret.errors = errors

    for ii in sorted(updated_messages):
      if ii in self.RADAR_A_MSGS:
        cpt = self.rcp.vl[ii]

        if cpt['LONG_DIST'] >= 255 or cpt['NEW_TRACK']:
          self.valid_cnt[ii] = 0    # reset counter
        if cpt['VALID'] and cpt['LONG_DIST'] < 255:
          self.valid_cnt[ii] += 1
        else:
          self.valid_cnt[ii] = max(self.valid_cnt[ii] - 1, 0)

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
