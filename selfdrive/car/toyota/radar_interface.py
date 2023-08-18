#!/usr/bin/env python3
from collections import defaultdict

from opendbc.can.parser import CANParser
from cereal import car
from selfdrive.car.toyota.values import DBC, TSS2_CAR
from selfdrive.car.interfaces import RadarInterfaceBase


def _create_radar_can_parser(car_fingerprint):
  if car_fingerprint in TSS2_CAR:
    RADAR_A_MSGS = list(range(0x180, 0x190))
    RADAR_B_MSGS = list(range(0x190, 0x1a0))
  else:
    RADAR_A_MSGS = list(range(0x210, 0x220))
    RADAR_B_MSGS = list(range(0x220, 0x230))

  msg_a_n = len(RADAR_A_MSGS)
  msg_b_n = len(RADAR_B_MSGS)
  messages = list(zip(RADAR_A_MSGS + RADAR_B_MSGS, [20] * (msg_a_n + msg_b_n), strict=True))

  return CANParser(DBC[car_fingerprint]['radar'], messages, 1)

class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.track_id = 0
    self.radar_ts = CP.radarTimeStep

    if CP.carFingerprint in TSS2_CAR:
      self.RADAR_A_MSGS = list(range(0x180, 0x190))
      self.RADAR_B_MSGS = list(range(0x190, 0x1a0))
    else:
      self.RADAR_A_MSGS = list(range(0x210, 0x220))
      self.RADAR_B_MSGS = list(range(0x220, 0x230))

    self.valid_cnt = {key: 0 for key in self.RADAR_A_MSGS}

    self.rcp = None if CP.radarUnavailable else _create_radar_can_parser(CP.carFingerprint)
    self.trigger_msg = self.RADAR_B_MSGS[-1]
    self.updated_values = defaultdict(lambda: defaultdict(list))

  def update(self, can_strings):
    if self.rcp is None:
      return None

    addresses = self.rcp.update_strings(can_strings)
    for addr in addresses:
      vals_dict = self.rcp.vl_all[addr]
      for sig_name, vals in vals_dict.items():
        self.updated_values[addr][sig_name].extend(vals)

    if self.trigger_msg not in self.updated_values:
      return None

    radar_data = self._radar_msg_from_buffer(self.updated_values, self.rcp.can_valid)
    self.updated_values.clear()

    return radar_data

  def _radar_msg_from_buffer(self, updated_values, can_valid):
    ret = car.RadarData.new_message()
    errors = []
    if not can_valid:
      errors.append("canError")
    ret.errors = errors

    for ii in sorted(updated_values):
      if ii not in self.RADAR_A_MSGS:
        continue

      radar_a_msgs = updated_values[ii]
      radar_b_msgs = updated_values[ii+16]

      n_vals_per_addr = len(list(radar_a_msgs.values())[0])
      cpts = [
        {k: v[i] for k, v in  radar_a_msgs.items()}
        for i in range(n_vals_per_addr)
      ]

      for index, cpt in enumerate(cpts):
        if cpt['LONG_DIST'] >= 255 or cpt['NEW_TRACK']:
          self.valid_cnt[ii] = 0    # reset counter
        if cpt['VALID'] and cpt['LONG_DIST'] < 255:
          self.valid_cnt[ii] += 1
        else:
          self.valid_cnt[ii] = max(self.valid_cnt[ii] - 1, 0)

        n_b_scores = len(radar_b_msgs['SCORE'])
        if n_b_scores > 0:
          score_index = min(index, n_b_scores - 1)
          score = radar_b_msgs['SCORE'][score_index]
        else:
          score = None

        # radar point only valid if it's a valid measurement and score is above 50
        if cpt['VALID'] or (score and score > 50 and cpt['LONG_DIST'] < 255 and self.valid_cnt[ii] > 0):
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
