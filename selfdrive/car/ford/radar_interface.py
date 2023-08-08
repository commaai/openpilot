#!/usr/bin/env python3
from math import cos, sin
from cereal import car
from opendbc.can.parser import CANParser
from common.conversions import Conversions as CV
from selfdrive.car.ford.fordcan import CanBus
from selfdrive.car.ford.values import DBC, RADAR
from selfdrive.car.interfaces import RadarInterfaceBase

DELPHI_ESR_RADAR_MSGS = list(range(0x500, 0x540))

DELPHI_MRR_RADAR_START_ADDR = 0x120
DELPHI_MRR_RADAR_MSG_COUNT = 64


def _create_delphi_esr_radar_can_parser(CP) -> CANParser:
  msg_n = len(DELPHI_ESR_RADAR_MSGS)
  signals = list(zip(['X_Rel'] * msg_n + ['Angle'] * msg_n + ['V_Rel'] * msg_n,
                     DELPHI_ESR_RADAR_MSGS * 3))
  checks = list(zip(DELPHI_ESR_RADAR_MSGS, [20] * msg_n))

  return CANParser(RADAR.DELPHI_ESR, signals, checks, CanBus(CP).radar)


def _create_delphi_mrr_radar_can_parser(CP) -> CANParser:
  signals = []
  checks = []

  for i in range(1, DELPHI_MRR_RADAR_MSG_COUNT + 1):
    msg = f"MRR_Detection_{i:03d}"
    signals += [
      (f"CAN_DET_VALID_LEVEL_{i:02d}", msg),
      (f"CAN_DET_AZIMUTH_{i:02d}", msg),
      (f"CAN_DET_RANGE_{i:02d}", msg),
      (f"CAN_DET_RANGE_RATE_{i:02d}", msg),
      (f"CAN_DET_AMPLITUDE_{i:02d}", msg),
      (f"CAN_SCAN_INDEX_2LSB_{i:02d}", msg),
    ]
    checks += [(msg, 20)]

  return CANParser(RADAR.DELPHI_MRR, signals, checks, CanBus(CP).radar)


class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP):
    super().__init__(CP)

    self.updated_messages = set()
    self.track_id = 0
    self.radar = DBC[CP.carFingerprint]['radar']
    if self.radar is None or CP.radarUnavailable:
      self.rcp = None
    elif self.radar == RADAR.DELPHI_ESR:
      self.rcp = _create_delphi_esr_radar_can_parser(CP)
      self.trigger_msg = DELPHI_ESR_RADAR_MSGS[-1]
      self.valid_cnt = {key: 0 for key in DELPHI_ESR_RADAR_MSGS}
    elif self.radar == RADAR.DELPHI_MRR:
      self.rcp = _create_delphi_mrr_radar_can_parser(CP)
      self.trigger_msg = DELPHI_MRR_RADAR_START_ADDR + DELPHI_MRR_RADAR_MSG_COUNT - 1
    else:
      raise ValueError(f"Unsupported radar: {self.radar}")

  def update(self, can_strings):
    if self.rcp is None:
      return super().update(None)

    vls = self.rcp.update_strings(can_strings)
    self.updated_messages.update(vls)

    if self.trigger_msg not in self.updated_messages:
      return None

    ret = car.RadarData.new_message()
    errors = []
    if not self.rcp.can_valid:
      errors.append("canError")
    ret.errors = errors

    if self.radar == RADAR.DELPHI_ESR:
      self._update_delphi_esr()
    elif self.radar == RADAR.DELPHI_MRR:
      self._update_delphi_mrr()

    ret.points = list(self.pts.values())
    self.updated_messages.clear()
    return ret

  def _update_delphi_esr(self):
    for ii in sorted(self.updated_messages):
      cpt = self.rcp.vl[ii]

      if cpt['X_Rel'] > 0.00001:
        self.valid_cnt[ii] = 0    # reset counter

      if cpt['X_Rel'] > 0.00001:
        self.valid_cnt[ii] += 1
      else:
        self.valid_cnt[ii] = max(self.valid_cnt[ii] - 1, 0)
      #print ii, self.valid_cnt[ii], cpt['VALID'], cpt['X_Rel'], cpt['Angle']

      # radar point only valid if there have been enough valid measurements
      if self.valid_cnt[ii] > 0:
        if ii not in self.pts:
          self.pts[ii] = car.RadarData.RadarPoint.new_message()
          self.pts[ii].trackId = self.track_id
          self.track_id += 1
        self.pts[ii].dRel = cpt['X_Rel']  # from front of car
        self.pts[ii].yRel = cpt['X_Rel'] * cpt['Angle'] * CV.DEG_TO_RAD  # in car frame's y axis, left is positive
        self.pts[ii].vRel = cpt['V_Rel']
        self.pts[ii].aRel = float('nan')
        self.pts[ii].yvRel = float('nan')
        self.pts[ii].measured = True
      else:
        if ii in self.pts:
          del self.pts[ii]

  def _update_delphi_mrr(self):
    for ii in range(1, DELPHI_MRR_RADAR_MSG_COUNT + 1):
      msg = self.rcp.vl[f"MRR_Detection_{ii:03d}"]

      # SCAN_INDEX rotates through 0..3 on each message
      # treat these as separate points
      scanIndex = msg[f"CAN_SCAN_INDEX_2LSB_{ii:02d}"]
      i = (ii - 1) * 4 + scanIndex

      if i not in self.pts:
        self.pts[i] = car.RadarData.RadarPoint.new_message()
        self.pts[i].trackId = self.track_id
        self.pts[i].aRel = float('nan')
        self.pts[i].yvRel = float('nan')
        self.track_id += 1

      valid = bool(msg[f"CAN_DET_VALID_LEVEL_{ii:02d}"])
      amplitude = msg[f"CAN_DET_AMPLITUDE_{ii:02d}"]            # dBsm [-64|63]

      if valid and 0 < amplitude <= 15:
        azimuth = msg[f"CAN_DET_AZIMUTH_{ii:02d}"]              # rad [-3.1416|3.13964]
        dist = msg[f"CAN_DET_RANGE_{ii:02d}"]                   # m [0|255.984]
        distRate = msg[f"CAN_DET_RANGE_RATE_{ii:02d}"]          # m/s [-128|127.984]

        # *** openpilot radar point ***
        self.pts[i].dRel = cos(azimuth) * dist                  # m from front of car
        self.pts[i].yRel = -sin(azimuth) * dist                 # in car frame's y axis, left is positive
        self.pts[i].vRel = distRate                             # m/s

        self.pts[i].measured = True

      else:
        del self.pts[i]
