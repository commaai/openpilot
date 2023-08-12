#!/usr/bin/env python3
from collections import defaultdict

from cereal import car
from opendbc.can.parser import CANParser
from selfdrive.car.interfaces import RadarInterfaceBase
from selfdrive.car.honda.values import DBC


def _create_nidec_can_parser(car_fingerprint):
  radar_messages = [0x400] + list(range(0x430, 0x43A)) + list(range(0x440, 0x446))
  messages = [(m, 20) for m in radar_messages]
  return CANParser(DBC[car_fingerprint]['radar'], messages, 1)


class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.track_id = 0
    self.radar_fault = False
    self.radar_wrong_config = False
    self.radar_off_can = CP.radarUnavailable
    self.radar_ts = CP.radarTimeStep

    self.delay = int(round(0.1 / CP.radarTimeStep))   # 0.1s delay of radar

    # Nidec
    if self.radar_off_can:
      self.rcp = None
    else:
      self.rcp = _create_nidec_can_parser(CP.carFingerprint)
    self.trigger_msg = 0x445
    self.updated_values = defaultdict(lambda: defaultdict(list))

  def update(self, can_strings):
    # in Bosch radar and we are only steering for now, so sleep 0.05s to keep
    # radard at 20Hz and return no points
    if self.radar_off_can:
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

    for ii in sorted(updated_values):
      msgs = updated_values[ii]
      n_vals_per_addr = len(list(msgs.values())[0])
      cpts = [
        {k: v[i] for k, v in  msgs.items()}
        for i in range(n_vals_per_addr)
      ]

      for cpt in cpts:
        if ii == 0x400:
          # check for radar faults
          self.radar_fault = cpt['RADAR_STATE'] != 0x79
          self.radar_wrong_config = cpt['RADAR_STATE'] == 0x69
        elif cpt['LONG_DIST'] < 255:
          if ii not in self.pts or cpt['NEW_TRACK']:
            self.pts[ii] = car.RadarData.RadarPoint.new_message()
            self.pts[ii].trackId = self.track_id
            self.track_id += 1
          self.pts[ii].dRel = cpt['LONG_DIST']  # from front of car
          self.pts[ii].yRel = -cpt['LAT_DIST']  # in car frame's y axis, left is positive
          self.pts[ii].vRel = cpt['REL_SPEED']
          self.pts[ii].aRel = float('nan')
          self.pts[ii].yvRel = float('nan')
          self.pts[ii].measured = True
        else:
          if ii in self.pts:
            del self.pts[ii]

    errors = []
    if not can_valid:
      errors.append("canError")
    if self.radar_fault:
      errors.append("fault")
    if self.radar_wrong_config:
      errors.append("wrongConfig")
    ret.errors = errors

    ret.points = list(self.pts.values())

    return ret
