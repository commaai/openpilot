#!/usr/bin/env python
import os
import time
from cereal import car
from selfdrive.can.parser import CANParser
from common.realtime import sec_since_boot

def _create_nidec_can_parser():
  dbc_f = 'acura_ilx_2016_nidec.dbc'
  radar_messages = [0x400] + range(0x430, 0x43A) + range(0x440, 0x446)
  signals = list(zip(['RADAR_STATE'] +
                ['LONG_DIST'] * 16 + ['NEW_TRACK'] * 16 + ['LAT_DIST'] * 16 +
                ['REL_SPEED'] * 16,
                [0x400] + radar_messages[1:] * 4,
                [0] + [255] * 16 + [1] * 16 + [0] * 16 + [0] * 16))
  checks = list(zip([0x445], [20]))

  return CANParser(os.path.splitext(dbc_f)[0], signals, checks, 1)


class RadarInterface(object):
  def __init__(self, CP):
    # radar
    self.pts = {}
    self.track_id = 0
    self.radar_fault = False
    self.radar_wrong_config = False
    self.radar_off_can = CP.radarOffCan

    self.delay = 0.1  # Delay of radar

    # Nidec
    self.rcp = _create_nidec_can_parser()
    self.trigger_msg = 0x445
    self.updated_messages = set()

  def update(self, can_strings):
    # in Bosch radar and we are only steering for now, so sleep 0.05s to keep
    # radard at 20Hz and return no points
    if self.radar_off_can:
      time.sleep(0.05)
      return car.RadarData.new_message()

    tm = int(sec_since_boot() * 1e9)
    vls = self.rcp.update_strings(tm, can_strings)
    self.updated_messages.update(vls)

    if self.trigger_msg not in self.updated_messages:
      return None

    rr = self._update(self.updated_messages)
    self.updated_messages.clear()
    return rr


  def _update(self, updated_messages):
    ret = car.RadarData.new_message()

    for ii in updated_messages:
      cpt = self.rcp.vl[ii]
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
    if not self.rcp.can_valid:
      errors.append("canError")
    if self.radar_fault:
      errors.append("fault")
    if self.radar_wrong_config:
      errors.append("wrongConfig")
    ret.errors = errors

    ret.points = self.pts.values()

    return ret
