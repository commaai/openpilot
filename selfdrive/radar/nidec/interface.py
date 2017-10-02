#!/usr/bin/env python
import os
import numpy as np

from selfdrive.can.parser import CANParser

from selfdrive.boardd.boardd import can_capnp_to_can_list

from cereal import car
from common.realtime import sec_since_boot

import zmq
from selfdrive.services import service_list
import selfdrive.messaging as messaging

def _create_nidec_can_parser():
  dbc_f = 'acura_ilx_2016_nidec.dbc'
  radar_messages = [0x400] + range(0x430, 0x43A) + range(0x440, 0x446)
  signals = zip(['RADAR_STATE'] + 
                ['LONG_DIST'] * 16 + ['NEW_TRACK'] * 16 + ['LAT_DIST'] * 16 +
                ['REL_SPEED'] * 16,
                [0x400] + radar_messages[1:] * 4,
                [0] + [255] * 16 + [1] * 16 + [0] * 16 + [0] * 16)
  checks = zip([0x445], [20])

  return CANParser(os.path.splitext(dbc_f)[0], signals, checks, 1)

class RadarInterface(object):
  def __init__(self):
    # radar
    self.pts = {}
    self.track_id = 0
    self.radar_fault = False

    # Nidec
    self.rcp = _create_nidec_can_parser()

    context = zmq.Context()
    self.logcan = messaging.sub_sock(context, service_list['can'].port)

  def update(self):
    canMonoTimes = []

    updated_messages = set()
    while 1:
      tm = int(sec_since_boot() * 1e9)
      updated_messages.update(self.rcp.update(tm, True))
      if 0x445 in updated_messages:
        break

    for ii in updated_messages:
      cpt = self.rcp.vl[ii]
      if ii == 0x400:
        # check for radar faults
        self.radar_fault = cpt['RADAR_STATE'] != 0x79
      elif cpt['LONG_DIST'] < 255:
        if ii not in self.pts or cpt['NEW_TRACK']:
          self.pts[ii] = car.RadarState.RadarPoint.new_message()
          self.pts[ii].trackId = self.track_id
          self.track_id += 1
        self.pts[ii].dRel = cpt['LONG_DIST']  # from front of car
        self.pts[ii].yRel = -cpt['LAT_DIST']  # in car frame's y axis, left is positive
        self.pts[ii].vRel = cpt['REL_SPEED']
        self.pts[ii].aRel = float('nan')
        self.pts[ii].yvRel = float('nan')
      else:
        if ii in self.pts:
          del self.pts[ii]

    ret = car.RadarState.new_message()
    errors = []
    if not self.rcp.can_valid:
      errors.append("commIssue")
    if self.radar_fault:
      errors.append("fault")
    ret.errors = errors
    ret.canMonoTimes = canMonoTimes

    ret.points = self.pts.values()
    return ret

if __name__ == "__main__":
  RI = RadarInterface()
  while 1:
    ret = RI.update()
    print(chr(27) + "[2J")
    print ret


