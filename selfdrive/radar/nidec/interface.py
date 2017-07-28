#!/usr/bin/env python
import os
import numpy as np

from selfdrive.car.honda.can_parser import CANParser
from selfdrive.can.parser import CANParser as CANParserC

from selfdrive.boardd.boardd import can_capnp_to_can_list

from cereal import car
from common.realtime import sec_since_boot

import zmq
from selfdrive.services import service_list
import selfdrive.messaging as messaging

NEW_CAN = os.getenv("OLD_CAN") is None

def _create_radard_can_parser():
  dbc_f = 'acura_ilx_2016_nidec.dbc'
  radar_messages = range(0x430, 0x43A) + range(0x440, 0x446)
  signals = zip(['LONG_DIST'] * 16 + ['NEW_TRACK'] * 16 + ['LAT_DIST'] * 16 +
                ['REL_SPEED'] * 16, radar_messages * 4,
                [255] * 16 + [1] * 16 + [0] * 16 + [0] * 16)
  checks = zip(radar_messages, [20]*16)

  if NEW_CAN:
    return CANParserC(os.path.splitext(dbc_f)[0], signals, checks, 1)
  else:
    return CANParser(dbc_f, signals, checks)

class RadarInterface(object):
  def __init__(self):
    # radar
    self.pts = {}
    self.track_id = 0

    # Nidec
    self.rcp = _create_radard_can_parser()

    context = zmq.Context()
    self.logcan = messaging.sub_sock(context, service_list['can'].port)

  def update(self):
    canMonoTimes = []

    if NEW_CAN:
      updated_messages = set()
      while 1:
        tm = int(sec_since_boot() * 1e9)
        updated_messages.update(self.rcp.update(tm, True))
        if 0x445 in updated_messages:
          break
    else:
      can_pub_radar = []

      # TODO: can hang if no packets show up
      while 1:
        for a in messaging.drain_sock(self.logcan, wait_for_one=True):
          canMonoTimes.append(a.logMonoTime)
          can_pub_radar.extend(can_capnp_to_can_list(a.can, [1, 3]))

        # only run on the 0x445 packets, used for timing
        if any(x[0] == 0x445 for x in can_pub_radar):
          break

      updated_messages = self.rcp.update_can(can_pub_radar)

    ret = car.RadarState.new_message()
    errors = []
    if not self.rcp.can_valid:
      errors.append("notValid")
    ret.errors = errors
    ret.canMonoTimes = canMonoTimes

    for ii in updated_messages:
      cpt = self.rcp.vl[ii]
      #print cpt
      if cpt['LONG_DIST'] < 255:
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

    ret.points = self.pts.values()
    return ret

if __name__ == "__main__":
  RI = RadarInterface()
  while 1:
    ret = RI.update()
    print(chr(27) + "[2J")
    print ret


