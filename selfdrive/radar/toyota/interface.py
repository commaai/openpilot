#!/usr/bin/env python
import os
import numpy as np
from selfdrive.can.parser import CANParser
from cereal import car
from common.realtime import sec_since_boot
import zmq
from selfdrive.services import service_list
import selfdrive.messaging as messaging


RADAR_MSGS = range(0x210, 0x220)

def _create_radard_can_parser():
  dbc_f = 'toyota_prius_2017_adas.dbc'
  msg_n = len(RADAR_MSGS)
  msg_last = RADAR_MSGS[-1]
  signals = zip(['LONG_DIST'] * msg_n + ['NEW_TRACK'] * msg_n + ['LAT_DIST'] * msg_n +
                ['REL_SPEED'] * msg_n + ['VALID'] * msg_n,
                RADAR_MSGS * 5,
                [255] * msg_n + [1] * msg_n + [0] * msg_n + [0] * msg_n + [0] * msg_n)
  checks = zip(RADAR_MSGS, [20]*msg_n)

  return CANParser(os.path.splitext(dbc_f)[0], signals, checks, 1)

class RadarInterface(object):
  def __init__(self):
    # radar
    self.pts = {}
    self.ptsValid = {key: False for key in RADAR_MSGS}
    self.track_id = 0

    self.delay = 0.0  # Delay of radar

    # Nidec
    self.rcp = _create_radard_can_parser()

    context = zmq.Context()
    self.logcan = messaging.sub_sock(context, service_list['can'].port)

  def update(self):
    canMonoTimes = []

    updated_messages = set()
    while 1:
      tm = int(sec_since_boot() * 1e9)
      updated_messages.update(self.rcp.update(tm, True))
      # TODO: use msg_last
      if 0x21f in updated_messages:
        break

    ret = car.RadarState.new_message()
    errors = []
    if not self.rcp.can_valid:
      errors.append("commIssue")
    ret.errors = errors
    ret.canMonoTimes = canMonoTimes
    #print "NEW TRACKS"
    for ii in updated_messages:
      cpt = self.rcp.vl[ii]

      # a point needs one valid measurement before being considered
      #if cpt['NEW_TRACK'] or cpt['LONG_DIST'] >= 255:
      #  self.ptsValid[ii] = False    # reset validity
      # TODO: find better way to eliminate both false positive and false negative
      if cpt['VALID'] and cpt['LONG_DIST'] < 255:
        self.ptsValid[ii] = True
      else:
        self.ptsValid[ii] = False

      if self.ptsValid[ii]:
        #print "%5s %5s %5s" % (round(cpt['LONG_DIST'], 1), round(cpt['LAT_DIST'], 1), round(cpt['REL_SPEED'], 1))
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
