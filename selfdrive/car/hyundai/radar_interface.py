#!/usr/bin/env python
import os
from selfdrive.can.parser import CANParser
from cereal import car
from common.realtime import sec_since_boot
import zmq
from selfdrive.services import service_list
import selfdrive.messaging as messaging


RADAR_MSGS = list(range(0x420, 0x421))

def _create_radard_can_parser():
  dbc_f = 'kia_sorento_2018.dbc'
  msg_n = len(RADAR_MSGS)
  signals = zip(['ACC_ObjDist'] * msg_n + ['ACC_ObjStatus'] * msg_n + ['ACC_ObjLatPos'] * msg_n +
                ['ACC_ObjRelSpeed'] * msg_n + ['ObjValid'] * msg_n,
                RADAR_MSGS * 5,
                [255] * msg_n + [1] * msg_n + [0] * msg_n + [0] * msg_n + [0] * msg_n)
  checks = zip(RADAR_MSGS, [20]*msg_n)

  return CANParser(os.path.splitext(dbc_f)[0], signals, checks, 1)

class RadarInterface(object):
  def __init__(self, CP):
    # radar
    self.pts = {}
    self.validCnt = {key: 0 for key in RADAR_MSGS}
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
      # TODO: do not hardcode last msg
      if 0x21f in updated_messages:
        break

    ret = car.RadarState.new_message()
    errors = []
    if not self.rcp.can_valid:
      errors.append("commIssue")
    ret.errors = errors
    ret.canMonoTimes = canMonoTimes

    for ii in updated_messages:
      cpt = self.rcp.vl[ii]

      if cpt['ACC_ObjDist'] >=255 or cpt['ACC_ObjStatus']:
        self.validCnt[ii] = 0    # reset counter

      if cpt['ObjValid'] and cpt['ACC_ObjDist'] < 255:
        self.validCnt[ii] += 1
      else:
        self.validCnt[ii] = max(self.validCnt[ii] -1, 0)
      #print ii, self.validCnt[ii], cpt['ObjValid'], cpt['ACC_ObjDist'], cpt['ACC_ObjLatPos']

      # radar point only valid if there have been enough valid measurements
      if self.validCnt[ii] > 0:
        if ii not in self.pts or cpt['ACC_ObjStatus']:
          self.pts[ii] = car.RadarState.RadarPoint.new_message()
          self.pts[ii].trackId = self.track_id
          self.track_id += 1
        self.pts[ii].dRel = cpt['ACC_ObjDist']  # from front of car
        self.pts[ii].yRel = -cpt['ACC_ObjLatPos']  # in car frame's y axis, left is positive
        self.pts[ii].vRel = cpt['ACC_ObjRelSpeed']
        self.pts[ii].aRel = float('nan')
        self.pts[ii].yvRel = float('nan')
        self.pts[ii].measured = bool(cpt['ObjValid'])
      else:
        if ii in self.pts:
          del self.pts[ii]

    ret.points = self.pts.values()
    return ret

if __name__ == "__main__":
  RI = RadarInterface(None)
  while 1:
    ret = RI.update()
    print(chr(27) + "[2J")
    print(ret)
