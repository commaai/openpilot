#!/usr/bin/env python
import zmq
import time
from collections import namedtuple
from cereal import car
from selfdrive.can.parser import CANParser
from selfdrive.car.honda.values import DBC
from common.realtime import sec_since_boot
from selfdrive.services import service_list
import selfdrive.messaging as messaging
import selfdrive.car.rcp_helpers as rcp


RADAR_HEADER_MSG = 0x400
RADAR_TARGET_MSG = range(0x430, 0x43A) + range(0x440, 0x446)

def _create_nidec_can_parser(car_fingerprint):
  dbc_f = DBC[car_fingerprint]['radar']
  if dbc_f is not None:
    radar_messages = [RADAR_HEADER_MSG] + RADAR_TARGET_MSG

    sig = namedtuple('sig', 'name_value msg')
    headers = [('RADAR_STATE', 0)]
    header_sig = sig(headers, [RADAR_HEADER_MSG])

    targets = [('LONG_DIST', 255),
               ('NEW_TRACK', 1),
               ('LAT_DIST', 0),
               ('REL_SPEED', 0)]
    target_sig = sig(targets, RADAR_TARGET_MSG)

    signals = rcp.create_radar_signals(header_sig, target_sig)
    checks = rcp.create_radar_checks(radar_messages, select = "last")

    return CANParser(dbc_f, signals, checks, 1)
  else:
    return None


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
    self.rcp = _create_nidec_can_parser(CP.carFingerprint)

    context = zmq.Context()
    self.logcan = messaging.sub_sock(context, service_list['can'].port)

  def update(self):
    canMonoTimes = []

    updated_messages = set()
    ret = car.RadarState.new_message()

    # in Bosch radar and we are only steering for now, so sleep 0.05s to keep
    # radard at 20Hz and return no points
    if self.radar_off_can:
      time.sleep(0.05)
      return ret

    while 1:
      tm = int(sec_since_boot() * 1e9)
      updated_messages.update(self.rcp.update(tm, True))
      if RADAR_TARGET_MSG[-1] in updated_messages:
        break

    for ii in updated_messages:
      cpt = self.rcp.vl[ii]
      if ii == RADAR_HEADER_MSG:
        # check for radar faults
        self.radar_fault = cpt['RADAR_STATE'] != 0x79
        self.radar_wrong_config = cpt['RADAR_STATE'] == 0x69
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
        self.pts[ii].measured = True
      else:
        if ii in self.pts:
          del self.pts[ii]

    errors = []
    if not self.rcp.can_valid:
      errors.append("commIssue")
    if self.radar_fault:
      errors.append("fault")
    if self.radar_wrong_config:
      errors.append("wrongConfig")
    ret.errors = errors
    ret.canMonoTimes = canMonoTimes

    ret.points = self.pts.values()

    return ret


if __name__ == "__main__":
  class CarParams:
    radarOffCan = False

  RI = RadarInterface(CarParams)
  while 1:
    ret = RI.update()
    print(chr(27) + "[2J")
    print ret
