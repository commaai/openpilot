#!/usr/bin/env python
import numpy as np
from selfdrive.can.parser import CANParser
from cereal import car
from common.realtime import sec_since_boot
import zmq
from selfdrive.services import service_list
import selfdrive.messaging as messaging

def create_radar_signals(*signals):
  # accepts multiple namedtuples in the form ([('name', value)],[msg])
  name_value = []
  msgs = []
  repetitions = []
  for s in signals:
    name_value += [nv for nv in s.name_value]
    name_value_n = len(s.name_value)
    msgs_n = [len(s.msg)]
    repetitions += msgs_n * name_value_n
    msgs += s.msg * name_value_n

  name_value = sum([[nv] * r for nv, r in zip(name_value, repetitions)], [])
  names = [n for n, v in name_value]
  vals  = [v for n, v in name_value]
  return zip(names, msgs, vals)

def create_radar_checks(msgs, select, rate = [20]):
  if select == "all":
    return zip(msgs, rate * len(msgs))
  if select == "last":
    return zip([msgs[-1]], rate)
  if select == "none":
    return []
  return []

RADAR_MSGS = range(0x500, 0x540)

def _create_radard_can_parser(car_fingerprint):
  dbc_f = DBC[car_fingerprint]['radar']

  sig = namedtuple('sig','name_value msg')
  trackers = [('X_Rel', 0),
              ('Angle', 0),
              ('V_Rel', 0)]
  tracker_sig = sig(trackers, RADAR_MSGS)

  signals = create_radar_signals(tracker_sig)
  checks = create_radar_checks(RADAR_MSGS, select = "all")

  return CANParser(dbc_f, signals, checks, 1)

class RadarInterface(object):
  def __init__(self, CP):
    # radar
    self.pts = {}
    self.validCnt = {key: 0 for key in RADAR_MSGS}
    self.track_id = 0

    self.delay = 0.0  # Delay of radar

    self.rcp = _create_radard_can_parser()

    context = zmq.Context()
    self.logcan = messaging.sub_sock(context, service_list['can'].port)

  def update(self):
    canMonoTimes = []

    updated_messages = set()
    while 1:
      tm = int(sec_since_boot() * 1e9)
      updated_messages.update(self.rcp.update(tm, True))

      if RADAR_MSGS[-1] in updated_messages:
        break

    ret = car.RadarState.new_message()
    errors = []
    if not self.rcp.can_valid:
      errors.append("commIssue")
    ret.errors = errors
    ret.canMonoTimes = canMonoTimes

    for ii in updated_messages:
      cpt = self.rcp.vl[ii]

      if cpt['X_Rel'] > 0.00001:
        self.validCnt[ii] = 0    # reset counter

      if cpt['X_Rel'] > 0.00001:
        self.validCnt[ii] += 1
      else:
        self.validCnt[ii] = max(self.validCnt[ii] -1, 0)
      #print ii, self.validCnt[ii], cpt['VALID'], cpt['X_Rel'], cpt['Angle']

      # radar point only valid if there have been enough valid measurements
      if self.validCnt[ii] > 0:
        if ii not in self.pts:
          self.pts[ii] = car.RadarState.RadarPoint.new_message()
          self.pts[ii].trackId = self.track_id
          self.track_id += 1
        self.pts[ii].dRel = cpt['X_Rel']  # from front of car
        self.pts[ii].yRel = cpt['X_Rel'] * cpt['Angle'] * np.pi / 180.  # in car frame's y axis, left is positive
        self.pts[ii].vRel = cpt['V_Rel']
        self.pts[ii].aRel = float('nan')
        self.pts[ii].yvRel = float('nan')
        self.pts[ii].measured = True
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
    print ret
