#!/usr/bin/env python
import zmq
import time
from selfdrive.can.parser import CANParser
from cereal import car
from common.realtime import sec_since_boot
from selfdrive.services import service_list
import selfdrive.messaging as messaging
from selfdrive.car.toyota.values import NO_DSU_CAR


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


RADAR_TARGET_MSGS = list(range(0x210, 0x220))
RADAR_SCORE_MSGS = list(range(0x220, 0x230))

def _create_radard_can_parser(car_fingerprint):
  dbc_f = DBC[car_fingerprint]['radar']
  radar_messages = RADAR_TARGET_MSGS + RADAR_SCORE_MSGS

  sig = namedtuple('sig','name_value msg')
  targets = [('LONG_DIST', 255),
              ('NEW_TRACK', 1),
              ('LAT_DIST', 0),
              ('REL_SPEED', 0),
              ('VALID', 0)]
  target_sig = sig(targets, RADAR_TARGET_MSGS)

  scores = [('SCORE', 0)]
  score_sig = sig(scores, RADAR_SCORE_MSGS)

  signals = create_radar_signals(target_sig, score_sig)
  checks = create_radar_checks(radar_messages, select = "all")

  return CANParser(dbc_f, signals, checks, 1)


class RadarInterface(object):
  def __init__(self, CP):
    # radar
    self.pts = {}
    self.valid_cnt = {key: 0 for key in RADAR_TARGET_MSGS}
    self.track_id = 0

    self.delay = 0.0  # Delay of radar

    self.rcp = _create_radard_can_parser(CP.carFingerprint)
    self.no_dsu_car = CP.carFingerprint in NO_DSU_CAR

    context = zmq.Context()
    self.logcan = messaging.sub_sock(context, service_list['can'].port)

  def update(self):

    ret = car.RadarState.new_message()
    if self.no_dsu_car:
      # TODO: make a adas dbc file for dsu-less models
      time.sleep(0.05)
      return ret

    canMonoTimes = []
    updated_messages = set()
    while 1:
      tm = int(sec_since_boot() * 1e9)
      updated_messages.update(self.rcp.update(tm, True))
      if RADAR_SCORE_MSGS[-1] in updated_messages:
        break

    errors = []
    if not self.rcp.can_valid:
      errors.append("commIssue")
    ret.errors = errors
    ret.canMonoTimes = canMonoTimes

    for ii in updated_messages:
      if ii in RADAR_TARGET_MSGS:
        cpt = self.rcp.vl[ii]

        if cpt['LONG_DIST'] >=255 or cpt['NEW_TRACK']:
          self.valid_cnt[ii] = 0    # reset counter
        if cpt['VALID'] and cpt['LONG_DIST'] < 255:
          self.valid_cnt[ii] += 1
        else:
          self.valid_cnt[ii] = max(self.valid_cnt[ii] -1, 0)

        score = self.rcp.vl[ii+16]['SCORE']
        # print ii, self.valid_cnt[ii], score, cpt['VALID'], cpt['LONG_DIST'], cpt['LAT_DIST']

        # radar point only valid if it's a valid measurement and score is above 50
        if cpt['VALID'] or (score > 50 and cpt['LONG_DIST'] < 255 and self.valid_cnt[ii] > 0):
          if ii not in self.pts or cpt['NEW_TRACK']:
            self.pts[ii] = car.RadarState.RadarPoint.new_message()
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

    ret.points = self.pts.values()
    return ret

if __name__ == "__main__":
  RI = RadarInterface(None)
  while 1:
    ret = RI.update()
    print(chr(27) + "[2J")
    print(ret)
