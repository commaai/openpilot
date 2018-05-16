#!/usr/bin/env python
import os
from selfdrive.can.parser import CANParser
from cereal import car
from common.realtime import sec_since_boot
import zmq
from selfdrive.services import service_list
import selfdrive.messaging as messaging


RADAR_MSGS_C = range(0x2c2, 0x2d4+2, 2)  # c_ messages 706,...,724
RADAR_MSGS_D = range(0x2a2, 0x2b4+2, 2)  # d_ messages
LAST_MSG = max(RADAR_MSGS_C + RADAR_MSGS_D)

def _create_radard_can_parser():
  dbc_f = 'chrysler_pacifica_2017_hybrid_private_fusion.dbc'
  msg_n = len(RADAR_MSGS_C)
  # list of [(signal name, message name or number, initial values), (...)]
  # [('RADAR_STATE', 1024, 0),
  #  ('LONG_DIST', 1072, 255),
  #  ('LONG_DIST', 1073, 255),
  #  ('LONG_DIST', 1074, 255),
  #  ('LONG_DIST', 1075, 255),

  # The factor and offset are applied by the dbc parsing library, so the
  # default values should be after the factor/offset are applied.
  signals = zip(['LONG_DIST'] * msg_n +
                ['LAT_DIST'] * msg_n +
                ['REL_SPEED'] * msg_n,
                RADAR_MSGS_C * 2 +  # LONG_DIST, LAT_DIST
                RADAR_MSGS_D,    # REL_SPEED
                [0] * msg_n +  # LONG_DIST
                [-1000] * msg_n +    # LAT_DIST
                [-146.278] * msg_n)  # REL_SPEED set to 0, factor/offset to this
  # TODO what are the checks actually used for?
  # honda only checks the last message,
  # toyota checks all the messages. Which do we want?
  checks = zip(RADAR_MSGS_C +
               RADAR_MSGS_D,
               [20]*msg_n +  # 20Hz (0.05s)
               [20]*msg_n)  # 20Hz (0.05s)

  return CANParser(os.path.splitext(dbc_f)[0], signals, checks, 1)

class RadarInterface(object):
  def __init__(self):
    # radar
    self.pts = {}
    self.validCnt = {key: 0 for key in RADAR_MSGS}
    self.track_id = 0

    self.delay = 0.0  # Delay of radar  #TUNE

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
      if LAST_MSG in updated_messages:
        break

    ret = car.RadarState.new_message()
    errors = []
    if not self.rcp.can_valid:
      errors.append("commIssue")
    ret.errors = errors
    ret.canMonoTimes = canMonoTimes

    for ii in updated_messages:
      cpt = self.rcp.vl[ii]

      if cpt['LONG_DIST'] == 0:
        self.validCnt[ii] = 0    # reset counter, new track.

      if cpt['LONG_DIST'] > 0:
        self.validCnt[ii] += 1
      else:
        self.validCnt[ii] = max(self.validCnt[ii] -1, 0)
      #print ii, self.validCnt[ii], cpt['VALID'], cpt['LONG_DIST'], cpt['LAT_DIST']

      # radar point only valid if there have been enough valid measurements
      if self.validCnt[ii] > 0:
        if ii not in self.pts:
          self.pts[ii] = car.RadarState.RadarPoint.new_message()
          # TODO instead get trackId from the c_* d_* message ID.
          self.pts[ii].trackId = self.track_id
          self.track_id += 1
        self.pts[ii].dRel = cpt['LONG_DIST']  # from front of car
        # our lat_dist is positive to the right in car's frame.
        # TODO what does yRel want?
        self.pts[ii].yRel = cpt['LAT_DIST']  # in car frame's y axis, left is positive
        self.pts[ii].vRel = cpt['REL_SPEED']
        self.pts[ii].aRel = float('nan')
        self.pts[ii].yvRel = float('nan')
        self.pts[ii].measured = True
      else:
        if ii in self.pts:
          del self.pts[ii]

    ret.points = self.pts.values()
    return ret

if __name__ == "__main__":
  RI = RadarInterface()
  while 1:
    ret = RI.update()
    print(chr(27) + "[2J")  # clear screen
    print ret
