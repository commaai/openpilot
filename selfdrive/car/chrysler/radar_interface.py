#!/usr/bin/env python3
import os
from opendbc.can.parser import CANParser
from cereal import car
from selfdrive.car.interfaces import RadarInterfaceBase

RADAR_MSGS_C = list(range(0x2c2, 0x2d4+2, 2))  # c_ messages 706,...,724
RADAR_MSGS_D = list(range(0x2a2, 0x2b4+2, 2))  # d_ messages
LAST_MSG = max(RADAR_MSGS_C + RADAR_MSGS_D)
NUMBER_MSGS = len(RADAR_MSGS_C) + len(RADAR_MSGS_D)

def _create_radar_can_parser():
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
  signals = list(zip(['LONG_DIST'] * msg_n +
                ['LAT_DIST'] * msg_n +
                ['REL_SPEED'] * msg_n,
                RADAR_MSGS_C * 2 +  # LONG_DIST, LAT_DIST
                RADAR_MSGS_D,    # REL_SPEED
                [0] * msg_n +  # LONG_DIST
                [-1000] * msg_n +    # LAT_DIST
                [-146.278] * msg_n))  # REL_SPEED set to 0, factor/offset to this
  # TODO what are the checks actually used for?
  # honda only checks the last message,
  # toyota checks all the messages. Which do we want?
  checks = list(zip(RADAR_MSGS_C +
               RADAR_MSGS_D,
               [20]*msg_n +  # 20Hz (0.05s)
               [20]*msg_n))  # 20Hz (0.05s)

  return CANParser(os.path.splitext(dbc_f)[0], signals, checks, 1)

def _address_to_track(address):
  if address in RADAR_MSGS_C:
    return (address - RADAR_MSGS_C[0]) // 2
  if address in RADAR_MSGS_D:
    return (address - RADAR_MSGS_D[0]) // 2
  raise ValueError("radar received unexpected address %d" % address)

class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.rcp = _create_radar_can_parser()
    self.updated_messages = set()
    self.trigger_msg = LAST_MSG

  def update(self, can_strings):
    vls = self.rcp.update_strings(can_strings)
    self.updated_messages.update(vls)

    if self.trigger_msg not in self.updated_messages:
      return None

    ret = car.RadarData.new_message()
    errors = []
    if not self.rcp.can_valid:
      errors.append("canError")
    ret.errors = errors

    for ii in self.updated_messages:  # ii should be the message ID as a number
      cpt = self.rcp.vl[ii]
      trackId = _address_to_track(ii)

      if trackId not in self.pts:
        self.pts[trackId] = car.RadarData.RadarPoint.new_message()
        self.pts[trackId].trackId = trackId
        self.pts[trackId].aRel = float('nan')
        self.pts[trackId].yvRel = float('nan')
        self.pts[trackId].measured = True

      if 'LONG_DIST' in cpt:  # c_* message
        self.pts[trackId].dRel = cpt['LONG_DIST']  # from front of car
        # our lat_dist is positive to the right in car's frame.
        # TODO what does yRel want?
        self.pts[trackId].yRel = cpt['LAT_DIST']  # in car frame's y axis, left is positive
      else:  # d_* message
        self.pts[trackId].vRel = cpt['REL_SPEED']

    # We want a list, not a dictionary. Filter out LONG_DIST==0 because that means it's not valid.
    ret.points = [x for x in self.pts.values() if x.dRel != 0]

    self.updated_messages.clear()
    return ret
