#!/usr/bin/env python
import sys
try:
    sys.path.index('/data/openpilot/')
except ValueError:
    sys.path.append('/data/openpilot/')

from cereal import car
import time
import os
import zmq
from selfdrive.can.parser import CANParser
from common.realtime import sec_since_boot
from selfdrive.services import service_list
import selfdrive.messaging as messaging
from selfdrive.car.tesla.readconfig import read_config_file,CarSettings

#RADAR_A_MSGS = list(range(0x371, 0x37F , 3))
#RADAR_B_MSGS = list(range(0x372, 0x37F, 3))
BOSCH_MAX_DIST = 150. #max distance for radar
RADAR_A_MSGS = list(range(0x310, 0x36F , 3))
RADAR_B_MSGS = list(range(0x311, 0x36F, 3))
OBJECT_MIN_PROBABILITY = 20.
CLASS_MIN_PROBABILITY = 20.
#for calibration we only want fixed objects within 1 m of the center line and between 2.5 and 4.5 m far from radar
MINX = 2.5 
MAXX = 4.5
MINY = -1.0
MAXY = 1.0


# Tesla Bosch firmware has 32 objects in all objects or a selected set of the 5 we should look at
# definetly switch to all objects when calibrating but most likely use select set of 5 for normal use
USE_ALL_OBJECTS = True

def _create_radard_can_parser():
  dbc_f = 'teslaradar.dbc'

  msg_a_n = len(RADAR_A_MSGS)
  msg_b_n = len(RADAR_B_MSGS)

  signals = zip(['LongDist'] * msg_a_n +  ['LatDist'] * msg_a_n +
                ['LongSpeed'] * msg_a_n + ['LongAccel'] * msg_a_n + 
                ['Valid'] * msg_a_n + ['Tracked'] * msg_a_n + 
                ['Meas'] * msg_a_n + ['ProbExist'] * msg_a_n + 
                ['Index'] * msg_a_n + ['ProbObstacle'] * msg_a_n + 
                ['LatSpeed'] * msg_b_n + ['Index2'] * msg_b_n +
                ['Class'] * msg_b_n + ['ProbClass'] * msg_b_n + 
                ['Length'] * msg_b_n + ['dZ'] * msg_b_n + ['MovingState'] * msg_b_n,
                RADAR_A_MSGS * 10 + RADAR_B_MSGS * 7,
                [255.] * msg_a_n + [0.] * msg_a_n + [0.] * msg_a_n + [0.] * msg_a_n + 
                [0] * msg_a_n + [0] * msg_a_n + [0] * msg_a_n + [0.] * msg_a_n +
                [0] * msg_a_n + [0.] * msg_a_n + [0.] * msg_b_n + [0] * msg_b_n +
                [0] * msg_b_n + [0.] * msg_b_n + [0.] * msg_b_n +[0.] * msg_b_n + [0]* msg_b_n)

  checks = zip(RADAR_A_MSGS + RADAR_B_MSGS, [20]*(msg_a_n + msg_b_n))

  return CANParser(os.path.splitext(dbc_f)[0], signals, checks, 1)


class RadarInterface(object):
  def __init__(self):
    # radar
    self.pts = {}
    self.delay = 0.1
    self.useTeslaRadar = CarSettings().get_value("useTeslaRadar")
    self.TRACK_LEFT_LANE = True
    self.TRACK_RIGHT_LANE = True
    if self.useTeslaRadar:
      self.pts = {}
      self.valid_cnt = {key: 0 for key in RADAR_A_MSGS}
      self.delay = 0.05  # Delay of radar
      self.rcp = _create_radard_can_parser()
      context = zmq.Context()
      self.logcan = messaging.sub_sock(context, service_list['can'].port)

  def update(self):

    ret = car.RadarState.new_message()
    if not self.useTeslaRadar:
      time.sleep(0.05)
      return ret

    canMonoTimes = []
    updated_messages = set()
    while 1:
      tm = int(sec_since_boot() * 1e9)
      updated_messages.update(self.rcp.update(tm, True))
      if RADAR_B_MSGS[-1] in updated_messages:
        break
    errors = []
    if not self.rcp.can_valid:
      errors.append("commIssue")
    ret.errors = errors
    ret.canMonoTimes = canMonoTimes
    for ii in updated_messages:
      if ii in RADAR_A_MSGS:
        cpt = self.rcp.vl[ii]
        if (cpt['LongDist'] >= BOSCH_MAX_DIST) or (cpt['LongDist']==0) or (not cpt['Tracked']):
          self.valid_cnt[ii] = 0    # reset counter
        if cpt['Valid'] and (cpt['LongDist'] < BOSCH_MAX_DIST) and (cpt['LongDist'] > 0) and (cpt['ProbExist'] >= OBJECT_MIN_PROBABILITY):
          self.valid_cnt[ii] += 1
        else:
          self.valid_cnt[ii] = max(self.valid_cnt[ii] -1, 0)

        if (cpt['Valid'] or cpt['Tracked'])and (cpt['LongDist']>=MINX) and (cpt['LongDist'] <= MAXX) and \
            (cpt['Index'] == self.rcp.vl[ii+1]['Index2']) and (self.valid_cnt[ii] > 10) and \
            (cpt['ProbExist'] >= OBJECT_MIN_PROBABILITY) and (cpt['LatDist']>=MINY) and (cpt['LatDist']<=MAXY):
          if ii not in self.pts and ( cpt['Tracked']):
            self.pts[ii] = car.RadarState.RadarPoint.new_message()
            self.pts[ii].trackId = int((ii - 0x310)/3) 
          if ii in self.pts:
            self.pts[ii].dRel = cpt['LongDist']  # from front of car
            self.pts[ii].yRel = cpt['LatDist']  # in car frame's y axis, left is positive
            self.pts[ii].vRel = cpt['LongSpeed']
            self.pts[ii].aRel = cpt['LongAccel']
            self.pts[ii].yvRel = self.rcp.vl[ii+1]['LatSpeed']
            self.pts[ii].measured = bool(cpt['Meas'])
            self.pts[ii].dz = self.rcp.vl[ii+1]['dZ']
            self.pts[ii].movingState = self.rcp.vl[ii+1]['MovingState']
            self.pts[ii].length = self.rcp.vl[ii+1]['Length']
            self.pts[ii].obstacleProb = cpt['ProbObstacle']
            if self.rcp.vl[ii+1]['Class'] >= CLASS_MIN_PROBABILITY:
              self.pts[ii].objectClass = self.rcp.vl[ii+1]['Class']
              # for now we will use class 0- unknown stuff to show trucks
              # we will base that on being a class 1 and length of 2 (hoping they meant width not length, but as germans could not decide)
              # 0-unknown 1-four wheel vehicle 2-two wheel vehicle 3-pedestrian 4-construction element
              # going to 0-unknown 1-truck 2-car 3/4-motorcycle/bicycle 5 pedestrian - we have two bits so
              if self.pts[ii].objectClass == 0:
                self.pts[ii].objectClass = 1
              if (self.pts[ii].objectClass == 1) and ((self.pts[ii].length >= 1.8) or (1.6 < self.pts[ii].dz < 4.5)):
                self.pts[ii].objectClass = 0
              if self.pts[ii].objectClass == 4:
                self.pts[ii].objectClass = 1
            else:
              self.pts[ii].objectClass = 1
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
