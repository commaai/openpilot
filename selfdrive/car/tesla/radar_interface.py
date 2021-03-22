#!/usr/bin/env python3
from cereal import car
from opendbc.can.parser import CANParser
from selfdrive.car.tesla.values import DBC
from selfdrive.car.interfaces import RadarInterfaceBase


BOSCH_MAX_DIST = 250. #max distance for radar
#use these for tracks (5 tracks)
#RADAR_A_MSGS = list(range(0x371, 0x37F , 3))
#RADAR_B_MSGS = list(range(0x372, 0x37F, 3))
#use these for point cloud  (32 points)
RADAR_A_MSGS = list(range(0x310, 0x36F , 3))
RADAR_B_MSGS = list(range(0x311, 0x36F, 3))
OBJECT_MIN_PROBABILITY = 50.
CLASS_MIN_PROBABILITY = 50.
RADAR_MESSAGE_FREQUENCY = 0.050 * 1e9 #time in ns, radar sends data at 0.06 s
VALID_MESSAGE_COUNT_THRESHOLD = 4


# Tesla Bosch firmware has 32 objects in all objects or a selected set of the 5 we should look at
# definetly switch to all objects when calibrating but most likely use select set of 5 for normal use
USE_ALL_OBJECTS = True

def _create_tesla_bosch_radard_can_parser(car_fingerprint):
  msg_a_n = len(RADAR_A_MSGS)
  msg_b_n = len(RADAR_B_MSGS)

  signals = list(zip(['LongDist'] * msg_a_n +  ['LatDist'] * msg_a_n +
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
                [0] * msg_b_n + [0.] * msg_b_n + [0.] * msg_b_n +[0.] * msg_b_n + [0]* msg_b_n))

  checks = list(zip(RADAR_A_MSGS + RADAR_B_MSGS, [6]*(msg_a_n + msg_b_n)))


  return CANParser(DBC[car_fingerprint]['radar'], signals, checks, 1)


class RadarInterface(RadarInterfaceBase):

  def __init__(self,CP):
    # radar
    self.pts = {}
    self.delay = 0 
    self.TRACK_LEFT_LANE = True
    self.TRACK_RIGHT_LANE = True
    self.updated_messages = set()
    self.canErrorCounter = 0
    self.track_id = 0
    self.radar_fault = False
    self.radar_wrong_config = False
    self.radar_off_can = CP.radarOffCan
    self.radar_ts = CP.radarTimeStep
    if not self.radar_off_can:
      self.pts = {}
      self.valid_cnt = {key: 0 for key in RADAR_A_MSGS}
      self.rcp = _create_tesla_bosch_radard_can_parser(CP.carFingerprint)
      self.trackId = 1
      self.trigger_start_msg = RADAR_A_MSGS[0]
      self.trigger_end_msg = RADAR_B_MSGS[-1]

    self.delay = int(round(0.1 / CP.radarTimeStep))   # 0.1s delay of radar

  def update(self, can_strings):
    # radard at 20Hz and return no points
    if self.radar_off_can:
      return super().update(None)

    if can_strings is not None:
      vls = self.rcp.update_strings(can_strings)
      self.updated_messages.update(vls)

    if self.trigger_start_msg not in self.updated_messages:
      return None

    if self.trigger_end_msg not in self.updated_messages:
      return None

    rr = self._update(self.updated_messages)
    self.updated_messages.clear()
    return rr


  def _update(self, updated_messages):
    ret = car.RadarData.new_message()
    for message in sorted(updated_messages):
      if not(message in RADAR_A_MSGS):
        if message in self.pts:
          del self.pts[message]
        continue
      cpt = self.rcp.vl[message]
      if not (message + 1 in updated_messages):
        continue
      cpt2 = self.rcp.vl[message+1]
      # ensure the two messages are from the same frame reading
      if cpt['Index'] != cpt2['Index2']:
        continue
      if (cpt['LongDist'] >= BOSCH_MAX_DIST) or (cpt['LongDist']==0) or (not cpt['Tracked']) or (not cpt['Valid']):
        self.valid_cnt[message] = 0    # reset counter
        if message in self.pts:
          del self.pts[message]
      elif cpt['Valid'] and (cpt['LongDist'] < BOSCH_MAX_DIST) and (cpt['LongDist'] > 0) and (cpt['ProbExist'] >= OBJECT_MIN_PROBABILITY):
        self.valid_cnt[message] += 1
      else:
        self.valid_cnt[message] = max(self.valid_cnt[message] -20, 0)
        if (self.valid_cnt[message]==0) and (message in self.pts):
          del self.pts[message]

      # radar point only valid if it's a valid measurement and score is above 50
      # bosch radar data needs to match Index and Index2 for validity
      # also for now ignore construction elements
      if (cpt['Valid'] or cpt['Tracked']) and (cpt['LongDist']>0) and (cpt['LongDist'] < BOSCH_MAX_DIST) and \
          (self.valid_cnt[message] > VALID_MESSAGE_COUNT_THRESHOLD) and (cpt['ProbExist'] >= OBJECT_MIN_PROBABILITY): 
        if message not in self.pts and ( cpt['Tracked']):
          self.pts[message] = car.RadarData.RadarPoint.new_message()
          self.pts[message].trackId = self.trackId 
          self.trackId = (self.trackId + 1) & 0xFFFFFFFFFFFFFFFF
          if self.trackId ==0:
            self.trackId = 1
        if message in self.pts:
          self.pts[message].dRel = cpt['LongDist']  # from front of car
          self.pts[message].yRel = cpt['LatDist']  # in car frame's y axis, left is positive
          self.pts[message].vRel = cpt['LongSpeed']
          self.pts[message].aRel = cpt['LongAccel']
          self.pts[message].yvRel = cpt2['LatSpeed']
          self.pts[message].measured = bool(cpt['Meas'])

    ret.points = list(self.pts.values())
    errors = []
    if not self.rcp.can_valid:
      errors.append("canError")
      self.canErrorCounter += 1
    else:
      self.canErrorCounter = 0
    #BB: Only trigger canError for 3 consecutive errors
    if self.canErrorCounter > 9:
      ret.errors = errors
    else:
      ret.errors = []
    return ret

# radar_interface standalone tester
if __name__ == "__main__":
  CP = None
  RI = RadarInterface(CP)
  while 1:
    ret = RI.update(can_strings = None)
    print(chr(27) + "[2J")
    print(ret)
