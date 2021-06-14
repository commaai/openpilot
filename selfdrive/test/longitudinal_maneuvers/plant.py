#!/usr/bin/env python3
import os
import time
import numpy as np

from opendbc import DBC_PATH

from cereal import log
from common.realtime import Ratekeeper
import cereal.messaging as messaging
from selfdrive.car.honda.values import CAR

from selfdrive.car.honda.interface import CarInterface
from selfdrive.controls.lib.longcontrol import LongCtrlState
from opendbc.can.dbc import dbc
from selfdrive.modeld.constants import T_IDXS
honda = dbc(os.path.join(DBC_PATH, "honda_civic_touring_2016_can_generated.dbc"))

# Trick: set 0x201 (interceptor) in fingerprints for gas is controlled like if there was an interceptor
CP = CarInterface.get_params(CAR.CIVIC, {0: {0x201: 6}, 1: {}, 2: {}, 3: {}})


class Plant():
  messaging_initialized = False

  def __init__(self, lead_relevancy=False, rate=20, speed=0.0, distance_lead=2.0):
    self.rate = rate

    if not Plant.messaging_initialized:
      Plant.model = messaging.pub_sock('modelV2')
      Plant.controls_state = messaging.pub_sock('controlsState')
      Plant.car_state = messaging.pub_sock('carState')
      Plant.plan = messaging.sub_sock('longitudinalPlan')
      Plant.messaging_initialized = True

    self.frame = 0

    self.distance = 0.
    self.speed = speed
    self.steer_error, self.brake_error, self.steer_not_allowed = 0, 0, 0
    self.cruise_setting = 0

    self.steer_torque, self.v_cruise, self.acc_status = 0, 0, 0  # v_cruise is reported from can, not the one used for controls

    self.lead_relevancy = lead_relevancy

    # lead car
    self.distance_lead = distance_lead

    self.rk = Ratekeeper(rate, print_delay_threshold=100.0)
    self.ts = 1./rate
    time.sleep(1)
    self.sm = messaging.SubMaster(['longitudinalPlan'], poll=['longitudinalPlan'])

  def close(self):
    Plant.model.close()

  def current_time(self):
    return float(self.rk.frame) / self.rate

  def step(self, v_lead=0.0, cruise_buttons=None, grade=0.0, publish_model=True):
    # ******** publish a fake model going straight and fake calibration ********
    # note that this is worst case for MPC, since model will delay long mpc by one time step
    md = messaging.new_message('modelV2')
    control = messaging.new_message('controlsState')
    car_state = messaging.new_message('carState')
    md.modelV2.frameId = 0

    if self.lead_relevancy:
      d_rel = np.maximum(0., self.distance_lead - self.distance)
      v_rel = v_lead[0] - self.speed
      prob = 1.0
    else:
      d_rel = 200.
      v_rel = 0.
      prob = 0.0

    md.modelV2.position.t = [float(t) for t in T_IDXS]
    lead = log.ModelDataV2.LeadDataV3.new_message()
    lead_x = [float(d_rel),]
    for i in range(5):
      lead_x.append(lead_x[-1] + 2.0*v_lead[i])
    lead.x = [float(x) for x in lead_x]
    lead.y = [0.0 for i in range(0,12,2)]
    lead.v = [float(v) for v in v_lead]
    lead.a = [0.0 for i in range(0,12,2)]

    lead.prob = prob
    md.modelV2.leads = [lead, lead]


    control.controlsState.longControlState = LongCtrlState.pid
    control.controlsState.vCruise = 130
    car_state.carState.vEgo = self.speed
    Plant.model.send(md.to_bytes())
    Plant.controls_state.send(control.to_bytes())
    Plant.car_state.send(car_state.to_bytes())


    # ******** get controlsState messages for plotting ***
    self.sm.update()
    while True:
      time.sleep(0.01)
      if self.sm.updated['longitudinalPlan']:
        plan = self.sm['longitudinalPlan']
        self.speed = plan.speeds[5]
        self.acceleration = plan.accels[5]
        fcw = plan.fcw
        break


    self.distance_lead = self.distance_lead + v_lead[0] * self.ts

    # ******** run the car ********
    #print(self.distance, speed)
    if self.speed <= 0:
      self.speed = 0
      self.acceleration = 0
    self.distance = self.distance + self.speed * self.ts

    # *** radar model ***
    if self.lead_relevancy:
      d_rel = np.maximum(0., self.distance_lead - self.distance)
      v_rel = v_lead[0] - self.speed
    else:
      d_rel = 200.
      v_rel = 0.

    # print at 5hz
    if (self.frame % (self.rate//5)) == 0:
      print("%2.2f sec   %6.2f m  %6.2f m/s  %6.2f m/s2   lead_rel: %6.2f m  %6.2f m/s"
            % (self.current_time(), self.distance, self.speed, self.acceleration, d_rel, v_rel))


    # ******** update prevs ********
    self.frame += 1
    self.rk.monitor_time()

    return {
      "distance": self.distance,
      "speed": self.speed,
      "acceleration": self.acceleration,
      "distance_lead": self.distance_lead,
      "fcw": fcw,
    }

# simple engage in standalone mode
def plant_thread(rate=20):
  plant = Plant(rate)
  while 1:
    plant.step()


if __name__ == "__main__":
  plant_thread()
