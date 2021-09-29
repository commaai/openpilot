#!/usr/bin/env python3
import time
import numpy as np

from cereal import log
import cereal.messaging as messaging
from common.realtime import Ratekeeper, DT_MDL
from selfdrive.controls.lib.longcontrol import LongCtrlState
from selfdrive.controls.lib.longitudinal_planner import Planner


class Plant():
  messaging_initialized = False

  def __init__(self, lead_relevancy=False, speed=0.0, distance_lead=2.0,
               only_lead2=False, only_radar=False):
    self.rate = 1. / DT_MDL

    if not Plant.messaging_initialized:
      Plant.model = messaging.pub_sock('modelV2')
      Plant.controls_state = messaging.pub_sock('controlsState')
      Plant.car_state = messaging.pub_sock('carState')
      Plant.plan = messaging.sub_sock('longitudinalPlan')
      Plant.messaging_initialized = True

    self.v_lead_prev = 0.0

    self.distance = 0.
    self.speed = speed
    self.acceleration = 0.0

    # lead car
    self.distance_lead = distance_lead
    self.lead_relevancy = lead_relevancy
    self.only_lead2=only_lead2
    self.only_radar=only_radar

    self.rk = Ratekeeper(self.rate, print_delay_threshold=100.0)
    self.ts = 1. / self.rate
    time.sleep(1)
    self.sm = messaging.SubMaster(['longitudinalPlan'])

    from selfdrive.car.honda.values import CAR
    from selfdrive.car.honda.interface import CarInterface
    self.CP = CarInterface.get_params(CAR.CIVIC)
    self.planner = Planner(self.CP, init_v=self.speed)

  def current_time(self):
    return float(self.rk.frame) / self.rate

  def step(self, v_lead=0.0, prob=1.0, v_cruise=50.):
    # ******** publish a fake model going straight and fake calibration ********
    # note that this is worst case for MPC, since model will delay long mpc by one time step
    model = messaging.new_message('modelV2')
    control = messaging.new_message('controlsState')
    car_state = messaging.new_message('carState')
    self.v_lead_prev = v_lead

    if self.lead_relevancy:
      d_rel = np.maximum(0., self.distance_lead - self.distance)
      v_rel = v_lead - self.speed
    else:
      d_rel = 200.
      v_rel = 0.
      prob = 0.0

    leads = []
    for i in range(2):
      lead = log.ModelDataV2.LeadDataV3.new_message()
      lead.t = [float(i) for i in range(0,12,2)]
      lead.x = [float(d_rel + v_lead*i) for i in range(0,12,2)]
      lead.v = [float(v_lead) for i in range(0,12,2)]
      if self.only_lead2 and i == 0:
        lead.prob = float(0.0)
      else:
        lead.prob = float(prob)
      leads.append(lead)
    model.modelV2.leadsV3 = leads


    control.controlsState.longControlState = LongCtrlState.pid
    control.controlsState.vCruise = float(v_cruise * 3.6)
    car_state.carState.vEgo = float(self.speed)


    # ******** get controlsState messages for plotting ***
    sm = {'modelV2': model.modelV2,
          'carState': car_state.carState,
          'controlsState': control.controlsState}
    self.planner.update(sm, self.CP)
    self.speed = self.planner.v_desired
    self.acceleration = self.planner.a_desired
    fcw = self.planner.fcw
    self.distance_lead = self.distance_lead + v_lead * self.ts

    # ******** run the car ********
    #print(self.distance, speed)
    if self.speed <= 0:
      self.speed = 0
      self.acceleration = 0
    self.distance = self.distance + self.speed * self.ts

    # *** radar model ***
    if self.lead_relevancy:
      d_rel = np.maximum(0., self.distance_lead - self.distance)
      v_rel = v_lead - self.speed
    else:
      d_rel = 200.
      v_rel = 0.

    # print at 5hz
    if (self.rk.frame % (self.rate // 5)) == 0:
      print("%2.2f sec   %6.2f m  %6.2f m/s  %6.2f m/s2   lead_rel: %6.2f m  %6.2f m/s"
            % (self.current_time(), self.distance, self.speed, self.acceleration, d_rel, v_rel))


    # ******** update prevs ********
    self.rk.monitor_time()

    return {
      "distance": self.distance,
      "speed": self.speed,
      "acceleration": self.acceleration,
      "distance_lead": self.distance_lead,
      "fcw": fcw,
    }

# simple engage in standalone mode
def plant_thread():
  plant = Plant()
  while 1:
    plant.step()


if __name__ == "__main__":
  plant_thread()
