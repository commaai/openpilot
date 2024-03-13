#!/usr/bin/env python3
import time
import numpy as np

from cereal import log
import cereal.messaging as messaging
from openpilot.common.realtime import Ratekeeper, DT_MDL
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.radard import _LEAD_ACCEL_TAU


class Plant:
  messaging_initialized = False

  def __init__(self, lead_relevancy=False, speed=0.0, distance_lead=2.0,
               enabled=True, only_lead2=False, only_radar=False, e2e=False, personality=0, force_decel=False):
    self.rate = 1. / DT_MDL

    if not Plant.messaging_initialized:
      Plant.radar = messaging.pub_sock('radarState')
      Plant.controls_state = messaging.pub_sock('controlsState')
      Plant.car_state = messaging.pub_sock('carState')
      Plant.plan = messaging.sub_sock('longitudinalPlan')
      Plant.messaging_initialized = True

    self.v_lead_prev = 0.0

    self.distance = 0.
    self.speed = speed
    self.acceleration = 0.0
    self.speeds = []

    # lead car
    self.lead_relevancy = lead_relevancy
    self.distance_lead = distance_lead
    self.enabled = enabled
    self.only_lead2 = only_lead2
    self.only_radar = only_radar
    self.e2e = e2e
    self.personality = personality
    self.force_decel = force_decel

    self.rk = Ratekeeper(self.rate, print_delay_threshold=100.0)
    self.ts = 1. / self.rate
    time.sleep(0.1)
    self.sm = messaging.SubMaster(['longitudinalPlan'])

    from openpilot.selfdrive.car.honda.values import CAR
    from openpilot.selfdrive.car.honda.interface import CarInterface

    self.planner = LongitudinalPlanner(CarInterface.get_non_essential_params(CAR.CIVIC), init_v=self.speed)

  @property
  def current_time(self):
    return float(self.rk.frame) / self.rate

  def step(self, v_lead=0.0, prob=1.0, v_cruise=50.):
    # ******** publish a fake model going straight and fake calibration ********
    # note that this is worst case for MPC, since model will delay long mpc by one time step
    radar = messaging.new_message('radarState')
    control = messaging.new_message('controlsState')
    car_state = messaging.new_message('carState')
    model = messaging.new_message('modelV2')
    a_lead = (v_lead - self.v_lead_prev)/self.ts
    self.v_lead_prev = v_lead

    if self.lead_relevancy:
      d_rel = np.maximum(0., self.distance_lead - self.distance)
      v_rel = v_lead - self.speed
      if self.only_radar:
        status = True
      elif prob > .5:
        status = True
      else:
        status = False
    else:
      d_rel = 200.
      v_rel = 0.
      prob = 0.0
      status = False

    lead = log.RadarState.LeadData.new_message()
    lead.dRel = float(d_rel)
    lead.yRel = float(0.0)
    lead.vRel = float(v_rel)
    lead.aRel = float(a_lead - self.acceleration)
    lead.vLead = float(v_lead)
    lead.vLeadK = float(v_lead)
    lead.aLeadK = float(a_lead)
    # TODO use real radard logic for this
    lead.aLeadTau = float(_LEAD_ACCEL_TAU)
    lead.status = status
    lead.modelProb = float(prob)
    if not self.only_lead2:
      radar.radarState.leadOne = lead
    radar.radarState.leadTwo = lead

    # Simulate model predicting slightly faster speed
    # this is to ensure lead policy is effective when model
    # does not predict slowdown in e2e mode
    position = log.XYZTData.new_message()
    position.x = [float(x) for x in (self.speed + 0.5) * np.array(ModelConstants.T_IDXS)]
    model.modelV2.position = position
    velocity = log.XYZTData.new_message()
    velocity.x = [float(x) for x in (self.speed + 0.5) * np.ones_like(ModelConstants.T_IDXS)]
    model.modelV2.velocity = velocity
    acceleration = log.XYZTData.new_message()
    acceleration.x = [float(x) for x in np.zeros_like(ModelConstants.T_IDXS)]
    model.modelV2.acceleration = acceleration

    control.controlsState.longControlState = LongCtrlState.pid if self.enabled else LongCtrlState.off
    control.controlsState.vCruise = float(v_cruise * 3.6)
    control.controlsState.experimentalMode = self.e2e
    control.controlsState.personality = self.personality
    control.controlsState.forceDecel = self.force_decel
    car_state.carState.vEgo = float(self.speed)
    car_state.carState.standstill = self.speed < 0.01

    # ******** get controlsState messages for plotting ***
    sm = {'radarState': radar.radarState,
          'carState': car_state.carState,
          'controlsState': control.controlsState,
          'modelV2': model.modelV2}
    self.planner.update(sm)
    self.speed = self.planner.v_desired_filter.x
    self.acceleration = self.planner.a_desired
    self.speeds = self.planner.v_desired_trajectory.tolist()
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
    # if (self.rk.frame % (self.rate // 5)) == 0:
    #   print("%2.2f sec   %6.2f m  %6.2f m/s  %6.2f m/s2   lead_rel: %6.2f m  %6.2f m/s"
    #         % (self.current_time, self.distance, self.speed, self.acceleration, d_rel, v_rel))


    # ******** update prevs ********
    self.rk.monitor_time()

    return {
      "distance": self.distance,
      "speed": self.speed,
      "acceleration": self.acceleration,
      "speeds": self.speeds,
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
