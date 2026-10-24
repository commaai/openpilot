#!/usr/bin/env python3
import numpy as np

from openpilot.cereal import log
import openpilot.cereal.messaging as messaging
from openpilot.common.realtime import Ratekeeper, DT_MDL
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.radard import _LEAD_ACCEL_TAU


class Plant:
  def __init__(self, lead_relevancy=False, speed=0.0, distance_lead=2.0,
               enabled=True, only_lead2=False, only_radar=False, personality=0, force_decel=False):
    self.rate = 1. / DT_MDL

    self.v_lead_prev = 0.0

    self.distance = 0.
    self.speed = speed
    self.should_stop = False
    self.acceleration = 0.0

    # lead car
    self.lead_relevancy = lead_relevancy
    self.distance_lead = distance_lead
    self.enabled = enabled
    self.only_lead2 = only_lead2
    self.only_radar = only_radar
    self.personality = personality
    self.force_decel = force_decel

    self.rk = Ratekeeper(self.rate, print_delay_threshold=100.0)
    self.ts = 1. / self.rate

    from opendbc.car.honda.values import CAR
    from opendbc.car.honda.interface import CarInterface

    self.planner = LongitudinalPlanner(CarInterface.get_non_essential_params(CAR.HONDA_CIVIC), init_v=self.speed)

  @property
  def current_time(self):
    return float(self.rk.frame) / self.rate

  def step(self, v_lead=0.0, prob_lead=1.0, v_cruise=50.):
    radar = messaging.new_message('radarState')
    control = messaging.new_message('controlsState')
    ss = messaging.new_message('selfdriveState')
    car_state = messaging.new_message('carState')
    model = messaging.new_message('modelV2')
    a_lead = (v_lead - self.v_lead_prev)/self.ts
    self.v_lead_prev = v_lead

    if self.lead_relevancy:
      d_rel = np.maximum(0., self.distance_lead - self.distance)
      v_rel = v_lead - self.speed
      if self.only_radar:
        status = True
      elif prob_lead > .5:
        status = True
      else:
        status = False
    else:
      d_rel = 200.
      v_rel = 0.
      prob_lead = 0.0
      status = False

    lead = log.RadarState.LeadData.new_message()
    lead.dRel = float(d_rel)
    lead.yRel = 0.0
    lead.vRel = float(v_rel)
    lead.vLead = float(v_lead)
    lead.vLeadK = float(v_lead)
    lead.aLeadK = float(a_lead)
    # TODO use real radard logic for this
    lead.aLeadTau = float(_LEAD_ACCEL_TAU)
    lead.present = status
    lead.modelProb = float(prob_lead)
    lead.radar = True
    if not self.only_lead2:
      radar.radarState.leadOne = lead
    radar.radarState.leadTwo = lead

    # Model requests more acceleration so the lead policy must handle slowdowns.
    model.modelV2.action.desiredAcceleration = float(self.acceleration + 0.5)

    control.controlsState.longControlState = LongCtrlState.pid if self.enabled else LongCtrlState.off
    ss.selfdriveState.personality = self.personality
    control.controlsState.forceDecel = self.force_decel
    car_state.carState.vEgo = float(self.speed)
    car_state.carState.standstill = bool(self.speed < 0.01)
    car_state.carState.vCruise = float(v_cruise * 3.6)

    sm = {'radarState': radar.radarState,
          'carState': car_state.carState,
          'controlsState': control.controlsState,
          'selfdriveState': ss.selfdriveState,
          'modelV2': model.modelV2}
    self.planner.update(sm)
    self.acceleration = self.planner.output_a_target
    if self.planner.output_should_stop:
      self.acceleration = min(-0.5, self.acceleration)
    self.speed = self.speed + self.acceleration * self.ts
    self.should_stop = self.planner.output_should_stop
    fcw = self.planner.fcw
    self.distance_lead = self.distance_lead + v_lead * self.ts

    # ******** run the car ********
    if self.speed <= 0:
      self.speed = 0
      self.acceleration = 0
    self.distance = self.distance + self.speed * self.ts

    # ******** update prevs ********
    self.rk.monitor_time()

    return {
      "distance": self.distance,
      "speed": self.speed,
      "acceleration": self.acceleration,
      "should_stop": self.should_stop,
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
