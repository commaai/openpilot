import numpy as np

from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from opendbc.car.structs import car
from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.radard import _LEAD_ACCEL_TAU


class Maneuver:
  def __init__(self, title, duration, **kwargs):
    # scenario
    self.title = title
    initial_v_ego = kwargs.get("initial_speed", 0.0)
    initial_x_lead = kwargs.get("initial_distance_lead", 200.0)

    breakpoints = kwargs.get("breakpoints", [0.0, duration])
    v_lead_values = kwargs.get("speed_lead_values", [0.0] * len(breakpoints))
    v_cruise_values = kwargs.get("cruise_values", [50.0] * len(breakpoints))

    self.enabled = kwargs.get("enabled", True)
    self.personality = kwargs.get("personality", 0)
    self.force_decel = kwargs.get("force_decel", False)

    # eval
    self.ensure_start = kwargs.get("ensure_start", False)

    # perception
    self.lead_relevancy = kwargs.get("lead_relevancy", False)
    self.only_lead2 = kwargs.get("only_lead2", False)
    only_radar = kwargs.get("only_radar", False)
    prob_lead_values = kwargs.get("prob_lead_values", [1.0] * len(breakpoints))

    # ego state, to be simulated
    n_steps = round(duration / DT_MDL)
    self.t = np.arange(n_steps + 1) * DT_MDL
    self.x_ego = np.zeros_like(self.t)
    self.v_ego = np.zeros_like(self.t)
    self.a_ego = np.zeros_like(self.t)
    self.v_ego[0] = initial_v_ego

    # lead state, known in advance
    self.v_lead = np.interp(self.t, breakpoints, v_lead_values)
    self.a_lead = np.diff(self.v_lead, prepend=0.0) / DT_MDL
    self.x_lead = np.cumsum(np.r_[initial_x_lead, self.v_lead[:-1] * DT_MDL])
    self.prob_lead = np.interp(self.t, breakpoints, prob_lead_values)
    self.lead_present = np.logical_and(self.lead_relevancy, np.logical_or(only_radar, self.prob_lead > 0.5))
    self.v_cruise = np.interp(self.t, breakpoints, v_cruise_values)

  def _make_messages(self, i):
    v_ego = float(self.v_ego[i])
    v_lead = float(self.v_lead[i])
    prob = float(self.prob_lead[i]) if self.lead_relevancy else 0.0
    lead = log.RadarState.LeadData(
      dRel=max(0., float(self.x_lead[i] - self.x_ego[i])) if self.lead_relevancy else 200.,
      vRel=v_lead - v_ego if self.lead_relevancy else 0.,
      vLead=v_lead, vLeadK=v_lead, aLeadK=float(self.a_lead[i]), aLeadTau=_LEAD_ACCEL_TAU,
      present=bool(self.lead_present[i]), modelProb=prob, radar=True,
    )
    radar = log.RadarState(leadTwo=lead)
    if not self.only_lead2:
      radar.leadOne = lead

    # Model requests more acceleration so the lead policy must handle slowdowns.
    model = log.ModelDataV2()
    model.action.desiredAcceleration = float(self.a_ego[i] + 0.5)
    return {
      'radarState': radar,
      'modelV2': model,
      'carState': car.CarState(vEgo=v_ego, standstill=v_ego < 0.01, vCruise=float(self.v_cruise[i] * CV.MS_TO_KPH)),
      'controlsState': log.ControlsState(longControlState=LongCtrlState.pid if self.enabled else LongCtrlState.off, forceDecel=self.force_decel),
      'selfdriveState': log.SelfdriveState(personality=self.personality),
      'vehicleParameters': log.VehicleParameters(),
    }

  def _step(self, i):
    self.planner.update(self._make_messages(i))
    a_ego = self.planner.output_a_target
    if self.planner.output_should_stop:
      a_ego = min(-0.5, a_ego)
    v_ego = self.v_ego[i] + a_ego * DT_MDL
    if v_ego <= 0.:
      v_ego = a_ego = 0.
    self.v_ego[i + 1] = v_ego
    self.a_ego[i + 1] = a_ego
    self.x_ego[i + 1] = self.x_ego[i] + v_ego * DT_MDL

  def evaluate(self):
    CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC)
    self.planner = LongitudinalPlanner(CP, init_v=self.v_ego[0])
    for i in range(len(self.t) - 1):
      self._step(i)

    d_rel = self.x_lead - self.x_ego

    valid = True
    if np.any(d_rel[self.lead_present] < 0.4):
      print('Crashed!!!!')
      valid = False

    if self.ensure_start and self.lead_relevancy:
      not_starting_since = None
      not_starting = (self.v_lead > self.v_ego) & (self.a_ego < 1e-3)
      for now, waiting in zip(self.t, not_starting, strict=True):
        if not waiting:
          not_starting_since = None
        elif not_starting_since is None:
          not_starting_since = now
        elif now - not_starting_since > 0.5:
          print('LongitudinalPlanner not starting!')
          valid = False
          break

    if self.force_decel and self.v_ego[-1] > 0.1 and self.a_ego[-1] > -0.04:
      print('Not stopping with force decel')
      valid = False

    print('maneuver end', valid)
    logs = np.column_stack((self.t, self.x_ego, self.x_lead, self.v_ego, self.v_lead, self.a_ego, d_rel))
    return valid, logs
