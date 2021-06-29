import os
import math

from selfdrive.swaglog import cloudlog
from common.realtime import sec_since_boot
from selfdrive.controls.lib.radar_helpers import _LEAD_ACCEL_TAU
from selfdrive.controls.lib.drive_helpers import MPC_COST_LONG
from selfdrive.controls.lib.lead_mpc_lib import libmpc_py

LOG_MPC = os.environ.get('LOG_MPC', False)


class LeadMpc():
  def __init__(self, lead_id):
    self.lead_id = lead_id

    self.reset_mpc()
    self.status = False

    self.last_cloudlog_t = 0.0
    self.n_its = 0

  def reset_mpc(self):
    ffi, self.libmpc = libmpc_py.get_libmpc(self.lead_id)
    self.libmpc.init(MPC_COST_LONG.TTC, MPC_COST_LONG.DISTANCE,
                     MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)

    self.mpc_solution = ffi.new("log_t *")
    self.cur_state = ffi.new("state_t *")
    self.cur_state[0].v_ego = 0
    self.cur_state[0].a_ego = 0
    self.a_lead_tau = _LEAD_ACCEL_TAU

  def set_cur_state(self, v, a):
    v_safe = max(v, 1e-3)
    a_safe = a
    self.cur_state[0].v_ego = v_safe
    self.cur_state[0].a_ego = a_safe

  def update(self, carstate, radarstate, v_cruise):
    if self.lead_id == 0:
      lead = radarstate.leadOne
    else:
      lead = radarstate.leadOne
    x_lead = lead.dRel
    v_lead = max(0.0, lead.vLead)
    a_lead = lead.aLeadK
    self.a_lead_tau = lead.aLeadTau
    self.status = lead.modelProb > 0.5
    if (v_lead < 0.1):
      v_lead = 0.0

    v_ego = carstate.vEgo

    # Setup current mpc state
    self.cur_state[0].x_ego = 0.0
    self.cur_state[0].x_l = x_lead
    self.cur_state[0].v_l = v_lead

    # Calculate mpc
    t = sec_since_boot()
    self.n_its = self.libmpc.run_mpc(self.cur_state, self.mpc_solution,
                                     self.a_lead_tau, a_lead)

    # Reset if NaN or goes through lead car
    crashing = any(lead - ego < -50 for (lead, ego) in zip(self.mpc_solution[0].x_l, self.mpc_solution[0].x_ego))
    nans = any(math.isnan(x) for x in self.mpc_solution[0].v_ego)
    backwards = min(self.mpc_solution[0].v_ego) < -0.01

    if (backwards or crashing or nans):
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Longitudinal mpc %d reset - backwards: %s crashing: %s nan: %s" % (
                          self.lead_id, backwards, crashing, nans))

      self.reset_mpc()
      self.cur_state[0].v_ego = v_ego
