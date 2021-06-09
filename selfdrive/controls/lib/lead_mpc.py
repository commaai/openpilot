import os
import math

import numpy as np
from selfdrive.modeld.constants import T_IDXS
from selfdrive.swaglog import cloudlog
from common.realtime import sec_since_boot
from selfdrive.controls.lib.lead_mpc_lib import libmpc_py
from selfdrive.controls.lib.drive_helpers import MPC_COST_LONG
from selfdrive.controls.lib.drive_helpers import LON_MPC_N

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

  def set_cur_state(self, v, a):
    v_safe = max(v, 1e-3)
    a_safe = a
    self.cur_state[0].v_ego = v_safe
    self.cur_state[0].a_ego = a_safe

  def update(self, carstate, model, v_cruise):
    lead = model.leads[self.lead_id]
    v_ego = carstate.vEgo

    # Setup current mpc state
    self.cur_state[0].x_ego = 0.0

    if lead is not None:
      self.status = lead.prob > 0.5
      x_lead = lead.x[0]
      v_lead = max(0.0, lead.v[0])
      a_lead = lead.a[0]

      if (v_lead < 0.1 or -a_lead / 2.0 > v_lead):
        v_lead = 0.0
        a_lead = 0.0

      self.cur_state[0].x_l = x_lead
      self.cur_state[0].v_l = v_lead

    # Calculate mpc
    t = sec_since_boot()
    model_t = [0., 2., 4., 6., 8., 10.]
    # -2m to since measurements are from the camera
    lead_x_interp = np.interp(T_IDXS[:LON_MPC_N+1], model_t, lead.x) - 2.0
    lead_v_interp = np.interp(T_IDXS[:LON_MPC_N+1], model_t, lead.v)
    self.n_its = self.libmpc.run_mpc(self.cur_state, self.mpc_solution,
                                     list(lead_x_interp), list(lead_v_interp))

    # Reset if NaN or goes through lead car
    crashing = any(lead - ego < -50 for (lead, ego) in zip(lead_x_interp, self.mpc_solution[0].x_ego))
    nans = any(math.isnan(x) for x in self.mpc_solution[0].v_ego)
    backwards = min(self.mpc_solution[0].v_ego) < -0.01

    if (backwards or crashing or nans):
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Longitudinal mpc %d reset - backwards: %s crashing: %s nan: %s" % (
                          self.lead_id, backwards, crashing, nans))

      self.reset_mpc()
      self.cur_state[0].v_ego = v_ego
