import numpy as np
import math

from selfdrive.swaglog import cloudlog
from common.realtime import sec_since_boot
from selfdrive.controls.lib.longitudinal_mpc_lib import libmpc_py
from selfdrive.controls.lib.drive_helpers import LON_MPC_N
from selfdrive.modeld.constants import T_IDXS


class LongitudinalMpc():
  def __init__(self):
    self.reset_mpc()
    self.last_cloudlog_t = 0.0
    self.ts = list(range(10))
    self.status = True
    self.min_a = -1.2
    self.max_a = 1.2


  def reset_mpc(self):
    self.libmpc = libmpc_py.libmpc
    self.libmpc.init(0.0, 1.0, 0.0, 50.0, 10000.0)

    self.mpc_solution = libmpc_py.ffi.new("log_t *")
    self.cur_state = libmpc_py.ffi.new("state_t *")

    self.cur_state[0].x_ego = 0
    self.cur_state[0].v_ego = 0
    self.cur_state[0].a_ego = 0

    self.v_solution = [0.0 for i in range(len(T_IDXS))]
    self.a_solution = [0.0 for i in range(len(T_IDXS))]
    self.j_solution = [0.0 for i in range(len(T_IDXS)-1)]

  def set_accel_limits(self, min_a, max_a):
    self.min_a = min_a
    self.max_a = max_a

  def set_cur_state(self, v, a):
    v_safe = max(v, 1e-2)
    a_safe = min(a, self.max_a - 1e-2)
    a_safe = max(a_safe, self.min_a + 1e-2)
    self.cur_state[0].x_ego = 0.0
    self.cur_state[0].v_ego = v_safe
    self.cur_state[0].a_ego = a_safe

  def update(self, carstate, model, v_cruise):
    v_cruise_clipped = np.clip(v_cruise, self.cur_state[0].v_ego - 10., self.cur_state[0].v_ego + 10.0)
    poss = v_cruise_clipped * np.array(T_IDXS[:LON_MPC_N+1])
    speeds = v_cruise_clipped * np.ones(LON_MPC_N+1)
    accels = np.zeros(LON_MPC_N+1)

    # Calculate mpc
    self.libmpc.run_mpc(self.cur_state, self.mpc_solution,
                        list(poss), list(speeds), list(accels),
                        self.min_a, self.max_a)

    self.v_solution = list(self.mpc_solution.v_ego)
    self.a_solution = list(self.mpc_solution.a_ego)
    self.j_solution = list(self.mpc_solution.j_ego)

    # Reset if NaN or goes through lead car
    nans = any(math.isnan(x) for x in self.mpc_solution[0].v_ego)

    t = sec_since_boot()
    if nans:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Longitudinal model mpc reset - nans")
      self.reset_mpc()
