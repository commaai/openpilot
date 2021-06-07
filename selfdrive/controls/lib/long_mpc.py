import numpy as np
import math

from selfdrive.swaglog import cloudlog
from common.realtime import sec_since_boot
from selfdrive.controls.lib.longitudinal_mpc_lib import libmpc_py


class LongitudinalMpc():
  def __init__(self):
    self.reset_mpc()
    self.last_cloudlog_t = 0.0
    self.ts = list(range(10))


  def reset_mpc(self):
    self.libmpc = libmpc_py.libmpc
    self.libmpc.init(0.0, 1.0, 1.0, 10.0)

    self.mpc_solution = libmpc_py.ffi.new("log_t *")
    self.cur_state = libmpc_py.ffi.new("state_t *")

    self.cur_state[0].x_ego = 0
    self.cur_state[0].v_ego = 0
    self.cur_state[0].a_ego = 0
    self.min_a = -1.2
    self.max_a = 1.2

  def set_accel_limits(self, min_a, max_a):
    self.min_a = min_a
    self.max_a = max_a

  def set_cur_state(self, v, a):
    self.cur_state[0].x_ego = 0.0
    self.cur_state[0].v_ego = v
    self.cur_state[0].a_ego = a

  def update(self, carstate, model, v_cruise):
    v_cruise_clipped = np.clip(v_cruise, self.cur_state[0].v_ego - 10., self.cur_state[0].v_ego + 10.0)
    mpc_t = [0.0, .2, .4, .6, .8] + list(np.arange(1.0, 10.1, .6))
    poss = v_cruise_clipped * np.array(mpc_t)
    speeds = v_cruise_clipped * np.ones(len(mpc_t))
    accels = np.zeros(len(mpc_t))

    # Calculate mpc
    self.libmpc.run_mpc(self.cur_state, self.mpc_solution,
                        list(poss), list(speeds), list(accels),
                        self.min_a, self.max_a)

    # Reset if NaN or goes through lead car
    nans = any(math.isnan(x) for x in self.mpc_solution[0].v_ego)

    t = sec_since_boot()
    if nans:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Longitudinal model mpc reset - nans")
      self.reset_mpc()
