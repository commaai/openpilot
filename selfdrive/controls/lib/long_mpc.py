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
    self.libmpc.init(0.0, 1.0, 1.0, .0, 10.0)

    self.mpc_solution = libmpc_py.ffi.new("log_t *")
    self.cur_state = libmpc_py.ffi.new("state_t *")

    self.cur_state[0].x_ego = 0
    self.cur_state[0].v_ego = 0
    self.cur_state[0].a_ego = 0

  def set_cur_state(self, v, a):
    self.cur_state[0].x_ego = 0.0
    self.cur_state[0].v_ego = v
    self.cur_state[0].a_ego = a

  def update(self, carstate, model, v_cruise):
    v_ego = carstate.v_ego
    v_cruise_clipped = np.clip(v_cruise, v_ego - 10.0, v_ego + 5.0)
    poss = v_cruise_clipped * np.arange(0.,10.,1.0)
    speeds = v_cruise_clipped * np.ones(10)
    accels = np.zeros(10)
    x_poly = list(map(float, np.polyfit(self.ts, poss, 3)))
    v_poly = list(map(float, np.polyfit(self.ts, speeds, 3)))
    a_poly = list(map(float, np.polyfit(self.ts, accels, 3)))

    # Calculate mpc
    self.libmpc.run_mpc(self.cur_state, self.mpc_solution, x_poly, v_poly, a_poly)

    # Reset if NaN or goes through lead car
    nans = any(math.isnan(x) for x in self.mpc_solution[0].v_ego)

    t = sec_since_boot()
    if nans:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Longitudinal model mpc reset - nans")
      self.reset_mpc()
