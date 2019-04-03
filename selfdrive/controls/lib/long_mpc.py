import numpy as np

import selfdrive.messaging as messaging
from selfdrive.swaglog import cloudlog
from common.realtime import sec_since_boot
from selfdrive.controls.lib.radar_helpers import _LEAD_ACCEL_TAU
from selfdrive.controls.lib.longitudinal_mpc import libmpc_py
from selfdrive.controls.lib.drive_helpers import MPC_COST_LONG
from scipy import interpolate
import math


class LongitudinalMpc(object):
  def __init__(self, mpc_id, live_longitudinal_mpc):
    self.live_longitudinal_mpc = live_longitudinal_mpc
    self.mpc_id = mpc_id

    self.setup_mpc()
    self.v_mpc = 0.0
    self.v_mpc_future = 0.0
    self.a_mpc = 0.0
    self.v_cruise = 0.0
    self.prev_lead_status = False
    self.prev_lead_x = 0.0
    self.new_lead = False
    self.lastTR = 2
    self.last_cloudlog_t = 0.0
    self.last_cost = 0
    self.accel_dict = {"self": [], "lead": []}

    try:
      with open("/data/openpilot/gas-interceptor", "r") as f:
        self.gas_interceptor = bool(f.read())
    except:
      self.gas_interceptor = False

  def save_velocities(self, self_vel, lead_vel):
    if len(self.accel_dict["self"]) > 200:  # 100hz, so 200 items is 2 seconds
      self.accel_dict["self"].pop(0)
    self.accel_dict["self"].append(self_vel)

    if len(self.accel_dict["lead"]) > 200:  # 100hz, so 200 items is 2 seconds
      self.accel_dict["lead"].pop(0)
    self.accel_dict["lead"].append(lead_vel)

  def calculate_tr(self, v_ego, car_state, relative_velocity):
    """
    Returns a follow time gap in seconds based on car state values

    Parameters:
    v_ego: Vehicle speed [m/s]
    read_distance_lines: ACC setting showing how much follow distance the user has set [1|2|3]
    """

    read_distance_lines = car_state.readdistancelines
    if v_ego < 2.0 and read_distance_lines != 2 and self.gas_interceptor:  # if under 2m/s, not dynamic follow, and user has comma pedal
      return 1.8  # under 7km/hr use a TR of 1.8 seconds

    if car_state.leftBlinker or car_state.rightBlinker:  # if car is changing lanes and not already .9s
      if self.last_cost != 1.0:
        self.libmpc.init(MPC_COST_LONG.TTC, 1.0, MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)
        self.last_cost = 1.0
      return 0.9

    if read_distance_lines == 1:
      if read_distance_lines != self.lastTR or self.last_cost != 1.0:
        self.libmpc.init(MPC_COST_LONG.TTC, 1.0, MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)
        self.lastTR = read_distance_lines
        self.last_cost = 1.0
      return 0.9  # 10m at 40km/hr

    if read_distance_lines == 2:
      self.save_velocities(v_ego, v_ego + relative_velocity)

      generatedTR = self.generateTR(v_ego, relative_velocity)
      generated_cost = self.generate_cost(generatedTR)

      if abs(generated_cost - self.last_cost) > .15:
        self.libmpc.init(MPC_COST_LONG.TTC, generated_cost, MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)
        self.last_cost = generated_cost
      return generatedTR

    if read_distance_lines == 3:
      if read_distance_lines != self.lastTR or self.last_cost != 0.05:
        self.libmpc.init(MPC_COST_LONG.TTC, 0.05, MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)
        self.lastTR = read_distance_lines
        self.last_cost = 0.05
      return 2.7  # 30m at 40km/hr

    return 1.8  # if readdistancelines = 0

  def send_mpc_solution(self, qp_iterations, calculation_time):
    qp_iterations = max(0, qp_iterations)
    dat = messaging.new_message()
    dat.init('liveLongitudinalMpc')
    dat.liveLongitudinalMpc.xEgo = list(self.mpc_solution[0].x_ego)
    dat.liveLongitudinalMpc.vEgo = list(self.mpc_solution[0].v_ego)
    dat.liveLongitudinalMpc.aEgo = list(self.mpc_solution[0].a_ego)
    dat.liveLongitudinalMpc.xLead = list(self.mpc_solution[0].x_l)
    dat.liveLongitudinalMpc.vLead = list(self.mpc_solution[0].v_l)
    dat.liveLongitudinalMpc.cost = self.mpc_solution[0].cost
    dat.liveLongitudinalMpc.aLeadTau = self.a_lead_tau
    dat.liveLongitudinalMpc.qpIterations = qp_iterations
    dat.liveLongitudinalMpc.mpcId = self.mpc_id
    dat.liveLongitudinalMpc.calculationTime = calculation_time
    self.live_longitudinal_mpc.send(dat.to_bytes())

  def setup_mpc(self):
    ffi, self.libmpc = libmpc_py.get_libmpc(self.mpc_id)
    self.libmpc.init(MPC_COST_LONG.TTC, MPC_COST_LONG.DISTANCE,
                     MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)

    self.mpc_solution = ffi.new("log_t *")
    self.cur_state = ffi.new("state_t *")
    self.cur_state[0].v_ego = 0
    self.cur_state[0].a_ego = 0
    self.a_lead_tau = _LEAD_ACCEL_TAU

  def set_cur_state(self, v, a):
    self.cur_state[0].v_ego = v
    self.cur_state[0].a_ego = a

  def get_acceleration(self,
                       velocity_list):  # calculate car's own acceleration to generate more accurate following distances
    a = (velocity_list[-1] - velocity_list[0])  # first half of acceleration formula
    a = a / (len(
      velocity_list) / 100.0)  # divide difference in velocity by how long in seconds the velocity list has been tracking velocity (2 sec)
    if abs(a) < 0.11176:  # if abs(acceleration) is less than 0.25 mph/s, return 0
      return 0.0
    else:
      return a

  def generateTR(self, velocity, relative_velocity):  # in m/s
    x = [0.0, 1.86267, 3.72533, 5.588, 7.45067, 9.31333, 11.55978, 13.645, 22.352, 31.2928, 33.528, 35.7632,
         40.2336]  # velocity
    y = [1.03, 1.05363, 1.07879, 1.11493, 1.16969, 1.25071, 1.36325, 1.43, 1.6, 1.7, 1.75618, 1.85, 2.0]  # distances

    TR = interpolate.interp1d(x, y, fill_value='extrapolate')  # extrapolate above 90 mph

    TR = TR(velocity)[()]

    x = [-11.176, -7.84276, -4.67716, -2.12623, 0, 1.34112,
         2.68224]  # relative velocity values, mph: [-25, -17.5, -10.5, -4.75, 0, 3, 6]
    y = [(TR + .45), (TR + .34), (TR + .26), (TR + .13), TR, (TR - .18),
         (TR - .3)]  # modification values, less modification with less difference in velocity

    TR = np.interp(relative_velocity, x, y)  # interpolate as to not modify too much

    x = [-4.4704, -2.2352, -0.89408, 0, 1.34112]  # self acceleration values, mph: [-10, -5, -2, 0, 3]
    y = [(TR + .158), (TR + .062), (TR + .009), TR, (TR - .13)]  # modification values

    TR = np.interp(self.get_acceleration(self.accel_dict["self"]), x, y)  # factor in self acceleration

    x = [-4.4704, -2.2352, -0.89408, 0, 1.34112]  # lead acceleration values, mph: [-10, -5, -2, 0, 3]
    y = [(TR + .237), (TR + .093), (TR + .0135), TR, (TR - .195)]  # modification values

    TR = np.interp(self.get_acceleration(self.accel_dict["lead"]), x,
                   y)  # factor in lead car's acceleration; should perform better

    if TR < 0.65:
      return 0.65
    else:
      return round(TR, 2)

  def generate_cost(self, distance):
    x = [.9, 1.8, 2.7]
    y = [1.0, .1, .05]

    return round(float(np.interp(distance, x, y)),
                 2)  # used to cause stuttering, but now we're doing a percentage change check before changing

  def update(self, CS, lead, v_cruise_setpoint, relative_velocity):
    v_ego = CS.carState.vEgo

    # Setup current mpc state
    self.cur_state[0].x_ego = 0.0

    if lead is not None and lead.status:
      x_lead = max(0, lead.dRel - 1)
      v_lead = max(0.0, lead.vLead)
      a_lead = lead.aLeadK

      if (v_lead < 0.1 or -a_lead / 2.0 > v_lead):
        v_lead = 0.0
        a_lead = 0.0

      self.a_lead_tau = max(lead.aLeadTau, (a_lead ** 2 * math.pi) / (2 * (v_lead + 0.01) ** 2))
      self.new_lead = False
      if not self.prev_lead_status or abs(x_lead - self.prev_lead_x) > 2.5:
        self.libmpc.init_with_simulation(self.v_mpc, x_lead, v_lead, a_lead, self.a_lead_tau)
        self.new_lead = True

      self.prev_lead_status = True
      self.prev_lead_x = x_lead
      self.cur_state[0].x_l = x_lead
      self.cur_state[0].v_l = v_lead
    else:
      self.prev_lead_status = False
      # Fake a fast lead car, so mpc keeps running
      self.cur_state[0].x_l = 50.0
      self.cur_state[0].v_l = v_ego + 10.0
      a_lead = 0.0
      self.a_lead_tau = _LEAD_ACCEL_TAU

    # Calculate mpc
    t = sec_since_boot()
    TR = self.calculate_tr(v_ego, CS.carState, relative_velocity)
    n_its = self.libmpc.run_mpc(self.cur_state, self.mpc_solution, self.a_lead_tau, a_lead, TR)
    duration = int((sec_since_boot() - t) * 1e9)
    self.send_mpc_solution(n_its, duration)

    # Get solution. MPC timestep is 0.2 s, so interpolation to 0.05 s is needed
    self.v_mpc = self.mpc_solution[0].v_ego[1]
    self.a_mpc = self.mpc_solution[0].a_ego[1]
    self.v_mpc_future = self.mpc_solution[0].v_ego[10]

    # Reset if NaN or goes through lead car
    dls = np.array(list(self.mpc_solution[0].x_l)) - np.array(list(self.mpc_solution[0].x_ego))
    crashing = min(dls) < -50.0
    nans = np.any(np.isnan(list(self.mpc_solution[0].v_ego)))
    backwards = min(list(self.mpc_solution[0].v_ego)) < -0.01

    if ((backwards or crashing) and self.prev_lead_status) or nans:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Longitudinal mpc %d reset - backwards: %s crashing: %s nan: %s" % (
          self.mpc_id, backwards, crashing, nans))

      self.libmpc.init(MPC_COST_LONG.TTC, MPC_COST_LONG.DISTANCE,
                       MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)
      self.cur_state[0].v_ego = v_ego
      self.cur_state[0].a_ego = 0.0
      self.v_mpc = v_ego
      self.a_mpc = CS.carState.aEgo
      self.prev_lead_status = False