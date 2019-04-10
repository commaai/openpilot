from common.numpy_fast import interp
import numpy as np
import selfdrive.messaging as messaging
from selfdrive.swaglog import cloudlog
from common.realtime import sec_since_boot
from selfdrive.controls.lib.radar_helpers import _LEAD_ACCEL_TAU
from selfdrive.controls.lib.longitudinal_mpc import libmpc_py
from selfdrive.controls.lib.drive_helpers import MPC_COST_LONG
from scipy import interpolate
import math
import json


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
    self.dynamic_follow_dict = {"self_vels": [], "lead_vels": [], "traffic_vels": []}
    self.mpc_frame = 0  # idea thanks to kegman
    self.relative_velocity = None
    self.relative_distance = None
    self.dict_builder = {}

    try:
      with open("/data/openpilot/gas-interceptor", "r") as f:
        self.gas_interceptor = bool(f.read())
    except:
      self.gas_interceptor = False

  def save_car_data(self, self_vel):
    if len(self.dynamic_follow_dict["self_vels"]) >= 200:  # 100hz, so 200 items is 2 seconds
      del self.dynamic_follow_dict["self_vels"][0]
    self.dynamic_follow_dict["self_vels"].append(self_vel)

    if self.relative_velocity is not None:
      if len(self.dynamic_follow_dict["lead_vels"]) >= 300:
        del self.dynamic_follow_dict["lead_vels"][0]
      self.dynamic_follow_dict["lead_vels"].append(self_vel + self.relative_velocity)

      if self.mpc_frame >= 50:  # add to traffic list every half second so we're not working with a huge list
        if len(self.dynamic_follow_dict["traffic_vels"]) >= 360:  # 360 half seconds is 3 minutes of traffic logging
          del self.dynamic_follow_dict["traffic_vels"][0]
        self.dynamic_follow_dict["traffic_vels"].append(self_vel + self.relative_velocity)
        self.mpc_frame = 0  # reset every half second
      self.mpc_frame += 1  # increment every frame

    else:  # if no car, reset lead car list; ignore for traffic
      self.dynamic_follow_dict["lead_vels"] = []

  def calculate_tr(self, v_ego, car_state):
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
      self.save_car_data(v_ego)
      generatedTR = self.dynamic_follow(v_ego)
      generated_cost = self.generate_cost(generatedTR, v_ego)

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

  def get_traffic_level(self, lead_vels):  # generate a value to modify TR by based on fluctuations in lead speed
    if len(lead_vels) < 40:
      self.dict_builder["traffic_modifier"] = None
      return 1.0  # if less than 20 seconds of traffic data do nothing to TR
    lead_vel_diffs = []
    for idx, vel in enumerate(lead_vels):
      if idx != 0:
        lead_vel_diffs.append(abs(vel - lead_vels[idx - 1]))
    x = [0, len(lead_vels)]
    y = [1.25, .9]  # min and max values to modify TR by, need to tune
    traffic = interp(sum(lead_vel_diffs), x, y)
    self.dict_builder["traffic_modifier"] = traffic
    return traffic

  def get_acceleration(self, velocity_list, is_self):  # calculate acceleration to generate more accurate following distances
    if is_self:
      a = (velocity_list[-1] - velocity_list[0]) / (len(velocity_list) / 100.0)
    else:
      if len(velocity_list) >= 300:
        a_short = (velocity_list[-1] - velocity_list[-150]) / 1.5  # calculate lead accel last 1.5 s
        a_long = (velocity_list[-1] - velocity_list[-300]) / 3.0  # divide difference in velocity by how long in sec we're tracking velocity

        if abs(sum([a_short, a_long])) < 0.22352:  # if abs(sum) is less than .5 mph/s, average the two
          a = (a_short + a_long) / 2.0
        elif sum([a_short, a_long]) >= 0:  # return value furthest from 0
          a = max([a_short, a_long])
        else:
          a = min([a_short, a_long])
      else:
        a = (velocity_list[-1] - velocity_list[0]) / (len(velocity_list) / 100.0)

    return a

  def dynamic_follow(self, velocity):  # in m/s
    x = [0.0, 1.86267, 3.72533, 5.588, 7.45067, 9.31333, 11.55978, 13.645, 22.352, 31.2928, 33.528, 35.7632, 40.2336]  # velocity
    y = [1.03, 1.05363, 1.07879, 1.11493, 1.16969, 1.25071, 1.36325, 1.43, 1.6, 1.7, 1.75618, 1.85, 2.0]  # distances
    TR = interpolate.interp1d(x, y, fill_value='extrapolate')(velocity)[()]  # extrapolate above 90 mph

    if self.relative_velocity is not None:
      x = [6.0064, 4.1456, 1.7336, 0.2863, 0.0107, -2.5393, -4.4, -5.5716, -6.6743, -9.7755, -13.0146, -16.9428, -20.5265, -24.4547]  # relative velocity values
      y = [-0.3004, -0.2402, -0.1371, -0.0554, 0.0005, 0.0327, 0.0757, 0.1402, 0.1961, 0.2584, 0.3057, 0.34, 0.3766, 0.3852]  # modification values
      TR = TR + interp(self.relative_velocity, x, y)  # interpolate as to not modify too much

      x = [-4.4704, -2.2352, -0.8941, 0.0, 1.3411]   # self acceleration values
      y = [0.158, 0.058, 0.016, 0, -0.13]  # modification values
      TR = TR + interp(self.get_acceleration(self.dynamic_follow_dict["self_vels"], True), x, y)  # factor in self acceleration
      self.dict_builder["self_accel"] = self.get_acceleration(self.dynamic_follow_dict["self_vels"], True)

      x = [-4.4704, -2.2, -0.958, -0.447, -0.2235, 0.0, 0.5588, 1.3411]  # lead acceleration values
      y = [0.3555, 0.2788, 0.165, 0.1, 0.0375, 0, -0.115, -0.195]  # modification values
      TR = TR + interp(self.get_acceleration(self.dynamic_follow_dict["lead_vels"], False), x, y)  # factor in lead car's acceleration; should perform better
      self.dict_builder["lead_accel"] = self.get_acceleration(self.dynamic_follow_dict["lead_vels"], False)

      TR = TR * self.get_traffic_level(self.dynamic_follow_dict["traffic_vels"])  # modify TR based on last minute of traffic data

    self.dict_builder["TR"] = TR

    if TR < 0.65:
      return 0.65
    else:
      return round(TR, 3)

  def generate_cost(self, TR, v_ego):
    x = [.9, 1.8, 2.7]
    y = [1.0, .1, .05]
    if v_ego != 0:
      real_TR = self.relative_distance / float(v_ego)  # switched to cost generation using actual distance from lead car; should be safer
      if abs(real_TR - TR) >= .25:  # use real TR if diff is greater than x safety threshold
        TR = real_TR

    cost = round(float(interp(TR, x, y)), 3)
    self.dict_builder["generated_cost"] = cost
    return cost

  def update(self, CS, lead, v_cruise_setpoint):
    v_ego = CS.carState.vEgo

    try:
      self.relative_velocity = lead.vRel
      self.relative_distance = lead.dRel
    except: #if no lead car
      self.relative_velocity = None
      self.relative_distance = None

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
    TR = self.calculate_tr(v_ego, CS.carState)
    with open("/data/op-tools-data/dynamic-follow.json", "w") as d_f:
      d_f.write(json.dumps(self.dict_builder))
    self.dict_builder = {}
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