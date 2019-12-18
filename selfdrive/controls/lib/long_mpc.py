import os
import math
import time

import cereal.messaging as messaging
from selfdrive.swaglog import cloudlog
from common.realtime import sec_since_boot
from selfdrive.controls.lib.radar_helpers import _LEAD_ACCEL_TAU
from selfdrive.controls.lib.longitudinal_mpc import libmpc_py
from selfdrive.controls.lib.drive_helpers import MPC_COST_LONG
from common.op_params import opParams
from common.numpy_fast import interp, clip
from common.travis_checker import travis

LOG_MPC = os.environ.get('LOG_MPC', False)


class LongitudinalMpc():
  def __init__(self, mpc_id):
    self.mpc_id = mpc_id
    self.MPH_TO_MS = 0.44704
    self.op_params = opParams()

    self.setup_mpc()
    self.v_mpc = 0.0
    self.v_mpc_future = 0.0
    self.a_mpc = 0.0
    self.v_cruise = 0.0
    self.prev_lead_status = False
    self.prev_lead_x = 0.0
    self.new_lead = False
    self.last_cloudlog_t = 0.0

    self.CS = None
    self.car_data = {'v_ego': 0.0, 'a_ego': 0.0}
    self.lead_data = {'v_lead': None, 'x_lead': None, 'a_lead': None, 'status': False}
    self.df_data = {"v_leads": [], "v_egos": []}  # dynamic follow data
    self.last_cost = 0.0
    self.customTR = self.op_params.get('following_distance', None)

  def send_mpc_solution(self, pm, qp_iterations, calculation_time):
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
    pm.send('liveLongitudinalMpc', dat)

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

  def get_TR(self):
    if not self.lead_data['status'] or travis:
      TR = 1.8
    elif self.customTR is not None:
      TR = clip(self.customTR, 0.9, 2.7)
    else:
      self.store_lead_data()
      TR = self.dynamic_follow()

    self.change_cost(TR)
    return TR

  def change_cost(self, TR):
    TRs = [0.9, 1.8, 2.7]
    costs = [1.0, 0.1, 0.05]
    cost = interp(TR, TRs, costs)
    if self.last_cost != cost:
      self.libmpc.change_tr(MPC_COST_LONG.TTC, cost, MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)
      self.last_cost = cost

  def store_lead_data(self):
    v_lead_retention = 2.0  # seconds
    v_ego_retention = 1.5

    if self.lead_data['status']:
      self.df_data['v_leads'] = [sample for sample in self.df_data['v_leads'] if
                                 time.time() - sample['time'] <= v_lead_retention
                                 and not self.new_lead]  # reset when new lead
      self.df_data['v_leads'].append({'v_lead': self.lead_data['v_lead'], 'time': time.time()})

    self.df_data['v_egos'] = [sample for sample in self.df_data['v_egos'] if time.time() - sample['time'] <= v_ego_retention]
    self.df_data['v_egos'].append({'v_ego': self.car_data['v_ego'], 'time': time.time()})

  def lead_accel_over_time(self):
    min_consider_time = 1.0  # minimum amount of time required to consider calculation
    a_lead = self.lead_data['a_lead']
    if len(self.df_data['v_leads']):  # if not empty
      elapsed = self.df_data['v_leads'][-1]['time'] - self.df_data['v_leads'][0]['time']
      if elapsed > min_consider_time:  # if greater than min time (not 0)
        v_diff = self.df_data['v_leads'][-1]['v_lead'] - self.df_data['v_leads'][0]['v_lead']
        calculated_accel = v_diff / elapsed
        if abs(calculated_accel) > abs(a_lead):  # if a_lead is greater than calculated accel (over last 1.5s, use that)
          a_lead = calculated_accel
    return a_lead  # if above doesn't execute, we'll return a_lead from radar

  def dynamic_follow(self):
    x_vel = [0.0, 1.8627, 3.7253, 5.588, 7.4507, 9.3133, 11.5598, 13.645, 22.352, 31.2928, 33.528, 35.7632, 40.2336]  # velocities
    y_mod = [1.102, 1.12, 1.14, 1.168, 1.21, 1.273, 1.36, 1.411, 1.543, 1.62, 1.664, 1.736, 1.853]  # TRs

    sng_TR = 1.7  # stop and go parameters
    sng_speed = 15.0 * self.MPH_TO_MS

    if self.car_data['v_ego'] >= sng_speed or self.df_data['v_egos'][-1]['v_ego'] >= self.car_data['v_ego']:  # if above 15 mph OR we're decelerating to a stop, keep shorter TR. when we reaccelerate, use 1.8s and slowly decrease
      TR = interp(self.car_data['v_ego'], x_vel, y_mod)
    else:  # this allows us to get closer to the lead car when stopping, while being able to have smooth stop and go when reaccelerating
      x = [sng_speed / 3.0, sng_speed]  # decrease TR between 5 and 15 mph from 1.8s to defined TR above at 15mph while accelerating
      y = [sng_TR, interp(sng_speed, x_vel, y_mod)]
      TR = interp(self.car_data['v_ego'], x, y)

    # Dynamic follow modifications (the secret sauce)
    x = [-20, -15.655, -11.1702, -7.8235, -4.6665, -2.5663, -1.1843, 0, 1.0107, 1.89, 2.6909]  # relative velocity values
    y = [0.65, 0.525, 0.44, 0.341, 0.26, 0.159, 0.049, 0, -0.082, -0.18, -0.28]  # modification values
    TR_mod = interp(self.lead_data['v_lead'] - self.car_data['v_ego'], x, y)

    x = [-4.4704, -1.77, -0.3145, 0, 0.446, 1.3411]  # lead acceleration values
    y = [0.237, 0.12, 0.027, 0, -0.105, -0.195]  # modification values
    TR_mod += interp(self.lead_accel_over_time(), x, y)  # todo: test if these modifications are too much

    TR += TR_mod

    if self.CS.leftBlinker or self.CS.rightBlinker:
      old_TR = float(TR)
      x = [8.9408, 22.352, 31.2928]  # 20, 50, 70 mph
      y = [1.0, .7, .65]  # reduce TR when changing lanes
      TR *= interp(self.car_data['v_ego'], x, y)
      with open('/data/blinker_debug', 'a') as f:
        f.write('{}\n'.format([self.CS.leftBlinker, self.CS.rightBlinker, self.car_data['v_ego'], old_TR, TR]))

    # TR *= self.get_traffic_level()  # modify TR based on last minute of traffic data  # todo: look at getting this to work, a model could be used

    return clip(round(TR, 3), 0.9, 2.7)

  def process_lead(self, v_lead, a_lead, x_lead, status):
    self.lead_data['v_lead'] = v_lead
    self.lead_data['a_lead'] = a_lead
    self.lead_data['x_lead'] = x_lead
    self.lead_data['status'] = status

  # def get_traffic_level(self, lead_vels):  # generate a value to modify TR by based on fluctuations in lead speed
  #   if len(lead_vels) < 60:
  #     return 1.0  # if less than 30 seconds of traffic data do nothing to TR
  #   lead_vel_diffs = []
  #   for idx, vel in enumerate(lead_vels):
  #     try:
  #       lead_vel_diffs.append(abs(vel - lead_vels[idx - 1]))
  #     except:
  #       pass
  #
  #   x = [0, len(lead_vels)]
  #   y = [1.15, .9]  # min and max values to modify TR by, need to tune
  #   traffic = interp(sum(lead_vel_diffs), x, y)
  #
  #   return traffic

  def update(self, pm, CS, lead, v_cruise_setpoint):
    v_ego = CS.vEgo
    self.CS = CS
    self.car_data = {'v_ego': CS.vEgo, 'a_ego': CS.aEgo}

    # Setup current mpc state
    self.cur_state[0].x_ego = 0.0

    if lead is not None and lead.status:
      x_lead = lead.dRel
      v_lead = max(0.0, lead.vLead)
      a_lead = lead.aLeadK

      if v_lead < 0.1 or -a_lead / 2.0 > v_lead:
        v_lead = 0.0
        a_lead = 0.0
      self.process_lead(v_lead, a_lead, x_lead, lead.status)

      self.a_lead_tau = lead.aLeadTau
      self.new_lead = False
      if not self.prev_lead_status or abs(x_lead - self.prev_lead_x) > 2.5:
        self.libmpc.init_with_simulation(self.v_mpc, x_lead, v_lead, a_lead, self.a_lead_tau)
        self.new_lead = True

      self.prev_lead_status = True
      self.prev_lead_x = x_lead
      self.cur_state[0].x_l = x_lead
      self.cur_state[0].v_l = v_lead
    else:
      self.process_lead(None, None, None, False)
      self.prev_lead_status = False
      # Fake a fast lead car, so mpc keeps running
      self.cur_state[0].x_l = 50.0
      self.cur_state[0].v_l = v_ego + 10.0
      a_lead = 0.0
      self.a_lead_tau = _LEAD_ACCEL_TAU

    # Calculate mpc
    t = sec_since_boot()
    n_its = self.libmpc.run_mpc(self.cur_state, self.mpc_solution, self.a_lead_tau, a_lead, self.get_TR())
    duration = int((sec_since_boot() - t) * 1e9)

    if not travis:
      with open('/data/mpc_debug.{}'.format(self.mpc_id), 'a') as f:
        f.write('{}\n'.format(self.lead_data))

    if LOG_MPC:
      self.send_mpc_solution(pm, n_its, duration)

    # Get solution. MPC timestep is 0.2 s, so interpolation to 0.05 s is needed
    self.v_mpc = self.mpc_solution[0].v_ego[1]
    self.a_mpc = self.mpc_solution[0].a_ego[1]
    self.v_mpc_future = self.mpc_solution[0].v_ego[10]

    # Reset if NaN or goes through lead car
    crashing = any(lead - ego < -50 for (lead, ego) in zip(self.mpc_solution[0].x_l, self.mpc_solution[0].x_ego))
    nans = any(math.isnan(x) for x in self.mpc_solution[0].v_ego)
    backwards = min(self.mpc_solution[0].v_ego) < -0.01

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
      self.a_mpc = CS.aEgo
      self.prev_lead_status = False
