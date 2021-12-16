import math
import numpy as np
import cereal.messaging as messaging
from common.realtime import sec_since_boot, DT_MDL
from selfdrive.controls.lib.drive_helpers import MPC_COST_LONG
from common.op_params import opParams
from common.numpy_fast import interp, clip, mean
from selfdrive.config import Conversions as CV
from cereal.messaging import SubMaster

from selfdrive.controls.lib.dynamic_follow.auto_df import predict
from selfdrive.controls.lib.dynamic_follow.df_manager import dfManager
from selfdrive.controls.lib.dynamic_follow.support import LeadData, CarData, dfData, dfProfiles
from common.data_collector import DataCollector

travis = False
DEFAULT_TR = 1.45


class DistanceModController:
  def __init__(self, k_i, k_d, x_clip, mods):
    self._k_i = k_i
    self._k_d = k_d
    self._to_clip = x_clip  # reaches this with v_rel=3.5 mph for 4 seconds
    self._mods = mods

    self.i = 0  # never resets, even when new lead
    self.last_error = 0

  def update(self, error):
    """
    Relative velocity is a good starting point
    Returns: Multiplier for final y_dist output
    """

    if (d := self._k_d * (error - self.last_error)) < 0:  # only add if it will add distance
      self.i += d

    self.i += error * DT_MDL * self._k_i
    self.i = clip(self.i, self._to_clip[0], self._to_clip[-1])  # clip to reasonable range
    self._slow_reset()  # slowly reset from max to 0

    fact = interp(self.i, self._to_clip, self._mods)
    self.last_error = float(error)

    # print("I: {}, FACT: {}".format(round(self.i, 4), round(fact, 3)))
    return fact

  def _slow_reset(self):
    if abs(self.i) > 0.01:  # oscillation starts around 0.006
      reset_time = 15  # in x seconds i goes from max to 0
      sign = 1 if self.i > 0 else -1
      self.i -= sign * max(self._to_clip) / (reset_time / DT_MDL)


class DynamicFollow:
  def __init__(self):
    self.op_params = opParams()
    self.df_profiles = dfProfiles()
    self.df_manager = dfManager()
    self.dmc_v_rel = DistanceModController(k_i=0.042, k_d=0.08, x_clip=[-1, 0, 0.66], mods=[1.15, 1., 0.95])
    self.dmc_a_rel = DistanceModController(k_i=0.042 * 1.05, k_d=0.08, x_clip=[-1, 0, 0.33], mods=[1.15, 1., 0.98])  # a_lead loop is 5% faster

    if not travis:
      self.pm = messaging.PubMaster(['dynamicFollowData'])
    else:
      self.pm = None

    # Model variables
    self.model_scales = {'v_ego': [-0.06112159043550491, 37.96522521972656], 'a_lead': [-3.109330892562866, 3.3612186908721924], 'v_lead': [0.0, 35.27671432495117], 'x_lead': [2.4600000381469727, 141.44000244140625]}
    self.predict_rate = 1 / 4.
    self.skip_every = round(0.25 / DT_MDL)
    self.model_input_len = round(45 / DT_MDL)

    # Dynamic follow variables
    self.TR = DEFAULT_TR
    self.v_ego_retention = 2.5
    self.v_rel_retention = 1.75

    self.sng_TR = DEFAULT_TR  # reacceleration stop and go TR
    self.sng_speed = 18.0 * CV.MPH_TO_MS

    self._setup_collector()
    self._setup_changing_variables()

  def _setup_collector(self):
    self.sm_collector = SubMaster(['liveTracks', 'laneSpeed'])
    self.log_auto_df = self.op_params.get('log_auto_df')
    if not isinstance(self.log_auto_df, bool):
      self.log_auto_df = False
    self.data_collector = DataCollector(file_path='/data/df_data', keys=['v_ego', 'a_ego', 'a_lead', 'v_lead', 'x_lead', 'left_lane_speeds', 'middle_lane_speeds', 'right_lane_speeds', 'left_lane_distances', 'middle_lane_distances', 'right_lane_distances', 'profile', 'time'], log_data=self.log_auto_df)

  def _setup_changing_variables(self):
    self.TR = DEFAULT_TR
    self.user_profile = self.df_profiles.stock  # just a starting point
    self.model_profile = self.df_profiles.stock

    self.last_effective_profile = self.user_profile
    self.profile_change_time = 0

    self.sng = False
    self.car_data = CarData()
    self.lead_data = LeadData()
    self.df_data = dfData()  # dynamic follow data

    self.last_cost = 0.0
    self.last_predict_time = 0.0
    self.auto_df_model_data = []
    self._get_live_params()  # so they're defined just in case

  def update(self, CS):
    self._get_live_params()
    self._update_car(CS)
    self._get_profiles()

    if self.log_auto_df:
      self._gather_data()

    if not self.lead_data.status:
      self.TR = DEFAULT_TR
    else:
      self._store_df_data()
      self.TR = self._get_TR()

    if not travis:
      self._send_cur_state()

    return self.TR

  def _get_profiles(self):
    """This receives profile change updates from dfManager and runs the auto-df prediction if auto mode"""
    df_out = self.df_manager.update()
    self.user_profile = df_out.user_profile
    if df_out.is_auto:  # todo: find some way to share prediction between the two mpcs to reduce processing overhead
      self._get_pred()  # sets self.model_profile, all other checks are inside function

  def _gather_data(self):
    self.sm_collector.update(0)
    # live_tracks = [[i.dRel, i.vRel, i.aRel, i.yRel] for i in self.sm_collector['liveTracks']]
    if self.car_data.cruise_enabled:
      self.data_collector.update([self.car_data.v_ego,
                                  self.car_data.a_ego,
                                  self.lead_data.a_lead,
                                  self.lead_data.v_lead,
                                  self.lead_data.x_lead,
                                  list(self.sm_collector['laneSpeed'].leftLaneSpeeds),
                                  list(self.sm_collector['laneSpeed'].middleLaneSpeeds),
                                  list(self.sm_collector['laneSpeed'].rightLaneSpeeds),

                                  list(self.sm_collector['laneSpeed'].leftLaneDistances),
                                  list(self.sm_collector['laneSpeed'].middleLaneDistances),
                                  list(self.sm_collector['laneSpeed'].rightLaneDistances),
                                  self.user_profile,
                                  sec_since_boot()])

  def _norm(self, x, name):
    self.x = x
    return interp(x, self.model_scales[name], [0, 1])

  def _send_cur_state(self):
    if self.pm is not None:
      dat = messaging.new_message('dynamicFollowData')
      dat.dynamicFollowData.mpcTR = self.TR
      dat.dynamicFollowData.profilePred = self.model_profile
      self.pm.send('dynamicFollowData', dat)

  def _store_df_data(self):
    cur_time = sec_since_boot()
    # Store custom relative accel over time
    if self.lead_data.status:
      if self.lead_data.new_lead:
        self.df_data.v_rels = []  # reset when new lead
      else:
        self.df_data.v_rels = self._remove_old_entries(self.df_data.v_rels, cur_time, self.v_rel_retention)
      self.df_data.v_rels.append({'v_ego': self.car_data.v_ego, 'v_lead': self.lead_data.v_lead, 'time': cur_time})

    # Store our velocity for better sng
    self.df_data.v_egos = self._remove_old_entries(self.df_data.v_egos, cur_time, self.v_ego_retention)
    self.df_data.v_egos.append({'v_ego': self.car_data.v_ego, 'time': cur_time})

    # Store data for auto-df model
    self.auto_df_model_data.append([self._norm(self.car_data.v_ego, 'v_ego'),
                                    self._norm(self.lead_data.v_lead, 'v_lead'),
                                    self._norm(self.lead_data.a_lead, 'a_lead'),
                                    self._norm(self.lead_data.x_lead, 'x_lead')])
    while len(self.auto_df_model_data) > self.model_input_len:
      del self.auto_df_model_data[0]

  def _get_pred(self):
    cur_time = sec_since_boot()
    if self.car_data.cruise_enabled and self.lead_data.status:
      if cur_time - self.last_predict_time > self.predict_rate:
        if len(self.auto_df_model_data) == self.model_input_len:
          pred = predict(np.array(self.auto_df_model_data[::self.skip_every], dtype=np.float32).flatten())
          self.last_predict_time = cur_time
          self.model_profile = int(np.argmax(pred))

  @staticmethod
  def _remove_old_entries(lst, cur_time, retention):
    return [sample for sample in lst if cur_time - sample['time'] <= retention]

  def _relative_accel_mod(self):
    """
    Returns relative acceleration mod calculated from list of lead and ego velocities over time (longer than 1s)
    If min_consider_time has not been reached, uses lead accel and ego accel from openpilot (kalman filtered)
    """
    a_ego = self.car_data.a_ego
    a_lead = self.lead_data.a_lead
    min_consider_time = 0.75  # minimum amount of time required to consider calculation
    if len(self.df_data.v_rels) > 0:  # if not empty
      elapsed_time = self.df_data.v_rels[-1]['time'] - self.df_data.v_rels[0]['time']
      if elapsed_time > min_consider_time:
        a_ego = (self.df_data.v_rels[-1]['v_ego'] - self.df_data.v_rels[0]['v_ego']) / elapsed_time
        a_lead = (self.df_data.v_rels[-1]['v_lead'] - self.df_data.v_rels[0]['v_lead']) / elapsed_time

    mods_x = [-1.5, -.75, 0]
    mods_y = [1, 1.25, 1.3]
    if a_lead < 0:  # more weight to slight lead decel
      a_lead *= interp(a_lead, mods_x, mods_y)

    if a_lead - a_ego > 0:  # return only if adding distance
      return 0

    rel_x = [-2.6822, -1.7882, -0.8941, -0.447, -0.2235, 0.0, 0.2235, 0.447, 0.8941, 1.7882, 2.6822]
    mod_y = [0.3245 * 1.1, 0.277 * 1.08, 0.11075 * 1.06, 0.08106 * 1.045, 0.06325 * 1.035, 0.0, -0.09, -0.09375, -0.125, -0.3, -0.35]
    return interp(a_lead - a_ego, rel_x, mod_y)

  def global_profile_mod(self, x_vel, y_dist):
    """
    This function modifies the y_dist list used by dynamic follow in accordance with global_df_mod
    """
    if self.global_df_mod == 1.:
      return y_dist
    global_df_mod = 1 - self.global_df_mod

    # Calculate new TRs
    speeds, mods = [0.], [1.]  # if increasing don't limit
    if self.global_df_mod < 1:  # if reducing distance
      speeds = [0, self.sng_speed, 18, x_vel[-1]]  # [0, 18 mph, ~40 mph, highest profile mod speed (~78 mph)]
      mods = [0, 0.25, 0.75, 1]  # how much to limit global_df_mod at each speed, 1 is full effect

    return [y - (y * global_df_mod * interp(x, speeds, mods)) for x, y in zip(x_vel, y_dist)]

  def _get_TR(self):
    if self.df_manager.is_auto:  # decide which profile to use, model profile will be updated before this
      df_profile = self.model_profile
    else:
      df_profile = self.user_profile

    if df_profile != self.last_effective_profile:
      self.profile_change_time = sec_since_boot()
    self.last_effective_profile = df_profile

    if df_profile == self.df_profiles.traffic:  # for in congested traffic
      x_vel = [0.0, 1.892, 3.7432, 5.8632, 8.0727, 10.7301, 14.343, 17.6275, 22.4049, 28.6752, 34.8858, 40.35]  # velocities
      y_dist = [1.3781, 1.3791, 1.3457, 1.3134, 1.3145, 1.318, 1.3485, 1.257, 1.144, 0.979, 0.9461, 0.9156]
    elif df_profile == self.df_profiles.stock:  # default to stock
      return 1.45
    elif df_profile == self.df_profiles.roadtrip:  # previous stock following distance
      return 1.8
    else:
      raise Exception('Unknown profile type: {}'.format(df_profile))

    # Global df mod
    y_dist = self.global_profile_mod(x_vel, y_dist)

    v_rel_dist_factor = self.dmc_v_rel.update(self.lead_data.v_lead - self.car_data.v_ego)
    a_lead_dist_factor = self.dmc_a_rel.update(self.lead_data.a_lead - self.car_data.a_ego)

    if self.car_data.v_ego > self.sng_speed:  # keep sng distance until we're above sng speed again
      self.sng = False

    if (self.car_data.v_ego >= self.sng_speed or self.df_data.v_egos[0]['v_ego'] >= self.car_data.v_ego) and not self.sng:
      # if above 15 mph OR we're decelerating to a stop, keep shorter TR. when we reaccelerate, use sng_TR and slowly decrease
      TR = interp(self.car_data.v_ego, x_vel, y_dist)
    else:  # this allows us to get closer to the lead car when stopping, while being able to have smooth stop and go when reaccelerating
      self.sng = True
      x = [self.sng_speed * 0.7, self.sng_speed]  # decrease TR between 12.6 and 18 mph from 1.8s to defined TR above at 18mph while accelerating
      y = [self.sng_TR, interp(self.sng_speed, x_vel, y_dist)]
      TR = interp(self.car_data.v_ego, x, y)

    TR *= v_rel_dist_factor
    TR *= a_lead_dist_factor

    return float(clip(TR, self.min_TR, 2.7))

    # TR_mods = []
    # # Dynamic follow modifications (the secret sauce)
    # x = [-26, -15.6464, -9.8422, -6.0, -4.0, -2.68, -2.3, -1.8, -1.26, -0.61, 0, 0.61, 1.26, 2.1, 2.68, 4.4704]  # relative velocity values
    # y = [1.76, 1.504, 1.34, 1.29, 1.25, 1.22, 1.19, 1.13, 1.053, 1.017, 1.0, 0.985, 0.958, 0.87, 0.81, 0.685]  # multiplier values
    # y = np.array(y) - 1  # converts back to original abs mod
    # y *= 1.1  # multiplier for how much to mod
    # y = y / TR + 1  # converts back to multipliers
    # TR_mods.append(interp(self.lead_data.v_lead - self.car_data.v_ego, x, y))
    #
    # x = [-4.4795, -2.8122, -1.5727, -1.1129, -0.6611, -0.2692, 0.0, 0.1466, 0.5144, 0.6903, 0.9302]  # lead acceleration values
    # y = [1.16, 1.1067, 1.0613, 1.0343, 1.0203, 1.0147, 1.0, 0.9898, 0.972, 0.9647, 0.9607]  # multiplier values
    # converted_with_TR = 1.5  # todo: do without numpy and simplify by converting with TR of 1, so only subtract
    # absolute_y_TR_mod = np.array(y) * converted_with_TR - converted_with_TR  # converts back to original abs mod
    # absolute_y_TR_mod *= 1.2  # multiplier for how much to mod
    # y = absolute_y_TR_mod / TR + 1  # converts back to multipliers with accel mod of 1.4 taking current TR into account
    # TR_mods.append(interp(self.get_rel_accel(), x, y))  # todo: make this over more than 1 sec
    #
    # # deadzone = self.car_data.v_ego / 3  # 10 mph at 30 mph  # todo: tune pedal to react similarly to without before adding/testing this
    # # if self.lead_data.v_lead - deadzone > self.car_data.v_ego:
    # #   TR_mods.append(self._relative_accel_mod())
    #
    # # x = [self.sng_speed, self.sng_speed / 5.0]  # as we approach 0, apply x% more distance
    # # y = [1.0, 1.05]
    #
    # TR *= mean(TR_mods)  # with mods as multipliers, profile mods shouldn't be needed
    #
    # # if (self.car_data.left_blinker or self.car_data.right_blinker) and df_profile != self.df_profiles.traffic:
    # #   x = [8.9408, 22.352, 31.2928]  # 20, 50, 70 mph
    # #   y = [1.0, .75, .65]
    # #   TR *= interp(self.car_data.v_ego, x, y)  # reduce TR when changing lanes
    #
    # return float(clip(TR, self.min_TR, 2.7))

  def update_lead(self, v_lead=None, a_lead=None, x_lead=None, status=False, new_lead=False):
    self.lead_data.v_lead = v_lead
    self.lead_data.a_lead = a_lead
    self.lead_data.x_lead = x_lead

    self.lead_data.status = status
    self.lead_data.new_lead = new_lead

  def _update_car(self, CS):
    self.car_data.v_ego = CS.vEgo
    self.car_data.a_ego = CS.aEgo

    self.car_data.left_blinker = CS.leftBlinker
    self.car_data.right_blinker = CS.rightBlinker
    self.car_data.cruise_enabled = CS.cruiseState.enabled

  def _get_live_params(self):
    self.global_df_mod = self.op_params.get('global_df_mod')
    if self.global_df_mod != 1.:
      self.global_df_mod = clip(self.global_df_mod, 0.85, 2.5)

    self.min_TR = self.op_params.get('min_TR')
    if self.min_TR != 1.:
      self.min_TR = clip(self.min_TR, 0.85, 2.7)
