from common.op_params import opParams
from selfdrive.config import Conversions as CV
from common.numpy_fast import clip, interp
import numpy as np


class DynamicLaneSpeed:
  def __init__(self):
    self.op_params = opParams()
    self.use_dynamic_lane_speed = self.op_params.get('use_dynamic_lane_speed', default=True)
    self.min_dynamic_lane_speed = max(self.op_params.get('min_dynamic_lane_speed', default=20.), 5.) * CV.MPH_TO_MS

    self.track_tolerance_v = 0.05 * CV.MPH_TO_MS
    self.MPC_TIME_STEP = 1 / 20.
    self.lane_vels = [i * CV.MPH_TO_MS for i in [5, 40, 70]]
    self.margins = [0.36, 0.4675, 0.52]
    self.max_TR = 2.5  # the maximum TR we'll allow for each track

    self.track_TR = [0.6, 1.8, 2.5]
    self.track_importance = [1.2, 1.0, 0.25]

  def get_track_average(self, v_ego, v_cruise, track_data):
    tracks = []
    for track in track_data:
      valid = all([True if abs(trk['v_lead'] - track['v_lead']) >= self.track_tolerance_v else False for trk in tracks])  # radar sometimes reports multiple points for one vehicle, especially semis
      if valid:
        tracks.append(track)

    track_speed_margin = interp(v_ego, self.lane_vels, self.margins)  # tracks must be within this times v_ego

    tracks = [trk for trk in tracks if (v_ego * track_speed_margin) <= trk['v_lead'] <= v_cruise]  # filter out tracks not in margins
    tracks = [trk for trk in tracks if trk['v_lead'] > 0.0 and (trk['x_lead'] / trk['v_lead']) <= self.max_TR]  # filter out tracks greater than max following distance
    track_weights = [np.interp(trk['x_lead'] / trk['v_lead'], self.track_TR, self.track_importance) for trk in tracks]  # calculate importance from track TR

    if len(tracks) > 0 and sum(track_weights) > 0:
      return sum([trk['v_lead'] * weight for trk, weight in zip(tracks, track_weights)]) / sum(track_weights), len(tracks)  # weighted average based off TR
    return 0, 0

  def update(self, v_target, v_target_future, v_cruise, a_target, v_ego, track_data, lead_data):
    if self.use_dynamic_lane_speed and v_ego > self.min_dynamic_lane_speed and len(track_data) > 0:
      v_cruise *= CV.KPH_TO_MS  # convert to m/s

      weighted_average, len_tracks = self.get_track_average(v_ego, v_cruise, track_data)
      if weighted_average < v_target and weighted_average < v_target_future and len_tracks > 0:
        x = [1, 3, 6, 19]
        y = [0.075, 0.36, .46, 0.52]
        track_speed_weight = interp(len_tracks, x, y)
        if lead_data['status']:  # if lead, give more weight to surrounding tracks todo: this if check might need to be flipped, so if not lead...
          track_speed_weight = clip(1.05 * track_speed_weight, min(y), max(y))
        # v_ego_v_cruise = (v_ego + v_cruise) / 2.0
        v_target_slow = (v_cruise * (1 - track_speed_weight)) + (weighted_average * track_speed_weight)  # average set speed and average of tracks
        if v_target_slow < v_target and v_target_slow < v_target_future:  # just a sanity check, don't want to run into any leads if we somehow predict faster velocity
          future_time = 1.0
          a_target_slow = self.MPC_TIME_STEP * ((v_target_slow - v_target) / future_time)  # long_mpc runs at 20 hz, so interpolate assuming a_target is 1 second into future? or since long_control is 100hz, should we interpolate using that?
          a_target = a_target_slow
          v_target = v_target_slow
          v_target_future = v_target_slow

    return v_target, v_target_future, a_target
