from common.op_params import opParams
from selfdrive.config import Conversions as CV
from common.numpy_fast import clip, interp
import numpy as np


class DynamicLaneSpeed:
  def __init__(self):
    self.op_params = opParams()
    self.use_dynamic_lane_speed = self.op_params.get('use_dynamic_lane_speed', default=True)
    self.min_dynamic_lane_speed = max(self.op_params.get('min_dynamic_lane_speed', default=20.), 5.) * CV.MPH_TO_MS

  def update(self, v_target, v_target_future, v_cruise, a_target, v_ego, track_data, lead_data):
    if self.use_dynamic_lane_speed:
      v_cruise *= CV.KPH_TO_MS  # convert to m/s
      MPC_TIME_STEP = 1 / 20.
      track_tolerance_v = 0.05 * CV.MPH_TO_MS

      vels = [i * CV.MPH_TO_MS for i in [5, 40, 70]]
      margins = [0.36, 0.4675, 0.52]
      track_speed_margin = interp(v_ego, vels, margins)  # tracks must be within this times v_ego

      max_TR = 2.0  # the maximum TR we'll allow for each track

      if v_ego > self.min_dynamic_lane_speed and len(track_data) > 0:
        tracks = []
        for track in track_data:
          valid = all([True if abs(trk['v_lead'] - track['v_lead']) >= track_tolerance_v else False for trk in tracks])  # radar sometimes reports multiple points for one vehicle, especially semis
          if valid:
            tracks.append(track)
        tracks = [trk for trk in tracks if (v_ego * track_speed_margin) <= trk['v_lead'] <= v_cruise]
        tracks = [trk['v_lead'] for trk in tracks if trk['v_lead'] > 0.0 and (trk['x_lead'] / trk['v_lead']) <= max_TR]
        average_track_speed = np.mean(tracks)
        if average_track_speed < v_target and average_track_speed < v_target_future:
          x = [0, 3, 6, 19]
          y = [.05, 0.2, .4, 0.5]
          # todo: give less weight to further away tracks, but increase the above!
          track_speed_weight = interp(len(tracks), x, y)
          if lead_data['status']:  # if lead, give more weight to surrounding tracks (todo: this if check might need to be flipped, so if not lead...)
            track_speed_weight = clip(1.05 * track_speed_weight, min(y), max(y))
          # v_ego_v_cruise = (v_ego + v_cruise) / 2.0
          v_target_slow = (v_cruise * (1 - track_speed_weight)) + (average_track_speed * track_speed_weight)  # average set speed and average of tracks
          if v_target_slow < v_target and v_target_slow < v_target_future:  # just a sanity check, don't want to run into any leads if we somehow predict faster velocity
            a_target_slow = MPC_TIME_STEP * ((v_target_slow - v_target) / 1.0)  # long_mpc runs at 20 hz, so interpolate assuming a_target is 1 second into future? or since long_control is 100hz, should we interpolate using that?
            a_target = a_target_slow
            v_target = v_target_slow
            v_target_future = v_target_slow

    return v_target, v_target_future, a_target
