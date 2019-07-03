#!/usr/bin/env python
import numpy as np
import numpy.matlib
import importlib
from collections import defaultdict, deque

import selfdrive.messaging as messaging
from selfdrive.services import service_list
from selfdrive.controls.lib.latcontrol_helpers import calc_lookahead_offset
from selfdrive.controls.lib.model_parser import ModelParser
from selfdrive.controls.lib.radar_helpers import Track, Cluster, \
                                                 RDR_TO_LDR, NO_FUSION_SCORE

from selfdrive.controls.lib.cluster.fastcluster_py import cluster_points_centroid
from selfdrive.controls.lib.vehicle_model import VehicleModel
from selfdrive.swaglog import cloudlog
from cereal import car
from common.params import Params
from common.realtime import set_realtime_priority, Ratekeeper, DT_MDL
from common.kalman.ekf import EKF, SimpleSensor

DEBUG = False

#vision point
DIMSV = 2
XV, SPEEDV = 0, 1
VISION_POINT = -1


class EKFV1D(EKF):
  def __init__(self):
    super(EKFV1D, self).__init__(False)
    self.identity = numpy.matlib.identity(DIMSV)
    self.state = np.matlib.zeros((DIMSV, 1))
    self.var_init = 1e2   # ~ model variance when probability is 70%, so good starting point
    self.covar = self.identity * self.var_init

    self.process_noise = np.matlib.diag([0.5, 1])

  def calc_transfer_fun(self, dt):
    tf = np.matlib.identity(DIMSV)
    tf[XV, SPEEDV] = dt
    tfj = tf
    return tf, tfj


## fuses camera and radar data for best lead detection
def radard_thread(gctx=None):
  set_realtime_priority(2)

  # wait for stats about the car to come in from controls
  cloudlog.info("radard is waiting for CarParams")
  CP = car.CarParams.from_bytes(Params().get("CarParams", block=True))
  mocked = CP.carName == "mock"
  VM = VehicleModel(CP)
  cloudlog.info("radard got CarParams")

  # import the radar from the fingerprint
  cloudlog.info("radard is importing %s", CP.carName)
  RadarInterface = importlib.import_module('selfdrive.car.%s.radar_interface' % CP.carName).RadarInterface

  sm = messaging.SubMaster(['model', 'controlsState', 'liveParameters'])

  # Default parameters
  live_parameters = messaging.new_message()
  live_parameters.init('liveParameters')
  live_parameters.liveParameters.valid = True
  live_parameters.liveParameters.steerRatio = CP.steerRatio
  live_parameters.liveParameters.stiffnessFactor = 1.0

  MP = ModelParser()
  RI = RadarInterface(CP)

  last_md_ts = 0
  last_controls_state_ts = 0

  # *** publish radarState and liveTracks
  radarState = messaging.pub_sock(service_list['radarState'].port)
  liveTracks = messaging.pub_sock(service_list['liveTracks'].port)

  path_x = np.arange(0.0, 140.0, 0.1)    # 140 meters is max

  # Time-alignment
  rate = 1. / DT_MDL  # model and radar are both at 20Hz
  v_len = 20   # how many speed data points to remember for t alignment with rdr data

  active = 0
  steer_angle = 0.
  steer_override = False

  tracks = defaultdict(dict)

  # Kalman filter stuff:
  ekfv = EKFV1D()
  speedSensorV = SimpleSensor(XV, 1, 2)

  # v_ego
  v_ego = 0.
  v_ego_hist_t = deque([0], maxlen=v_len)
  v_ego_hist_v = deque([0], maxlen=v_len)
  v_ego_t_aligned = 0.

  rk = Ratekeeper(rate, print_delay_threshold=None)
  while 1:
    rr = RI.update()

    ar_pts = {}
    for pt in rr.points:
      ar_pts[pt.trackId] = [pt.dRel + RDR_TO_LDR, pt.yRel, pt.vRel, pt.measured]

    sm.update(0)

    if sm.updated['liveParameters']:
        VM.update_params(sm['liveParameters'].stiffnessFactor, sm['liveParameters'].steerRatio)

    if sm.updated['controlsState']:
      active = sm['controlsState'].active
      v_ego = sm['controlsState'].vEgo
      steer_angle = sm['controlsState'].angleSteers
      steer_override = sm['controlsState'].steerOverride

      v_ego_hist_v.append(v_ego)
      v_ego_hist_t.append(float(rk.frame)/rate)

      last_controls_state_ts = sm.logMonoTime['controlsState']

    if sm.updated['model']:
      last_md_ts = sm.logMonoTime['model']
      MP.update(v_ego, sm['model'])


    # run kalman filter only if prob is high enough
    if MP.lead_prob > 0.7:
      reading = speedSensorV.read(MP.lead_dist, covar=np.matrix(MP.lead_var))
      ekfv.update_scalar(reading)
      ekfv.predict(DT_MDL)

      # When changing lanes the distance to the lead car can suddenly change,
      # which makes the Kalman filter output large relative acceleration
      if mocked and abs(MP.lead_dist - ekfv.state[XV]) > 2.0:
        ekfv.state[XV] = MP.lead_dist
        ekfv.covar = (np.diag([MP.lead_var, ekfv.var_init]))
        ekfv.state[SPEEDV] = 0.

      ar_pts[VISION_POINT] = (float(ekfv.state[XV]), np.polyval(MP.d_poly, float(ekfv.state[XV])),
                              float(ekfv.state[SPEEDV]), False)
    else:
      ekfv.state[XV] = MP.lead_dist
      ekfv.covar = (np.diag([MP.lead_var, ekfv.var_init]))
      ekfv.state[SPEEDV] = 0.

      if VISION_POINT in ar_pts:
        del ar_pts[VISION_POINT]

    # *** compute the likely path_y ***
    if (active and not steer_override) or mocked:
      # use path from model (always when mocking as steering is too noisy)
      path_y = np.polyval(MP.d_poly, path_x)
    else:
      # use path from steer, set angle_offset to 0 it does not only report the physical offset
      path_y = calc_lookahead_offset(v_ego, steer_angle, path_x, VM, angle_offset=live_parameters.liveParameters.angleOffsetAverage)[0]

    # *** remove missing points from meta data ***
    for ids in tracks.keys():
      if ids not in ar_pts:
        tracks.pop(ids, None)

    # *** compute the tracks ***
    for ids in ar_pts:
      # ignore standalone vision point, unless we are mocking the radar
      if ids == VISION_POINT and not mocked:
        continue
      rpt = ar_pts[ids]

      # align v_ego by a fixed time to align it with the radar measurement
      cur_time = float(rk.frame)/rate
      v_ego_t_aligned = np.interp(cur_time - RI.delay, v_ego_hist_t, v_ego_hist_v)

      d_path = np.sqrt(np.amin((path_x - rpt[0]) ** 2 + (path_y - rpt[1]) ** 2))
      # add sign
      d_path *= np.sign(rpt[1] - np.interp(rpt[0], path_x, path_y))

      # create the track if it doesn't exist or it's a new track
      if ids not in tracks:
        tracks[ids] = Track()
      tracks[ids].update(rpt[0], rpt[1], rpt[2], d_path, v_ego_t_aligned, rpt[3], steer_override)

    # allow the vision model to remove the stationary flag if distance and rel speed roughly match
    if VISION_POINT in ar_pts:
      fused_id = None
      best_score = NO_FUSION_SCORE
      for ids in tracks:
        dist_to_vision = np.sqrt((0.5*(ar_pts[VISION_POINT][0] - tracks[ids].dRel)) ** 2 + (2*(ar_pts[VISION_POINT][1] - tracks[ids].yRel)) ** 2)
        rel_speed_diff = abs(ar_pts[VISION_POINT][2] - tracks[ids].vRel)
        tracks[ids].update_vision_score(dist_to_vision, rel_speed_diff)
        if best_score > tracks[ids].vision_score:
          fused_id = ids
          best_score = tracks[ids].vision_score

      if fused_id is not None:
        tracks[fused_id].vision_cnt += 1
        tracks[fused_id].update_vision_fusion()

    if DEBUG:
      print("NEW CYCLE")
      if VISION_POINT in ar_pts:
        print("vision", ar_pts[VISION_POINT])

    idens = list(tracks.keys())
    track_pts = np.array([tracks[iden].get_key_for_cluster() for iden in idens])

    # If we have multiple points, cluster them
    if len(track_pts) > 1:
      cluster_idxs = cluster_points_centroid(track_pts, 2.5)
      clusters = [None] * (max(cluster_idxs) + 1)

      for idx in xrange(len(track_pts)):
        cluster_i = cluster_idxs[idx]
        if clusters[cluster_i] is None:
          clusters[cluster_i] = Cluster()
        clusters[cluster_i].add(tracks[idens[idx]])

    elif len(track_pts) == 1:
      # TODO: why do we need this?
      clusters = [Cluster()]
      clusters[0].add(tracks[idens[0]])
    else:
      clusters = []

    if DEBUG:
      for i in clusters:
        print(i)
    # *** extract the lead car ***
    lead_clusters = [c for c in clusters
                     if c.is_potential_lead(v_ego)]
    lead_clusters.sort(key=lambda x: x.dRel)
    lead_len = len(lead_clusters)

    # *** extract the second lead from the whole set of leads ***
    lead2_clusters = [c for c in lead_clusters
                      if c.is_potential_lead2(lead_clusters)]
    lead2_clusters.sort(key=lambda x: x.dRel)
    lead2_len = len(lead2_clusters)

    # *** publish radarState ***
    dat = messaging.new_message()
    dat.init('radarState')
    dat.valid = sm.all_alive_and_valid(service_list=['controlsState'])
    dat.radarState.mdMonoTime = last_md_ts
    dat.radarState.canMonoTimes = list(rr.canMonoTimes)
    dat.radarState.radarErrors = list(rr.errors)
    dat.radarState.controlsStateMonoTime = last_controls_state_ts
    if lead_len > 0:
      dat.radarState.leadOne = lead_clusters[0].toRadarState()
      if lead2_len > 0:
        dat.radarState.leadTwo = lead2_clusters[0].toRadarState()
      else:
        dat.radarState.leadTwo.status = False
    else:
      dat.radarState.leadOne.status = False

    dat.radarState.cumLagMs = -rk.remaining*1000.
    radarState.send(dat.to_bytes())

    # *** publish tracks for UI debugging (keep last) ***
    dat = messaging.new_message()
    dat.init('liveTracks', len(tracks))

    for cnt, ids in enumerate(tracks.keys()):
      if DEBUG:
        print("id: %4.0f x:  %4.1f  y: %4.1f  vr: %4.1f d: %4.1f  va: %4.1f  vl: %4.1f  vlk: %4.1f alk: %4.1f  s: %1.0f  v: %1.0f" % \
          (ids, tracks[ids].dRel, tracks[ids].yRel, tracks[ids].vRel,
           tracks[ids].dPath, tracks[ids].vLat,
           tracks[ids].vLead, tracks[ids].vLeadK,
           tracks[ids].aLeadK,
           tracks[ids].stationary,
           tracks[ids].measured))
      dat.liveTracks[cnt] = {
        "trackId": ids,
        "dRel": float(tracks[ids].dRel),
        "yRel": float(tracks[ids].yRel),
        "vRel": float(tracks[ids].vRel),
        "aRel": float(tracks[ids].aRel),
        "stationary": bool(tracks[ids].stationary),
        "oncoming": bool(tracks[ids].oncoming),
      }
    liveTracks.send(dat.to_bytes())

    rk.monitor_time()

def main(gctx=None):
  radard_thread(gctx)

if __name__ == "__main__":
  main()
