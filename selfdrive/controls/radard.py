#!/usr/bin/env python
import gc
import zmq
import numpy as np
import numpy.matlib
import importlib
from collections import defaultdict
from fastcluster import linkage_vector
import selfdrive.messaging as messaging
from selfdrive.services import service_list
from selfdrive.controls.lib.latcontrol_helpers import calc_lookahead_offset
from selfdrive.controls.lib.pathplanner import PathPlanner
from selfdrive.controls.lib.radar_helpers import Track, Cluster, fcluster, \
                                                 RDR_TO_LDR, NO_FUSION_SCORE
from selfdrive.controls.lib.vehicle_model import VehicleModel
from selfdrive.swaglog import cloudlog
from cereal import car
from common.params import Params
from common.realtime import set_realtime_priority, Ratekeeper
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


# fuses camera and radar data for best lead detection
def radard_thread(gctx=None):
  gc.disable()
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
  context = zmq.Context()

  # *** subscribe to features and model from visiond
  poller = zmq.Poller()
  model = messaging.sub_sock(context, service_list['model'].port, conflate=True, poller=poller)
  live100 = messaging.sub_sock(context, service_list['live100'].port, conflate=True, poller=poller)

  PP = PathPlanner()
  RI = RadarInterface(CP)

  last_md_ts = 0
  last_l100_ts = 0

  # *** publish live20 and liveTracks
  live20 = messaging.pub_sock(context, service_list['live20'].port)
  liveTracks = messaging.pub_sock(context, service_list['liveTracks'].port)

  path_x = np.arange(0.0, 140.0, 0.1)    # 140 meters is max

  # Time-alignment
  rate = 20.   # model and radar are both at 20Hz
  tsv = 1./rate
  v_len = 20         # how many speed data points to remember for t alignment with rdr data

  active = 0
  steer_angle = 0.
  steer_override = False

  tracks = defaultdict(dict)

  # Kalman filter stuff:
  ekfv = EKFV1D()
  speedSensorV = SimpleSensor(XV, 1, 2)

  # v_ego
  v_ego = None
  v_ego_array = np.zeros([2, v_len])
  v_ego_t_aligned = 0.

  rk = Ratekeeper(rate, print_delay_threshold=np.inf)
  while 1:
    rr = RI.update()

    ar_pts = {}
    for pt in rr.points:
      ar_pts[pt.trackId] = [pt.dRel + RDR_TO_LDR, pt.yRel, pt.vRel, pt.measured]

    # receive the live100s
    l100 = None
    md = None

    for socket, event in poller.poll(0):
      if socket is live100:
        l100 = messaging.recv_one(socket)
      elif socket is model:
        md = messaging.recv_one(socket)

    if l100 is not None:
      active = l100.live100.active
      v_ego = l100.live100.vEgo
      steer_angle = l100.live100.angleSteers
      steer_override = l100.live100.steerOverride

      v_ego_array = np.append(v_ego_array, [[v_ego], [float(rk.frame)/rate]], 1)
      v_ego_array = v_ego_array[:, 1:]

      last_l100_ts = l100.logMonoTime

    if v_ego is None:
      continue

    if md is not None:
      last_md_ts = md.logMonoTime

    # *** get path prediction from the model ***
    PP.update(v_ego, md)

    # run kalman filter only if prob is high enough
    if PP.lead_prob > 0.7:
      reading = speedSensorV.read(PP.lead_dist, covar=np.matrix(PP.lead_var))
      ekfv.update_scalar(reading)
      ekfv.predict(tsv)
      ar_pts[VISION_POINT] = (float(ekfv.state[XV]), np.polyval(PP.d_poly, float(ekfv.state[XV])),
                              float(ekfv.state[SPEEDV]), False)
    else:
      ekfv.state[XV] = PP.lead_dist
      ekfv.covar = (np.diag([PP.lead_var, ekfv.var_init]))
      ekfv.state[SPEEDV] = 0.

      if VISION_POINT in ar_pts:
        del ar_pts[VISION_POINT]

    # *** compute the likely path_y ***
    if (active and not steer_override) or mocked:
      # use path from model (always when mocking as steering is too noisy)
      path_y = np.polyval(PP.d_poly, path_x)
    else:
      # use path from steer, set angle_offset to 0 it does not only report the physical offset
      path_y = calc_lookahead_offset(v_ego, steer_angle, path_x, VM, angle_offset=0)[0]

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
      v_ego_t_aligned = np.interp(cur_time - RI.delay, v_ego_array[1], v_ego_array[0])
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

    idens = tracks.keys()
    track_pts = np.array([tracks[iden].get_key_for_cluster() for iden in idens])

    # If we have multiple points, cluster them
    if len(track_pts) > 1:
      link = linkage_vector(track_pts, method='centroid')
      cluster_idxs = fcluster(link, 2.5, criterion='distance')
      clusters = [None]*max(cluster_idxs)

      for idx in xrange(len(track_pts)):
        cluster_i = cluster_idxs[idx]-1

        if clusters[cluster_i] == None:
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

    # *** publish live20 ***
    dat = messaging.new_message()
    dat.init('live20')
    dat.live20.mdMonoTime = last_md_ts
    dat.live20.canMonoTimes = list(rr.canMonoTimes)
    dat.live20.radarErrors = list(rr.errors)
    dat.live20.l100MonoTime = last_l100_ts
    if lead_len > 0:
      dat.live20.leadOne = lead_clusters[0].toLive20()
      if lead2_len > 0:
        dat.live20.leadTwo = lead2_clusters[0].toLive20()
      else:
        dat.live20.leadTwo.status = False
    else:
      dat.live20.leadOne.status = False

    dat.live20.cumLagMs = -rk.remaining*1000.
    live20.send(dat.to_bytes())

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
        "stationary": tracks[ids].stationary,
        "oncoming": tracks[ids].oncoming,
      }
    liveTracks.send(dat.to_bytes())

    rk.monitor_time()

def main(gctx=None):
  radard_thread(gctx)

if __name__ == "__main__":
  main()
