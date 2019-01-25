#!/usr/bin/env python

# Add phonelibs openblas to LD_LIBRARY_PATH if import fails
try:
  from scipy import spatial
except ImportError as e:
  import os
  import sys
  from common.basedir import BASEDIR

  openblas_path = os.path.join(BASEDIR, "phonelibs/openblas/")
  os.environ['LD_LIBRARY_PATH'] += ':' + openblas_path

  args = [sys.executable]
  args.extend(sys.argv)
  os.execv(sys.executable, args)

import os
import sys
import time
import zmq
import threading
import numpy as np
import overpy
from collections import defaultdict

from common.params import Params
from common.transformations.coordinates import geodetic2ecef
from selfdrive.services import service_list
import selfdrive.messaging as messaging
from mapd_helpers import MAPS_LOOKAHEAD_DISTANCE, Way, circle_through_points
import selfdrive.crash as crash
from selfdrive.version import version, dirty


OVERPASS_API_URL = "https://overpass.kumi.systems/api/interpreter"
OVERPASS_HEADERS = {
    'User-Agent': 'NEOS (comma.ai)'
}

last_gps = None
query_lock = threading.Lock()
last_query_result = None
last_query_pos = None
cache_valid = False


def setup_thread_excepthook():
  """
  Workaround for `sys.excepthook` thread bug from:
  http://bugs.python.org/issue1230540
  Call once from the main thread before creating any threads.
  Source: https://stackoverflow.com/a/31622038
  """
  init_original = threading.Thread.__init__

  def init(self, *args, **kwargs):
    init_original(self, *args, **kwargs)
    run_original = self.run

    def run_with_except_hook(*args2, **kwargs2):
      try:
        run_original(*args2, **kwargs2)
      except Exception:
        sys.excepthook(*sys.exc_info())

    self.run = run_with_except_hook

  threading.Thread.__init__ = init


def build_way_query(lat, lon, radius=50):
  """Builds a query to find all highways within a given radius around a point"""
  pos = "  (around:%f,%f,%f)" % (radius, lat, lon)
  q = """(
  way
  """ + pos + """
  [highway][highway!~"^(footway|path|bridleway|steps|cycleway|construction|bus_guideway|escape)$"];
  >;);out;
  """
  return q


def query_thread():
  global last_query_result, last_query_pos, cache_valid
  api = overpy.Overpass(url=OVERPASS_API_URL, headers=OVERPASS_HEADERS, timeout=10.)

  while True:
    time.sleep(1)
    if last_gps is not None:
      fix_ok = last_gps.flags & 1
      if not fix_ok:
        continue

      if last_query_pos is not None:
        cur_ecef = geodetic2ecef((last_gps.latitude, last_gps.longitude, last_gps.altitude))
        prev_ecef = geodetic2ecef((last_query_pos.latitude, last_query_pos.longitude, last_query_pos.altitude))
        dist = np.linalg.norm(cur_ecef - prev_ecef)
        if dist < 1000:
          continue

        if dist > 3000:
          cache_valid = False

      q = build_way_query(last_gps.latitude, last_gps.longitude, radius=3000)
      try:
        new_result = api.query(q)

        # Build kd-tree
        nodes = []
        real_nodes = []
        node_to_way = defaultdict(list)

        for n in new_result.nodes:
          nodes.append((float(n.lat), float(n.lon), 0))
          real_nodes.append(n)

        for way in new_result.ways:
          for n in way.nodes:
            node_to_way[n.id].append(way)

        nodes = np.asarray(nodes)
        nodes = geodetic2ecef(nodes)
        tree = spatial.cKDTree(nodes)

        query_lock.acquire()
        last_query_result = new_result, tree, real_nodes, node_to_way
        last_query_pos = last_gps
        cache_valid = True
        query_lock.release()

      except Exception as e:
        print e
        query_lock.acquire()
        last_query_result = None
        query_lock.release()


def mapsd_thread():
  global last_gps

  context = zmq.Context()
  gps_sock = messaging.sub_sock(context, service_list['gpsLocation'].port, conflate=True)
  gps_external_sock = messaging.sub_sock(context, service_list['gpsLocationExternal'].port, conflate=True)
  map_data_sock = messaging.pub_sock(context, service_list['liveMapData'].port)

  cur_way = None
  curvature_valid = False
  curvature = None
  upcoming_curvature = 0.
  dist_to_turn = 0.
  road_points = None

  while True:
    gps = messaging.recv_one(gps_sock)
    gps_ext = messaging.recv_one_or_none(gps_external_sock)

    if gps_ext is not None:
      gps = gps_ext.gpsLocationExternal
    else:
      gps = gps.gpsLocation

    last_gps = gps

    fix_ok = gps.flags & 1
    if not fix_ok or last_query_result is None or not cache_valid:
      cur_way = None
      curvature = None
      curvature_valid = False
      upcoming_curvature = 0.
      dist_to_turn = 0.
      road_points = None
      map_valid = False
    else:
      map_valid = True
      lat = gps.latitude
      lon = gps.longitude
      heading = gps.bearing
      speed = gps.speed

      query_lock.acquire()
      cur_way = Way.closest(last_query_result, lat, lon, heading, cur_way)
      if cur_way is not None:
        pnts, curvature_valid = cur_way.get_lookahead(last_query_result, lat, lon, heading, MAPS_LOOKAHEAD_DISTANCE)

        xs = pnts[:, 0]
        ys = pnts[:, 1]
        road_points = map(float, xs), map(float, ys)

        if speed < 10:
          curvature_valid = False
        if curvature_valid and pnts.shape[0] <= 3:
          curvature_valid = False

        # The curvature is valid when at least MAPS_LOOKAHEAD_DISTANCE of road is found
        if curvature_valid:
          # Compute the curvature for each point
          with np.errstate(divide='ignore'):
            circles = [circle_through_points(*p) for p in zip(pnts, pnts[1:], pnts[2:])]
            circles = np.asarray(circles)
            radii = np.nan_to_num(circles[:, 2])
            radii[radii < 10] = np.inf
            curvature = 1. / radii

          # Index of closest point
          closest = np.argmin(np.linalg.norm(pnts, axis=1))
          dist_to_closest = pnts[closest, 0]  # We can use x distance here since it should be close

          # Compute distance along path
          dists = list()
          dists.append(0)
          for p, p_prev in zip(pnts, pnts[1:, :]):
            dists.append(dists[-1] + np.linalg.norm(p - p_prev))
          dists = np.asarray(dists)
          dists = dists - dists[closest] + dist_to_closest
          dists = dists[1:-1]

          close_idx = np.logical_and(dists > 0, dists < 500)
          dists = dists[close_idx]
          curvature = curvature[close_idx]

          if len(curvature):
            # TODO: Determine left or right turn
            curvature = np.nan_to_num(curvature)

            # Outlier rejection
            new_curvature = np.percentile(curvature, 90, interpolation='lower')

            k = 0.6
            upcoming_curvature = k * upcoming_curvature + (1 - k) * new_curvature
            in_turn_indices = curvature > 0.8 * new_curvature

            if np.any(in_turn_indices):
              dist_to_turn = np.min(dists[in_turn_indices])
            else:
              dist_to_turn = 999
          else:
            upcoming_curvature = 0.
            dist_to_turn = 999

      query_lock.release()

    dat = messaging.new_message()
    dat.init('liveMapData')

    if last_gps is not None:
      dat.liveMapData.lastGps = last_gps

    if cur_way is not None:
      dat.liveMapData.wayId = cur_way.id

      # Seed limit
      max_speed = cur_way.max_speed
      if max_speed is not None:
        dat.liveMapData.speedLimitValid = True
        dat.liveMapData.speedLimit = max_speed

      # Curvature
      dat.liveMapData.curvatureValid = curvature_valid
      dat.liveMapData.curvature = float(upcoming_curvature)
      dat.liveMapData.distToTurn = float(dist_to_turn)
      if road_points is not None:
        dat.liveMapData.roadX, dat.liveMapData.roadY = road_points
      if curvature is not None:
        dat.liveMapData.roadCurvatureX = map(float, dists)
        dat.liveMapData.roadCurvature = map(float, curvature)

    dat.liveMapData.mapValid = map_valid

    map_data_sock.send(dat.to_bytes())


def main(gctx=None):
  params = Params()
  dongle_id = params.get("DongleId")
  crash.bind_user(id=dongle_id)
  crash.bind_extra(version=version, dirty=dirty, is_eon=True)
  crash.install()

  setup_thread_excepthook()
  main_thread = threading.Thread(target=mapsd_thread)
  main_thread.daemon = True
  main_thread.start()

  q_thread = threading.Thread(target=query_thread)
  q_thread.daemon = True
  q_thread.start()

  while True:
    time.sleep(0.1)


if __name__ == "__main__":
  main()
