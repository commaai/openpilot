#!/usr/bin/env python
import os
import zmq
import json
import time
import numpy as np
from numpy import linalg as LA
from threading import Thread
from scipy.spatial import cKDTree

from selfdrive.swaglog import cloudlog
from cereal.services import service_list
from common.realtime import Ratekeeper
from common.kalman.ned import geodetic2ecef, NED
import cereal.messaging as messaging
from cereal import log
import warnings
from selfdrive.config import Conversions as CV


if os.getenv('EON_LIVE') == '1':
  _REMOTE_ADDR = "192.168.5.11"
else:
  _REMOTE_ADDR = "127.0.0.1"

LOOP = 'small_loop'

TRACK_SNAP_DIST = 17.  # snap to a track below this distance
TRACK_LOST_DIST = 30.  # lose a track above this distance
INSTRUCTION_APPROACHING_DIST = 200.
INSTRUCTION_ACTIVE_DIST = 20.

ROT_CENTER_TO_LOC = 1.2

class INSTRUCTION_STATE:
  NONE = log.UiNavigationEvent.Status.none
  PASSIVE = log.UiNavigationEvent.Status.passive
  APPROACHING = log.UiNavigationEvent.Status.approaching
  ACTIVE = log.UiNavigationEvent.Status.active


def convert_ecef_to_capnp(points):
  points_capnp = []
  for p in points:
    point = log.ECEFPoint.new_message()
    point.x, point.y, point.z = map(float, p[0:3])
    points_capnp.append(point)
  return points_capnp


def get_spaced_points(track, start_index, cur_ecef, v_ego):
  active_points = []
  look_ahead = 5.0 + 1.5 * v_ego  # 5m + 1.5s

  # forward and backward passes for better poly fit
  for idx_sign in [1, -1]:
    for i in range(0, 1000):
      index = start_index + i * idx_sign
      # loop around 
      p = track[index % len(track)]

      distance = LA.norm(cur_ecef - p[0:3])
      if i > 5 and distance > look_ahead:
        break

      active_points.append([p, index])

  # sort points by index
  active_points = sorted(active_points, key=lambda pt: pt[1])
  active_points = [p[0] for p in active_points]

  return active_points


def fit_poly(points, cur_ecef, cur_heading, ned_converter):
  relative_points = []
  for point in points.points:
    p = np.array([point.x, point.y, point.z])
    relative_points.append(ned_converter.ecef_to_ned_matrix.dot(p - cur_ecef))

  relative_points = np.matrix(np.vstack(relative_points))

  # Calculate relative postions and rotate wrt to heading of car
  c, s = np.cos(-cur_heading), np.sin(-cur_heading)
  R = np.array([[c, -s], [s, c]])

  n, e = relative_points[:, 0], relative_points[:, 1]
  relative_points = np.hstack([e, n])
  rotated_points = relative_points.dot(R)

  rotated_points = np.array(rotated_points)
  x, y = rotated_points[:, 1], -rotated_points[:, 0]

  warnings.filterwarnings('error')

  # delete points that go backward
  max_x = x[0]
  x_new = []
  y_new = []

  for xi, yi in zip(x, y):
    if xi > max_x:
      max_x = xi
      x_new.append(xi)
      y_new.append(yi)

  x = np.array(x_new)
  y = np.array(y_new)

  if len(x) > 10:
    poly = map(float, np.polyfit(x + ROT_CENTER_TO_LOC, y, 3))  # 1.2m in front
  else:
    poly = [0.0, 0.0, 0.0, 0.0]
  return poly, float(max_x + ROT_CENTER_TO_LOC)


def get_closest_track(tracks, track_trees, cur_ecef):

  track_list = [(name, track_trees[name].query(cur_ecef, 1)) for name in track_trees]
  closest_name, [closest_distance, closest_idx] = min(track_list, key=lambda x: x[1][0])

  return {'name': closest_name,
          'distance': closest_distance,
          'idx': closest_idx,
          'speed': tracks[closest_name][closest_idx][3],
          'accel': tracks[closest_name][closest_idx][4]}


def get_track_from_name(tracks, track_trees, track_name, cur_ecef):
  if track_name is None:
    return None
  else:
    track_distance, track_idx = track_trees[track_name].query(cur_ecef, 1)
  return {'name': track_name,
          'distance': track_distance,
          'idx': track_idx,
          'speed': tracks[track_name][track_idx][3],
          'accel': tracks[track_name][track_idx][4]}


def get_tracks_from_instruction(tracks,instruction, track_trees, cur_ecef):
  if instruction is None:
    return None, None
  else:
    source_track = get_track_from_name(tracks, track_trees, instruction['source'], cur_ecef)
    target_track = get_track_from_name(tracks, track_trees, instruction['target'], cur_ecef)
    return source_track, target_track


def get_next_instruction_distance(track, instruction, cur_ecef):
  if instruction is None:
    return None
  else:
    return np.linalg.norm(cur_ecef - track[instruction['start_idx']][0:3])


def update_current_track(tracks, cur_track, cur_ecef, track_trees):

  closest_track = get_closest_track(tracks, track_trees, cur_ecef)

  # have we lost current track?
  if cur_track is not None:
    cur_track = get_track_from_name(tracks, track_trees, cur_track['name'], cur_ecef)
    if cur_track['distance'] > TRACK_LOST_DIST:
      cur_track = None

  # did we snap to a new track?
  if cur_track is None and closest_track['distance'] < TRACK_SNAP_DIST:
    cur_track = closest_track

  return cur_track, closest_track


def update_instruction(instruction, instructions, cur_track, source_track, state, cur_ecef, tracks):

  if state == INSTRUCTION_STATE.ACTIVE:  # instruction frozen, just update distance
    instruction['distance'] = get_next_instruction_distance(tracks[source_track['name']], instruction, cur_ecef)
    return instruction

  elif cur_track is None:
    return None

  else:
    instruction_list = [i for i in instructions[cur_track['name']] if i['start_idx'] > cur_track['idx']]
    if len(instruction_list) > 0:
      next_instruction = min(instruction_list, key=lambda x: x['start_idx'])
      next_instruction['distance'] = get_next_instruction_distance(tracks[cur_track['name']], next_instruction, cur_ecef)
      return next_instruction
    else:
      return None


def calc_instruction_state(state, cur_track, closest_track, source_track, target_track, instruction):

  lost_track_or_instruction = cur_track is None or instruction is None

  if state == INSTRUCTION_STATE.NONE:
    if lost_track_or_instruction:
      pass
    else:
      state = INSTRUCTION_STATE.PASSIVE

  elif state == INSTRUCTION_STATE.PASSIVE:
    if lost_track_or_instruction:
      state = INSTRUCTION_STATE.NONE
    elif instruction['distance'] < INSTRUCTION_APPROACHING_DIST:
      state = INSTRUCTION_STATE.APPROACHING

  elif state == INSTRUCTION_STATE.APPROACHING:
    if lost_track_or_instruction:
      state = INSTRUCTION_STATE.NONE
    elif instruction['distance'] < INSTRUCTION_ACTIVE_DIST:
      state = INSTRUCTION_STATE.ACTIVE

  elif state == INSTRUCTION_STATE.ACTIVE:
    if lost_track_or_instruction:
      state = INSTRUCTION_STATE.NONE
    elif target_track['distance'] < TRACK_SNAP_DIST and \
         source_track['idx'] > instruction['start_idx'] and \
         instruction['distance'] > 10.:
      state = INSTRUCTION_STATE.NONE
      cur_track = target_track

  return state, cur_track


def gps_planner_point_selection():

  DECIMATION = 1

  cloudlog.info("Starting gps_plannerd point selection")

  rk = Ratekeeper(10.0, print_delay_threshold=np.inf)

  context = zmq.Context()
  live_location = messaging.sub_sock(context, 'liveLocation', conflate=True, addr=_REMOTE_ADDR)
  car_state = messaging.sub_sock(context, 'carState', conflate=True)
  gps_planner_points = messaging.pub_sock(context, 'gpsPlannerPoints')
  ui_navigation_event = messaging.pub_sock(context, 'uiNavigationEvent')

  # Load tracks and instructions from disk
  basedir = os.environ['BASEDIR']
  tracks = np.load(os.path.join(basedir, 'selfdrive/controls/tracks/%s.npy' % LOOP)).item()
  instructions = json.loads(open(os.path.join(basedir, 'selfdrive/controls/tracks/instructions_%s.json' % LOOP)).read())

  # Put tracks into KD-trees
  track_trees = {}
  for name in tracks:
    tracks[name] = tracks[name][::DECIMATION]
    track_trees[name] = cKDTree(tracks[name][:,0:3]) # xyz
  cur_track = None
  source_track = None
  target_track = None
  instruction = None
  v_ego = 0.
  state = INSTRUCTION_STATE.NONE

  counter = 0

  while True:
    counter += 1
    ll = messaging.recv_one(live_location)
    ll = ll.liveLocation
    cur_ecef = geodetic2ecef((ll.lat, ll.lon, ll.alt))
    cs = messaging.recv_one_or_none(car_state)
    if cs is not None:
      v_ego = cs.carState.vEgo

    cur_track, closest_track = update_current_track(tracks, cur_track, cur_ecef, track_trees)
    #print cur_track

    instruction = update_instruction(instruction, instructions, cur_track, source_track, state, cur_ecef, tracks)

    source_track, target_track = get_tracks_from_instruction(tracks, instruction, track_trees, cur_ecef)

    state, cur_track = calc_instruction_state(state, cur_track, closest_track, source_track, target_track, instruction)

    active_points = []

    # Make list of points used by gpsPlannerPlan
    if cur_track is not None:
      active_points = get_spaced_points(tracks[cur_track['name']], cur_track['idx'], cur_ecef, v_ego)

    cur_pos = log.ECEFPoint.new_message()
    cur_pos.x, cur_pos.y, cur_pos.z = map(float, cur_ecef)
    m = messaging.new_message()
    m.init('gpsPlannerPoints')
    m.gpsPlannerPoints.curPos = cur_pos
    m.gpsPlannerPoints.points = convert_ecef_to_capnp(active_points)
    m.gpsPlannerPoints.valid = len(active_points) > 10
    m.gpsPlannerPoints.trackName = "none" if cur_track is None else cur_track['name']
    m.gpsPlannerPoints.speedLimit = 100. if cur_track is None else float(cur_track['speed'])
    m.gpsPlannerPoints.accelTarget = 0. if cur_track is None else float(cur_track['accel'])
    gps_planner_points.send(m.to_bytes())

    m = messaging.new_message()
    m.init('uiNavigationEvent')
    m.uiNavigationEvent.status = state
    m.uiNavigationEvent.type = "none" if instruction is None else instruction['type']
    m.uiNavigationEvent.distanceTo = 0. if instruction is None else float(instruction['distance'])
    endRoadPoint = log.ECEFPoint.new_message()
    m.uiNavigationEvent.endRoadPoint = endRoadPoint
    ui_navigation_event.send(m.to_bytes())

    rk.keep_time()


def gps_planner_plan():

  context = zmq.Context()

  live_location = messaging.sub_sock(context, 'liveLocation', conflate=True, addr=_REMOTE_ADDR)
  gps_planner_points = messaging.sub_sock(context, 'gpsPlannerPoints', conflate=True)
  gps_planner_plan = messaging.pub_sock(context, 'gpsPlannerPlan')

  points = messaging.recv_one(gps_planner_points).gpsPlannerPoints

  target_speed = 100. * CV.MPH_TO_MS
  target_accel = 0.

  last_ecef = np.array([0., 0., 0.])

  while True:
    ll = messaging.recv_one(live_location)
    ll = ll.liveLocation
    p = messaging.recv_one_or_none(gps_planner_points)
    if p is not None:
      points = p.gpsPlannerPoints
      target_speed = p.gpsPlannerPoints.speedLimit
      target_accel = p.gpsPlannerPoints.accelTarget

    cur_ecef = geodetic2ecef((ll.lat, ll.lon, ll.alt))

    # TODO: make NED initialization much faster so we can run this every time step
    if np.linalg.norm(last_ecef - cur_ecef) > 200.:
      ned_converter = NED(ll.lat, ll.lon, ll.alt)
      last_ecef = cur_ecef

    cur_heading = np.radians(ll.heading)

    if points.valid:
      poly, x_lookahead = fit_poly(points, cur_ecef, cur_heading, ned_converter)
    else:
      poly, x_lookahead = [0.0, 0.0, 0.0, 0.0], 0.

    valid = points.valid

    m = messaging.new_message()
    m.init('gpsPlannerPlan')
    m.gpsPlannerPlan.valid = valid
    m.gpsPlannerPlan.poly = poly
    m.gpsPlannerPlan.trackName = points.trackName
    r = []
    for p in points.points:
      point = log.ECEFPoint.new_message()
      point.x, point.y, point.z = p.x, p.y, p.z
      r.append(point)
    m.gpsPlannerPlan.points = r
    m.gpsPlannerPlan.speed = target_speed
    m.gpsPlannerPlan.acceleration = target_accel
    m.gpsPlannerPlan.xLookahead = x_lookahead
    gps_planner_plan.send(m.to_bytes())


def main(gctx=None):
  cloudlog.info("Starting gps_plannerd main thread")

  point_thread = Thread(target=gps_planner_point_selection)
  point_thread.daemon = True
  control_thread = Thread(target=gps_planner_plan)
  control_thread.daemon = True

  point_thread.start()
  control_thread.start()

  while True:
    time.sleep(1)


if __name__ == "__main__":
  main()
