#!/usr/bin/env python3
"""parknavd: computes the rotation toward a GPS destination for parking navigation.

Inputs:
  - gpsLocation            quectel GNSS fix (absolute position, heading while moving)
  - deviceMotion           locationd kalman pose (IMU + camera odometry, device frame)
  - extrinsicsCalibration  camera mount calibration

Publishes parkNavSignal at ~4 Hz; modeld injects its content as turn desires
(selfdrive/modeld/park_nav.py). Destination comes from the "ParkingDestination"
param as JSON: {"latitude": float, "longitude": float}.

deviceMotion's NED yaw has no absolute reference (the locationd filter does not
consume GNSS), so it is dead-reckoned by pose deltas and anchored with the GNSS
heading via a complementary filter whenever the car moves fast enough for the
GNSS heading to be trustworthy.
"""
import math
import time

import numpy as np

from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.selfdrive.locationd.helpers import Pose, PoseCalibrator
from openpilot.system.parknav.geometry import geodetic_to_local_ned, wrap_angle

PARKNAV_FREQ = 4.
PUB_DIVISOR = int(20. / PARKNAV_FREQ)

MAX_HEADING_STD = 0.15       # rad
MAX_BEARING_ACCURACY = 10.   # deg
MAX_FIX_AGE = 5.0            # s
ARRIVAL_RADIUS = 5.         # m
MAX_NAV_DIST = 500.          # m

MIN_GPS_HEADING_SPEED = 1.5  # m/s
MIN_INTEGRATION_SPEED = 0.1  # m/s
MAX_INTEGRATION_DT = 0.2     # s
GPS_YAW_CORRECTION_GAIN = 0.2


class ParkNavEstimator:
  """Dead reckoning between GNSS fixes + GNSS heading anchoring.

  Position is tracked in a local NED frame anchored at the destination
  (the destination is the origin). Heading is the kalman yaw with its
  drift corrected against the GNSS bearing.
  """
  def __init__(self):
    self.yaw = 0.                  # corrected NED yaw [rad]
    self.yaw_initialized = False
    self.heading_anchored = False
    self.raw_yaw_prev = 0.         # uncorrected kalman yaw of the previous update
    self.yaw_valid = False

    self.pos_ned = np.zeros(2)     # [north, east] meters relative to the destination
    self.last_t: float | None = None

    self.last_fix_t: float | None = None
    self.arrived = False
    self.last_destination: tuple[float, float] | None = None


def get_destination(params: Params) -> tuple[float, float] | None:
  dest = params.get("ParkingDestination")
  if dest is None:
    return None
  try:
    return float(dest["latitude"]), float(dest["longitude"])
  except Exception:
    cloudlog.exception("bad ParkingDestination param")
    return None


def update_pose(state: ParkNavEstimator, sm, calibrator: PoseCalibrator) -> None:
  """Update yaw (dead reckoning + GNSS anchoring) and position (velocity integration)."""
  if not calibrator.calib_valid:
    state.yaw_valid = False
    state.last_t = None
    return
  t = time.monotonic()

  dm = sm['deviceMotion']
  pose = calibrator.build_calibrated_pose(Pose.from_device_motion(dm))

  # heading: dead-reckon the corrected yaw by the kalman pose delta
  raw_yaw = float(pose.orientation.yaw)
  yaw_std = float(dm.orientationNED.zStd)
  state.yaw_valid = dm.orientationNED.valid and 0. <= yaw_std < MAX_HEADING_STD and math.isfinite(raw_yaw)
  if state.yaw_valid:
    if state.yaw_initialized:
      state.yaw = wrap_angle(state.yaw + wrap_angle(raw_yaw - state.raw_yaw_prev))
    else:
      state.yaw = raw_yaw
      state.yaw_initialized = True
    state.raw_yaw_prev = raw_yaw

  # Rotate velocity with the same absolute heading used for the target bearing.
  orientation = pose.orientation.xyz.copy()
  orientation[2] = state.yaw
  v_ned = rot_from_euler(orientation) @ pose.velocity.xyz
  speed = float(np.linalg.norm(v_ned[:2]))
  if state.yaw_valid and state.heading_anchored and state.last_t is not None and speed > MIN_INTEGRATION_SPEED:
    dt = min(max(t - state.last_t, 0.), MAX_INTEGRATION_DT)
    if dt > 0:
      state.pos_ned += v_ned[:2] * dt
  state.last_t = t


def update_gps(state: ParkNavEstimator, sm, destination: tuple[float, float] | None) -> None:
  if not sm.updated['gpsLocationExternal'] or not sm.valid['gpsLocationExternal']:
    return
  gps = sm['gpsLocationExternal']
  t = time.monotonic()

  if gps.hasFix and destination is not None:
    state.pos_ned = np.array(geodetic_to_local_ned(gps.latitude, gps.longitude, *destination))
    state.last_fix_t = t

  if gps.hasFix and 0. <= gps.bearingAccuracyDeg < MAX_BEARING_ACCURACY and math.isfinite(gps.bearingDeg) and abs(gps.speed) > MIN_GPS_HEADING_SPEED:
    gps_bearing = math.radians(gps.bearingDeg)
    if state.yaw_initialized and state.yaw_valid:
      # Consume each course measurement once. The first fixes the arbitrary NED origin.
      if state.heading_anchored:
        state.yaw = wrap_angle(state.yaw + GPS_YAW_CORRECTION_GAIN * wrap_angle(gps_bearing - state.yaw))
      else:
        state.yaw = wrap_angle(gps_bearing)
        state.heading_anchored = True


def make_signal(state: ParkNavEstimator, destination: tuple[float, float] | None):
  """Build the parkNavSignal message; None means don't publish (no destination)."""
  if destination is None:
    return None

  if destination != state.last_destination:
    state.arrived = False
    state.last_destination = destination

  now = time.monotonic()
  fix_fresh = state.last_fix_t is not None and (now - state.last_fix_t) < MAX_FIX_AGE
  bearing_valid = state.heading_anchored and state.yaw_valid

  total_dist = float(np.linalg.norm(state.pos_ned))
  bearing_to_target = math.atan2(-state.pos_ned[1], -state.pos_ned[0])
  rel_bearing = wrap_angle(bearing_to_target - state.yaw)
  if fix_fresh and total_dist <= ARRIVAL_RADIUS:
    state.arrived = True
  arrived = state.arrived
  if arrived:
    rel_bearing = 0.
  valid = fix_fresh and bearing_valid and total_dist < MAX_NAV_DIST

  msg = messaging.new_message('parkNavSignal')
  sig = msg.parkNavSignal
  sig.relBearing = rel_bearing
  sig.lateralOffset = total_dist * math.sin(rel_bearing)
  sig.forwardDist = total_dist * math.cos(rel_bearing)
  sig.totalDist = total_dist
  sig.arrived = arrived
  sig.valid = valid
  return msg


def main():
  cloudlog.warning("parknavd init")
  params = Params()
  calibrator = PoseCalibrator()
  state = ParkNavEstimator()

  sm = messaging.SubMaster(['gpsLocationExternal', 'deviceMotion', 'extrinsicsCalibration'], poll='deviceMotion')
  pm = messaging.PubMaster(['parkNavSignal'])

  frame = 0
  while True:
    sm.update()
    frame += 1

    if sm.updated['extrinsicsCalibration']:
      calibrator.feed_extrinsics_calibration(sm['extrinsicsCalibration'])

    update_pose(state, sm, calibrator)

    nav_enabled = params.get_bool("ParkingNavEnabled")
    destination = get_destination(params) if nav_enabled else None

    update_gps(state, sm, destination)

    if frame % PUB_DIVISOR != 0:
      continue

    msg = make_signal(state, destination)
    if msg is not None:
      pm.send('parkNavSignal', msg)


if __name__ == "__main__":
  main()
