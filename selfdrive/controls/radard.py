#!/usr/bin/env python3
import importlib
from typing import Optional

from cereal import messaging, car
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper, Priority, config_realtime_process
from openpilot.common.swaglog import cloudlog


# radar tracks
SPEED, ACCEL = 0, 1     # Kalman filter states enum

# stationary qualification parameters
V_EGO_STATIONARY = 4.   # no stationary object flag below this speed

RADAR_TO_CENTER = 2.7   # (deprecated) RADAR is ~ 2.7m ahead from center of car
RADAR_TO_CAMERA = 1.52  # RADAR is ~ 1.5m ahead from center of mesh frame


class RadarD:
  def __init__(self, radar_ts: float, delay: int = 0):
    self.points: dict[int, tuple[float, float, float]] = {}

    self.radar_tracks_valid = False

  def update(self, rr: Optional[car.RadarData]):
    radar_points = []
    radar_errors = []
    if rr is not None:
      radar_points = rr.points
      radar_errors = rr.errors

    self.radar_tracks_valid = len(radar_errors) == 0

    self.points = {}
    for pt in radar_points:
      self.points[pt.trackId] = (pt.dRel, pt.yRel, pt.vRel)

  def publish(self):
    tracks_msg = messaging.new_message('liveTracks', len(self.points))
    tracks_msg.valid = self.radar_tracks_valid
    for index, tid in enumerate(sorted(self.points.keys())):
      tracks_msg.liveTracks[index] = {
        "trackId": tid,
        "dRel": float(self.points[tid][0]) + RADAR_TO_CAMERA,
        "yRel": -float(self.points[tid][1]),
        "vRel": float(self.points[tid][2]),
      }

    return tracks_msg


# publishes radar tracks
def main():
  config_realtime_process(5, Priority.CTRL_LOW)

  # wait for stats about the car to come in from controls
  cloudlog.info("radard is waiting for CarParams")
  with car.CarParams.from_bytes(Params().get("CarParams", block=True)) as msg:
    CP = msg
  cloudlog.info("radard got CarParams")

  # import the radar from the fingerprint
  cloudlog.info("radard is importing %s", CP.carName)
  RadarInterface = importlib.import_module(f'selfdrive.car.{CP.carName}.radar_interface').RadarInterface

  # *** setup messaging
  can_sock = messaging.sub_sock('can')
  pub_sock = messaging.pub_sock('liveTracks')

  RI = RadarInterface(CP)

  # TODO timing is different between cars, need a single time step for all cars
  # TODO just take the fastest one for now, and keep resending same messages for slower radars
  rk = Ratekeeper(1.0 / CP.radarTimeStep, print_delay_threshold=None)
  RD = RadarD(CP.radarTimeStep, RI.delay)

  while 1:
    can_strings = messaging.drain_sock_raw(can_sock, wait_for_one=True)
    rr = RI.update(can_strings)
    if rr is None:
      continue

    RD.update(rr)
    msg = RD.publish()
    pub_sock.send(msg.to_bytes())

    rk.monitor_time()


if __name__ == "__main__":
  main()
