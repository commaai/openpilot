#!/usr/bin/env python3
import importlib
import cereal.messaging as messaging

from cereal import car
from common.params import Params
from common.realtime import Ratekeeper, Priority, config_realtime_process
from selfdrive.controls.lib.radar_helpers import RadarD
from system.swaglog import cloudlog



# fuses camera and radar data for best lead detection
def radard_thread(sm=None, pm=None, can_sock=None):
  config_realtime_process(5, Priority.CTRL_LOW)

  # wait for stats about the car to come in from controls
  cloudlog.info("radard is waiting for CarParams")
  CP = car.CarParams.from_bytes(Params().get("CarParams", block=True))
  cloudlog.info("radard got CarParams")

  # import the radar from the fingerprint
  cloudlog.info("radard is importing %s", CP.carName)
  RadarInterface = importlib.import_module(f'selfdrive.car.{CP.carName}.radar_interface').RadarInterface

  # *** setup messaging
  if can_sock is None:
    can_sock = messaging.sub_sock('can')
  if sm is None:
    sm = messaging.SubMaster(['modelV2', 'carState'], ignore_avg_freq=['modelV2', 'carState'])  # Can't check average frequency, since radar determines timing
  if pm is None:
    pm = messaging.PubMaster(['radarState', 'liveTracks'])

  RI = RadarInterface(CP)

  rk = Ratekeeper(1.0 / CP.radarTimeStep, print_delay_threshold=None)
  RD = RadarD(CP.radarTimeStep, RI.delay)

  while 1:
    can_strings = messaging.drain_sock_raw(can_sock, wait_for_one=True)
    rr = RI.update(can_strings)

    if rr is None:
      continue

    sm.update(0)

    dat = RD.update(sm, rr)
    dat.radarState.cumLagMs = -rk.remaining*1000.

    pm.send('radarState', dat)

    # *** publish tracks for UI debugging (keep last) ***
    tracks = RD.tracks
    dat = messaging.new_message('liveTracks', len(tracks))

    for cnt, ids in enumerate(sorted(tracks.keys())):
      dat.liveTracks[cnt] = {
        "trackId": ids,
        "dRel": float(tracks[ids].dRel),
        "yRel": float(tracks[ids].yRel),
        "vRel": float(tracks[ids].vRel),
      }
    pm.send('liveTracks', dat)

    rk.monitor_time()


def main(sm=None, pm=None, can_sock=None):
  radard_thread(sm, pm, can_sock)


if __name__ == "__main__":
  main()
