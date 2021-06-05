#!/usr/bin/env python3
import threading
import time

import cereal.messaging as messaging
from common.realtime import Ratekeeper
from selfdrive.manager.process import ensure_running
from selfdrive.manager.process_config import managed_processes
from tools.lib.route import Route
from tools.lib.logreader import LogReader

def replay_service(s, dt, msgs):
  pm = messaging.PubMaster([s, ])
  rk = Ratekeeper(1 / dt, print_delay_threshold=None)
  s_msgs = [m for m in msgs if m.which() == s]
  for m in s_msgs:
    pm.send(s, m.as_builder())
    rk.keep_time()

# TODO: send real frames
def replay_cameras():
  pm = messaging.PubMaster(["roadCameraState", "driverCameraState"])
  rk = Ratekeeper(1 / 0.05, print_delay_threshold=None)

  frame = 0
  while True:
    m = messaging.new_message("roadCameraState")
    m.roadCameraState.frameId = frame
    pm.send("roadCameraState", m)

    m = messaging.new_message("driverCameraState")
    m.driverCameraState.frameId = frame
    pm.send("driverCameraState", m)

    frame += 1
    rk.keep_time()

def regen_segment(route, seg):

  r = Route(route)
  lr = list(LogReader(r.log_paths()[seg]))

  fake_daemons = {
    'sensord': [
      threading.Thread(target=replay_service, args=('sensorEvents', 0.01, lr)),
    ],
    'pandad': [
      threading.Thread(target=replay_service, args=('can', 0.01, lr)),
      threading.Thread(target=replay_service, args=('pandaState', 0.01, lr)),
    ],
    'camerad': [
      threading.Thread(target=replay_cameras),
    ],
  }

  # startup procs
  ensure_running(managed_processes.values(), started=True, not_run=fake_daemons)
  for threads in fake_daemons.values():
    for t in threads:
      t.start()

  # run for 10s
  time.sleep(10)

  # kill everything
  for p in managed_processes.values():
    p.stop()


if __name__ == "__main__":
  regen_segment("ef895f46af5fd73f|2021-05-22--14-06-35", 15)
