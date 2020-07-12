#!/usr/bin/env python3
import signal
import time
import multiprocessing

import cereal.messaging as messaging
from common.params import Params
from common.realtime import DT_CTRL, DT_DMON
import selfdrive.manager as manager


def send_controls_packet(pm):
  while True:
    dat = messaging.new_message('controlsState')
    dat.controlsState = {
      "rearViewCam": True,
    }
    pm.send('controlsState', dat)
    time.sleep(DT_CTRL)


def send_dmon_packet(pm, rhd, should_exit=False):
  dat = messaging.new_message('dMonitoringState')
  dat.dMonitoringState = {
    "isRHD": rhd,
    "rhdChecked": True,
    "isPreview": should_exit,
  }
  pm.send('dMonitoringState', dat)


def main():
  pm = messaging.PubMaster(['controlsState', 'dMonitoringState'])
  controls_sender = multiprocessing.Process(target=send_controls_packet, args=[pm])
  controls_sender.start()

  # TODO: procs really shouldn't be started outside manager
  manager.start_managed_process('camerad')
  manager.start_managed_process('dmonitoringmodeld')

  is_rhd = Params().get("IsRHD") == b"1"

  def terminate(signalNumber, frame):
    print('got SIGTERM, exiting..')
    send_dmon_packet(pm, is_rhd, should_exit=True)
    manager.kill_managed_process('camerad')
    manager.kill_managed_process('dmonitoringmodeld')
    controls_sender.terminate()
    exit()
  signal.signal(signal.SIGTERM, terminate)

  while True:
    send_dmon_packet(pm, is_rhd)
    time.sleep(DT_DMON)


if __name__ == '__main__':
  main()
