#!/usr/bin/env python3
import time
import signal
import multiprocessing
import cereal.messaging as messaging
from selfdrive.manager import start_managed_process, kill_managed_process
from common.params import Params


def send_controls_packet(pm):
  while True:
    dat = messaging.new_message('controlsState')
    dat.controlsState = {
      "rearViewCam": True,
    }
    pm.send('controlsState', dat)
    time.sleep(0.01)


def send_dmon_packet(pm, d):
  dat = messaging.new_message('dMonitoringState')
  dat.dMonitoringState = {
    "isRHD": d[0],
    "rhdChecked": d[1],
    "isPreview": d[2],
  }
  pm.send('dMonitoringState', dat)


def main():
  pm = messaging.PubMaster(['controlsState', 'dMonitoringState'])
  controls_sender = multiprocessing.Process(target=send_controls_packet, args=[pm])
  controls_sender.start()

  start_managed_process('dmonitoringmodeld')
  start_managed_process('camerad')

  params = Params()
  is_rhd = False
  is_rhd_checked = False
  should_exit = False

  def terminate():
    print('got SIGTERM, exiting..')
    should_exit = True
    send_dmon_packet(pm, [is_rhd, is_rhd_checked, not should_exit])
    kill_managed_process('dmonitoringmodeld')
    kill_managed_process('camerad')
    controls_sender.terminate()
    exit()

  signal.signal(signal.SIGTERM, terminate)

  while True:
    send_dmon_packet(pm, [is_rhd, is_rhd_checked, not should_exit])
    if not is_rhd_checked:
      is_rhd = params.get("IsRHD") == b"1"
      is_rhd_checked = True

    time.sleep(0.01)


if __name__ == '__main__':
  main()
