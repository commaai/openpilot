#!/usr/bin/env python3
import os
import subprocess
import multiprocessing
import signal
import time

import cereal.messaging as messaging
from common.params import Params

from common.basedir import BASEDIR

KILL_TIMEOUT = 15


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

  # TODO: refactor with manager start/kill
  proc_cam = subprocess.Popen(os.path.join(BASEDIR, "selfdrive/camerad/camerad"), cwd=os.path.join(BASEDIR, "selfdrive/camerad"))
  proc_mon = subprocess.Popen(os.path.join(BASEDIR, "selfdrive/modeld/dmonitoringmodeld"), cwd=os.path.join(BASEDIR, "selfdrive/modeld"))

  params = Params()
  is_rhd = False
  is_rhd_checked = False
  should_exit = False

  def terminate(signalNumber, frame):
    print('got SIGTERM, exiting..')
    should_exit = True
    send_dmon_packet(pm, [is_rhd, is_rhd_checked, not should_exit])
    proc_cam.send_signal(signal.SIGINT)
    proc_mon.send_signal(signal.SIGINT)
    kill_start = time.time()
    while proc_cam.poll() is None:
      if time.time() - kill_start > KILL_TIMEOUT:
        from selfdrive.swaglog import cloudlog
        cloudlog.critical("FORCE REBOOTING PHONE!")
        os.system("date >> /sdcard/unkillable_reboot")
        os.system("reboot")
        raise RuntimeError
      continue
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
