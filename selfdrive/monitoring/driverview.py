#!/usr/bin/env python3
import os
import sys
import time
import signal
import subprocess
import multiprocessing
import cereal.messaging as messaging

from common.params import Params
from common.basedir import BASEDIR

KILL_TIMEOUT = 15


def send_controls_packet(pm, d):
  while True:
    dat = messaging.new_message('controlsState')
    dat.controlsState = {
      "rearViewCam": d,
    }
    pm.send('controlsState', dat)
    time.sleep(1 / 100)


def send_thermal_packet(pm):
  while True:
    dat = messaging.new_message('thermal')
    dat.thermal = {
      'started': True,
    }
    pm.send('thermal', dat)
    time.sleep(1 / 2.)  # 2 hz


def send_dmon_packet(pm, d):
  dat = messaging.new_message('dMonitoringState')
  dat.dMonitoringState = {
    "isRHD": d[0],
    "rhdChecked": d[1],
    "isPreview": d[2],
  }
  pm.send('dMonitoringState', dat)


def main(driverview=True, uiview=False):
  senders = ['controlsState']
  if driverview:
    senders.append('dMonitoringState')
  elif uiview:
    senders.append('thermal')
  pm = messaging.PubMaster(senders)

  rearViewCam = True
  if uiview:
    rearViewCam = False
    thermal_sender = multiprocessing.Process(target=send_thermal_packet, args=[pm])
    thermal_sender.start()

  controls_sender = multiprocessing.Process(target=send_controls_packet, args=[pm, rearViewCam])
  controls_sender.start()

  # TODO: refactor with manager start/kill
  proc_cam = subprocess.Popen(os.path.join(BASEDIR, "selfdrive/camerad/camerad"), cwd=os.path.join(BASEDIR, "selfdrive/camerad"))
  if driverview:
    proc_mon = subprocess.Popen(os.path.join(BASEDIR, "selfdrive/modeld/dmonitoringmodeld"), cwd=os.path.join(BASEDIR, "selfdrive/modeld"))
  elif uiview:
    proc_ui = subprocess.Popen(os.path.join(BASEDIR, "selfdrive/ui/ui"), cwd=os.path.join(BASEDIR, "selfdrive/ui"))

  params = Params()
  is_rhd = False
  is_rhd_checked = False
  should_exit = False

  def terminate(signalNumber, frame):
    print('got SIGTERM, exiting..')
    proc_cam.send_signal(signal.SIGINT)
    kill_start = time.time()
    while proc_cam.poll() is None:
      if time.time() - kill_start > KILL_TIMEOUT:
        from selfdrive.swaglog import cloudlog
        cloudlog.critical("FORCE REBOOTING PHONE!")
        os.system("date >> /sdcard/unkillable_reboot")
        os.system("reboot")
        raise RuntimeError
      continue

    if driverview:
      should_exit = True
      send_dmon_packet(pm, [is_rhd, is_rhd_checked, not should_exit])
      proc_mon.send_signal(signal.SIGINT)
    elif uiview:
      proc_ui.send_signal(signal.SIGINT)
      thermal_sender.terminate()
    controls_sender.terminate()
    exit()

  signal.signal(signal.SIGTERM, terminate)
  signal.signal(signal.SIGINT, terminate)  # catch ctrl-c as well

  if driverview:
    while True:
      send_dmon_packet(pm, [is_rhd, is_rhd_checked, not should_exit])
      if not is_rhd_checked:
        is_rhd = params.get("IsRHD") == b"1"
        is_rhd_checked = True

      time.sleep(0.01)


if __name__ == '__main__':
  args = {}
  if len(sys.argv) > 1:
    arg = sys.argv[1].lower().strip()
    if arg in ['--front', '-f']:
      args['driverview'], args['uiview'] = False, True
  main(**args)
