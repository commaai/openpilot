#!/usr/bin/env python3
import sys
import time
import signal
import multiprocessing
import cereal.messaging as messaging
from selfdrive.manager import start_managed_process, kill_managed_process
from common.params import Params


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


def main(driverview, uiview):
  pm = messaging.PubMaster(['controlsState', 'dMonitoringState', 'thermal'])

  if driverview:
    controls_sender = multiprocessing.Process(target=send_controls_packet, args=[pm, True])
  elif uiview:
    controls_sender = multiprocessing.Process(target=send_controls_packet, args=[pm, False])
    thermal_sender = multiprocessing.Process(target=send_thermal_packet, args=[pm])
    thermal_sender.start()
  controls_sender.start()


  start_managed_process('camerad')
  if driverview:
    start_managed_process('dmonitoringmodeld')
  elif uiview:
    start_managed_process('ui')

  params = Params()
  is_rhd = False
  is_rhd_checked = False
  should_exit = False

  def terminate(signalNumber, frame):
    print('got SIGTERM, exiting..')
    kill_managed_process('camerad')
    if driverview:
      should_exit = True
      send_dmon_packet(pm, [is_rhd, is_rhd_checked, not should_exit])
      kill_managed_process('dmonitoringmodeld')
    elif uiview:
      kill_managed_process('ui')
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
  driverview, uiview = True, False
  if len(sys.argv) > 1:
    arg = sys.argv[1].lower().strip()
    if arg in ['--front', '--ui', '-f', '-u']:
      driverview, uiview = False, True
  main(driverview=driverview, uiview=uiview)
