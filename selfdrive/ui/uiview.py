#!/usr/bin/env python3
import time
import signal
import multiprocessing
import cereal.messaging as messaging
from selfdrive.manager import start_managed_process, kill_managed_process


def send_controls_packet(pm):
  while True:
    dat = messaging.new_message('controlsState')
    dat.controlsState = {
      "rearViewCam": False,
    }
    pm.send('controlsState', dat)
    time.sleep(0.01)


def send_thermal_packet(pm):
  while True:
    dat = messaging.new_message('thermal')
    dat.thermal = {
      'started': True,
    }
    pm.send('thermal', dat)
    time.sleep(0.01)


def main():
  pm = messaging.PubMaster(['controlsState', 'thermal'])
  controls_sender = multiprocessing.Process(target=send_controls_packet, args=[pm])
  thermal_sender = multiprocessing.Process(target=send_thermal_packet, args=[pm])
  controls_sender.start()
  thermal_sender.start()

  start_managed_process('ui')
  start_managed_process('camerad')

  def terminate():
    print('got SIGTERM, exiting..')
    kill_managed_process('ui')
    kill_managed_process('camerad')
    controls_sender.terminate()
    thermal_sender.terminate()
    exit()

  signal.signal(signal.SIGTERM, terminate)


if __name__ == '__main__':
  main()
