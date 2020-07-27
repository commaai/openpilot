#!/usr/bin/env python3
import os
import time
import signal
import subprocess
import multiprocessing
import cereal.messaging as messaging
from common.basedir import BASEDIR


def send_packets(pm, idx=0):
  while True:
    dat = messaging.new_message('controlsState')
    dat.controlsState.rearViewCam = False
    pm.send('controlsState', dat)
    if idx % 50 == 0:  # thermal runs at 2 hz
      dat = messaging.new_message('thermal')
      dat.thermal.started = True
      pm.send('thermal', dat)
    time.sleep(1 / 100.)
    idx += 1


def main():
  pm = messaging.PubMaster(['controlsState', 'thermal'])
  packet_sender = multiprocessing.Process(target=send_packets, args=[pm])
  packet_sender.start()
  proc_cam = subprocess.Popen(os.path.join(BASEDIR, "selfdrive/camerad/camerad"), cwd=os.path.join(BASEDIR, "selfdrive/camerad"))
  proc_ui = subprocess.Popen(os.path.join(BASEDIR, "selfdrive/ui/ui"), cwd=os.path.join(BASEDIR, "selfdrive/ui"))

  def terminate(signalNumber, frame):
    proc_cam.send_signal(signal.SIGINT)
    proc_ui.send_signal(signal.SIGINT)
    packet_sender.terminate()
    exit()

  signal.signal(signal.SIGINT, terminate)  # catch ctrl-c


if __name__ == '__main__':
  main()
