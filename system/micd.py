#!/usr/bin/env python3
import sounddevice as sd
import numpy as np

from cereal import messaging
from common.realtime import Ratekeeper


class Microphone:
  def __init__(self):
    self.pm = messaging.PubMaster(['microphone'])
    self.last_volume = 0

  def update(self):
    msg = messaging.new_message('microphone')
    microphone = msg.microphone
    microphone.noiseLevel = float(self.last_volume)
    self.pm.send('microphone', msg)

  def calculate_volume(self, indata, frames, time, status):
    self.last_volume = np.linalg.norm(indata)


def main():
  mic = Microphone()
  rk = Ratekeeper(1.0)

  with sd.InputStream(callback=mic.calculate_volume):
    while True:
      rk.keep_time()
      mic.update()


if __name__ == "__main__":
  main()
