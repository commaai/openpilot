#!/usr/bin/env python3
import time

import sounddevice as sd
import numpy as np

from cereal import messaging
from common.filter_simple import FirstOrderFilter
from common.realtime import Ratekeeper
from system.swaglog import cloudlog

RATE = 10
DT_MIC = 1. / RATE

MUTE_TIME = 5


class Mic:
  def __init__(self, pm, sm):
    self.pm = pm
    self.sm = sm
    self.rk = Ratekeeper(RATE)

    self.measurements = np.array([])
    self.filter = FirstOrderFilter(1, 7, DT_MIC)
    self.last_alert_time = 0

  def update(self):
    self.sm.update(0)

    if self.sm.updated['controlsState']:
      if self.sm['controlsState'].alertSound > 0:
        self.last_alert_time = time.time()

    muted = time.time() - self.last_alert_time < MUTE_TIME

    if not muted and len(self.measurements) > 0:
      noise_level_raw = float(np.linalg.norm(self.measurements))
      self.filter.update(noise_level_raw)
    else:
      noise_level_raw = 0
    self.measurements = np.array([])

    msg = messaging.new_message('microphone')
    microphone = msg.microphone
    microphone.ambientNoiseLevelRaw = noise_level_raw
    microphone.filteredAmbientNoiseLevel = self.filter.x

    self.pm.send('microphone', msg)
    self.rk.keep_time()

  def callback(self, indata, frames, time, status):
    self.measurements = np.concatenate((self.measurements, indata[:, 0]))

  def micd_thread(self, device=None):
    if device is None:
      device = "sysdefault"

    with sd.InputStream(device=device, channels=1, samplerate=44100, callback=self.callback) as stream:
      cloudlog.info(f"micd stream started: {stream.samplerate=} {stream.channels=} {stream.dtype=} {stream.device=}")
      while True:
        self.update()


def main(pm=None, sm=None):
  if pm is None:
    pm = messaging.PubMaster(['microphone'])
  if sm is None:
    sm = messaging.SubMaster(['controlsState'])

  mic = Mic(pm, sm)
  mic.micd_thread()


if __name__ == "__main__":
  main()
