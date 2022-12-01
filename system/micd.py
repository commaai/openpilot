#!/usr/bin/env python3
import sounddevice as sd
import numpy as np

from cereal import messaging
from common.filter_simple import FirstOrderFilter
from common.realtime import Ratekeeper
from system.swaglog import cloudlog

RATE = 10
DT_MIC = 1. / RATE
REFERENCE_SPL = 2 * 10 ** -5  # newtons/m^2


class Mic:
  def __init__(self, pm):
    self.pm = pm
    self.rk = Ratekeeper(RATE)

    self.measurements = np.empty(0)
    self.spl_filter = FirstOrderFilter(0, 4, DT_MIC, initialized=False)

  def update(self):
    # self.measurements contains amplitudes from -1 to 1 which we use to
    # calculate an uncalibrated sound pressure level
    if len(self.measurements) > 0:
      # https://www.engineeringtoolbox.com/sound-pressure-d_711.html
      sound_pressure = np.sqrt(np.mean(self.measurements ** 2))  # RMS of amplitudes
      sound_pressure_level = 20 * np.log10(sound_pressure / REFERENCE_SPL) if sound_pressure > 0 else 0  # dB
      self.spl_filter.update(sound_pressure_level)
    else:
      sound_pressure = 0
      sound_pressure_level = 0

    self.measurements = np.empty(0)

    msg = messaging.new_message('microphone')
    msg.microphone.soundPressure = float(sound_pressure)
    msg.microphone.soundPressureDb = float(sound_pressure_level)
    msg.microphone.filteredSoundPressureDb = float(self.spl_filter.x)

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

  mic = Mic(pm)
  mic.micd_thread()


if __name__ == "__main__":
  main()
