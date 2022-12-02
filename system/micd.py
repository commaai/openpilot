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
SAMPLE_RATE = 44100


def calculate_spl(measurements):
  # https://www.engineeringtoolbox.com/sound-pressure-d_711.html
  sound_pressure = np.sqrt(np.mean(measurements ** 2))  # RMS of amplitudes
  sound_pressure_level = 20 * np.log10(sound_pressure / REFERENCE_SPL) if sound_pressure > 0 else 0  # dB
  return sound_pressure, sound_pressure_level


def apply_a_weighting(measurements: np.ndarray) -> np.ndarray:
  # Generate a Hanning window of the same length as the audio measurements
  hanning_window = np.hanning(len(measurements))
  measurements_windowed = measurements * hanning_window

  # Calculate the frequency axis for the signal
  freqs = np.fft.fftfreq(measurements_windowed.size, d=1 / SAMPLE_RATE)

  # Calculate the A-weighting filter
  # TODO: create global for 3.5041384e16
  A = 3.5041384e16 * freqs ** 4 / ((freqs ** 2 + 20.598997 ** 2) * (freqs ** 2 + 12194 ** 2) * np.sqrt((freqs ** 2 + 107.65265 ** 2) * (freqs ** 2 + 737.86223 ** 2)))
  A /= np.max(A)  # Normalize the filter

  # Apply the A-weighting filter to the signal
  return np.fft.ifft(np.fft.fft(measurements_windowed) * A)


class Mic:
  def __init__(self, pm):
    self.pm = pm
    self.rk = Ratekeeper(RATE)

    self.measurements = np.empty(0)
    self.spl_filter = FirstOrderFilter(0, 4, DT_MIC, initialized=False)
    self.a_weighted_spl_filter = FirstOrderFilter(0, 4, DT_MIC, initialized=False)

  def update(self):
    """
    Using amplitude measurements, calculate an uncalibrated sound pressure and sound pressure level.
    Then apply A-weighting to the raw amplitudes and run the same calculations again.

    Logged A-weighted equivalents are rough approximations of the human-perceived loudness.
    """

    if len(self.measurements) > 0:
      sound_pressure, sound_pressure_level = calculate_spl(self.measurements)
      self.spl_filter.update(sound_pressure_level)

      measurements_weighted = apply_a_weighting(self.measurements)
      sound_pressure_weighted, sound_pressure_level_weighted = calculate_spl(measurements_weighted)
      self.a_weighted_spl_filter.update(sound_pressure_level_weighted)
    else:
      sound_pressure = 0
      sound_pressure_level = 0
      sound_pressure_weighted = 0
      sound_pressure_level_weighted = 0

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

    with sd.InputStream(device=device, channels=1, samplerate=SAMPLE_RATE, callback=self.callback) as stream:
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
