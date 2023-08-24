#!/usr/bin/env python3
import numpy as np

from cereal import messaging
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import Ratekeeper
from openpilot.system.swaglog import cloudlog

RATE = 10
FFT_SAMPLES = 4096
REFERENCE_SPL = 2e-5  # newtons/m^2
SAMPLE_RATE = 44100
FILTER_DT = 1. / (SAMPLE_RATE / FFT_SAMPLES)


def calculate_spl(measurements):
  # https://www.engineeringtoolbox.com/sound-pressure-d_711.html
  sound_pressure = np.sqrt(np.mean(measurements ** 2))  # RMS of amplitudes
  if sound_pressure > 0:
    sound_pressure_level = 20 * np.log10(sound_pressure / REFERENCE_SPL)  # dB
  else:
    sound_pressure_level = 0
  return sound_pressure, sound_pressure_level


def apply_a_weighting(measurements: np.ndarray) -> np.ndarray:
  # Generate a Hanning window of the same length as the audio measurements
  measurements_windowed = measurements * np.hanning(len(measurements))

  # Calculate the frequency axis for the signal
  freqs = np.fft.fftfreq(measurements_windowed.size, d=1 / SAMPLE_RATE)

  # Calculate the A-weighting filter
  # https://en.wikipedia.org/wiki/A-weighting
  A = 12194 ** 2 * freqs ** 4 / ((freqs ** 2 + 20.6 ** 2) * (freqs ** 2 + 12194 ** 2) * np.sqrt((freqs ** 2 + 107.7 ** 2) * (freqs ** 2 + 737.9 ** 2)))
  A /= np.max(A)  # Normalize the filter

  # Apply the A-weighting filter to the signal
  return np.abs(np.fft.ifft(np.fft.fft(measurements_windowed) * A))


class Mic:
  def __init__(self, pm):
    self.pm = pm
    self.rk = Ratekeeper(RATE)

    self.measurements = np.empty(0)

    self.sound_pressure = 0
    self.sound_pressure_weighted = 0
    self.sound_pressure_level_weighted = 0

    self.spl_filter_weighted = FirstOrderFilter(0, 2.5, FILTER_DT, initialized=False)

  def update(self):
    msg = messaging.new_message('microphone')
    msg.microphone.soundPressure = float(self.sound_pressure)
    msg.microphone.soundPressureWeighted = float(self.sound_pressure_weighted)

    msg.microphone.soundPressureWeightedDb = float(self.sound_pressure_level_weighted)
    msg.microphone.filteredSoundPressureWeightedDb = float(self.spl_filter_weighted.x)

    self.pm.send('microphone', msg)
    self.rk.keep_time()

  def callback(self, indata, frames, time, status):
    """
    Using amplitude measurements, calculate an uncalibrated sound pressure and sound pressure level.
    Then apply A-weighting to the raw amplitudes and run the same calculations again.

    Logged A-weighted equivalents are rough approximations of the human-perceived loudness.
    """

    self.measurements = np.concatenate((self.measurements, indata[:, 0]))

    while self.measurements.size >= FFT_SAMPLES:
      measurements = self.measurements[:FFT_SAMPLES]

      self.sound_pressure, _ = calculate_spl(measurements)
      measurements_weighted = apply_a_weighting(measurements)
      self.sound_pressure_weighted, self.sound_pressure_level_weighted = calculate_spl(measurements_weighted)
      self.spl_filter_weighted.update(self.sound_pressure_level_weighted)

      self.measurements = self.measurements[FFT_SAMPLES:]

  def micd_thread(self):
    # sounddevice must be imported after forking processes
    import sounddevice as sd  # pylint: disable=import-outside-toplevel

    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=self.callback) as stream:
      cloudlog.info(f"micd stream started: {stream.samplerate=} {stream.channels=} {stream.dtype=} {stream.device=}")
      while True:
        self.update()


def main(pm=None):
  if pm is None:
    pm = messaging.PubMaster(['microphone'])

  mic = Mic(pm)
  mic.micd_thread()


if __name__ == "__main__":
  main()
