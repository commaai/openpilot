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



def calc_loudness(measurements):
  # https://www.engineeringtoolbox.com/sound-pressure-d_711.html
  sound_pressure = np.sqrt(np.mean(measurements ** 2))  # RMS of amplitudes
  sound_pressure_level = 20 * np.log10(sound_pressure / REFERENCE_SPL) if sound_pressure > 0 else 0  # dB
  return sound_pressure, sound_pressure_level


def a_weight(signal: np.ndarray) -> np.ndarray:
  # Calculate the frequency axis for the signal
  freqs = np.fft.fftfreq(signal.size, d=1 / 44100)

  # Calculate the A-weighting filter
  A = 3.5041384e16 * freqs ** 4 / ((freqs ** 2 + 20.598997 ** 2) * (freqs ** 2 + 12194 ** 2) * np.sqrt((freqs ** 2 + 107.65265 ** 2) * (freqs ** 2 + 737.86223 ** 2)))
  # A = 12194 ** 2 * freqs ** 4 / ((freqs ** 2 + 20.6 ** 2) * (freqs ** 2 + 12194 ** 2) * np.sqrt((freqs ** 2 + 107.7 ** 2) * (freqs ** 2 + 737.9 ** 2)))
  A /= np.max(A)  # Normalize the filter

  # Apply the A-weighting filter to the signal
  return np.fft.ifft(np.fft.fft(signal) * A)

def apply_a_weighting(signal: np.ndarray) -> np.ndarray:
  # Compute the A-weighting filter
  f = np.array([20.598997, 107.65265, 737.86223, 12194.217])
  A1000 = 1.9997
  num = (2 * np.pi * f) ** 2 * A1000
  den = np.array([1, 2 * np.pi * f, (2 * np.pi * f)**2])
  H = num / den

  # Filter the signal with the A-weighting filter
  return np.convolve(signal, H, mode="same")


def extract_frequencies(audio_data, sample_rate):
  # # Generate a Hanning window of the same length as the audio data
  # hanning_window = np.hanning(len(audio_data))
  #
  # # Multiply the audio data by the Hanning window to apply the window
  # audio_data_windowed = audio_data * hanning_window

  # Perform the FFT on the windowed audio data
  fft_result = np.fft.fft(audio_data)

  # Get the absolute value of the complex FFT result
  fft_magnitudes = np.abs(fft_result)

  # Calculate the frequencies corresponding to the FFT result
  fft_frequencies = np.fft.fftfreq(len(audio_data), d=1/sample_rate)

  # Return the frequencies and their corresponding magnitudes
  return (fft_frequencies, fft_magnitudes)

def inverse_dft(frequencies, magnitudes, sample_rate=44100):
  # Compute the length of the audio signal
  signal_length = len(frequencies)

  # Create an array of complex numbers representing the spectrum of the signal
  spectrum = magnitudes * np.exp(1j * 2 * np.pi * frequencies / sample_rate)

  # Compute the inverse Fourier transform of the spectrum
  amplitudes = np.fft.ifft(spectrum)

  # Return the time-domain amplitudes
  return amplitudes


class Mic:
  def __init__(self, pm):
    self.pm = pm
    self.rk = Ratekeeper(RATE)

    self.measurements = np.empty(0)
    self.spl_filter = FirstOrderFilter(0, 4, DT_MIC, initialized=False)
    self.all_measurements = np.empty(0)

  def update(self):
    # self.measurements contains amplitudes from -1 to 1 which we use to
    # calculate an uncalibrated sound pressure level
    if len(self.measurements) > 0:
      self.all_measurements = np.concatenate((self.all_measurements, self.measurements))
      np.save('/data/test_recording', self.all_measurements)
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
