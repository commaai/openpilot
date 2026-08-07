#!/usr/bin/env python3
import numpy as np
from functools import cache
import threading
import time

from openpilot.cereal import messaging
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.system.audio_device import AutomaticAudioDevice

RATE = 10
FFT_SAMPLES = 1600 # 100ms
REFERENCE_SPL = 2e-5  # newtons/m^2
SAMPLE_RATE = 16000
SAMPLE_BUFFER = 160  # 10ms, matching WebRTC audio processing cadence


def patch_sounddevice(sd):
  # TODO: remove once sounddevice uses np.reshape internally.
  def sounddevice_array(buffer, channels, dtype):
    return np.frombuffer(buffer, dtype=dtype).reshape(-1, channels)

  sd._array = sounddevice_array


@cache
def get_a_weighting_filter():
  # Calculate the A-weighting filter
  # https://en.wikipedia.org/wiki/A-weighting
  freqs = np.fft.fftfreq(FFT_SAMPLES, d=1 / SAMPLE_RATE)
  A = 12194 ** 2 * freqs ** 4 / ((freqs ** 2 + 20.6 ** 2) * (freqs ** 2 + 12194 ** 2) * np.sqrt((freqs ** 2 + 107.7 ** 2) * (freqs ** 2 + 737.9 ** 2)))
  return A / np.max(A)


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

  # Apply the A-weighting filter to the signal
  return np.abs(np.fft.ifft(np.fft.fft(measurements_windowed) * get_a_weighting_filter()))


class Mic:
  def __init__(self):
    self.rk = Ratekeeper(RATE)
    self.pm = messaging.PubMaster(['soundPressure', 'rawAudioData'])

    self.measurements = np.empty(0)

    self.sound_pressure = 0
    self.sound_pressure_weighted = 0
    self.sound_pressure_level_weighted = 0

    self.lock = threading.Lock()
    self.input_rate = SAMPLE_RATE
    self.device_selector = AutomaticAudioDevice("input")

  def update(self):
    with self.lock:
      sound_pressure = self.sound_pressure
      sound_pressure_weighted = self.sound_pressure_weighted
      sound_pressure_level_weighted = self.sound_pressure_level_weighted

    msg = messaging.new_message('soundPressure', valid=True)
    msg.soundPressure.soundPressure = float(sound_pressure)
    msg.soundPressure.soundPressureWeighted = float(sound_pressure_weighted)
    msg.soundPressure.soundPressureWeightedDb = float(sound_pressure_level_weighted)

    self.pm.send('soundPressure', msg)
    self.rk.keep_time()

  def callback(self, indata, frames, time, status):
    """
    Using amplitude measurements, calculate an uncalibrated sound pressure and sound pressure level.
    Then apply A-weighting to the raw amplitudes and run the same calculations again.

    Logged A-weighted equivalents are rough approximations of the human-perceived loudness.
    """
    if status:
      cloudlog.warning(f"micd stream over/underflow: {status}")
    samples = indata[:, 0]
    if self.input_rate != SAMPLE_RATE:
      output_samples = round(len(samples) * SAMPLE_RATE / self.input_rate)
      samples = np.interp(np.arange(output_samples) * self.input_rate / SAMPLE_RATE,
                          np.arange(len(samples)), samples).astype(np.float32)

    msg = messaging.new_message('rawAudioData', valid=True)
    audio_data_int_16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    msg.rawAudioData.data = audio_data_int_16.tobytes()
    msg.rawAudioData.sampleRate = SAMPLE_RATE
    self.pm.send('rawAudioData', msg)

    with self.lock:
      self.measurements = np.concatenate((self.measurements, samples))

      while self.measurements.size >= FFT_SAMPLES:
        measurements = self.measurements[:FFT_SAMPLES]

        self.sound_pressure, _ = calculate_spl(measurements)
        measurements_weighted = apply_a_weighting(measurements)
        self.sound_pressure_weighted, self.sound_pressure_level_weighted = calculate_spl(measurements_weighted)

        self.measurements = self.measurements[FFT_SAMPLES:]

  def supported_rate(self, sd, device: int | None) -> int:
    device_info = sd.query_devices(device, "input")
    rates = [SAMPLE_RATE, int(device_info["default_samplerate"]), 48000, 44100]
    for rate in dict.fromkeys(rates):
      try:
        sd.check_input_settings(device=device, channels=1, samplerate=rate)
        return rate
      except Exception:
        continue
    raise RuntimeError(f"no supported capture sample rate for {device_info['name']}")

  def get_stream(self, sd, device: int | None = None):
    # reload sounddevice to reinitialize portaudio
    sd._terminate()
    sd._initialize()
    self.input_rate = self.supported_rate(sd, device)
    blocksize = round(self.input_rate * SAMPLE_BUFFER / SAMPLE_RATE)
    return sd.InputStream(channels=1, samplerate=self.input_rate, callback=self.callback, blocksize=blocksize, device=device)

  def micd_thread(self):
    # sounddevice must be imported after forking processes
    import sounddevice as sd
    patch_sounddevice(sd)

    failed_until: dict[int | None, float] = {}
    while True:
      try:
        sd._terminate()
        sd._initialize()
        excluded = {device for device, until in failed_until.items() if until > time.monotonic()}
        device, _ = self.device_selector.select(sd, force=True, exclude=excluded)
        candidates = [device] + [candidate for candidate in self.device_selector.alternatives(sd) if candidate not in excluded]
      except Exception:
        cloudlog.exception("micd failed to enumerate audio input devices")
        time.sleep(1.)
        continue
      for candidate in dict.fromkeys(candidates):
        try:
          with self.get_stream(sd, candidate) as stream:
            self.device_selector.selected = candidate
            failed_until.pop(candidate, None)
            cloudlog.info(f"micd stream started: {stream.samplerate=} {stream.channels=} {stream.dtype=} {stream.device=}, {stream.blocksize=}")
            while stream.active:
              self.update()
              excluded = {failed_device for failed_device, until in failed_until.items() if until > time.monotonic()}
              _, changed = self.device_selector.select(sd, exclude=excluded)
              if changed:
                break
          break
        except Exception:
          cloudlog.exception(f"micd failed to open input device {candidate}")
          failed_until[candidate] = time.monotonic() + 5.
      else:
        time.sleep(1.)


def main():
  mic = Mic()
  mic.micd_thread()


if __name__ == "__main__":
  main()
