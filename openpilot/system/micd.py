#!/usr/bin/env python3
from functools import cache
from pathlib import Path
import re
import threading
import time

import numpy as np

from openpilot.cereal import messaging
from openpilot.common.realtime import Ratekeeper
from openpilot.common.utils import retry
from openpilot.common.swaglog import cloudlog

RATE = 10
FFT_SAMPLES = 1600 # 100ms
REFERENCE_SPL = 2e-5  # newtons/m^2
SAMPLE_RATE = 16000
SAMPLE_BUFFER = 160  # 10ms
SOUND_DEVICE_CHECK_INTERVAL = 1.0
INPUT_SAMPLE_RATE_FALLBACKS = (48000, 44100)

AudioDevice = dict[str, object]
AudioDeviceSignature = tuple[int, str, int, int, int]

ALSA_CARD_RE = re.compile(r"^\s*(\d+)\s+\[([^\]]+)\]:.*usb", re.IGNORECASE)
ALSA_DEVICE_CARD_RE = re.compile(r"(?:^|[:,(])CARD=([^,):]+)")
ALSA_DEVICE_HW_RE = re.compile(r"(?:^|[(:])(?:plug)?hw:([^,)]+)")


def _device_value(device: AudioDevice, key: str, default: object = None) -> object:
  return device.get(key, default)


def _device_int(device: AudioDevice, key: str, default: int = 0) -> int:
  try:
    return int(device.get(key, default))
  except (TypeError, ValueError):
    return default


def _device_sample_rate(device: AudioDevice) -> int:
  try:
    return int(float(device.get("default_samplerate", 0)))
  except (TypeError, ValueError):
    return 0


def sample_buffer_for_sample_rate(sample_rate: int) -> int:
  return max(1, round(SAMPLE_BUFFER * sample_rate / SAMPLE_RATE))


def unique_valid_sample_rates(sample_rates: list[int]) -> list[int]:
  valid_sample_rates: list[int] = []
  for sample_rate in sample_rates:
    if sample_rate > 0 and sample_rate not in valid_sample_rates:
      valid_sample_rates.append(sample_rate)

  return valid_sample_rates


def get_audio_device_signature(index: int, device: AudioDevice) -> AudioDeviceSignature:
  return (
    index,
    str(_device_value(device, "name", "")),
    _device_int(device, "hostapi", -1),
    _device_int(device, "max_input_channels"),
    _device_sample_rate(device),
  )


def stable_audio_device_signature(signature: AudioDeviceSignature) -> tuple[str, int, int, int]:
  return signature[1:]


def audio_device_can_input(device: AudioDevice) -> bool:
  return _device_int(device, "max_input_channels") > 0


def query_audio_input_devices(sd) -> dict[AudioDeviceSignature, AudioDevice]:
  return {
    get_audio_device_signature(index, device): device
    for index, device in enumerate(sd.query_devices())
    if audio_device_can_input(device)
  }


def query_audio_input_device(sd, device: int | None) -> AudioDevice | None:
  try:
    return sd.query_devices(device, "input")
  except Exception:
    return None


def find_audio_input_device(sd, signature: AudioDeviceSignature) -> int | None:
  stable_signature = stable_audio_device_signature(signature)
  fallback_index = None

  for index, device in enumerate(sd.query_devices()):
    if not audio_device_can_input(device):
      continue

    device_signature = get_audio_device_signature(index, device)
    if device_signature == signature:
      return index

    if stable_audio_device_signature(device_signature) == stable_signature:
      fallback_index = index

  return fallback_index


def get_usb_audio_cards(asound_root: Path = Path("/proc/asound")) -> set[str]:
  usb_cards: set[str] = set()

  try:
    for card_path in asound_root.glob("card*"):
      card_number = card_path.name.removeprefix("card")
      if not card_number.isdigit() or not (card_path / "usbid").exists():
        continue

      usb_cards.add(card_number)
      try:
        usb_cards.add((card_path / "id").read_text(encoding="utf-8").strip())
      except OSError:
        pass
  except OSError:
    pass

  try:
    cards = (asound_root / "cards").read_text(encoding="utf-8")
  except OSError:
    return {card for card in usb_cards if card}

  for line in cards.splitlines():
    match = ALSA_CARD_RE.match(line)
    if match is not None:
      usb_cards.add(match.group(1).strip())
      usb_cards.add(match.group(2).strip())

  return {card for card in usb_cards if card}


def get_alsa_card_tokens_from_device_name(name: str) -> set[str]:
  tokens: set[str] = set()

  for pattern in (ALSA_DEVICE_CARD_RE, ALSA_DEVICE_HW_RE):
    for match in pattern.finditer(name):
      token = match.group(1).strip()
      if token:
        tokens.add(token)

  return tokens


def normalize_audio_card_tokens(tokens: set[str]) -> set[str]:
  return tokens | {token.lower() for token in tokens}


def is_usb_audio_input_device(device: AudioDevice, usb_audio_cards: set[str] | None = None) -> bool:
  if not audio_device_can_input(device):
    return False

  device_name = str(_device_value(device, "name", ""))
  if "usb" in device_name.lower():
    return True

  if usb_audio_cards is None:
    usb_audio_cards = get_usb_audio_cards()

  device_card_tokens = normalize_audio_card_tokens(get_alsa_card_tokens_from_device_name(device_name))
  return bool(device_card_tokens & normalize_audio_card_tokens(usb_audio_cards))


def audio_input_device_is_present(sd, signature: AudioDeviceSignature, usb_audio_cards: set[str]) -> bool:
  device_card_tokens = normalize_audio_card_tokens(get_alsa_card_tokens_from_device_name(signature[1]))
  if device_card_tokens and not (device_card_tokens & normalize_audio_card_tokens(usb_audio_cards)):
    return False

  return find_audio_input_device(sd, signature) is not None


def describe_audio_input_device(signature: AudioDeviceSignature) -> str:
  index, name, hostapi, channels, samplerate = signature
  return f"index={index} name={name!r} hostapi={hostapi} input_channels={channels} default_samplerate={samplerate}"


def get_input_sample_rate_candidates(sd, device: int | None, signature: AudioDeviceSignature | None) -> list[int]:
  sample_rates = [SAMPLE_RATE]
  if signature is not None:
    sample_rates.append(signature[4])

  device_info = query_audio_input_device(sd, device)
  if device_info is not None:
    sample_rates.append(_device_sample_rate(device_info))

  sample_rates.extend(INPUT_SAMPLE_RATE_FALLBACKS)
  return unique_valid_sample_rates(sample_rates)


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
    self.input_device_signature: AudioDeviceSignature | None = None
    self.known_input_devices: set[AudioDeviceSignature] = set()
    self.known_usb_audio_cards: set[str] = set()
    self.last_sound_device_check_time = 0.
    self.input_sample_rate = SAMPLE_RATE
    self.resample_input_sample_rate = SAMPLE_RATE
    self.resample_position = 0.0
    self.resample_last_sample: float | None = None

    self.sound_pressure = 0
    self.sound_pressure_weighted = 0
    self.sound_pressure_level_weighted = 0

    self.lock = threading.Lock()

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

  def reset_audio_resampler(self, input_sample_rate: int | None = None) -> None:
    self.resample_input_sample_rate = self.input_sample_rate if input_sample_rate is None else input_sample_rate
    self.resample_position = 0.0
    self.resample_last_sample = None

  def resample_to_output_sample_rate(self, samples: np.ndarray, input_sample_rate: int) -> np.ndarray:
    if samples.size == 0:
      return samples

    if input_sample_rate == SAMPLE_RATE:
      self.reset_audio_resampler(input_sample_rate)
      return samples

    if input_sample_rate != self.resample_input_sample_rate:
      self.reset_audio_resampler(input_sample_rate)

    step = input_sample_rate / SAMPLE_RATE
    start = self.resample_position
    stop = samples.size - 1

    if self.resample_last_sample is None:
      start = max(start, 0.0)
      sample_positions = np.arange(samples.size)
      input_samples = samples
    else:
      sample_positions = np.arange(-1, samples.size)
      input_samples = np.concatenate(([self.resample_last_sample], samples))

    if start <= stop:
      output_positions = np.arange(start, stop + np.finfo(float).eps, step)
      output_samples = np.interp(output_positions, sample_positions, input_samples).astype(samples.dtype)
      self.resample_position = output_positions[-1] + step - samples.size
    else:
      output_samples = np.empty(0, dtype=samples.dtype)
      self.resample_position = start - samples.size

    self.resample_last_sample = float(samples[-1])
    return output_samples

  def callback(self, indata, frames, time, status):
    """
    Using amplitude measurements, calculate an uncalibrated sound pressure and sound pressure level.
    Then apply A-weighting to the raw amplitudes and run the same calculations again.

    Logged A-weighted equivalents are rough approximations of the human-perceived loudness.
    """
    audio_samples = self.resample_to_output_sample_rate(indata[:, 0], self.input_sample_rate)
    if audio_samples.size == 0:
      return

    msg = messaging.new_message('rawAudioData', valid=True)
    audio_data_int_16 = (audio_samples * 32767).astype(np.int16)
    msg.rawAudioData.data = audio_data_int_16.tobytes()
    msg.rawAudioData.sampleRate = SAMPLE_RATE
    self.pm.send('rawAudioData', msg)

    with self.lock:
      self.measurements = np.concatenate((self.measurements, audio_samples))

      while self.measurements.size >= FFT_SAMPLES:
        measurements = self.measurements[:FFT_SAMPLES]

        self.sound_pressure, _ = calculate_spl(measurements)
        measurements_weighted = apply_a_weighting(measurements)
        self.sound_pressure_weighted, self.sound_pressure_level_weighted = calculate_spl(measurements_weighted)

        self.measurements = self.measurements[FFT_SAMPLES:]

  def reset_audio_device_tracking(self, sd) -> dict[AudioDeviceSignature, AudioDevice]:
    self.known_usb_audio_cards = get_usb_audio_cards()
    input_devices = query_audio_input_devices(sd)
    self.known_input_devices = set(input_devices)
    self.last_sound_device_check_time = time.monotonic()
    return input_devices

  def select_preferred_usb_input_device(self, input_devices: dict[AudioDeviceSignature, AudioDevice],
                                        usb_audio_cards: set[str],
                                        preferred_usb_audio_cards: set[str] | None = None,
                                        allow_existing_usb_devices: bool = False) -> AudioDeviceSignature | None:
    candidates: list[tuple[int, int, AudioDeviceSignature]] = []
    new_input_devices = set(input_devices) - self.known_input_devices
    preferred_usb_audio_cards = preferred_usb_audio_cards or set()
    normalized_preferred_usb_audio_cards = normalize_audio_card_tokens(preferred_usb_audio_cards)

    for signature, device in input_devices.items():
      if not is_usb_audio_input_device(device, usb_audio_cards):
        continue

      device_card_tokens = normalize_audio_card_tokens(get_alsa_card_tokens_from_device_name(str(_device_value(device, "name", ""))))
      preferred_card_match = bool(device_card_tokens & normalized_preferred_usb_audio_cards)
      new_input_device = signature in new_input_devices

      if not preferred_card_match and not new_input_device and not allow_existing_usb_devices:
        continue

      priority = 2 if preferred_card_match else 1 if new_input_device else 0
      candidates.append((priority, signature[0], signature))

    if not candidates:
      return None

    return max(candidates)[2]

  def refresh_usb_input_device(self, sd, preferred_usb_audio_cards: set[str],
                               allow_existing_usb_devices: bool = False) -> AudioDeviceSignature | None:
    sd._terminate()
    sd._initialize()

    usb_audio_cards = get_usb_audio_cards()
    input_devices = query_audio_input_devices(sd)
    input_device_signature = self.select_preferred_usb_input_device(input_devices, usb_audio_cards, preferred_usb_audio_cards,
                                                                    allow_existing_usb_devices)
    self.known_usb_audio_cards = usb_audio_cards
    self.known_input_devices.update(input_devices)

    return input_device_signature

  def check_for_input_device_changes(self, sd) -> tuple[AudioDeviceSignature | None, set[str], bool]:
    now = time.monotonic()
    if now - self.last_sound_device_check_time < SOUND_DEVICE_CHECK_INTERVAL:
      return None, set(), False

    self.last_sound_device_check_time = now
    usb_audio_cards = get_usb_audio_cards()
    new_usb_audio_cards = usb_audio_cards - self.known_usb_audio_cards
    selected_input_removed = False

    try:
      input_devices = query_audio_input_devices(sd)
      if self.input_device_signature is not None:
        selected_input_removed = not audio_input_device_is_present(sd, self.input_device_signature, usb_audio_cards)

      input_device_signature = self.select_preferred_usb_input_device(input_devices, usb_audio_cards, new_usb_audio_cards)
      self.known_input_devices.update(input_devices)
    except Exception:
      cloudlog.exception("micd failed to query audio input devices")
      input_device_signature = None

    self.known_usb_audio_cards = usb_audio_cards
    return input_device_signature, new_usb_audio_cards, selected_input_removed

  @retry(attempts=10, delay=3)
  def get_stream(self, sd):
    # reload sounddevice to reinitialize portaudio
    sd._terminate()
    sd._initialize()

    device = None
    if self.input_device_signature is not None:
      device = find_audio_input_device(sd, self.input_device_signature)
      if device is None:
        cloudlog.warning(f"micd could not find selected audio input device: {describe_audio_input_device(self.input_device_signature)}")
        self.input_device_signature = None

    sample_rates = get_input_sample_rate_candidates(sd, device, self.input_device_signature)
    last_exception: Exception | None = None
    for sample_rate in sample_rates:
      try:
        stream = sd.InputStream(channels=1, samplerate=sample_rate, callback=self.callback,
                                blocksize=sample_buffer_for_sample_rate(sample_rate), device=device)
      except Exception as e:
        last_exception = e
        cloudlog.warning(f"micd failed to open audio input at {sample_rate} Hz with device={device}: {e}")
        continue

      self.input_sample_rate = int(stream.samplerate)
      self.reset_audio_resampler()
      return stream

    if last_exception is not None:
      raise last_exception
    raise RuntimeError("micd found no audio input sample rate candidates")

  def micd_thread(self):
    # sounddevice must be imported after forking processes
    import sounddevice as sd

    input_device_signature = self.refresh_usb_input_device(sd, set(), allow_existing_usb_devices=True)
    if input_device_signature is not None:
      self.input_device_signature = input_device_signature
      cloudlog.info(f"micd starting with USB audio input: {describe_audio_input_device(input_device_signature)}")
    startup_usb_input_check = input_device_signature is None

    pending_usb_audio_cards: set[str] = set()
    while True:
      if pending_usb_audio_cards:
        input_device_signature = self.refresh_usb_input_device(sd, pending_usb_audio_cards)
        pending_usb_audio_cards = set()
        if input_device_signature is not None:
          self.input_device_signature = input_device_signature
          cloudlog.info(f"micd switching to new USB audio input: {describe_audio_input_device(input_device_signature)}")

      with self.get_stream(sd) as stream:
        input_devices = self.reset_audio_device_tracking(sd)

        if startup_usb_input_check and self.input_device_signature is None:
          startup_usb_input_check = False
          input_device_signature = self.select_preferred_usb_input_device(input_devices, self.known_usb_audio_cards,
                                                                          allow_existing_usb_devices=True)
          if input_device_signature is not None:
            self.input_device_signature = input_device_signature
            cloudlog.info(f"micd switching to USB audio input present at startup: {describe_audio_input_device(input_device_signature)}")
            continue

        cloudlog.info(f"micd stream started: {stream.samplerate=} {stream.channels=} {stream.dtype=} {stream.device=}, {stream.blocksize=}")
        while True:
          self.update()

          input_device_signature, new_usb_audio_cards, selected_input_removed = self.check_for_input_device_changes(sd)
          if selected_input_removed:
            cloudlog.info("micd selected USB audio input removed; switching to system default")
            self.input_device_signature = None
            break
          if input_device_signature is not None:
            self.input_device_signature = input_device_signature
            cloudlog.info(f"micd switching to new USB audio input: {describe_audio_input_device(input_device_signature)}")
            break
          if new_usb_audio_cards:
            pending_usb_audio_cards = new_usb_audio_cards
            cloudlog.info(f"micd found new USB audio card ids: {sorted(new_usb_audio_cards)}")
            break

          if not stream.active:
            if self.input_device_signature is not None:
              cloudlog.warning("micd selected audio input stream stopped; switching to system default")
              self.input_device_signature = None
            else:
              cloudlog.warning("micd audio input stream stopped; reopening system default")
            break


def main():
  mic = Mic()
  mic.micd_thread()


if __name__ == "__main__":
  main()
