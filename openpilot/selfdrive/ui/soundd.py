from collections import deque
import math
from pathlib import Path
import re
import threading
import time
import wave

import numpy as np

from openpilot.cereal import log, messaging
from openpilot.common.basedir import BASEDIR
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import Ratekeeper
from openpilot.common.utils import retry
from openpilot.common.swaglog import cloudlog

from openpilot.system import micd
from openpilot.common.hardware import HARDWARE

SAMPLE_RATE = 48000
SAMPLE_BUFFER = 4096 # (approx 100ms)
WEBRTC_AUDIO_BUFFER_FRAMES = SAMPLE_RATE # 1s max queued remote audio
WEBRTC_AUDIO_PREBUFFER_FRAMES = SAMPLE_RATE // 5 # 200ms before starting remote audio playback
MAX_VOLUME = 1.0
MIN_VOLUME = 0.1
ALERT_RAMP_TIME = 4 # seconds to ramp to max volume for warningImmediate
SELFDRIVE_STATE_TIMEOUT = 5 # 5 seconds
FILTER_DT = 1. / (micd.SAMPLE_RATE / micd.FFT_SAMPLES)
SOUND_DEVICE_CHECK_INTERVAL = 1.0

AMBIENT_DB = 24 # DB where MIN_VOLUME is applied
DB_SCALE = 30 # AMBIENT_DB + DB_SCALE is where MAX_VOLUME is applied

VOLUME_BASE = 20
if HARDWARE.get_device_type() == "tizi":
  AMBIENT_DB = 30
  VOLUME_BASE = 10

AudibleAlert = log.SelfdriveState.AudibleAlert

AudioDevice = dict[str, object]
AudioDeviceSignature = tuple[int, str, int, int, int]

ALSA_CARD_RE = re.compile(r"^\s*(\d+)\s+\[([^\]]+)\]:.*usb", re.IGNORECASE)
ALSA_DEVICE_CARD_RE = re.compile(r"(?:^|[:,(])CARD=([^,):]+)")
ALSA_DEVICE_HW_RE = re.compile(r"(?:^|[(:])(?:plug)?hw:([^,)]+)")

sound_list: dict[int, tuple[str, int | None, float]] = {
  # AudibleAlert, file name, play count (none for infinite)
  AudibleAlert.engage: ("engage.wav", 1, MAX_VOLUME),
  AudibleAlert.disengage: ("disengage.wav", 1, MAX_VOLUME),
  AudibleAlert.refuse: ("refuse.wav", 1, MAX_VOLUME),

  AudibleAlert.prompt: ("prompt.wav", 1, MAX_VOLUME),
  AudibleAlert.promptRepeat: ("prompt.wav", None, MAX_VOLUME),
  AudibleAlert.promptDistracted: ("prompt_distracted.wav", None, MAX_VOLUME),

  AudibleAlert.preAlert: ("pre_alert.wav", 1, MAX_VOLUME),

  AudibleAlert.warningSoft: ("warning_soft.wav", None, MAX_VOLUME),
  AudibleAlert.warningImmediate: ("warning_immediate.wav", None, MAX_VOLUME),
}
if HARDWARE.get_device_type() == "tizi":
  sound_list.update({
    AudibleAlert.engage: ("engage_tizi.wav", 1, MAX_VOLUME),
    AudibleAlert.disengage: ("disengage_tizi.wav", 1, MAX_VOLUME),
  })

def check_selfdrive_timeout_alert(sm):
  ss_missing = time.monotonic() - sm.recv_time['selfdriveState']

  if ss_missing > SELFDRIVE_STATE_TIMEOUT:
    if sm['selfdriveState'].enabled and (ss_missing - SELFDRIVE_STATE_TIMEOUT) < 10:
      return True

  return False


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


def get_audio_device_signature(index: int, device: AudioDevice) -> AudioDeviceSignature:
  return (
    index,
    str(_device_value(device, "name", "")),
    _device_int(device, "hostapi", -1),
    _device_int(device, "max_output_channels"),
    _device_sample_rate(device),
  )


def stable_audio_device_signature(signature: AudioDeviceSignature) -> tuple[str, int, int, int]:
  return signature[1:]


def audio_device_can_output(device: AudioDevice) -> bool:
  return _device_int(device, "max_output_channels") > 0


def query_audio_output_devices(sd) -> dict[AudioDeviceSignature, AudioDevice]:
  return {
    get_audio_device_signature(index, device): device
    for index, device in enumerate(sd.query_devices())
    if audio_device_can_output(device)
  }


def find_audio_output_device(sd, signature: AudioDeviceSignature) -> int | None:
  stable_signature = stable_audio_device_signature(signature)
  fallback_index = None

  for index, device in enumerate(sd.query_devices()):
    if not audio_device_can_output(device):
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


def is_usb_audio_output_device(device: AudioDevice, usb_audio_cards: set[str] | None = None) -> bool:
  if not audio_device_can_output(device):
    return False

  device_name = str(_device_value(device, "name", ""))
  if "usb" in device_name.lower():
    return True

  if usb_audio_cards is None:
    usb_audio_cards = get_usb_audio_cards()

  device_card_tokens = normalize_audio_card_tokens(get_alsa_card_tokens_from_device_name(device_name))
  return bool(device_card_tokens & normalize_audio_card_tokens(usb_audio_cards))


def describe_audio_device(signature: AudioDeviceSignature) -> str:
  index, name, hostapi, channels, samplerate = signature
  return f"index={index} name={name!r} hostapi={hostapi} output_channels={channels} default_samplerate={samplerate}"


class Soundd:
  def __init__(self):
    self.load_sounds()

    self.current_alert = AudibleAlert.none
    self.current_volume = MIN_VOLUME
    self.current_sound_frame = 0

    self.ramp_start_volume = MIN_VOLUME
    self.ramp_start_time = 0.

    self.selfdrive_timeout_alert = False

    self.spl_filter_weighted = FirstOrderFilter(0, 2.5, FILTER_DT, initialized=False)
    self.webrtc_audio: deque[np.ndarray] = deque()
    self.webrtc_audio_frames = 0
    self.webrtc_audio_lock = threading.Lock()
    self.webrtc_audio_playing = False
    self.output_device_signature: AudioDeviceSignature | None = None
    self.known_output_devices: set[AudioDeviceSignature] = set()
    self.known_usb_audio_cards: set[str] = set()
    self.last_sound_device_check_time = 0.

  def load_sounds(self):
    self.loaded_sounds: dict[int, np.ndarray] = {}

    # Load all sounds
    for sound in sound_list:
      filename, play_count, volume = sound_list[sound]

      with wave.open(BASEDIR + "/openpilot/selfdrive/assets/sounds/" + filename, 'r') as wavefile:
        assert wavefile.getnchannels() == 1
        assert wavefile.getsampwidth() == 2
        assert wavefile.getframerate() == SAMPLE_RATE

        length = wavefile.getnframes()
        self.loaded_sounds[sound] = np.frombuffer(wavefile.readframes(length), dtype=np.int16).astype(np.float32) / (2**16/2)

  def get_sound_data(self, frames): # get "frames" worth of data from the current alert sound, looping when required

    ret = np.zeros(frames, dtype=np.float32)

    if self.current_alert != AudibleAlert.none:
      num_loops = sound_list[self.current_alert][1]
      sound_data = self.loaded_sounds[self.current_alert]
      written_frames = 0

      current_sound_frame = self.current_sound_frame % len(sound_data)
      loops = self.current_sound_frame // len(sound_data)

      while written_frames < frames and (num_loops is None or loops < num_loops):
        available_frames = sound_data.shape[0] - current_sound_frame
        frames_to_write = min(available_frames, frames - written_frames)
        ret[written_frames:written_frames+frames_to_write] = sound_data[current_sound_frame:current_sound_frame+frames_to_write]
        written_frames += frames_to_write
        self.current_sound_frame += frames_to_write
        current_sound_frame = self.current_sound_frame % len(sound_data)
        loops = self.current_sound_frame // len(sound_data)

    return ret * self.current_volume

  def add_webrtc_audio(self, audio_data) -> None:
    data = np.frombuffer(audio_data.data, dtype=np.int16).astype(np.float32) / 32768.0
    if data.size == 0:
      return

    with self.webrtc_audio_lock:
      self.webrtc_audio.append(data)
      self.webrtc_audio_frames += data.size
      while self.webrtc_audio_frames > WEBRTC_AUDIO_BUFFER_FRAMES:
        self.webrtc_audio_frames -= self.webrtc_audio.popleft().size

  def get_webrtc_audio_data(self, frames: int) -> np.ndarray:
    ret = np.zeros(frames, dtype=np.float32)

    with self.webrtc_audio_lock:
      if not self.webrtc_audio_playing:
        if self.webrtc_audio_frames < WEBRTC_AUDIO_PREBUFFER_FRAMES:
          return ret
        self.webrtc_audio_playing = True

      written_frames = 0
      while written_frames < frames and self.webrtc_audio:
        data = self.webrtc_audio[0]
        frames_to_write = min(data.size, frames - written_frames)
        ret[written_frames:written_frames+frames_to_write] = data[:frames_to_write]
        written_frames += frames_to_write
        self.webrtc_audio_frames -= frames_to_write

        if frames_to_write == data.size:
          self.webrtc_audio.popleft()
        else:
          self.webrtc_audio[0] = data[frames_to_write:]

      if written_frames < frames:
        self.webrtc_audio_playing = False

    return ret

  def webrtc_audio_thread(self) -> None:
    webrtc_audio_sock = messaging.sub_sock('webrtcAudioData')
    while True:
      msg = messaging.recv_one(webrtc_audio_sock)
      if msg is not None:
        self.add_webrtc_audio(msg.webrtcAudioData)

  def callback(self, data_out: np.ndarray, frames: int, time, status) -> None:
    if status:
      cloudlog.warning(f"soundd stream over/underflow: {status}")
    audio = self.get_sound_data(frames) + self.get_webrtc_audio_data(frames)
    data_out[:frames, 0] = np.clip(audio, -1.0, 1.0)

  def update_alert(self, new_alert):
    current_alert_played_once = self.current_alert == AudibleAlert.none or self.current_sound_frame >= len(self.loaded_sounds[self.current_alert])
    if self.current_alert != new_alert and (new_alert != AudibleAlert.none or current_alert_played_once):
      if new_alert == AudibleAlert.warningImmediate:
        self.ramp_start_volume = self.current_volume
        self.ramp_start_time = time.monotonic()
      self.current_alert = new_alert
      self.current_sound_frame = 0

  def get_audible_alert(self, sm):
    if sm.updated['selfdriveState']:
      new_alert = sm['selfdriveState'].alertSound.raw
      self.update_alert(new_alert)
    elif check_selfdrive_timeout_alert(sm):
      self.update_alert(AudibleAlert.warningImmediate)
      self.selfdrive_timeout_alert = True
    elif self.selfdrive_timeout_alert:
      self.update_alert(AudibleAlert.none)
      self.selfdrive_timeout_alert = False

  def calculate_volume(self, weighted_db):
    volume = ((weighted_db - AMBIENT_DB) / DB_SCALE) * (MAX_VOLUME - MIN_VOLUME) + MIN_VOLUME
    return math.pow(VOLUME_BASE, (np.clip(volume, MIN_VOLUME, MAX_VOLUME) - 1))

  def reset_audio_device_tracking(self, sd) -> None:
    self.known_usb_audio_cards = get_usb_audio_cards()
    self.known_output_devices = set(query_audio_output_devices(sd))
    self.last_sound_device_check_time = time.monotonic()

  def select_preferred_usb_output_device(self, output_devices: dict[AudioDeviceSignature, AudioDevice],
                                         usb_audio_cards: set[str],
                                         preferred_usb_audio_cards: set[str] | None = None) -> AudioDeviceSignature | None:
    candidates: list[tuple[int, int, AudioDeviceSignature]] = []
    new_output_devices = set(output_devices) - self.known_output_devices
    preferred_usb_audio_cards = preferred_usb_audio_cards or set()
    normalized_preferred_usb_audio_cards = normalize_audio_card_tokens(preferred_usb_audio_cards)

    for signature, device in output_devices.items():
      if not is_usb_audio_output_device(device, usb_audio_cards):
        continue

      device_card_tokens = normalize_audio_card_tokens(get_alsa_card_tokens_from_device_name(str(_device_value(device, "name", ""))))
      preferred_card_match = bool(device_card_tokens & normalized_preferred_usb_audio_cards)
      new_output_device = signature in new_output_devices

      if not preferred_card_match and not new_output_device:
        continue

      candidates.append((2 if preferred_card_match else 1, signature[0], signature))

    if not candidates:
      return None

    return max(candidates)[2]

  def refresh_usb_output_device(self, sd, preferred_usb_audio_cards: set[str]) -> AudioDeviceSignature | None:
    sd._terminate()
    sd._initialize()

    usb_audio_cards = get_usb_audio_cards()
    output_devices = query_audio_output_devices(sd)
    output_device_signature = self.select_preferred_usb_output_device(output_devices, usb_audio_cards, preferred_usb_audio_cards)
    self.known_usb_audio_cards = usb_audio_cards
    self.known_output_devices.update(output_devices)

    return output_device_signature

  def check_for_new_usb_output_device(self, sd) -> tuple[AudioDeviceSignature | None, set[str]]:
    now = time.monotonic()
    if now - self.last_sound_device_check_time < SOUND_DEVICE_CHECK_INTERVAL:
      return None, set()

    self.last_sound_device_check_time = now
    usb_audio_cards = get_usb_audio_cards()
    new_usb_audio_cards = usb_audio_cards - self.known_usb_audio_cards

    try:
      output_devices = query_audio_output_devices(sd)
      output_device_signature = self.select_preferred_usb_output_device(output_devices, usb_audio_cards, new_usb_audio_cards)
      self.known_output_devices.update(output_devices)
    except Exception:
      cloudlog.exception("soundd failed to query audio output devices")
      output_device_signature = None

    self.known_usb_audio_cards = usb_audio_cards
    return output_device_signature, new_usb_audio_cards

  @retry(attempts=10, delay=3)
  def get_stream(self, sd):
    # reload sounddevice to reinitialize portaudio
    sd._terminate()
    sd._initialize()

    device = None
    if self.output_device_signature is not None:
      device = find_audio_output_device(sd, self.output_device_signature)
      if device is None:
        cloudlog.warning(f"soundd could not find selected audio output device: {describe_audio_device(self.output_device_signature)}")
        self.output_device_signature = None

    return sd.OutputStream(channels=1, samplerate=SAMPLE_RATE, callback=self.callback, blocksize=SAMPLE_BUFFER, device=device)

  def soundd_thread(self):
    # sounddevice must be imported after forking processes
    import sounddevice as sd

    sm = messaging.SubMaster(['selfdriveState', 'soundPressure'])
    threading.Thread(target=self.webrtc_audio_thread, daemon=True).start()

    pending_usb_audio_cards: set[str] = set()
    while True:
      if pending_usb_audio_cards:
        output_device_signature = self.refresh_usb_output_device(sd, pending_usb_audio_cards)
        pending_usb_audio_cards = set()
        if output_device_signature is not None:
          self.output_device_signature = output_device_signature
          cloudlog.info(f"soundd switching to new USB audio output: {describe_audio_device(output_device_signature)}")

      with self.get_stream(sd) as stream:
        rk = Ratekeeper(20)
        self.reset_audio_device_tracking(sd)

        cloudlog.info(f"soundd stream started: {stream.samplerate=} {stream.channels=} {stream.dtype=} {stream.device=}, {stream.blocksize=}")
        while True:
          sm.update(0)

          # freeze volume during alerts to avoid mic feedback increasing volume
          if sm.updated['soundPressure']:
            self.spl_filter_weighted.update(sm["soundPressure"].soundPressureWeightedDb)
            if self.current_alert == AudibleAlert.none:
              self.current_volume = self.calculate_volume(float(self.spl_filter_weighted.x))

          self.get_audible_alert(sm)

          # Ramp up immediate warning sound over 4s
          if self.current_alert == AudibleAlert.warningImmediate:
            elapsed = time.monotonic() - self.ramp_start_time
            ramp_vol = float(np.interp(elapsed, [0, ALERT_RAMP_TIME], [self.ramp_start_volume, MAX_VOLUME]))
            self.current_volume = max(self.current_volume, ramp_vol)

          output_device_signature, new_usb_audio_cards = self.check_for_new_usb_output_device(sd)
          if output_device_signature is not None:
            self.output_device_signature = output_device_signature
            cloudlog.info(f"soundd switching to new USB audio output: {describe_audio_device(output_device_signature)}")
            break
          if new_usb_audio_cards:
            pending_usb_audio_cards = new_usb_audio_cards
            cloudlog.info(f"soundd found new USB audio card ids: {sorted(new_usb_audio_cards)}")
            break

          rk.keep_time()

          assert stream.active


def main():
  s = Soundd()
  s.soundd_thread()


if __name__ == "__main__":
  main()
