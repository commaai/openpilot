from collections import deque
import math
import threading
import time
import wave

import numpy as np

from openpilot.cereal import messaging
from opendbc.car.structs import car
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

AMBIENT_DB = 24 # DB where MIN_VOLUME is applied
DB_SCALE = 30 # AMBIENT_DB + DB_SCALE is where MAX_VOLUME is applied

VOLUME_BASE = 20
if HARDWARE.get_device_type() == "tizi":
  AMBIENT_DB = 30
  VOLUME_BASE = 10

AudibleAlert = car.CarControl.HUDControl.AudibleAlert


sound_list: dict[int, tuple[str, int | None, float]] = {
  # AudibleAlert, file name, play count (none for infinite)
  AudibleAlert.engage: ("engage.wav", 1, MAX_VOLUME),
  AudibleAlert.disengage: ("disengage.wav", 1, MAX_VOLUME),
  AudibleAlert.refuse: ("refuse.wav", 1, MAX_VOLUME),

  AudibleAlert.prompt: ("prompt.wav", 1, MAX_VOLUME),
  AudibleAlert.promptRepeat: ("prompt.wav", None, MAX_VOLUME),
  AudibleAlert.promptDistracted: ("prompt_distracted.wav", None, MAX_VOLUME),

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

  @retry(attempts=10, delay=3)
  def get_stream(self, sd):
    # reload sounddevice to reinitialize portaudio
    sd._terminate()
    sd._initialize()
    return sd.OutputStream(channels=1, samplerate=SAMPLE_RATE, callback=self.callback, blocksize=SAMPLE_BUFFER)

  def soundd_thread(self):
    # sounddevice must be imported after forking processes
    import sounddevice as sd

    sm = messaging.SubMaster(['selfdriveState', 'soundPressure'])
    threading.Thread(target=self.webrtc_audio_thread, daemon=True).start()

    with self.get_stream(sd) as stream:
      rk = Ratekeeper(20)

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

        rk.keep_time()

        assert stream.active


def main():
  s = Soundd()
  s.soundd_thread()


if __name__ == "__main__":
  main()
