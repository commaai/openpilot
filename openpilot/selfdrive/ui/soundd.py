from collections import deque
import math
import numpy as np
import threading
import time
import wave


from openpilot.cereal import log, messaging
from openpilot.common.basedir import BASEDIR
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process, Ratekeeper
from openpilot.common.swaglog import cloudlog

from openpilot.system import micd
from openpilot.system.audio_device import AutomaticAudioDevice
from openpilot.common.hardware import HARDWARE

SAMPLE_RATE = 48000
SAMPLE_BUFFER = 480  # 10ms
WEBRTC_MAX_BUFFER = SAMPLE_RATE * 200 // 1000
WEBRTC_START_BUFFER = SAMPLE_RATE * 30 // 1000
WEBRTC_FADE_SAMPLES = SAMPLE_RATE * 5 // 1000
AUDIO_DEVICE_POLL_INTERVAL = 1.0
MAX_VOLUME = 1.0
MIN_VOLUME = 0.1
ALERT_RAMP_TIME = 4 # seconds to ramp to max volume for warningImmediate
SELFDRIVE_STATE_TIMEOUT = 5 # 5 seconds
FILTER_DT = 1. / (micd.SAMPLE_RATE / micd.FFT_SAMPLES)

AMBIENT_DB = 26 # DB where MIN_VOLUME is applied
DB_SCALE = 30 # AMBIENT_DB + DB_SCALE is where MAX_VOLUME is applied

VOLUME_BASE = 20
if HARDWARE.get_device_type() == "tizi":
  AMBIENT_DB = 30
  VOLUME_BASE = 10

AudibleAlert = log.SelfdriveState.AudibleAlert


sound_list: dict[int, tuple[str, int | None, float]] = {
  # AudibleAlert, file name, play count (none for infinite)
  AudibleAlert.engage: ("engage.wav", 1, MAX_VOLUME),
  AudibleAlert.disengage: ("disengage.wav", 1, MAX_VOLUME),
  AudibleAlert.refuse: ("refuse.wav", 1, MAX_VOLUME),

  AudibleAlert.prompt: ("warning.wav", 1, MAX_VOLUME),
  AudibleAlert.promptRepeat: ("warning.wav", None, MAX_VOLUME),
  AudibleAlert.promptDistracted: ("dm_warning.wav", None, MAX_VOLUME),

  AudibleAlert.preAlert: ("pre_alert.wav", 1, MAX_VOLUME),

  AudibleAlert.warningSoft: ("critical.wav", None, MAX_VOLUME),
  AudibleAlert.warningImmediate: ("dm_critical.wav", None, MAX_VOLUME),
}

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
    self.pending_stop = False

    self.spl_filter_weighted = FirstOrderFilter(0, 2.5, FILTER_DT, initialized=False)
    self.webrtc_audio: deque[float] = deque()
    self.webrtc_audio_lock = threading.Lock()
    self.webrtc_reference: deque[bytes] = deque(maxlen=100)
    self.webrtc_playing = False
    self.speaker_volume = 1.0
    self.device_selector = AutomaticAudioDevice("output")

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
        if self.pending_stop and current_sound_frame == 0:
          self.current_alert = AudibleAlert.none
          self.pending_stop = False
          break

    return ret * self.current_volume

  def callback(self, data_out: np.ndarray, frames: int, time, status) -> None:
    if status:
      cloudlog.warning(f"soundd stream over/underflow: {status}")
    data_out[:frames, 0] = np.clip(self.get_sound_data(frames) + self.get_webrtc_audio(frames), -1., 1.)
    reference = np.clip(data_out[:frames, 0] * 32767, -32768, 32767).astype(np.int16).tobytes()
    with self.webrtc_audio_lock:
      self.webrtc_reference.append(reference)

  def add_webrtc_audio(self, data: bytes, sample_rate: int) -> None:
    if sample_rate != SAMPLE_RATE:
      cloudlog.warning(f"Ignoring WebRTC audio with unexpected sample rate: {sample_rate}")
      return
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.
    with self.webrtc_audio_lock:
      overflow = len(self.webrtc_audio) + len(samples) - WEBRTC_MAX_BUFFER
      queued_drop = min(len(self.webrtc_audio), max(0, overflow))
      for _ in range(queued_drop):
        self.webrtc_audio.popleft()
      if overflow > queued_drop:
        samples = samples[overflow - queued_drop:]
      if overflow > 0:
        cloudlog.warning(f"soundd dropped {overflow} stale WebRTC audio samples")
        self.webrtc_playing = False
      self.webrtc_audio.extend(samples)

  def get_webrtc_audio(self, frames: int) -> np.ndarray:
    ret = np.zeros(frames, dtype=np.float32)
    with self.webrtc_audio_lock:
      starting = False
      if not self.webrtc_playing and len(self.webrtc_audio) >= WEBRTC_START_BUFFER:
        self.webrtc_playing = True
        starting = True
      count = min(frames, len(self.webrtc_audio)) if self.webrtc_playing else 0
      for i in range(count):
        ret[i] = self.webrtc_audio.popleft()

      if count and starting:
        fade = min(count, WEBRTC_FADE_SAMPLES)
        ret[:fade] *= np.linspace(0., 1., fade, dtype=np.float32)
      if count < frames and self.webrtc_playing:
        fade = min(count, WEBRTC_FADE_SAMPLES)
        if fade:
          ret[count - fade:count] *= np.linspace(1., 0., fade, dtype=np.float32)
        self.webrtc_playing = False
    return ret * self.speaker_volume

  def update_speaker_volume(self, volume_sock) -> None:
    messages = messaging.drain_sock(volume_sock)
    if messages:
      self.speaker_volume = messages[-1].speakerVolume.volume / 100.

  def publish_speaker_reference(self, pm) -> None:
    with self.webrtc_audio_lock:
      frames = list(self.webrtc_reference)
      self.webrtc_reference.clear()
    for data in frames:
      msg = messaging.new_message("webRtcAudioReference", valid=True)
      msg.webRtcAudioReference.data = data
      msg.webRtcAudioReference.sampleRate = SAMPLE_RATE
      pm.send("webRtcAudioReference", msg)

  def update_alert(self, new_alert):
    current_alert_played_once = self.current_alert == AudibleAlert.none or self.current_sound_frame >= len(self.loaded_sounds[self.current_alert])
    # let looping sounds finish the current loop instead of cutting off mid tone
    if new_alert == AudibleAlert.none and self.current_alert != AudibleAlert.none and sound_list[self.current_alert][1] is None:
      if current_alert_played_once:
        self.pending_stop = True
      else:
        self.current_alert = AudibleAlert.none
        self.current_sound_frame = 0
      return
    self.pending_stop = False
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

  def get_stream(self, sd, device: int | None = None):
    # reload sounddevice to reinitialize portaudio
    sd._terminate()
    sd._initialize()
    return sd.OutputStream(channels=1, samplerate=SAMPLE_RATE, callback=self.callback, blocksize=SAMPLE_BUFFER, device=device)

  def soundd_thread(self):
    # sounddevice must be imported after forking processes
    import sounddevice as sd
    micd.patch_sounddevice(sd)

    sm = messaging.SubMaster(['selfdriveState', 'soundPressure'])
    webrtc_audio_sock = messaging.sub_sock('webRtcAudioData', conflate=False)
    speaker_volume_sock = messaging.sub_sock('speakerVolume', conflate=True)
    reference_pm = messaging.PubMaster(['webRtcAudioReference'])
    # Drain remote audio faster than the output callback consumes its 10 ms
    # blocks. A 20 Hz service loop delivers packets in 50 ms bursts, causing
    # periodic speaker underflows and a moving AEC reference delay.
    rk = Ratekeeper(100)

    failed_until: dict[int | None, float] = {}
    while True:
      try:
        sd._terminate()
        sd._initialize()
        excluded = {device for device, until in failed_until.items() if until > time.monotonic()}
        device, _ = self.device_selector.select(sd, force=True, exclude=excluded)
        candidates = [device] + [candidate for candidate in self.device_selector.alternatives(sd) if candidate not in excluded]
      except Exception:
        cloudlog.exception("soundd failed to enumerate audio output devices")
        time.sleep(1.)
        continue
      for candidate in dict.fromkeys(candidates):
        try:
          with self.get_stream(sd, candidate) as stream:
            self.device_selector.selected = candidate
            failed_until.pop(candidate, None)
            cloudlog.info(f"soundd stream started: {stream.samplerate=} {stream.channels=} {stream.dtype=} {stream.device=}, {stream.blocksize=}")
            next_device_poll = time.monotonic() + AUDIO_DEVICE_POLL_INTERVAL
            while stream.active:
              sm.update(0)
              for msg in messaging.drain_sock(webrtc_audio_sock):
                self.add_webrtc_audio(msg.webRtcAudioData.data, msg.webRtcAudioData.sampleRate)
              self.publish_speaker_reference(reference_pm)
              self.update_speaker_volume(speaker_volume_sock)
              if time.monotonic() >= next_device_poll:
                next_device_poll = time.monotonic() + AUDIO_DEVICE_POLL_INTERVAL
                excluded = {failed_device for failed_device, until in failed_until.items() if until > time.monotonic()}
                _, changed = self.device_selector.select(sd, exclude=excluded)
                if changed:
                  break

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
          break
        except Exception:
          cloudlog.exception(f"soundd failed to open output device {candidate}")
          failed_until[candidate] = time.monotonic() + 5.

      else:
        time.sleep(1.)


def main():
  config_realtime_process([0, 1, 2, 3], 1)
  s = Soundd()
  s.soundd_thread()


if __name__ == "__main__":
  main()
