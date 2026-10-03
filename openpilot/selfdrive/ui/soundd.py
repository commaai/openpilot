import math
import queue
import signal
import numpy as np
import time
import wave


from openpilot.cereal import log, messaging
from openpilot.common.basedir import BASEDIR
from openpilot.common.voice_eq import VoiceEQ
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import Ratekeeper
from openpilot.common.utils import retry
from openpilot.common.swaglog import cloudlog

from openpilot.selfdrive.ui.usb_speaker import USBSpeaker, find_usb_output
from openpilot.system import micd
from openpilot.common.hardware import HARDWARE

SAMPLE_RATE = 48000
SAMPLE_BUFFER = 960 # 20 ms, also used for livestream voice playback
SPEECH_CHUNK = 480  # 10 ms chunks allow a 150 ms startup buffer
SPEECH_BUFFER_PACKETS = 15
SPEECH_QUEUE_PACKETS = 35  # 350 ms maximum backlog
SPEECH_MAX_AGE_NS = 450_000_000
MAX_VOLUME = 1.0
MIN_VOLUME = 0.1
ALERT_RAMP_TIME = 4 # seconds to ramp critical alerts to max volume
ALERT_MAX_TIME = 8 # seconds before critical alerts switch to the max sound
SELFDRIVE_STATE_TIMEOUT = 5 # 5 seconds
FILTER_DT = 1. / (micd.SAMPLE_RATE / micd.FFT_SAMPLES)

AMBIENT_DB = 26 # DB where MIN_VOLUME is applied
DB_SCALE = 30 # AMBIENT_DB + DB_SCALE is where MAX_VOLUME is applied

VOLUME_BASE = 20
if HARDWARE.get_device_type() == "tizi":
  AMBIENT_DB = 30
  VOLUME_BASE = 10

AudibleAlert = log.SelfdriveState.AudibleAlert
CRITICAL_MAX = -1 # internal sound key, not an AudibleAlert


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
  CRITICAL_MAX: ("dm_critical_max.wav", None, MAX_VOLUME),
}

def check_selfdrive_timeout_alert(sm):
  ss_missing = time.monotonic() - sm.recv_time['selfdriveState']

  if ss_missing > SELFDRIVE_STATE_TIMEOUT:
    if sm['selfdriveState'].enabled and (ss_missing - SELFDRIVE_STATE_TIMEOUT) < 10:
      return True

  return False


class LivestreamPlayback:
  """Bounded, expiring PCM queue shared by soundd's main/audio threads."""
  def __init__(self):
    self.eq = VoiceEQ(SAMPLE_RATE)
    self.last_audio_time = 0
    self.queue = queue.Queue(maxsize=SPEECH_QUEUE_PACKETS)
    self.pending = np.empty(0, dtype=np.float32)
    self.pending_time = 0
    self.generation = 0
    self.playing_generation = 0
    self.buffering = True

  def clear(self):
    self.eq.reset()
    self.last_audio_time = 0
    self.generation += 1
    while True:
      try:
        self.queue.get_nowait()
      except queue.Empty:
        break

  def enqueue(self, msg):
    audio = msg.livestreamAudio
    if not msg.valid or audio.sampleRate != SAMPLE_RATE or len(audio.data) % 2 or len(audio.data) > SAMPLE_RATE * 2 * 0.12:
      return
    if not audio.data:
      self.clear()
      return
    if time.monotonic_ns() - msg.logMonoTime > 200_000_000:
      return
    pcm = np.frombuffer(audio.data, dtype=np.int16).astype(np.float32) / 32768
    if msg.logMonoTime - self.last_audio_time > 200_000_000:
      self.eq.reset()
    self.last_audio_time = msg.logMonoTime
    pcm = np.clip(self.eq.process(pcm), -1.0, 1.0)
    for offset in range(0, len(pcm), SPEECH_CHUNK):
      item = (msg.logMonoTime, pcm[offset:offset + SPEECH_CHUNK])
      try:
        self.queue.put_nowait(item)
      except queue.Full:
        try:
          self.queue.get_nowait()
        except queue.Empty:
          pass
        self.queue.put_nowait(item)

  def render(self, frames):
    result = np.zeros(frames, dtype=np.float32)
    if self.playing_generation != self.generation:
      self.pending = np.empty(0, dtype=np.float32)
      self.playing_generation = self.generation
      self.buffering = True
    # Prime 150 ms on startup/recovery to absorb packet arrival jitter.
    # Playback expiry includes this intentional delay.
    if self.buffering:
      if self.queue.qsize() < SPEECH_BUFFER_PACKETS:
        return result
      self.buffering = False
    written = 0
    while written < frames:
      if time.monotonic_ns() - self.pending_time > SPEECH_MAX_AGE_NS:
        self.pending = np.empty(0, dtype=np.float32)
      if not self.pending.size:
        try:
          self.pending_time, self.pending = self.queue.get_nowait()
        except queue.Empty:
          self.buffering = True
          break
        if time.monotonic_ns() - self.pending_time > SPEECH_MAX_AGE_NS:
          continue
      count = min(frames - written, self.pending.size)
      result[written:written + count] = self.pending[:count]
      self.pending = self.pending[count:]
      written += count
    return result


class Soundd:
  def __init__(self):
    self.load_sounds()
    self.livestream = LivestreamPlayback()
    self.usb_stream = None
    self.usb_device = None
    self.usb_retry_at = 0.0
    self.test_tone_frame = None
    self.voice_input_peak = 0.0
    self.voice_output_peak = 0.0

    self.current_alert = AudibleAlert.none
    self.current_sound = AudibleAlert.none
    self.current_volume = MIN_VOLUME
    self.current_sound_frame = 0

    self.ramp_start_volume = MIN_VOLUME
    self.ramp_start_time = 0.

    self.selfdrive_timeout_alert = False
    self.pending_stop = False

    self.spl_filter_weighted = FirstOrderFilter(0, 2.5, FILTER_DT, initialized=False)

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
      sound_data = self.loaded_sounds[self.current_sound]
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
    alert_active = self.current_alert != AudibleAlert.none
    alerts = self.get_sound_data(frames)
    # Use the same gain/limiter on either output; only one consumes speech.
    voice = self.render_voice(frames) if self.usb_stream is None else 0
    data_out[:frames, 0] = alerts if alert_active else voice

  def usb_callback(self, data_out: np.ndarray, frames: int, time, status) -> None:
    if status:
      cloudlog.warning(f"USB speech stream over/underflow: {status}")
    if self.usb_stream is None:
      data_out.fill(0)
      return
    voice = self.render_voice(frames)
    # Give driving alerts priority, even when speech uses USB.
    data_out[:] = voice[:, None] if self.current_alert == AudibleAlert.none else 0

  def request_test_sound(self, *_):
    # SIGUSR1 provides a local diagnostic without taking over a WebRTC session.
    self.test_tone_frame = 0

  def render_voice(self, frames):
    voice = self.livestream.render(frames)
    if self.test_tone_frame is not None:
      position = np.arange(frames) + self.test_tone_frame
      t = position / SAMPLE_RATE
      envelope = np.clip(np.minimum((t % 1) / 0.02, (0.32 - t % 1) / 0.02), 0, 1)
      voice = (0.12 * envelope * np.sin(2 * np.pi * 660 * t)).astype(np.float32)
      self.test_tone_frame += frames
      if self.test_tone_frame >= SAMPLE_RATE * 3:
        self.test_tone_frame = None
    self.voice_output_peak = max(self.voice_output_peak, float(np.max(np.abs(voice))))
    return voice

  def update_usb_stream(self):
    if self.usb_stream is not None:
      try:
        if self.usb_stream.active and find_usb_output() == self.usb_device:
          return
      except Exception:
        pass
      try:
        self.usb_stream.close()
      except Exception:
        cloudlog.exception("Closing USB speech stream failed")
      self.usb_stream = None
      self.livestream.clear()
    if time.monotonic() < self.usb_retry_at:
      return
    self.usb_retry_at = time.monotonic() + 3
    stream = None
    try:
      # Read the live ALSA device list; PortAudio only enumerates at startup.
      device = find_usb_output()
      if device is None:
        return
      stream = USBSpeaker(device, self.usb_callback)
      stream.start()
      self.usb_stream = stream
      self.usb_device = device
      cloudlog.info(f"USB speech stream started: {device=}")
    except Exception:
      if stream is not None:
        try:
          stream.close()
        except Exception:
          pass
      cloudlog.exception("USB speech unavailable; retrying without interrupting alerts")

  def update_alert(self, new_alert):
    current_alert_played_once = self.current_alert == AudibleAlert.none or self.current_sound_frame >= len(self.loaded_sounds[self.current_sound])
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
      if new_alert in (AudibleAlert.warningSoft, AudibleAlert.warningImmediate):
        self.ramp_start_volume = self.current_volume
        self.ramp_start_time = time.monotonic()
      self.current_alert = new_alert
      self.current_sound = new_alert
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
    micd.patch_sounddevice(sd)

    sm = messaging.SubMaster(['selfdriveState', 'soundPressure'])
    livestream = messaging.sub_sock('livestreamAudio')
    signal.signal(signal.SIGUSR1, self.request_test_sound)
    diagnostic_at = time.monotonic() + 5

    with self.get_stream(sd) as stream:
      rk = Ratekeeper(100)

      cloudlog.info(f"soundd stream started: {stream.samplerate=} {stream.channels=} {stream.dtype=} {stream.device=}, {stream.blocksize=}")
      while True:
        sm.update(0)
        self.update_usb_stream()
        for _ in range(16):
          msg = messaging.recv_one_or_none(livestream)
          if msg is None:
            break
          self.livestream.enqueue(msg)
          if msg.livestreamAudio.data and len(msg.livestreamAudio.data) % 2 == 0:
            samples = np.frombuffer(msg.livestreamAudio.data, dtype=np.int16).astype(np.float32) / 32768
            self.voice_input_peak = max(self.voice_input_peak, float(np.max(np.abs(samples))))

        # freeze volume during alerts to avoid mic feedback increasing volume
        if sm.updated['soundPressure']:
          self.spl_filter_weighted.update(sm["soundPressure"].soundPressureWeightedDb)
          if self.current_alert == AudibleAlert.none:
            self.current_volume = self.calculate_volume(float(self.spl_filter_weighted.x))

        self.get_audible_alert(sm)

        if self.current_alert in (AudibleAlert.warningSoft, AudibleAlert.warningImmediate):
          elapsed = time.monotonic() - self.ramp_start_time
          ramp_vol = float(np.interp(elapsed, [0, ALERT_RAMP_TIME], [self.ramp_start_volume, MAX_VOLUME]))
          self.current_volume = max(self.current_volume, ramp_vol)
          if elapsed >= ALERT_MAX_TIME and self.current_sound != CRITICAL_MAX:
            self.current_sound = CRITICAL_MAX
            self.current_sound_frame = 0

        if time.monotonic() >= diagnostic_at:
          cloudlog.info(f"Speech output: input_peak={self.voice_input_peak:.4f} output_peak={self.voice_output_peak:.4f} "
                        + f"usb={self.usb_stream is not None} alert={self.current_alert}")
          self.voice_input_peak = self.voice_output_peak = 0.0
          diagnostic_at = time.monotonic() + 5
        rk.keep_time()

        assert stream.active


def main():
  s = Soundd()
  s.soundd_thread()


if __name__ == "__main__":
  main()
