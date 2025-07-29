import time
import pickle
import numpy as np
from pathlib import Path

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes

from cereal import messaging
from cereal.messaging import PubMaster, SubMaster
from openpilot.common.realtime import config_realtime_process
from openpilot.common.swaglog import cloudlog

SAMPLE_RATE = 16000
CHUNK_SIZE = 800

PROCESS_NAME = "selfdrive.ui.feedback.feedbackd"

MODEL_PKL_PATH = Path(__file__).parent / "models/hey_comma_tinygrad.pkl"

class StreamingLogMelProcessor:
  def __init__(self, sr: int = 16000, config: dict = None):
    self.sr = sr
    self.cfg = config or {"n_fft": 512, "win_length": 400, "hop_length": 160, "n_mels": 26, "fmin": 300, "fmax": 3800}

    self.n_fft = self.cfg["n_fft"]
    self.win_length = self.cfg["win_length"] if self.cfg["win_length"] is not None else self.n_fft
    self.hop_length = self.cfg["hop_length"]
    self.n_mels = self.cfg["n_mels"]
    self.fmin = self.cfg.get("fmin", 0.0)
    self.fmax = self.cfg.get("fmax", sr / 2)
    self.power = self.cfg.get("power", 2.0)

    self._precompute_matrices()

    self.overlap_buffer = np.zeros(0, dtype=np.float32)

  def _precompute_matrices(self):
    n = np.arange(self.win_length, dtype=np.float32)
    self.window = 0.5 - 0.5 * np.cos(2 * np.pi * n / self.win_length)  # hann
    self.window = np.pad(self.window, (0, self.n_fft - self.win_length)).astype(np.float32)

    self.mel_basis = self._mel_filter_bank(self.sr, self.n_fft, self.n_mels, self.fmin, self.fmax)

  def _mel_frequencies(self, n_mels: int, fmin: float, fmax: float):
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mels = np.linspace(mel_min, mel_max, n_mels)
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

  def _mel_filter_bank(self, sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float):
    fftfreqs = np.linspace(0, sr / 2, 1 + n_fft // 2, dtype=np.float32)
    mel_f = self._mel_frequencies(n_mels + 2, fmin, fmax)
    fdiff = np.diff(mel_f)[:, np.newaxis]
    ramps = mel_f[:, np.newaxis] - fftfreqs[np.newaxis, :]
    lower = -ramps[:-2] / fdiff[:-1]
    upper = ramps[2:] / fdiff[1:]
    weights = np.maximum(0, np.minimum(lower, upper))
    # Slaney normalization
    enorm = 2.0 / (mel_f[2:] - mel_f[:-2])
    weights *= enorm[:, np.newaxis]
    return weights.astype(np.float32)

  def _power_to_db(self, S: np.ndarray, ref=1.0, amin: float = 1e-10, top_db: float = 80.0):
    S = np.asarray(S, dtype=np.float32)

    log_spec = 10 * np.log10(np.maximum(amin, S))
    log_spec -= 10 * np.log10(np.maximum(amin, ref))

    if top_db is not None:
      log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec.astype(np.float32)

  def process_chunk(self, audio_chunk: np.ndarray):
    audio_chunk = audio_chunk.astype(np.float32)
    buffer_array = np.concatenate([self.overlap_buffer, audio_chunk])

    if len(buffer_array) < self.n_fft:
      self.overlap_buffer = buffer_array
      return np.empty((self.n_mels, 0), dtype=np.float32)

    max_possible_frames = ((len(buffer_array) - self.n_fft) // self.hop_length) + 1
    frames = np.lib.stride_tricks.sliding_window_view(buffer_array, self.n_fft)[:: self.hop_length][:max_possible_frames]
    windowed_frames = frames * self.window[None, :]

    S = np.fft.rfft(windowed_frames, axis=1)
    power_S = np.abs(S) ** self.power
    mel_spec = np.dot(power_S, self.mel_basis.T)
    log_mel = self._power_to_db(mel_spec)

    overlap_length = self.n_fft - self.hop_length
    if len(buffer_array) > overlap_length:
      self.overlap_buffer = buffer_array[-overlap_length:]
    else:
      self.overlap_buffer = buffer_array

    return log_mel.T.astype(np.float32)  # (n_mels, n_frames_in_chunk)

class ModelState:
  def __init__(self):
    self.pm = PubMaster(['feedbackState', 'heyComma'])

    self.debounce_period = 2.0
    self.last_detection_time = 0

    # melspec and model parameters
    self.melspec_config = {"n_fft": 512, "win_length": 400, "hop_length": 160, "n_mels": 26, "fmin": 300, "fmax": 3800}
    self.required_frames = 150
    self.melspec_processor = StreamingLogMelProcessor(sr=SAMPLE_RATE, config=self.melspec_config)
    self.melspec_buffer = np.empty((self.melspec_config["n_mels"], 0), dtype=np.float32)

    self.audio_buffer = np.zeros(0, dtype=np.float32)

    cloudlog.warning("Loading wake word model...")
    try:
      with open(MODEL_PKL_PATH, "rb") as f:
        self.wakeword_model = pickle.load(f)
      cloudlog.warning("Wake word model loaded.")
    except Exception as e:
      cloudlog.error(f"Failed to load wake word model: {e}")
      self.wakeword_model = None

  def run_model(self, model, input_data):
    tensor = Tensor(input_data, dtype=dtypes.float32, device='NPY')
    output = model(input=tensor).realize().uop.base.buffer.numpy()  # grab directly from buffer without reshaping bc faster
    return output

  def process_wakeword_detection(self, wakeword_prob):
    current_time = time.monotonic()

    if wakeword_prob > 0.9:
      cloudlog.debug(f"Wake word segment detected! Score: {wakeword_prob:.3f}")
      if (current_time - self.last_detection_time) > self.debounce_period:
        cloudlog.info("Wake word detected!")
        self.last_detection_time = current_time
        msg = messaging.new_message('heyComma', valid=True)
        self.pm.send('heyComma', msg)
    else:
      self.consecutive_detections = 0

  def run(self, msg):
    if not self.wakeword_model:
      cloudlog.error("model not loaded, feedbackd cannot run")
      return

    audio_chunk_int16 = np.frombuffer(msg.data, dtype=np.int16)
    audio_chunk = audio_chunk_int16.astype(np.float32) / 32768.0

    if msg.sampleRate != SAMPLE_RATE:
      cloudlog.error(f"Sample rate mismatch: expected wakeword sample rate {SAMPLE_RATE}, got {msg.sampleRate}")

    self.audio_buffer = np.concatenate((self.audio_buffer, audio_chunk))

    while len(self.audio_buffer) >= 2 * CHUNK_SIZE: # 1600 samples = 100ms
      t1 = time.perf_counter()
      to_process = self.audio_buffer[:2 * CHUNK_SIZE]
      self.audio_buffer = self.audio_buffer[2 * CHUNK_SIZE:]

      new_melspec = self.melspec_processor.process_chunk(to_process)
      self.melspec_buffer = np.hstack((self.melspec_buffer, new_melspec))
      self.melspec_buffer = self.melspec_buffer[:, -self.required_frames :]

      if self.melspec_buffer.shape[1] == self.required_frames:
        input_data = self.melspec_buffer[None, :, :]  # (1, n_melspec, required_frames)

        logits = self.run_model(self.wakeword_model, input_data)

        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        wakeword_prob = probabilities[1]

        t2 = time.perf_counter()

        msg = messaging.new_message('feedbackState', valid=True)
        fs = msg.feedbackState
        fs.totalExecutionTime = float(t2 - t1)
        fs.wakewordProb = float(wakeword_prob)
        self.pm.send('feedbackState', msg)

        self.process_wakeword_detection(wakeword_prob)

def main():
  config_realtime_process([0, 1, 2, 3], 5)

  model = ModelState()
  sm = SubMaster(['rawAudioData'])

  while True:
    sm.update()
    if sm.updated['rawAudioData']:
      msg = sm['rawAudioData']
      model.run(msg)

if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    cloudlog.warning("got SIGINT")
