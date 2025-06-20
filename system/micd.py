#!/usr/bin/env python3
import os
import time
import pickle
import numpy as np
from functools import cache
import threading
from pathlib import Path
from openpilot.system.hardware import TICI
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes

if TICI:
  os.environ['QCOM'] = '1'
else:
  os.environ['LLVM'] = '1'
from cereal import messaging
from openpilot.common.realtime import Ratekeeper
from openpilot.common.retry import retry
from openpilot.common.swaglog import cloudlog

RATE = 10
FFT_SAMPLES = 4096
REFERENCE_SPL = 2e-5  # newtons/m^2
SAMPLE_RATE = 16000
SAMPLE_BUFFER = 1280  # approx 80ms

WW_SAMPLE_RATE = 16000
WW_CHUNK_SIZE = 1280
WW_BUFFER_SIZE = WW_CHUNK_SIZE + 480  # overlap with prev chunk
VAD_FRAME_SIZE = 640


@cache
def get_a_weighting_filter():
  # Calculate the A-weighting filter
  # https://en.wikipedia.org/wiki/A-weighting
  freqs = np.fft.fftfreq(FFT_SAMPLES, d=1 / SAMPLE_RATE)
  A = 12194**2 * freqs**4 / ((freqs**2 + 20.6**2) * (freqs**2 + 12194**2) * np.sqrt((freqs**2 + 107.7**2) * (freqs**2 + 737.9**2)))
  return A / np.max(A)


def calculate_spl(measurements):
  # https://www.engineeringtoolbox.com/sound-pressure-d_711.html
  sound_pressure = np.sqrt(np.mean(measurements**2))  # RMS of amplitudes
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


def resample_audio(audio_chunk: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
  if from_rate == to_rate:
    return audio_chunk
  # Resamples an audio chunk using linear interpolation.
  num_samples_in = len(audio_chunk)
  num_samples_out = int(num_samples_in * to_rate / from_rate)

  x_original = np.arange(num_samples_in)
  x_new = np.linspace(0, num_samples_in - 1, num_samples_out)
  resampled_chunk = np.interp(x_new, x_original, audio_chunk)

  return resampled_chunk


class Mic:
  def __init__(self):
    self.rk = Ratekeeper(RATE)
    self.pm = messaging.PubMaster(['microphone', 'heyComma'])

    self.measurements = np.empty(0)

    self.sound_pressure = 0
    self.sound_pressure_weighted = 0
    self.sound_pressure_level_weighted = 0

    self.lock = threading.Lock()

    # Audio saving attributes
    self.audio_save_duration = 3.0  # seconds to save
    self.audio_buffer_size = int(WW_SAMPLE_RATE * self.audio_save_duration)
    self.audio_circular_buffer = np.zeros(self.audio_buffer_size, dtype=np.int16)
    self.buffer_write_index = 0
    self.save_counter = 0  # for unique filenames

    cloudlog.warning("Loading wake word models...")
    try:  # TODO currently only built for QCOM, should build according to arch in scons
      model_path = Path(__file__).parent
      with open(model_path / "wakeword_models/melspectrogram_tinygrad_compile3.pkl", "rb") as f:
        self.melspec_model = pickle.load(f)
      with open(model_path / "wakeword_models/embedding_model_tinygrad_compile3.pkl", "rb") as f:
        self.embedding_model = pickle.load(f)
      with open(model_path / "wakeword_models/hey_comma_tinygrad_v17_compile3.pkl", "rb") as f:
        self.wakeword_model = pickle.load(f)
      cloudlog.warning("Wake word models loaded.")
    except Exception as e:
      cloudlog.error(f"Failed to load wake word models: {e}")
      self.wakeword_model = None  # Continue without wakeword detection if models fail to load

    cloudlog.warning("Loading VAD model...")
    try:  # TODO currently only built for QCOM, should build according to arch in scons
      model_path = Path(__file__).parent
      with open(model_path / "wakeword_models/silero_vad_v5_simplified_tinygrad.pkl", "rb") as f:
        self.vad_model = pickle.load(f)
    except Exception as e:
      cloudlog.error(f"Failed to load wake word models: {e}")
      self.vad_model = None  # Continue without VAD if models fail to load

    # Buffers for the wake word model
    self.ww_raw_data_buffer = np.array([], dtype=np.int16)
    self.ww_melspectrogram_buffer = np.ones((76, 32), dtype=np.float32)
    self.ww_feature_buffer = np.zeros((16, 96), dtype=np.float32)
    self.last_ww_detection_time = 0
    self.debounce_period = 3.0  # seconds
    self.consecutive_detections = 0

    self.vad_state = np.zeros((2, 1, 128), dtype=np.float32)
    self.vad_raw_data_buffer = np.array([], dtype=np.float32)

  def _get_melspectrogram(self, audio_data: np.ndarray) -> np.ndarray:
    audio_float32 = audio_data.astype(np.float32)
    audio_tensor = Tensor(audio_float32[None, :], dtype=dtypes.float32, device='NPY')
    spec = np.squeeze(self.melspec_model(input=audio_tensor).numpy())
    return (spec / 10) + 2

  def _predict_wakeword(self, features: np.ndarray) -> float:
    model_input = features[None, :].astype(np.float32)
    feature_tensor = Tensor(model_input, device='NPY')
    prediction = self.wakeword_model(input=feature_tensor).numpy()
    return prediction[0][0]

  def _process_wakeword(self, audio_chunk_16khz: np.ndarray):
    chunk_size = len(audio_chunk_16khz)  # Store audio in buffer for potential saving
    if self.buffer_write_index + chunk_size <= self.audio_buffer_size:  # Handle buffer wraparound
      self.audio_circular_buffer[self.buffer_write_index : self.buffer_write_index + chunk_size] = audio_chunk_16khz
    else:  # Split across buffer boundary
      first_part_size = self.audio_buffer_size - self.buffer_write_index
      self.audio_circular_buffer[self.buffer_write_index :] = audio_chunk_16khz[:first_part_size]
      self.audio_circular_buffer[: chunk_size - first_part_size] = audio_chunk_16khz[first_part_size:]
    self.buffer_write_index = (self.buffer_write_index + chunk_size) % self.audio_buffer_size

    self.ww_raw_data_buffer = np.concatenate([self.ww_raw_data_buffer, audio_chunk_16khz])
    while len(self.ww_raw_data_buffer) >= WW_BUFFER_SIZE:
      processing_chunk = self.ww_raw_data_buffer[:WW_BUFFER_SIZE]
      new_melspec = self._get_melspectrogram(processing_chunk)
      self.ww_melspectrogram_buffer = np.vstack([self.ww_melspectrogram_buffer, new_melspec])[-76:]

      window_batch = self.ww_melspectrogram_buffer[None, :, :, None].astype(np.float32)
      window_tensor = Tensor(window_batch, device='NPY')
      new_embedding = self.embedding_model(input_1=window_tensor).numpy().squeeze()
      self.ww_feature_buffer = np.vstack([self.ww_feature_buffer, new_embedding])[-16:]

      score = self._predict_wakeword(self.ww_feature_buffer)
      current_time = time.time()

      if score > 0.1:
        self.consecutive_detections += 1
        cloudlog.warning(f"Wake word segment detected! Score: {float(score):.3f}, Consecutive: {self.consecutive_detections}")
        if (self.consecutive_detections >= 2 or score > 0.5) and (current_time - self.last_ww_detection_time) > self.debounce_period:
          self.last_ww_detection_time = current_time
          cloudlog.warning("send heyComma message")

          # Calculate detection offset for saving buffer
          detection_offset = len(self.ww_raw_data_buffer) - WW_BUFFER_SIZE // 2
          self._save_audio_buffer(detection_offset)

          hey_comma = messaging.new_message('heyComma')
          hey_comma.valid = True
          self.pm.send('heyComma', hey_comma)
      else:
        self.consecutive_detections = 0

      self.ww_raw_data_buffer = self.ww_raw_data_buffer[WW_CHUNK_SIZE:]

  def _process_vad(self, audio_chunk_16khz: np.ndarray):
    self.vad_raw_data_buffer = np.concatenate([self.vad_raw_data_buffer, audio_chunk_16khz])
    # frame_predictions = []
    while len(self.vad_raw_data_buffer) >= VAD_FRAME_SIZE:
      processing_chunk = self.vad_raw_data_buffer[:VAD_FRAME_SIZE]
      input_tensor = Tensor(processing_chunk, dtype=dtypes.float32, device='NPY').unsqueeze(0)
      state_tensor = Tensor(self.vad_state, dtype=dtypes.float32, device='NPY')
      sr_tensor = Tensor(WW_SAMPLE_RATE, dtype=dtypes.long, device='NPY')
      out, new_state = self.vad_model(input=input_tensor, state=state_tensor, sr=sr_tensor)
      self.vad_state = new_state.numpy()
      # frame_predictions.append(out.numpy()[0][0])
      self.vad_raw_data_buffer = self.vad_raw_data_buffer[VAD_FRAME_SIZE:]
      score = float(out.numpy()[0][0])
      if score > 0.1:
        cloudlog.warning(f"VAD {score:.4f}")

  def _save_audio_buffer(self, detection_offset_samples=0):
    try:
      import wave

      pre_detection_samples = int(1.5 * WW_SAMPLE_RATE)  # 1.5 seconds before
      post_detection_samples = int(1.5 * WW_SAMPLE_RATE)  # 1.5 seconds after
      total_samples_to_save = pre_detection_samples + post_detection_samples

      # Find the detection point in the circular buffer
      detection_index = (self.buffer_write_index - detection_offset_samples) % self.audio_buffer_size

      start_index = (detection_index - pre_detection_samples) % self.audio_buffer_size
      end_index = (detection_index + post_detection_samples) % self.audio_buffer_size

      if start_index < end_index:  # No wraparound case
        audio_data = self.audio_circular_buffer[start_index:end_index]
      else:  # Wraparound case
        audio_data = np.concatenate([self.audio_circular_buffer[start_index:], self.audio_circular_buffer[:end_index]])

      if len(audio_data) != total_samples_to_save:
        if len(audio_data) < total_samples_to_save:
          audio_data = np.pad(audio_data, (0, total_samples_to_save - len(audio_data)), 'constant')
        else:
          audio_data = audio_data[:total_samples_to_save]

      timestamp = int(time.time())
      filename = f"/data/media/0/wakeword_audio_{timestamp}_{self.save_counter:03d}.wav"
      self.save_counter += 1

      with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit = 2 bytes
        wav_file.setframerate(WW_SAMPLE_RATE)  # 16kHz
        wav_file.writeframes(audio_data.tobytes())

      cloudlog.warning(f"Audio saved to {filename} (detection at sample {detection_index})")

    except Exception as e:
      cloudlog.error(f"Failed to save audio: {e}")

  def update(self):
    with self.lock:
      sound_pressure = self.sound_pressure
      sound_pressure_weighted = self.sound_pressure_weighted
      sound_pressure_level_weighted = self.sound_pressure_level_weighted

    msg = messaging.new_message('microphone', valid=True)
    msg.microphone.soundPressure = float(sound_pressure)
    msg.microphone.soundPressureWeighted = float(sound_pressure_weighted)
    msg.microphone.soundPressureWeightedDb = float(sound_pressure_level_weighted)

    self.pm.send('microphone', msg)
    self.rk.keep_time()

  def callback(self, indata, frames, time, status):
    """
    Using amplitude measurements, calculate an uncalibrated sound pressure and sound pressure level.
    Then apply A-weighting to the raw amplitudes and run the same calculations again.

    Logged A-weighted equivalents are rough approximations of the human-perceived loudness.
    """
    with self.lock:
      self.measurements = np.concatenate((self.measurements, indata[:, 0]))

      while self.measurements.size >= FFT_SAMPLES:
        measurements = self.measurements[:FFT_SAMPLES]

        self.sound_pressure, _ = calculate_spl(measurements)
        measurements_weighted = apply_a_weighting(measurements)
        self.sound_pressure_weighted, self.sound_pressure_level_weighted = calculate_spl(measurements_weighted)

        self.measurements = self.measurements[FFT_SAMPLES:]


    resampled_chunk = resample_audio(indata[:, 0], SAMPLE_RATE, WW_SAMPLE_RATE)
    if self.wakeword_model: # excl copy time ~3.88ms GPU time per 80ms chunk (0.38ms melspec, 3.45ms embedding, 0.05ms wakeword)
      audio_data_int16 = (resampled_chunk * 32767).astype(np.int16)
      self._process_wakeword(audio_data_int16)
    if self.vad_model: # excl copy time ~0.92ms GPU time per 80ms chunk
      self._process_vad(resampled_chunk)


  @retry(attempts=7, delay=3)
  def get_stream(self, sd):
    # reload sounddevice to reinitialize portaudio
    sd._terminate()
    sd._initialize()
    return sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=self.callback, blocksize=SAMPLE_BUFFER)

  def micd_thread(self):
    # sounddevice must be imported after forking processes
    import sounddevice as sd

    with self.get_stream(sd) as stream:
      cloudlog.info(f"micd stream started: {stream.samplerate=} {stream.channels=} {stream.dtype=} {stream.device=}, {stream.blocksize=}")
      while True:
        self.update()


def main():
  mic = Mic()
  mic.micd_thread()


if __name__ == "__main__":
  main()
