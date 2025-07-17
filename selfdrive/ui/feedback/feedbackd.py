#!/usr/bin/env python3

# Attribution: The models melspectrogram.onnx and embedding_model.onnx
# are derived from a speech embedding model originally created by Google
# (https://www.kaggle.com/models/google/speech-embedding/tensorFlow1/speech-embedding/1),
# re-implemented/modified by David Scripka in the openWakeWord project
# (https://github.com/dscripka/openWakeWord/), and adapted here with fixed input sizes.
#
# These models are licensed under the Apache License, Version 2.0 (the "License");
# you may not use these models except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, these models
# are distributed under the License on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Note: This file as a whole is licensed under the MIT License.
#       The above applies only to the referenced models.

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
CHUNK_SIZE = 1280
BUFFER_SIZE = CHUNK_SIZE + 480
VAD_FRAME_SIZE = 640

PROCESS_NAME = "selfdrive.ui.feedback.feedbackd"

MODEL_PKL_PATHS = {
  'melspec': Path(__file__).parent / "models/melspectrogram_tinygrad.pkl",
  'embedding': Path(__file__).parent / "models/embedding_model_tinygrad.pkl",
  'wakeword': Path(__file__).parent / "models/hey_comma_tinygrad.pkl",
  'vad': Path(__file__).parent / "models/silero_simplified_unrolled_tinygrad.pkl",
}


class ModelState:
  def __init__(self):
    self.raw_audio_buffer = np.array([], dtype=np.int16)
    self.melspec_buffer = np.ones((76, 32), dtype=np.float32)
    self.feature_buffer = np.zeros((16, 96), dtype=np.float32)
    self.pm = PubMaster(['feedbackState', 'heyComma', 'audioFeedback'])

    # VAD buffers
    self.vad_state = np.zeros((2, 1, 128), dtype=np.float32)
    self.vad_raw_data_buffer = np.array([], dtype=np.float32)

    # State tracking for wakeword vs VAD mode
    self.in_vad_mode = False
    self.vad_start_time = 0
    self.vad_low_start_time = None
    self.feedback_start_time = 0

    self.last_detection_time = 0
    self.consecutive_detections = 0

    cloudlog.warning("Loading wake word models...")
    try:
      with open(MODEL_PKL_PATHS['melspec'], "rb") as f:
        self.melspec_model = pickle.load(f)
      with open(MODEL_PKL_PATHS['embedding'], "rb") as f:
        self.embedding_model = pickle.load(f)
      with open(MODEL_PKL_PATHS['wakeword'], "rb") as f:
        self.wakeword_model = pickle.load(f)
      with open(MODEL_PKL_PATHS['vad'], "rb") as f:
        self.vad_model = pickle.load(f)
      cloudlog.warning("Wake word models loaded.")
    except Exception as e:
      cloudlog.error(f"Failed to load wake word models: {e}")
      self.melspec_model = None
      self.embedding_model = None
      self.wakeword_model = None
      self.vad_model = None

  def run_model(self, model, input_data):
    tensor = Tensor(input_data, dtype=dtypes.float32, device='NPY')
    output = model(input=tensor).realize().uop.base.buffer.numpy()  # grab directly from buffer without reshaping bc faster
    return output

  def process_vad(self, audio_chunk: np.ndarray):
    if not self.vad_model:
      return None

    audio_chunk_float32 = audio_chunk.astype(np.float32) / 32767.0
    self.vad_raw_data_buffer = np.concatenate([self.vad_raw_data_buffer, audio_chunk_float32])
    vad_score = None

    while len(self.vad_raw_data_buffer) >= VAD_FRAME_SIZE:
      t1 = time.perf_counter()
      processing_chunk = self.vad_raw_data_buffer[:VAD_FRAME_SIZE]
      input_tensor = Tensor(processing_chunk, dtype=dtypes.float32, device='NPY').unsqueeze(0)
      state_tensor = Tensor(self.vad_state, dtype=dtypes.float32, device='NPY')

      out = self.vad_model(input=input_tensor, state=state_tensor).realize().uop.base.buffer.numpy()
      self.vad_state = out[1:].reshape((2, 1, 128))
      self.vad_raw_data_buffer = self.vad_raw_data_buffer[VAD_FRAME_SIZE:]

      t2 = time.perf_counter()
      vad_score = float(out[0])

      # Check exit conditions and calculate timeout
      current_time = time.monotonic()
      time_since_start = current_time - self.vad_start_time

      # Update VAD low state tracking
      if vad_score < 0.1:
        if self.vad_low_start_time is None:
          self.vad_low_start_time = current_time
      else:
        self.vad_low_start_time = None

      # Check both exit conditions in one if statement
      low_vad_timeout = (self.vad_low_start_time is not None and
                        current_time - self.vad_low_start_time > 2.0)

      if time_since_start >= 15.0 or low_vad_timeout:
        reason = "15 seconds" if time_since_start >= 15.0 else "VAD below 0.1 for more than 2 seconds"
        cloudlog.info(f"VAD mode timeout reached ({reason}). Returning to wakeword detection.")
        self.in_vad_mode = False
        self.vad_low_start_time = None
        return vad_score

      # Calculate timeout (time until either condition is met)
      time_until_15s = 15.0 - time_since_start
      time_until_2s = (2.0 - (current_time - self.vad_low_start_time)
                      if self.vad_low_start_time is not None
                      else float('inf'))
      timeout = max(0.0, min(time_until_15s, time_until_2s))

      # Publish audioFeedback message with timeout
      msg = messaging.new_message('audioFeedback', valid=True)
      af = msg.audioFeedback
      af.feedbackStartTime = self.feedback_start_time
      af.vadProb = vad_score
      af.data.data = audio_chunk.tobytes()
      af.data.sampleRate = SAMPLE_RATE
      af.timeout = timeout
      self.pm.send('audioFeedback', msg)

      # Update feedbackState with VAD info
      msg = messaging.new_message('feedbackState', valid=True)
      fs = msg.feedbackState
      fs.vadExecutionTime = float(t2 - t1)
      fs.vadProb = vad_score
      self.pm.send('feedbackState', msg)

      if vad_score > 0.1:
        cloudlog.warning(f"VAD detected speech: {vad_score:.4f}")

    return vad_score

  def process_wakeword_detection(self, wakeword_prob):
    current_time = time.monotonic()

    if wakeword_prob > 0.2:
      self.consecutive_detections += 1
      cloudlog.debug(f"Wake word segment detected! Score: {wakeword_prob:.3f}, Consecutive: {self.consecutive_detections}")

      # Check if we should trigger wakeword detection
      if (self.consecutive_detections >= 2 or wakeword_prob > 0.5) and not self.in_vad_mode:
        cloudlog.info("Wake word detected! Switching to VAD mode.")
        self.last_detection_time = current_time
        self.in_vad_mode = True
        self.vad_start_time = current_time
        self.feedback_start_time = int(current_time * 1e9)  # Convert to nanoseconds
        self.vad_low_start_time = None

        # Reset wakeword buffers when switching to VAD mode
        self.raw_audio_buffer = np.array([], dtype=np.int16)
        self.melspec_buffer = np.ones((76, 32), dtype=np.float32)
        self.feature_buffer = np.zeros((16, 96), dtype=np.float32)

        # Send heyComma message
        msg = messaging.new_message('heyComma', valid=True)
        self.pm.send('heyComma', msg)
    else:
      self.consecutive_detections = 0


  def run(self, msg):
    if not all([self.melspec_model, self.embedding_model, self.wakeword_model]):
      cloudlog.error("models not loaded, feedbackmodeld cannot run")
      return

    # Extract audio data from message
    audio_chunk = np.frombuffer(msg.data, dtype=np.int16)

    # Check sample rate
    if msg.sampleRate != SAMPLE_RATE:
      cloudlog.error(f"Sample rate mismatch: expected wakeword sample rate {SAMPLE_RATE}, got {msg.sampleRate}")

    self.raw_audio_buffer = np.concatenate([self.raw_audio_buffer, audio_chunk])

    # Always run wakeword detection when not in VAD mode
    if not self.in_vad_mode:
      while len(self.raw_audio_buffer) >= BUFFER_SIZE:
        t1 = time.perf_counter()
        processing_chunk = self.raw_audio_buffer[:BUFFER_SIZE]

        new_melspec = self.run_model(self.melspec_model, processing_chunk[None, :])
        new_melspec = (new_melspec.reshape(8, 32) / 10) + 2  # done in openWakeWord to better match Google's implementation
        self.melspec_buffer = np.vstack([self.melspec_buffer, new_melspec])[-76:]

        new_embedding = self.run_model(self.embedding_model, self.melspec_buffer[None, :, :, None])
        self.feature_buffer = np.vstack([self.feature_buffer, new_embedding])[-16:]

        wakeword_prob = self.run_model(self.wakeword_model, self.feature_buffer[None, :])[0]

        self.raw_audio_buffer = self.raw_audio_buffer[CHUNK_SIZE:]

        t2 = time.perf_counter()

        # Send feedbackState with wakeword info
        msg_fb = messaging.new_message('feedbackState', valid=True)
        fs = msg_fb.feedbackState
        fs.wakewordExecutionTime = float(t2 - t1)
        fs.wakewordProb = float(wakeword_prob)
        self.pm.send('feedbackState', msg_fb)

        self.process_wakeword_detection(wakeword_prob)

    # Only run VAD when in VAD mode
    if self.in_vad_mode:
      self.process_vad(audio_chunk)


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
