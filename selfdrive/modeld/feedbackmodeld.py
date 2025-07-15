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

PROCESS_NAME = "selfdrive.modeld.feedbackmodeld"

MODEL_PKL_PATHS = {
  'melspec': Path(__file__).parent / "models/melspectrogram_tinygrad.pkl",
  'embedding': Path(__file__).parent / "models/embedding_model_tinygrad.pkl",
  'wakeword': Path(__file__).parent / "models/hey_comma_tinygrad.pkl",
}


def fast_numpy(x: Tensor):  # massive speedup when compiling with LLVM when used instead of .numpy()
  return x.cast(x.dtype.base).contiguous().realize().uop.base.buffer.numpy().reshape(x.shape)  # type: ignore[union-attr]


class ModelState:
  def __init__(self):
    self.raw_audio_buffer = np.array([], dtype=np.int16)
    self.melspec_buffer = np.ones((76, 32), dtype=np.float32)
    self.feature_buffer = np.zeros((16, 96), dtype=np.float32)
    self.pm = PubMaster(['feedbackState'])

    cloudlog.warning("Loading wake word models...")
    try:
      with open(MODEL_PKL_PATHS['melspec'], "rb") as f:
        self.melspec_model = pickle.load(f)
      with open(MODEL_PKL_PATHS['embedding'], "rb") as f:
        self.embedding_model = pickle.load(f)
      with open(MODEL_PKL_PATHS['wakeword'], "rb") as f:
        self.wakeword_model = pickle.load(f)
      cloudlog.warning("Wake word models loaded.")
    except Exception as e:
      cloudlog.error(f"Failed to load wake word models: {e}")
      self.melspec_model = None
      self.embedding_model = None
      self.wakeword_model = None

  def run(self, audio_chunk: np.ndarray, audio_eof: int):
    if not all([self.melspec_model, self.embedding_model, self.wakeword_model]):
      cloudlog.error("models not loaded, feedbackmodeld cannot run")
      return

    melspec_time = 0.0
    embedding_time = 0.0
    wakeword_time = 0.0
    wakeword_prob = 0.0

    self.raw_audio_buffer = np.concatenate([self.raw_audio_buffer, audio_chunk])

    while len(self.raw_audio_buffer) >= BUFFER_SIZE:
      processing_chunk = self.raw_audio_buffer[:BUFFER_SIZE]

      # Melspectrogram processing
      t1 = time.perf_counter()
      audio_tensor = Tensor(processing_chunk[None, :], dtype=dtypes.float32, device='NPY')
      new_melspec = fast_numpy(self.melspec_model(input=audio_tensor)).squeeze()
      new_melspec = (new_melspec / 10) + 2  # done in openWakeWord to better match Google's implementation
      t2 = time.perf_counter()
      melspec_time = t2 - t1

      self.melspec_buffer = np.vstack([self.melspec_buffer, new_melspec])[-76:]

      # Embedding processing
      t1 = time.perf_counter()
      window_tensor = Tensor(self.melspec_buffer[None, :, :, None], dtype=dtypes.float32, device='NPY')
      new_embedding = fast_numpy(self.embedding_model(input_1=window_tensor)).squeeze()
      t2 = time.perf_counter()
      embedding_time = t2 - t1

      self.feature_buffer = np.vstack([self.feature_buffer, new_embedding])[-16:]

      # Wakeword prediction
      t1 = time.perf_counter()
      feature_tensor = Tensor(self.feature_buffer[None, :], dtype=dtypes.float32, device='NPY')
      wakeword_prob = fast_numpy(self.wakeword_model(input=feature_tensor)).squeeze()
      t2 = time.perf_counter()
      wakeword_time = t2 - t1

      self.raw_audio_buffer = self.raw_audio_buffer[CHUNK_SIZE:]

      msg = messaging.new_message('feedbackState', valid=True)
      fs = msg.feedbackState

      fs.timestampEof = audio_eof
      fs.melspecExecutionTime = float(melspec_time)
      fs.embeddingExecutionTime = float(embedding_time)
      fs.wakewordExecutionTime = float(wakeword_time)
      fs.wakewordProb = float(wakeword_prob)

      self.pm.send('feedbackState', msg)


def main():
  config_realtime_process(6, 5)

  model = ModelState()

  sm = SubMaster(['rawAudioData'])

  while True:
    sm.update()

    if sm.updated['rawAudioData']:
      msg = sm['rawAudioData']
      audio_data = np.frombuffer(msg.data, dtype=np.int16)
      sample_rate = msg.sampleRate
      audio_eof = sm.logMonoTime['rawAudioData']

      if sample_rate != SAMPLE_RATE:
        cloudlog.error(f"Sample rate mismatch: expected wakeword sample rate {SAMPLE_RATE}, got {sample_rate}")

      model.run(audio_data, audio_eof)


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    cloudlog.warning("got SIGINT")
