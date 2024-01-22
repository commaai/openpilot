# Copyright 2022 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Union, List, Callable, Deque
from collections import deque
import pathlib
import os
import numpy as np
import requests
from tqdm import tqdm
import openpilot.system.assistant.openwakeword as openwakeword
import onnxruntime as ort


# Base class for computing audio features using Google's speech_embedding
# model (https://tfhub.dev/google/speech_embedding/1)
class AudioFeatures():
  def __init__(self,
              melspec_model_path: str = pathlib.Path(__file__).parent.parent / "models/melspectrogram.onnx",
              embedding_model_path: str = pathlib.Path(__file__).parent.parent / "models/embedding_model.onnx",
              sr: int = 16000,
              ncpu: int = 1,
              device: str = 'cpu'
              ):
    # Initialize ONNX options
    sessionOptions = ort.SessionOptions()
    sessionOptions.inter_op_num_threads = ncpu
    sessionOptions.intra_op_num_threads = ncpu
    # Melspectrogram model
    self.melspec_model = ort.InferenceSession(melspec_model_path, sess_options=sessionOptions,
                                                providers=["CUDAExecutionProvider"] if device == "gpu" else ["CPUExecutionProvider"])
    self.onnx_execution_provider = self.melspec_model.get_providers()[0]
    self.melspec_model_predict = lambda x: self.melspec_model.run(None, {'input': x})
    # Audio embedding model
    self.embedding_model = ort.InferenceSession(embedding_model_path, sess_options=sessionOptions,
                                                providers=["CUDAExecutionProvider"] if device == "gpu"
                                                else ["CPUExecutionProvider"])
    self.embedding_model_predict = lambda x: self.embedding_model.run(None, {'input_1': x})[0].squeeze()
    # Create databuffers
    self.raw_data_buffer: Deque = deque(maxlen=sr*10)
    self.melspectrogram_buffer = np.ones((76, 32))  # n_frames x num_features
    self.melspectrogram_max_len = 10*97  # 97 is the number of frames in 1 second of 16hz audio
    self.accumulated_samples = 0  # the samples added to the buffer since the audio preprocessor was last called
    self.raw_data_remainder = np.empty(0)
    self.feature_buffer = self._get_embeddings(np.random.randint(-1000, 1000, 16000*4).astype(np.int16))
    self.feature_buffer_max_len = 120  # ~10 seconds of feature buffer history

  def _get_melspectrogram(self, x: Union[np.ndarray, List], melspec_transform: Callable = lambda x: x/10 + 2):
    x = np.asarray(x, dtype=np.float32)  # Convert to numpy array of type float32 directly
    if x.ndim == 1:
      x = x[np.newaxis, :]  # Add new axis for single sample
    outputs = self.melspec_model_predict(x)
    spec = np.squeeze(outputs[0])

    return melspec_transform(spec)

  def _get_embeddings(self, x: np.ndarray, window_size: int = 76, step_size: int = 8, **kwargs):
    spec = self._get_melspectrogram(x, **kwargs)

    # Check if input is too short
    if spec.shape[0] < window_size:
      raise ValueError("Input is too short for the specified window size.")

    # Collect windows
    windows = [spec[i:i + window_size] for i in range(0, spec.shape[0] - window_size + 1, 8)
            if i + window_size <= spec.shape[0]]

    # Convert to batch format
    batch = np.array(windows)[..., np.newaxis].astype(np.float32)
    embedding = self.embedding_model_predict(batch)
    return embedding

  def _streaming_melspectrogram(self, n_samples):
    if len(self.raw_data_buffer) < 400:
      raise ValueError("The number of input frames must be at least 400 samples @ 16khz (25 ms)!")

    self.melspectrogram_buffer = np.vstack(
      (self.melspectrogram_buffer, self._get_melspectrogram(list(self.raw_data_buffer)[-n_samples-160*3:]))
    )

    if self.melspectrogram_buffer.shape[0] > self.melspectrogram_max_len:
      self.melspectrogram_buffer = self.melspectrogram_buffer[-self.melspectrogram_max_len:, :]

  def _buffer_raw_data(self, x):

    self.raw_data_buffer.extend(x.tolist() if isinstance(x, np.ndarray) else x)

  def _streaming_features(self, x):
    # Add raw audio data to buffer, temporarily storing extra frames if not an even number of 80 ms chunks
    processed_samples = 0

    if self.raw_data_remainder.shape[0] != 0:
      x = np.concatenate((self.raw_data_remainder, x))
      self.raw_data_remainder = np.empty(0)

    if self.accumulated_samples + x.shape[0] >= 1280:
      remainder = (self.accumulated_samples + x.shape[0]) % 1280
      if remainder != 0:
        x_even_chunks = x[0:-remainder]
        self._buffer_raw_data(x_even_chunks)
        self.accumulated_samples += len(x_even_chunks)
        self.raw_data_remainder = x[-remainder:]
      elif remainder == 0:
        self._buffer_raw_data(x)
        self.accumulated_samples += x.shape[0]
        self.raw_data_remainder = np.empty(0)
    else:
      self.accumulated_samples += x.shape[0]
      self._buffer_raw_data(x)

    # Only calculate melspectrogram once minimum samples are accumulated
    if self.accumulated_samples >= 1280 and self.accumulated_samples % 1280 == 0:
      self._streaming_melspectrogram(self.accumulated_samples)
      # Calculate new audio embeddings/features based on update melspectrograms
      for i in np.arange(self.accumulated_samples//1280-1, -1, -1):
        ndx = -8*i
        ndx = ndx if ndx != 0 else len(self.melspectrogram_buffer)
        x = self.melspectrogram_buffer[-76 + ndx:ndx].astype(np.float32)[None, :, :, None]
        if x.shape[1] == 76:
          self.feature_buffer = np.vstack((self.feature_buffer,
                                          self.embedding_model_predict(x)))
        # Reset raw data buffer counter
        processed_samples = self.accumulated_samples
        self.accumulated_samples = 0

    if self.feature_buffer.shape[0] > self.feature_buffer_max_len:
      self.feature_buffer = self.feature_buffer[-self.feature_buffer_max_len:, :]

    return processed_samples if processed_samples != 0 else self.accumulated_samples

  def get_features(self, n_feature_frames: int = 16, start_ndx: int = -1):
    if start_ndx != -1:
      end_ndx = start_ndx + int(n_feature_frames) \
        if start_ndx + n_feature_frames != 0 else len(self.feature_buffer)
      return self.feature_buffer[start_ndx:end_ndx, :][None, ].astype(np.float32)
    else:
      return self.feature_buffer[int(-1*n_feature_frames):, :][None, ].astype(np.float32)

  def __call__(self, x):
    return self._streaming_features(x)

def download_file(url, target_directory, file_size=None):
  local_filename = url.split('/')[-1]
  with requests.get(url, stream=True) as r:
    if file_size is not None:
      progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True, desc=f"{local_filename}")
    else:
      total_size = int(r.headers.get('content-length', 0))
      progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"{local_filename}")
    with open(os.path.join(target_directory, local_filename), 'wb') as f:
      for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)
        progress_bar.update(len(chunk))
  progress_bar.close()


# Function to download models from GitHub release assets
def download_models(model_names: List[str] = ["",],
  target_directory: str = os.path.join(pathlib.Path(__file__).parent.resolve(), "resources", "models")):
  if not isinstance(model_names, list):
    raise ValueError("The model_names argument must be a list of strings")
  # Always download melspectrogram and embedding models, if they don't already exist
  if not os.path.exists(target_directory):
    os.makedirs(target_directory)
  for feature_model in openwakeword.FEATURE_MODELS.values():
    if not os.path.exists(os.path.join(target_directory, feature_model["download_url"].split("/")[-1])):
      download_file(feature_model["download_url"], target_directory)
      download_file(feature_model["download_url"].replace(".tflite", ".onnx"), target_directory)
  # Always download VAD models, if they don't already exist
  for vad_model in openwakeword.VAD_MODELS.values():
    if not os.path.exists(os.path.join(target_directory, vad_model["download_url"].split("/")[-1])):
      download_file(vad_model["download_url"], target_directory)
  # Get all model urls
  official_model_urls = [i["download_url"] for i in openwakeword.MODELS.values()]
  official_model_names = [i["download_url"].split("/")[-1] for i in openwakeword.MODELS.values()]
  if model_names != []:
    for model_name in model_names:
      url = [i for i, j in zip(official_model_urls, official_model_names, strict=False) if model_name in j]
      if url != []:
        if not os.path.exists(os.path.join(target_directory, url[0].split("/")[-1])):
          download_file(url[0], target_directory)
          download_file(url[0].replace(".tflite", ".onnx"), target_directory)
  else:
    for official_model_url in official_model_urls:
      if not os.path.exists(os.path.join(target_directory, official_model_url.split("/")[-1])):
        download_file(official_model_url, target_directory)
        download_file(official_model_url.replace(".tflite", ".onnx"), target_directory)

def re_arg(kwarg_map):
  def decorator(func):
    def wrapped(*args, **kwargs):
      new_kwargs = {}
      for k, v in kwargs.items():
        new_kwargs[kwarg_map.get(k, k)] = v
      return func(*args, **new_kwargs)
    return wrapped
  return decorator
