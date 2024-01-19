#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from openpilot.system.assistant.openwakeword import Model
from openpilot.system.assistant.openwakeword.utils import download_models
from openpilot.common.params import Params
from cereal import messaging
from openpilot.system.micd import SAMPLE_BUFFER, SAMPLE_RATE


RATE = 12.5
PHRASE_MODEL_NAME = "alexa_v0.1"
MODEL_DIR = Path(__file__).parent / 'models'
PHRASE_MODEL_PATH = f'{MODEL_DIR}/{PHRASE_MODEL_NAME}.onnx'
MEL_MODEL_PATH = f'{MODEL_DIR}/melspectrogram.onnx'
EMB_MODEL_PATH = f'{MODEL_DIR}/embedding_model.onnx'
THRESHOLD = .5


class WakeWordListener:
  def __init__(self, model_path=PHRASE_MODEL_PATH):
    self.owwModel = Model(wakeword_models=[model_path], melspec_model_path=MEL_MODEL_PATH, embedding_model_path=EMB_MODEL_PATH, sr=SAMPLE_RATE)
    self.sm = messaging.SubMaster(['microphoneRaw'])
    self.params = Params()

    self.model_name = model_path.split("/")[-1].split(".onnx")[0]
    self.frame_index = 0
    self.frame_index_last = 0
    self.detected_last = False

  def update(self):
    self.frame_index = self.sm['microphoneRaw'].frameIndex
    if not (self.frame_index_last == self.frame_index or
            self.frame_index - self.frame_index_last == SAMPLE_BUFFER):
      print(f'skipped {(self.frame_index - self.frame_index_last)//SAMPLE_BUFFER-1} sample(s)') # TODO: Stop it from skipping
    if self.frame_index_last == self.frame_index:
      print("got the same frame")
      return
    self.frame_index_last = self.frame_index
    sample = np.frombuffer(self.sm['microphoneRaw'].rawSample, dtype=np.int16)
    prediction_score = self.owwModel.predict(sample)
    detected = prediction_score[self.model_name] >= THRESHOLD
    if detected != self.detected_last: # Catch the edges only
      print("wake word detected" if detected else "wake word not detected")
      self.params.put_bool("WakeWordDetected", detected)
    self.detected_last = detected

  def wake_word_runner(self):
    self.sm.update(0)
    if self.sm.updated['microphoneRaw']:
        self.update()

def main():
  download_models([PHRASE_MODEL_NAME], MODEL_DIR)
  wwl = WakeWordListener()
  while True:
    wwl.wake_word_runner()

if __name__ == "__main__":
  main()
