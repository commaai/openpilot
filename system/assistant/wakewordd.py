#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from openpilot.system.assistant.openwakeword import Model
from openpilot.system.assistant.openwakeword.utils import download_models
from openpilot.common.params import Params
from cereal import messaging
from openpilot.system.micd import SAMPLE_BUFFER, SAMPLE_RATE


RATE = 12.5
class WakeWordListener:
  def __init__(self, model):
    model_path = Path(__file__).parent / f'models/{model}.onnx'
    melspec_model_path = Path(__file__).parent / 'models/melspectrogram.onnx'
    embedding_model_path = Path(__file__).parent / 'models/embedding_model.onnx'
    self.owwModel = Model(wakeword_models=[model_path], melspec_model_path=melspec_model_path, embedding_model_path=embedding_model_path, sr=SAMPLE_RATE)

    self.sm = messaging.SubMaster(['microphoneRaw'])
    self.params = Params()

    self.frame_index = 0
    self.frame_index_last = 0


  def update(self):
    self.frame_index = self.sm['microphoneRaw'].frameIndex
    #print(f'{self.frame_index_last=}, {self.frame_index=}')
    if not (self.frame_index_last == self.frame_index or
            self.frame_index - self.frame_index_last == SAMPLE_BUFFER):
      print(f'skipped {(self.frame_index - self.frame_index_last)//SAMPLE_BUFFER-1} sample(s)')
    if self.frame_index_last == self.frame_index:
      print("got the same frame")
    self.frame_index_last = self.frame_index
    sample = np.frombuffer(self.sm['microphoneRaw'].rawSample, dtype=np.int16)
    self.owwModel.predict(sample)
    for mdl in self.owwModel.prediction_buffer.keys():
        scores = list(self.owwModel.prediction_buffer[mdl])
        detected = scores[-1] >= 0.5
        #curr_score = "{:.20f}".format(abs(scores[-1]))
    #print(curr_score)
    if detected:
      print("wake word detected")
    self.params.put_bool("WakeWordDetected", detected)

  def wake_word_listener_thread(self):
    while True:
        self.sm.update(0)
        if self.sm.updated['microphoneRaw']:
            print(self.sm['microphoneRaw'].frameIndex)
            self.update()

def main():
  model = "alexa_v0.1"
  download_models([model], Path(__file__).parent / 'models')
  wwl = WakeWordListener(model)
  wwl.wake_word_listener_thread()

if __name__ == "__main__":
  main()
