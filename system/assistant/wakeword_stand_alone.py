#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from openpilot.common.realtime import Ratekeeper
from openpilot.common.retry import retry
from openpilot.system.assistant.openwakeword import Model
from openpilot.system.assistant.openwakeword.utils import download_models
from openpilot.common.params import Params


RATE = 12.5
SAMPLE_RATE = 16000
SAMPLE_BUFFER = 1280 # (approx 100ms)


class WakeWordListener:
  def __init__(self, model):
    self.raw_sample = np.zeros(SAMPLE_BUFFER)
    self.sample_idx = 0
    self.inference_idx = 0
    model_path = Path(__file__).parent / f'models/{model}.onnx'
    melspec_model_path = Path(__file__).parent / 'models/melspectrogram.onnx'
    embedding_model_path = Path(__file__).parent / 'models/embedding_model.onnx'
    self.owwModel = Model(wakeword_models=[model_path], melspec_model_path=melspec_model_path, embedding_model_path=embedding_model_path)
    self.n_models = len(self.owwModel.models.keys())
    self.params = Params()
    
    self.detected = False
    self.curr_score = 0

  def update(self):
    print(self.curr_score)
    if self.detected:
      print("wake word detected")
    
    self.rk.keep_time()
    return self.detected

  def callback(self, indata, frames, time, status):
    if status:
      print(f"Stream error: {status}")
      return
    self.raw_sample[:] = indata[:, 0]
    self.sample_idx+=1
    self.owwModel.predict(self.raw_sample)
    for mdl in self.owwModel.prediction_buffer.keys():
        scores = list(self.owwModel.prediction_buffer[mdl])
        self.detected = scores[-1] >= 0.5
        self.curr_score = "{:.20f}".format(abs(scores[-1]))

  @retry(attempts=7, delay=3)
  def get_stream(self, sd):
    # reload sounddevice to reinitialize portaudio
    sd._terminate()
    sd._initialize()
    return sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=self.callback, blocksize=SAMPLE_BUFFER, dtype="int16")

  def wake_word_listener_thread(self):
    self.params.put_bool("WakeWordDetected", False)
    # sounddevice must be imported after forking processes
    import sounddevice as sd
    with self.get_stream(sd) as stream:
      print(f"micd stream started: {stream.samplerate=} {stream.channels=} {stream.dtype=} {stream.device=}, {stream.blocksize=}")
      self.rk = Ratekeeper(RATE)
      detected = False
      while True:
        detected = self.update() # just get it one time for now then exit
    #close sd so google-speech can open
    self.params.put_bool("WakeWordDetected", True) 

def main():
  model = "alexa_v0.1"
  download_models([model], Path(__file__).parent / 'models')
  wwl = WakeWordListener(model)
  wwl.wake_word_listener_thread()

if __name__ == "__main__":
  main()
