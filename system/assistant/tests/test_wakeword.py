import unittest
import numpy as np
import wave
import openpilot.system.micd as micd
from openpilot.system.assistant.wakewordd import WakeWordListener as WWL, download_models
from openpilot.common.params import Params
from pathlib import Path

SOUND_FILE_PATH = Path(__file__).parent / 'sounds'
EASY = f'{SOUND_FILE_PATH}/alexa_easy.wav'
MEDIUM = f'{SOUND_FILE_PATH}/alexa_medium.wav'
HARD = f'{SOUND_FILE_PATH}/alexa_hard.wav'
CONVERSATION = f'{SOUND_FILE_PATH}/random_conversation.wav'
sounds_and_detects = {EASY: True, MEDIUM: True, HARD: True, CONVERSATION: False}

class WakeWordListener(unittest.TestCase):

  def setUp(self):
    # Download models if necessary
    download_models([WWL.PHRASE_MODEL_NAME], "./models")
    self.wwl = WWL(model_path=WWL.PHRASE_MODEL_PATH,threshhold=0.5)
    self.params = Params()

  def test_wake_word(self):
    # Create a Mic instance
    mic_instance = micd.Mic()
    for file,should_detect in sounds_and_detects.items():
      self.params.put_bool("WakeWordDetected", False) # Reset
      print(f'testing {file}, {should_detect=}')
      with wave.open(file, 'rb') as wf:
        # Ensure the file is mono and has the correct sample rate
        assert wf.getnchannels() == 1
        assert wf.getframerate() == micd.SAMPLE_RATE
        detected = False
        while True:
          # Read a chunk of data
          frames = wf.readframes(micd.SAMPLE_BUFFER)
          if len(frames) == 0:
            break
          # Convert frames to numpy array
          indata = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768
          indata = indata.reshape(-1, 1)

          # Pass the chunk to the callback
          mic_instance.callback(indata, len(indata), None, None)
          mic_instance.update()
          self.wwl.wake_word_runner()
          if self.params.get_bool("WakeWordDetected"):
            detected = True
        self.assertEqual(detected,should_detect, f'{detected=} {should_detect=} for sound {file}')

if __name__ == '__main__':
  unittest.main()
