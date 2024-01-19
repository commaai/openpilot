import unittest
import numpy as np
import wave
import openpilot.system.micd as micd
from openpilot.system.assistant.wakewordd import WakeWordListener, download_models, MODEL_DIR, PHRASE_MODEL_NAME
from openpilot.common.params import Params
from pathlib import Path
import time

SOUND_FILE_PATH = Path(__file__).parent / 'sounds'
EASY = f'{SOUND_FILE_PATH}/alexa_easy.wav'
MEDIUM = f'{SOUND_FILE_PATH}/alexa_medium.wav'
HARD = f'{SOUND_FILE_PATH}/alexa_hard.wav'
CONVERSATION = f'{SOUND_FILE_PATH}/random_conversation.wav'



class TestMicd(unittest.TestCase):
    
  def setUp(self):
    # Download models if necessary
    download_models([PHRASE_MODEL_NAME], MODEL_DIR)
    self.wwl = WakeWordListener()
    self.params = Params()

  def test_callback_with_wav(self):
    # Create a Mic instance
    mic_instance = micd.Mic()
    sounds_and_detects = {EASY: True, MEDIUM: True, HARD: True, CONVERSATION: False}
    for file,should_detect in sounds_and_detects.items():
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
