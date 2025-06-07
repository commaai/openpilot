import unittest
import pathlib
from examples.whisper import init_whisper, load_file_waveform, transcribe_file, transcribe_waveform
from tinygrad.helpers import CI, fetch, Context
from tinygrad import Device, dtypes
from tinygrad.device import is_dtype_supported

# Audio generated with the command on MacOS:
# say "Could you please let me out of the box?" --file-format=WAVE  --data-format=LEUI8@16000 -o test
# We use the WAVE type because it's easier to decode in CI test environments
TEST_FILE_1 = str(pathlib.Path(__file__).parent / "whisper/test.wav")
TRANSCRIPTION_1 = "Could you please let me out of the box?"
TEST_FILE_2 = str(pathlib.Path(__file__).parent / "whisper/test2.wav")
TRANSCRIPTION_2 = "a slightly longer audio file so that we can test batch transcriptions of varying length."
# TODO this file will possibly not survive long. find another 1-2 minute sound file online to transcribe
TEST_FILE_3_URL = 'https://homepage.ntu.edu.tw/~karchung/miniconversations/mc45.mp3'
TRANSCRIPTION_3 = "Just lie back and relax. Is the level of pressure about right? Yes, it's fine, and I'd like conditioner please. Sure. I'm going to start the second lathering now. Would you like some Q-tips? How'd you like it cut? I'd like my bangs and the back trimmed, and I'd like the rest thinned out a bit and layered. Where would you like the part? On the left, right about here. Here, have a look. What do you think? It's fine. Here's a thousand anti-dollars. It's 30-ant extra for the rants. Here's your change and receipt. Thank you, and please come again. So how do you like it? It could have been worse, but you'll notice that I didn't ask her for her card. Hmm, yeah. Maybe you can try that place over there next time."   # noqa: E501

@unittest.skipIf(CI and Device.DEFAULT in ["CPU"], "slow")
@unittest.skipUnless(is_dtype_supported(dtypes.float16), "need float16 support")
class TestWhisper(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    model, enc = init_whisper("tiny.en", batch_size=2)
    cls.model = model
    cls.enc = enc
    # TODO: whisper has out of bounds access somewhere
    cls.context = Context(IGNORE_OOB=1)
    cls.context.__enter__()

  @classmethod
  def tearDownClass(cls):
    cls.context.__exit__(None, None, None)
    del cls.model
    del cls.enc

  def test_transcribe_file1(self):
    self.assertEqual(transcribe_file(self.model, self.enc, TEST_FILE_1),  TRANSCRIPTION_1)

  @unittest.expectedFailure  # Test for out of bounds access
  @unittest.skip("TODO: flaky")
  def test_transcribe_file1_OOB(self):
    with Context(IGNORE_OOB=0):
      self.assertEqual(transcribe_file(self.model, self.enc, TEST_FILE_1),  TRANSCRIPTION_1)

  @unittest.skipIf(CI or Device.DEFAULT == "LLVM", "too many tests for CI")
  def test_transcribe_file2(self):
    self.assertEqual(transcribe_file(self.model, self.enc, TEST_FILE_2),  TRANSCRIPTION_2)

  @unittest.skipIf(CI or Device.DEFAULT == "LLVM", "too many tests for CI")
  def test_transcribe_batch12(self):
    waveforms = [load_file_waveform(TEST_FILE_1), load_file_waveform(TEST_FILE_2)]
    transcriptions = transcribe_waveform(self.model, self.enc, waveforms)
    self.assertEqual(2, len(transcriptions))
    self.assertEqual(TRANSCRIPTION_1,  transcriptions[0])
    self.assertEqual(TRANSCRIPTION_2,  transcriptions[1])

  def test_transcribe_batch21(self):
    waveforms = [load_file_waveform(TEST_FILE_2), load_file_waveform(TEST_FILE_1)]
    transcriptions = transcribe_waveform(self.model, self.enc, waveforms)
    self.assertEqual(2, len(transcriptions))
    self.assertEqual(TRANSCRIPTION_2,  transcriptions[0])
    self.assertEqual(TRANSCRIPTION_1,  transcriptions[1])

  @unittest.skipIf(CI or Device.DEFAULT == "LLVM", "too long for CI")
  def test_transcribe_long(self):
    waveform = [load_file_waveform(fetch(TEST_FILE_3_URL))]
    transcription = transcribe_waveform(self.model, self.enc, waveform)
    self.assertEqual(TRANSCRIPTION_3, transcription)

  @unittest.skipIf(CI or Device.DEFAULT == "LLVM", "too long for CI")
  def test_transcribe_long_no_batch(self):
    waveforms = [load_file_waveform(fetch(TEST_FILE_3_URL)), load_file_waveform(TEST_FILE_1)]

    trancriptions = transcribe_waveform(self.model, self.enc, waveforms)
    self.assertEqual(2, len(trancriptions))
    self.assertEqual(TRANSCRIPTION_3, trancriptions[0])
    self.assertEqual(TRANSCRIPTION_1, trancriptions[1])

if __name__ == '__main__':
  unittest.main()
