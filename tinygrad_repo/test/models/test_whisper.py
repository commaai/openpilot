import unittest
import pathlib
from examples.whisper import init_whisper, load_file_waveform, transcribe_file, transcribe_waveform
from examples.audio_helpers import mel
import examples.mlperf.metrics as metrics
from tinygrad.helpers import fetch
from test.helpers import slow
from tinygrad import Tensor, Device, dtypes
from tinygrad.device import is_dtype_supported
import numpy as np

# Audio generated with the command on MacOS:
# say "Could you please let me out of the box?" --file-format=WAVE  --data-format=LEUI8@16000 -o test
# We use the WAVE type because it's easier to decode in CI test environments
TEST_FILE_1 = str(pathlib.Path(__file__).parent / "whisper/test.wav")
TRANSCRIPTION_1 = "Could you please let me out of the box?"
TEST_FILE_2 = str(pathlib.Path(__file__).parent / "whisper/test2.wav")
TRANSCRIPTION_2 = "a slightly longer audio file so that we can test batch transcriptions of varying length."
# TODO this file will possibly not survive long. find another 1-2 minute sound file online to transcribe
TEST_FILE_3_URL = 'https://homepage.ntu.edu.tw/~karchung/miniconversations/mc45.mp3'
TRANSCRIPTION_3 = """Just lie back and relax.
Is the level of pressure about right?
Yes, it's fine. And I'd like conditioner, please.
Sure. I'm going to start the second lathering now.
Would you like some Q-tips?
How'd you like it cut?
I'd like my bangs and the back trimmed,
and I'd like the rest thinned out a bit and layered.
Where would you like the part?
On the left, right about here.
Here, have a look. What do you think?
It's fine. Here's thousand NT dollars.
It's 30 NT extra for the rinse. Here's your change and receipt.
Thank you, and please come again!
So, how do you like it?
It could have been worse. But you'll notice that I didn't ask her for her card.
Hmm, yeah.
Mm, maybe you can try that place over there next time."""

TRANSCRIPTION_3_ALT = "Just lie back and relax. Is the level of pressure about right? Yes, it's fine. And I'd like conditioner please. Sure. I'm going to start the second lathering now. Would you like some Q-tips? How'd you like it cut? I'd like my bangs on the back trimmed, and I'd like the rest to stand out a bit and layered. Where would you like the part? On the left, right about here. Here. Have a look. What do you think? It's fine. Here's a thousand and eighty dollars. It's thirty and t extra for the rants. Here's your change and receipt. Thank you, and please come again. So how do you like it? It could have been worse, but you'll notice that I didn't ask her for her card. Hmm, yeah. Maybe you can try that place over there next time." #noqa: E501
# NOTE: same as TRANSCRIPTION_3 but with minor changes that should only amount to ~0.079 WER difference (see test_wer_same)
# 'and'     --> 'on'
# 'thinned' --> 'to stand'
# 'nt'      --> 'and eighty'
# '30 nt'   --> 'thirty and t'
# 'rinse'   --> 'rants'
# 'mm'      --> ''

def wer_helper(result: str, reference: str)->float:
  result = metrics.normalize_string(result)
  reference = metrics.normalize_string(reference)
  wer, _, _ = metrics.word_error_rate([result], [reference])
  return wer

@unittest.skipIf(Device.DEFAULT in ["CPU"], "slow")
@unittest.skipUnless(is_dtype_supported(dtypes.float16), "need float16 support")
# TODO: WEBGPU GPU dispatch dimensions limit
@unittest.skipIf(Device.DEFAULT == "WEBGPU", "WEBGPU GPU dispatch dimensions limit")
class TestWhisper(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    model, enc = init_whisper("tiny.en", batch_size=2)
    cls.model = model
    cls.enc = enc

  @classmethod
  def tearDownClass(cls):
    del cls.model
    del cls.enc

  def assertWER(self, actual: str, expected: str, threshold: float):
    __tracebackhide__ = True  # Hide traceback for py.test
    wer = wer_helper(actual, expected)
    if wer > threshold:
      err = f"WER={wer:.3f} > {threshold}"
      raise AssertionError(
        err
      )

  def test_transcribe_file1(self):
    self.assertEqual(transcribe_file(self.model, self.enc, TEST_FILE_1),  TRANSCRIPTION_1)

  @slow
  def test_transcribe_file2(self):
    self.assertEqual(transcribe_file(self.model, self.enc, TEST_FILE_2),  TRANSCRIPTION_2)

  @slow
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

  @unittest.skip("file 3 url is broken")
  @slow
  def test_transcribe_long(self):
    waveform = [load_file_waveform(fetch(TEST_FILE_3_URL))]
    transcription = transcribe_waveform(self.model, self.enc, waveform)
    self.assertWER(transcription, TRANSCRIPTION_3, 0.085)

  @unittest.skip("file 3 url is broken")
  @slow
  def test_transcribe_long_no_batch(self):
    waveforms = [load_file_waveform(fetch(TEST_FILE_3_URL)), load_file_waveform(TEST_FILE_1)]

    trancriptions = transcribe_waveform(self.model, self.enc, waveforms)
    self.assertEqual(2, len(trancriptions))
    self.assertWER(trancriptions[0], TRANSCRIPTION_3, 0.085)
    self.assertEqual(TRANSCRIPTION_1, trancriptions[1])

  def test_wer_same(self):
    reference = TRANSCRIPTION_3
    self.assertWER(TRANSCRIPTION_3_ALT, reference, 0.079)

  def test_wer_different(self):
    reference = TRANSCRIPTION_3
    self.assertWER("[no speech]", reference, 1.0)

  def test_wer_different_2(self):
    reference = TRANSCRIPTION_3
    self.assertWER("", reference, 1.0)

  def test_wer_different_3(self):
    reference = TRANSCRIPTION_3
    self.assertWER(reference[:len(reference)//2], reference, 0.524)

  def test_mel_filters(self):
    # reference = librosa.filters.mel(sr=16000, n_fft=16, n_mels=16)
    reference = Tensor([[-0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0021111054811626673, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.003133024089038372, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0017568661132827401, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0009823603322729468, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0007768510840833187, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0010490329004824162, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0011341988574713469, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.000231665835599415, 0.0006950111710466444, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.00040073052514344454, 0.0005822855746373534, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00033081238507293165, 0.0006097797304391861, 0.0]])
    np.testing.assert_allclose(mel(sr=16000, n_fft=16, n_mels=16, dtype=dtypes.float32).numpy(), reference.numpy(), atol=1e-6)

if __name__ == '__main__':
  unittest.main()
