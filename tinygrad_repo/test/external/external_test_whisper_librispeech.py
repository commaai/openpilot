import unittest
import torch
import tqdm
import torchaudio
import pathlib
import jiwer
import os
import numpy as np
from whisper.normalizers import EnglishTextNormalizer
from examples.whisper import init_whisper, transcribe_waveform

class TestWhisperLibriSpeech(unittest.TestCase):
  # reference WERs determined by running https://github.com/openai/whisper/blob/main/notebooks/LibriSpeech.ipynb
  # the values should be consistent with the paper D.1.1 https://cdn.openai.com/papers/whisper.pdf#page=22
  # tinygrad WERs do not perfectly match due to what seem to be precision differences vs torch
  def test_en_tiny(self):
    run_evaluation("tiny.en", 0.056629001883239174, 0.05655609406528749)

  def test_tiny(self):
    run_evaluation("tiny", 0.0771121409407306, 0.07558413638335187)

  def test_en_base(self):
    run_evaluation("base.en", 0.041412520064205455, 0.04271408904897505)

  def test_en_small(self):
    run_evaluation("small.en", 0.03369011117172363, 0.030531615969223228)

def run_evaluation(model_name, tinygrad_expected_wer, reference_wer):
  dataset = LibriSpeech()
  batch_size=16
  loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

  model, enc = init_whisper(model_name, batch_size=batch_size)

  hypotheses = []
  references = []

  for audio, texts in tqdm.tqdm(loader):
    transcriptions = transcribe_waveform(model, enc, audio.numpy(), truncate=True)
    hypotheses.extend(transcriptions)
    references.extend(texts)

  normalizer = EnglishTextNormalizer()
  normalized_hypotheses = [normalizer(text) for text in hypotheses]
  normalized_references = [normalizer(text) for text in references]
  wer = jiwer.wer(normalized_hypotheses, normalized_references)

  np.testing.assert_almost_equal(wer, tinygrad_expected_wer)
  print(f'tinygrad WER {wer} vs reference WER {reference_wer}')
  del model, enc

class LibriSpeech(torch.utils.data.Dataset):
  def __init__(self):
    folder = pathlib.Path(__file__).parent.parent.parent / "extra" / "datasets" / "librispeech"
    if not os.path.exists(folder):
      os.makedirs(folder)

    self.dataset = torchaudio.datasets.LIBRISPEECH(
      root=folder,
      url="test-clean",
      download=True,
    )

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, item):
    audio, sample_rate, text, _, _, _ = self.dataset[item]
    assert sample_rate == 16000
    return pad_or_trim_tensor(audio[0]), text

def pad_or_trim_tensor(tensor, target_len=480000):
  curr_len = len(tensor)
  if curr_len == target_len:
    return tensor
  elif curr_len < target_len:
    return torch.cat((tensor, torch.zeros(target_len - curr_len)))
  else:
    return tensor[:target_len]


if __name__ == '__main__':
  unittest.main()
