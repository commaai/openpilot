import json
import pathlib
import numpy as np
import librosa
import soundfile

"""
The dataset has to be downloaded manually from https://www.openslr.org/12/ and put in `extra/datasets/librispeech`.
For mlperf validation the dev-clean dataset is used.

Then all the flacs have to be converted to wav using something like:
```fish
for file in $(find * | grep flac); do ffmpeg -i $file -ar 16k "$(dirname $file)/$(basename $file .flac).wav"; done
```

Then this [file](https://github.com/mlcommons/inference/blob/master/speech_recognition/rnnt/dev-clean-wav.json) has to also be put in `extra/datasets/librispeech`.
"""
BASEDIR = pathlib.Path(__file__).parent / "librispeech"
with open(BASEDIR / "dev-clean-wav.json") as f:
  ci = json.load(f)

FILTER_BANK = np.expand_dims(librosa.filters.mel(sr=16000, n_fft=512, n_mels=80, fmin=0, fmax=8000), 0)
WINDOW = librosa.filters.get_window("hann", 320)

def feature_extract(x, x_lens):
  x_lens = np.ceil((x_lens / 160) / 3).astype(np.int32)

  # pre-emphasis
  x = np.concatenate((np.expand_dims(x[:, 0], 1), x[:, 1:] - 0.97 * x[:, :-1]), axis=1)

  # stft
  x = librosa.stft(x, n_fft=512, window=WINDOW, hop_length=160, win_length=320, center=True, pad_mode="reflect")
  x = np.stack((x.real, x.imag), axis=-1)

  # power spectrum
  x = (x**2).sum(-1)

  # mel filter bank
  x = np.matmul(FILTER_BANK, x)

  # log
  x = np.log(x + 1e-20)

  # feature splice
  seq = [x]
  for i in range(1, 3):
    tmp = np.zeros_like(x)
    tmp[:, :, :-i] = x[:, :, i:]
    seq.append(tmp)
  features = np.concatenate(seq, axis=1)[:, :, ::3]

  # normalize
  features_mean = np.zeros((features.shape[0], features.shape[1]), dtype=np.float32)
  features_std = np.zeros((features.shape[0], features.shape[1]), dtype=np.float32)
  for i in range(features.shape[0]):
    features_mean[i, :] = features[i, :, :x_lens[i]].mean(axis=1)
    features_std[i, :] = features[i, :, :x_lens[i]].std(axis=1, ddof=1)
  features_std += 1e-5
  features = (features - np.expand_dims(features_mean, 2)) / np.expand_dims(features_std, 2)

  return features.transpose(2, 0, 1), x_lens.astype(np.float32)

def load_wav(file):
  sample = soundfile.read(file)[0].astype(np.float32)
  return sample, sample.shape[0]

def iterate(bs=1, start=0):
  print(f"there are {len(ci)} samples in the dataset")
  for i in range(start, len(ci), bs):
    samples, sample_lens = zip(*[load_wav(BASEDIR / v["files"][0]["fname"]) for v in ci[i : i + bs]])
    samples = list(samples)
    # pad to same length
    max_len = max(sample_lens)
    for j in range(len(samples)):
      samples[j] = np.pad(samples[j], (0, max_len - sample_lens[j]), "constant")
    samples, sample_lens = np.array(samples), np.array(sample_lens)

    yield feature_extract(samples, sample_lens), np.array([v["transcript"] for v in ci[i : i + bs]])

if __name__ == "__main__":
  X, Y = next(iterate())
  print(X[0].shape, Y.shape)
