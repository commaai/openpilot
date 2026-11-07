import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SR = 16000
WIN = 20480  # 1.28 s
NFFT, HOP, NMEL = 512, 160, 40


def _mel_matrix() -> np.ndarray:
  def hz2mel(f): return 2595 * np.log10(1 + f / 700)
  pts = 700 * (10**(np.linspace(hz2mel(60), hz2mel(7600), NMEL + 2) / 2595) - 1)
  bins = np.floor((NFFT + 1) * pts / SR).astype(int)
  m = np.zeros((NMEL, NFFT // 2 + 1), np.float32)
  for i in range(NMEL):
    a, b, c = bins[i], bins[i + 1], bins[i + 2]
    m[i, a:b] = (np.arange(a, b) - a) / max(b - a, 1)
    m[i, b:c] = (c - np.arange(b, c)) / max(c - b, 1)
  return m


MEL = _mel_matrix()
WINDOW = np.hanning(400).astype(np.float32)


def logmel(x: np.ndarray) -> np.ndarray:
  n = 1 + (len(x) - 400) // HOP
  fr = sliding_window_view(x, 400)[::HOP][:n] * WINDOW
  p = np.abs(np.fft.rfft(fr, NFFT)) ** 2
  lm = np.log(p @ MEL.T + 1e-9)
  # per window band normalization makes the detector gain invariant
  return (lm - lm.mean(0)).T.astype(np.float32)


def _conv(x, w, b, stride=1, groups=1):
  # x (c, h, w), 3x3 or 1x1, same padding
  k = w.shape[-1]
  if k > 1:
    x = np.pad(x, ((0, 0), (1, 1), (1, 1)))
  win = sliding_window_view(x, (k, k), axis=(1, 2))[:, ::stride, ::stride]
  if groups == 1:
    c, h, ww = win.shape[:3]
    cols = np.ascontiguousarray(win.transpose(1, 2, 0, 3, 4).reshape(h * ww, -1))
    y = (cols @ w.reshape(w.shape[0], -1).T).T.reshape(-1, h, ww)
  else:
    y = np.einsum('chwij,cij->chw', win, w[:, 0])
  return y + b[:, None, None]


class Kws:
  def __init__(self, path: str):
    self.p = dict(np.load(path))

  def __call__(self, audio: np.ndarray) -> float:
    p = self.p
    x = np.maximum(_conv(logmel(audio)[None], p['c0.w'], p['c0.b'], stride=2), 0)
    for i in range(4):
      x = np.maximum(_conv(x, p[f'dw{i}.w'], p[f'dw{i}.b'], groups=x.shape[0]), 0)
      x = np.maximum(_conv(x, p[f'pw{i}.w'], p[f'pw{i}.b']), 0)
      if i == 1:
        h, w = x.shape[1] // 2 * 2, x.shape[2] // 2 * 2
        x = x[:, :h, :w].reshape(x.shape[0], h // 2, 2, w // 2, 2).max((2, 4))
    z = float(x.mean((1, 2)) @ p['fc.w'][0] + p['fc.b'][0])
    return 1 / (1 + np.exp(-z))
