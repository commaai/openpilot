"""Small streaming speech EQ, before gain/limiting. No extra audio dependencies."""
import numpy as np


class VoiceEQ:
  def __init__(self, sample_rate):
    # Symmetric FIR: 5 ms group delay, with persistent history across packets.
    half = round(sample_rate * 0.005)
    n = np.arange(-half, half + 1)
    window = np.hamming(len(n))

    def lowpass(frequency):
      kernel = 2 * frequency / sample_rate * np.sinc(2 * frequency / sample_rate * n) * window
      return kernel / kernel.sum()

    impulse = np.zeros(len(n))
    impulse[half] = 1
    bell = lowpass(600) - lowpass(200)
    bell /= np.sum(bell * np.cos(2 * np.pi * 400 / sample_rate * n))
    self.kernel = impulse + (10 ** (-4 / 20) - 1) * bell
    vocal_dip = lowpass(1400) - lowpass(700)
    response = np.cos(2 * np.pi * 1000 / sample_rate * n)
    vocal_dip /= np.sum(vocal_dip * response)
    self.kernel += (10 ** (-3 / 20) - np.sum(self.kernel * response)) * vocal_dip
    if sample_rate > 20000:
      self.kernel += (10 ** (2 / 20) - 1) * (impulse - lowpass(10000))
    self.history = None

  def reset(self):
    self.history = None

  def process(self, samples):
    if not len(samples):
      return samples
    if self.history is None:
      self.history = np.full(len(self.kernel) - 1, samples[0])
    padded = np.concatenate((self.history, samples))
    self.history = padded[-(len(self.kernel) - 1):].copy()
    return np.convolve(padded, self.kernel, mode='valid').astype(np.float32)
