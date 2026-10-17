"""Conservative delay-and-gain echo subtraction for the livestream speaker.

This handles a dominant direct echo, not a room's full impulse response. Weak
matches (including unrelated nearby speech) pass through without attenuation.
"""
import numpy as np

from openpilot.common.voice_eq import VoiceEQ


class SimpleEchoCanceller:
  RATE = 48000
  MAX_DELAY = 0.5  # Includes the 150 ms speech playback buffer.

  def __init__(self):
    self.eq = VoiceEQ(self.RATE)
    self.reset()

  def reset(self):
    self.reference = np.empty(0, dtype=np.float32)
    self.start = 0.0
    self.last_packet = 0.0
    self.eq.reset()

  def push(self, samples, timestamp):
    if timestamp < self.last_packet:
      return
    end = self.start + len(self.reference) / self.RATE
    if not len(self.reference) or abs(timestamp - end) > 0.1:
      self.reset()
      self.start = timestamp
    self.last_packet = timestamp
    # Match soundd's speech EQ and limiting, before the hardware volume control.
    pcm = np.clip(self.eq.process(samples), -1, 1)
    self.reference = np.concatenate((self.reference, pcm))
    excess = len(self.reference) - self.RATE
    if excess > 0:
      self.reference = self.reference[excess:].copy()
      self.start += excess / self.RATE

  def process(self, samples, end_time):
    if not len(self.reference) or end_time - self.last_packet > self.MAX_DELAY + 0.15:
      return samples
    count = len(samples)
    if count < 6 or count % 6:
      return samples
    delay = round(self.MAX_DELAY * self.RATE)
    first = round((end_time - count / self.RATE - self.MAX_DELAY - self.start) * self.RATE)
    indices = first + np.arange(count + delay)
    reference = np.zeros(len(indices), dtype=np.float32)
    valid = (indices >= 0) & (indices < len(self.reference))
    reference[valid] = self.reference[indices[valid]]
    # Search at 8 kHz, then refine the best delay at the original sample rate.
    mic_low = samples.reshape(-1, 6).mean(axis=1)
    ref_low = reference.reshape(-1, 6).mean(axis=1)
    mic_energy = float(mic_low @ mic_low)
    if mic_energy < 1e-8:
      return samples
    dot = np.correlate(ref_low, mic_low, mode='valid')
    energy = np.convolve(ref_low * ref_low, np.ones(len(mic_low), dtype=np.float32), mode='valid')
    score = dot / np.sqrt(np.maximum(energy * mic_energy, 1e-20))
    best = int(np.argmax(np.abs(score)))
    if abs(score[best]) < 0.65 or energy[best] < 1e-7:
      return samples
    best_offset, best_score = 0, 0.0
    for offset in range(max(0, best * 6 - 6), min(delay, best * 6 + 6) + 1):
      echo = reference[offset:offset + count]
      power = float(echo @ echo)
      cross = float(echo @ samples)
      match = abs(cross) / np.sqrt(max(power * float(samples @ samples), 1e-20))
      if match > best_score:
        best_offset, best_score = offset, match
    if best_score < 0.65:
      return samples
    echo = reference[best_offset:best_offset + count]
    gain = float(echo @ samples) / max(float(echo @ echo), 1e-12)
    if not 0 < abs(gain) <= 2:
      return samples
    return (samples - gain * echo).astype(np.float32)
