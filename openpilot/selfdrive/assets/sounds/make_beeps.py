import wave
import numpy as np


sr = 48000
max_int16 = 2**15 - 1

def harmonic_beep(freq, duration_seconds):
  n_total = int(sr * duration_seconds)

  signal = np.sin(2 * np.pi * freq * np.arange(n_total) / sr)
  x = np.arange(n_total)
  exp_scale = np.exp(-x/5.5e3)
  return max_int16 * signal * exp_scale

def write_wav(fn, samples):
  with wave.open(fn, "wb") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sr)
    f.writeframes(samples.astype(np.int16).tobytes())

write_wav("engage.wav", harmonic_beep(1661.219, 0.5))
write_wav("disengage.wav", harmonic_beep(1318.51, 0.5))
