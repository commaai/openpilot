import numpy as np
from scipy.io import wavfile


sr = 48000
max_int16 = 2**15 - 1

def harmonic_beep(freq, duration_seconds):
    n_total = int(sr * duration_seconds)

    signal = np.sin(2 * np.pi * freq * np.arange(n_total) / sr)
    x = np.arange(n_total)
    exp_scale = np.exp(-x/5.5e3)
    return max_int16 * signal * exp_scale

engage_beep = harmonic_beep(1661.219, 0.5)
wavfile.write("engage.wav", sr, engage_beep.astype(np.int16))
disengage_beep = harmonic_beep(1318.51, 0.5)
wavfile.write("disengage.wav", sr, disengage_beep.astype(np.int16))
