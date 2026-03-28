from __future__ import annotations

from pathlib import Path

import numpy as np
import sounddevice as sd

from openpilot.common.basedir import BASEDIR


def play_wav(path_from_repo_root: str) -> None:
  path = Path(BASEDIR) / path_from_repo_root
  if not path.exists():
    return

  import wave
  with wave.open(str(path), "rb") as wave_file:
    n_channels = wave_file.getnchannels()
    frames = wave_file.readframes(wave_file.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16)
    if n_channels > 1:
      audio = audio.reshape(-1, n_channels)
    sd.play(audio.astype(np.float32) / 32768.0, samplerate=wave_file.getframerate(), blocking=False)
