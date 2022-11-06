#!/usr/bin/env python3
import sounddevice as sd
import numpy as np


def print_sound(indata, frames, time, status):
  if status:
    print(status, end='')
  volume_norm = np.linalg.norm(indata) * 10
  print ("|" * int(volume_norm))


if __name__ == "__main__":
  with sd.InputStream(callback=print_sound):
    sd.sleep(10000)
