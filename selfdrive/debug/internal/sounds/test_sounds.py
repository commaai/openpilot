#!/usr/bin/env python3

import os
import subprocess
import time

from common.basedir import BASEDIR

if __name__ == "__main__":

  sound_dir = os.path.join(BASEDIR, "selfdrive/assets/sounds")
  sound_files = [f for f in os.listdir(sound_dir) if f.endswith(".wav")]

  play_sound = os.path.join(BASEDIR, "selfdrive/ui/test/play_sound")

  os.environ["LD_LIBRARY_PATH"] = ""

  while True:
    for volume in range(10, 16):
      for sound in sound_files:
        p = subprocess.Popen([play_sound, os.path.join(sound_dir, sound), str(volume)])
        time.sleep(1)
        p.terminate()
