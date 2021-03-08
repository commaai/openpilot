#!/usr/bin/env python3
import os
import random
import subprocess
import time
from pathlib import Path
from common.basedir import BASEDIR

os.environ["LD_LIBRARY_PATH"] = ""

# pull this from the provisioning tests
play_sound = os.path.join(BASEDIR, "selfdrive/ui/test/play_sound")
waste = os.path.join(BASEDIR, "scripts/waste")
sound_path = Path(os.path.join(BASEDIR, "selfdrive/assets/sounds"))

def sound_test():

  # max volume
  vol = 15
  sound_files = [p.absolute() for p in sound_path.iterdir() if str(p).endswith(".wav")]

  # start waste
  p = subprocess.Popen([waste], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

  start_time = time.monotonic()
  frame = 0
  while True:
    # start a few processes
    procs = []
    for _ in range(random.randint(5, 20)):
      sound = random.choice(sound_files)
      p = subprocess.Popen([play_sound, str(sound), str(vol)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      procs.append(p)
      time.sleep(random.uniform(0, 0.75))

    # and kill them
    time.sleep(random.uniform(0, 5))
    for p in procs:
      p.terminate()

    # write stats
    stats = f"running time {time.monotonic() - start_time}s, cycle {frame}"
    with open("/tmp/sound_stats.txt", "a") as f:
      f.write(stats)
    print(stats)
    frame +=1

if __name__ == "__main__":
  sound_test()
