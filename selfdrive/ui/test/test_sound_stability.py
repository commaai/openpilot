#!/usr/bin/env python3
import os
import random
import subprocess
import time

from cereal import car
from common.basedir import BASEDIR

os.environ["LD_LIBRARY_PATH"] = ""
play_sound = os.path.join(BASEDIR, "selfdrive/ui/test/play_sound")

def sound_test():

  sound_enums = car.CarControl.AudibleAlert.schema.enumerants.items()
  all_sounds = [v for k, v in  sound_enums if k != "none"]

  # max volume
  vol = 15

  start_time = time.monotonic()
  frame = 0
  while True:
    # start a few processes
    procs = []
    for _ in range(random.randint(5, 20)):
      sound = random.choice(all_sounds)
      p = subprocess.Popen([play_sound, str(sound), str(vol)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
