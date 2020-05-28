#!/usr/bin/env python3
import os
import subprocess
import time
import datetime
import random

from common.basedir import BASEDIR
import cereal.messaging as messaging

if __name__ == "__main__":

  sound_dir = os.path.join(BASEDIR, "selfdrive/assets/sounds")
  sound_files = [f for f in os.listdir(sound_dir) if f.endswith(".wav")]
  play_sound = os.path.join(BASEDIR, "selfdrive/ui/test/play_sound")

  print("disabling charging")
  os.system('echo "0" > /sys/class/power_supply/battery/charging_enabled')

  os.environ["LD_LIBRARY_PATH"] = ""

  sm = messaging.SubMaster(["thermal"])

  FNULL = open(os.devnull, "w")
  start_time = time.time()
  while True:
    volume = 15

    n = random.randint(5, 10)
    procs = []
    for _ in range(n):
      sound = random.choice(sound_files)
      p = subprocess.Popen([play_sound, os.path.join(sound_dir, sound), str(volume)], stdout=FNULL, stderr=FNULL)
      procs.append(p)
      time.sleep(random.uniform(0, 0.75))

    time.sleep(random.randint(0, 5))
    for p in procs:
      p.terminate()

    sm.update(0)
    s = time.time() - start_time
    hhmmss = str(datetime.timedelta(seconds=s)).split(".")[0]
    print("test duration:", hhmmss)
    print("\tbattery percent", sm["thermal"].batteryPercent)
