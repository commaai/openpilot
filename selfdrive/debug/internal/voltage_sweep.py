#!/usr/bin/env python3
import os
import time

os.system("echo 1 > /sys/kernel/debug/regulator/pm8994_s11/enable")

for i in range(900000, 465000, -10000):
  print("setting voltage to",i)
  os.system("echo %d %d > /sys/kernel/debug/regulator/pm8994_s11/voltage" % (i,i))
  time.sleep(1)


