#!/usr/bin/env python3
import os
import sys
import time

print("starting at")
os.system("cat /sys/kernel/debug/regulator/pm8994_s11/voltage")
print("volts")

os.system("echo 99e8000.cpr3-ctrl > /sys/devices/soc/spm-regulator-10/regulator/regulator.56/99e8000.cpr3-ctrl-vdd/driver/unbind")
os.system("echo 1 > /sys/kernel/debug/regulator/pm8994_s11/enable")

if len(sys.argv) > 1:
  i = int(sys.argv[1])
  os.system("echo %d %d > /sys/kernel/debug/regulator/pm8994_s11/voltage" % (i,i))
  os.system("cat /sys/kernel/debug/regulator/pm8994_s11/voltage")
else:
  for i in range(900000, 465000, -10000):
    print("setting voltage to",i)
    os.system("echo %d %d > /sys/kernel/debug/regulator/pm8994_s11/voltage" % (i,i))
    os.system("cat /sys/kernel/debug/regulator/pm8994_s11/voltage")
    time.sleep(1)

