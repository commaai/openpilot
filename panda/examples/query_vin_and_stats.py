#!/usr/bin/env python3
import time
import struct
from panda import Panda
from hexdump import hexdump
from panda.python.isotp import isotp_send, isotp_recv

# 0x7e0 = Toyota
# 0x18DB33F1 for Honda?


def get_current_data_for_pid(pid):
  # 01 xx = Show current data
  isotp_send(panda, b"\x01" + bytes([pid]), 0x7e0)
  return isotp_recv(panda, 0x7e8)

def get_supported_pids():
  ret = []
  pid = 0
  while 1:
    supported = struct.unpack(">I", get_current_data_for_pid(pid)[2:])[0]
    for i in range(1 + pid, 0x21 + pid):
      if supported & 0x80000000:
        ret.append(i)
      supported <<= 1
    pid += 0x20
    if pid not in ret:
      break
  return ret

if __name__ == "__main__":
  panda = Panda()
  panda.set_safety_mode(Panda.SAFETY_ELM327)
  panda.can_clear(0)

  # 09 02 = Get VIN
  isotp_send(panda, b"\x09\x02", 0x7df)
  ret = isotp_recv(panda, 0x7e8)
  hexdump(ret)
  print("VIN: %s" % "".join(map(chr, ret[:2])))

  # 03 = get DTCS
  isotp_send(panda, b"\x03", 0x7e0)
  dtcs = isotp_recv(panda, 0x7e8)
  print("DTCs:", "".join(map(chr, dtcs[:2])))

  supported_pids = get_supported_pids()
  print("Supported PIDs:", supported_pids)

  while 1:
    speed = struct.unpack(">B", get_current_data_for_pid(13)[2:])[0]                  # kph
    rpm = struct.unpack(">H", get_current_data_for_pid(12)[2:])[0] / 4.0                # revs
    throttle = struct.unpack(">B", get_current_data_for_pid(17)[2:])[0] / 255.0 * 100   # percent
    temp = struct.unpack(">B", get_current_data_for_pid(5)[2:])[0] - 40               # degrees C
    load = struct.unpack(">B", get_current_data_for_pid(4)[2:])[0] / 255.0 * 100        # percent
    print("%d KPH, %d RPM, %.1f%% Throttle, %d deg C, %.1f%% load" % (speed, rpm, throttle, temp, load))
    time.sleep(0.2)
