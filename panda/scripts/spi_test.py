#!/usr/bin/env python3
import os
import sys
import time
from datetime import datetime
from collections import defaultdict, deque

os.environ['NO_RETRY'] = '1'  # no spi retries
from panda import Panda, PandaSpiException


if __name__ == "__main__":
  s = defaultdict(lambda: deque(maxlen=30))
  def avg(k):
    return sum(s[k])/len(s[k])

  #core = 6
  #from openpilot.common.realtime import config_realtime_process
  #config_realtime_process(core, 10)
  #os.system(f"echo {core} | sudo tee /proc/irq/10/smp_affinity_list")
  #os.system("sudo chrt -f -p 1 $(pgrep -f spi0)")
  #print(f"sudo taskset -pc {core} $(pgrep -f spi0)")
  #os.system(f"sudo taskset -pc {core} $(pgrep -f spi0)")

  p = Panda()
  p.reset()
  start = datetime.now()
  le = p.health()['spi_error_count']
  while True:
    cnt = 0
    st = time.monotonic()
    while time.monotonic() - st < 1:
      try:
        p.get_type()   # get_type has no processing on panda side
      except PandaSpiException:
        print(f"ERROR after {datetime.now() - start}\n\n")
        sys.stdout.write("\033[1;32;40m" + p.serial_read(0).decode('utf8') + "\033[00m" + "\n")
        sys.stdout.flush()
        sys.exit()
      cnt += 1
    s['hz'].append(cnt)

    err = p.health()['spi_error_count']
    s['err'].append((err - le) & 0xFFFFFFFF)
    le = err

    print(
      f"{avg('hz'):04.0f}Hz   {avg('err'):04.0f} errors   [{cnt:04d}Hz   {s['err'][-1]:04d} errors]  {datetime.now() - start}"
    )

