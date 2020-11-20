#!/usr/bin/env python3
from selfdrive.manager import start_managed_process, kill_managed_process
import random
import os
import time
import cereal.messaging as messaging

if __name__ == "__main__":
  logmessage = messaging.sub_sock('logMessage')
  hitcount = 0
  hits = []
  ln = 0
  while 1:
    print("\n***** loop %d with hit count %d %r\n" % (ln, hitcount, hits))
    start_managed_process("camerad")
    #time.sleep(random.random())
    os.environ['LOGPRINT'] = "debug"
    start_managed_process("dmonitoringmodeld")
    os.environ['LOGPRINT'] = ""

    # drain all old messages
    messaging.drain_sock(logmessage, False)

    done = False
    cnt = 0
    best = 100
    for i in range(100):
      ret = messaging.drain_sock(logmessage, True)
      for r in ret:
        if 'dmonitoring process' in r.logMessage:
          cnt += 1
          done = r.logMessage
          ms = float(done.split('dmonitoring process: ')[1].split("ms")[0])
          best = min(ms, best)
      if cnt >= 5:
        break

    print(ln, best, done)
    #if best > 16:
    if best > 4:
      print("HIT HIT HIT")
      hitcount += 1
      hits.append(best)


    #start_managed_process("modeld")
    kill_managed_process("dmonitoringmodeld")
    #kill_managed_process("modeld")
    kill_managed_process("camerad")
    ln += 1

    if hitcount >= 1:
      break

