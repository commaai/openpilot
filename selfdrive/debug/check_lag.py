#!/usr/bin/env python3
import cereal.messaging as messaging
from cereal.services import service_list

TO_CHECK = ['carState']


if __name__ == "__main__":
  sm = messaging.SubMaster(TO_CHECK)

  prev_t = {}

  while True:
    sm.update()

    for s in TO_CHECK:
      if sm.updated[s]:
        t = sm.logMonoTime[s] / 1e9

        if s in prev_t:
          expected = 1.0 / (service_list[s].frequency)
          dt = t - prev_t[s]
          if dt > 10 * expected:
            print(t, s, dt)

        prev_t[s] = t
