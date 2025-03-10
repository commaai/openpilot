#!/usr/bin/env python3
# type: ignore

import os
import argparse
import struct
from collections import deque
from statistics import mean

from cereal import log
import cereal.messaging as messaging

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Sniff a communication socket')
  parser.add_argument('--addr', default='127.0.0.1')
  args = parser.parse_args()

  if args.addr != "127.0.0.1":
    os.environ["ZMQ"] = "1"
    messaging.reset_context()

  poller = messaging.Poller()
  messaging.sub_sock('can', poller, addr=args.addr)

  active = 0
  start_t = 0
  start_v = 0
  max_v = 0
  max_t = 0
  window = deque(maxlen=10)
  avg = 0
  while 1:
    polld = poller.poll(1000)
    for sock in polld:
      msg = sock.receive()
      with log.Event.from_bytes(msg) as log_evt:
        evt = log_evt

      for item in evt.can:
        if item.address == 0xe4 and item.src == 128:
          torque_req = struct.unpack('!h', item.dat[0:2])[0]
          # print(torque_req)
          active = abs(torque_req) > 0
          if abs(torque_req) < 100:
            if max_v > 5:
              print(f'{start_v} -> {max_v} = {round(max_v - start_v, 2)} over {round(max_t - start_t, 2)}s')
            start_t = evt.logMonoTime / 1e9
            start_v = avg
            max_t = 0
            max_v = 0
        if item.address == 0x1ab and item.src == 0:
          motor_torque = ((item.dat[0] & 0x3) << 8) + item.dat[1]
          window.append(motor_torque)
          avg = mean(window)
          #print(f'{evt.logMonoTime}: {avg}')
          if active and avg > max_v + 0.5:
            max_v = avg
            max_t = evt.logMonoTime / 1e9
