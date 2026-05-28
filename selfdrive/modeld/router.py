#!/usr/bin/env python3
import time
import cereal.messaging as messaging
from cereal.messaging import PubMaster, SubMaster
from cereal.services import SERVICE_LIST


def main():
  sm = SubMaster(['bigModelV2', 'smolModelV2'])
  pm = PubMaster(['modelV2'])

  # fall back to smol as soon as big misses a frame
  big_stale_dt = 1.5 / SERVICE_LIST['bigModelV2'].frequency
  while True:
    sm.update()
    big_fresh = sm.seen['bigModelV2'] and (time.monotonic() - sm.recv_time['bigModelV2']) < big_stale_dt
    src = 'bigModelV2' if big_fresh else 'smolModelV2'
    if sm.updated[src]:
      msg = messaging.new_message('modelV2')
      msg.valid = sm.valid[src]
      msg.modelV2 = sm[src]
      pm.send('modelV2', msg)


if __name__ == "__main__":
  main()
