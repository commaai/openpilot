#!/usr/bin/env python3
import os
import time
import cereal.messaging as messaging

logMessage = messaging.sub_sock("logMessage")

cnt = 0
last_missed_cycle = time.monotonic()
missed_cycle_count = 0
while True:
  # check for missed cycles
  msgs = messaging.drain_sock(logMessage)
  for msg in msgs:
    if "missed cycle" in msg.logMessage:
      missed_cycle_count += 1
      last_missed_cycle = time.monotonic()
      print("MISSED CYCLE", missed_cycle_count)
  
  cnt += 1
  if cnt % 50 == 0:
    print(f"{time.monotonic() - last_missed_cycle}s since last missed cycle, {missed_cycle_count} total")
  time.sleep(0.1)
