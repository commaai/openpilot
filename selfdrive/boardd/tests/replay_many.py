#!/usr/bin/env python3
import os
import sys
import time
import signal
import traceback
from panda import Panda
from multiprocessing import Pool

jungle = "JUNGLE" in os.environ
if jungle:
  from panda_jungle import PandaJungle # pylint: disable=import-error

import cereal.messaging as messaging
from selfdrive.boardd.boardd import can_capnp_to_can_list

def initializer():
  """Ignore CTRL+C in the worker process.
  source: https://stackoverflow.com/a/44869451 """
  signal.signal(signal.SIGINT, signal.SIG_IGN)

def send_thread(sender_serial):
  global jungle
  try:
    if jungle:
      sender = PandaJungle(sender_serial)
    else:
      sender = Panda(sender_serial)
      sender.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
    sender.set_can_loopback(False)

    can_sock = messaging.sub_sock('can')

    while True:
      # Send messages one bus 0 and 1
      tsc = messaging.recv_one(can_sock)
      snd = can_capnp_to_can_list(tsc.can)
      snd = list(filter(lambda x: x[-1] <= 2, snd))
      sender.can_send_many(snd)

      # Drain panda message buffer
      sender.can_recv()
  except Exception:
    traceback.print_exc()

if __name__ == "__main__":
  if jungle:
    serials = PandaJungle.list()
  else:
    serials = Panda.list()
  num_senders = len(serials)

  if num_senders == 0:
    print("No senders found. Exiting")
    sys.exit(1)
  else:
    print("%d senders found. Starting broadcast" % num_senders)

  pool = Pool(num_senders, initializer=initializer)
  pool.map_async(send_thread, serials)

  while True:
    try:
      time.sleep(10)
    except KeyboardInterrupt:
      pool.terminate()
      pool.join()
      raise
