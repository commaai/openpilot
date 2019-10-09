#!/usr/bin/env python3
import sys
import time
import signal
from panda import Panda
from multiprocessing import Pool

import selfdrive.messaging as messaging
from selfdrive.services import service_list
from selfdrive.boardd.boardd import can_capnp_to_can_list

def initializer():
    """Ignore CTRL+C in the worker process.
    source: https://stackoverflow.com/a/44869451 """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def send_thread(serial):
  panda = Panda(serial)
  panda.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
  panda.set_can_loopback(False)

  can_sock = messaging.sub_sock(service_list['can'].port)

  while True:
    # Send messages one bus 0 and 1
    tsc = messaging.recv_one(can_sock)
    snd = can_capnp_to_can_list(tsc.can)
    snd = filter(lambda x: x[-1] <= 2, snd)
    panda.can_send_many(snd)

    # Drain panda message buffer
    panda.can_recv()


if __name__ == "__main__":
  serials = Panda.list()
  num_pandas = len(serials)

  if num_pandas == 0:
    print("No pandas found. Exiting")
    sys.exit(1)
  else:
    print("%d pandas found. Starting broadcast" % num_pandas)

  pool = Pool(num_pandas, initializer=initializer)
  pool.map_async(send_thread, serials)

  while True:
    try:
      time.sleep(10)
    except KeyboardInterrupt:
      pool.terminate()
      pool.join()
      raise
