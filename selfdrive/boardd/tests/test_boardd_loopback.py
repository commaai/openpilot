#!/usr/bin/env python3
"""Run boardd with the BOARDD_LOOPBACK envvar before running this test."""

import os
import random
import time

from selfdrive.boardd.boardd import can_list_to_can_capnp
from cereal.messaging import drain_sock, pub_sock, sub_sock

def get_test_string():
  return b"test"+os.urandom(10)

BUS = 0

def main():
  rcv = sub_sock('can') # port 8006
  snd = pub_sock('sendcan') # port 8017
  time.sleep(0.3) # wait to bind before send/recv

  for i in range(10):
    print("Loop %d" % i)
    at = random.randint(1024, 2000)
    st = get_test_string()[0:8]
    snd.send(can_list_to_can_capnp([[at, 0, st, 0]], msgtype='sendcan').to_bytes())
    time.sleep(0.1)
    res = drain_sock(rcv, True)
    assert len(res) == 1

    res = res[0].can
    assert len(res) == 2

    msg0, msg1 = res

    assert msg0.dat == st
    assert msg1.dat == st

    assert msg0.address == at
    assert msg1.address == at

    assert msg0.src == 0x80 | BUS
    assert msg1.src == BUS

  print("Success")

if __name__ == "__main__":
  main()
