#!/usr/bin/env python3
import os
import random
import time

import cereal.messaging as messaging
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.car import make_can_msg
from selfdrive.test.helpers import with_processes


os.environ['BOARDD_LOOPBACK'] = '1'
@with_processes(['boardd'])
def test_boardd_loopback():

  can = messaging.sub_sock('can', timeout=1000)
  sendcan = messaging.pub_sock('sendcan')

  time.sleep(1)

  for _ in range(100):
    msgs = []
    for _ in range(random.randrange(5)):
      to_send = []
      for _ in range(random.randrange(20, 100)):
        to_send.append(make_can_msg(random.randrange(1, 0x800), b'\xff'*8, random.randrange(3)))
      msgs.extend(to_send)
      sendcan.send(can_list_to_can_capnp(to_send, msgtype='sendcan'))

    recvd = messaging.drain_sock(can, wait_for_one=True)
    print(recvd)

    recv_msgs = []
    for msg in recvd:
      for m in msg.can:
        recv_msgs.append(make_can_msg(m.address, m.dat, m.src))


    break
    #time.sleep(0.01)


def main():
  rcv = messaging.sub_sock('can')
  snd = messaging.pub_sock('sendcan')
  time.sleep(0.3)  # wait to bind before send/recv

  bus = 0
  for i in range(10):
    print("Loop %d" % i)
    at = random.randint(1024, 2000)
    #st = get_test_string()[0:8]
    st = b"\xff"*8
    snd.send(can_list_to_can_capnp([[at, 0, st, 0]], msgtype='sendcan').to_bytes())
    time.sleep(0.1)
    res = messaging.drain_sock(rcv, True)
    assert len(res) == 1

    res = res[0].can
    assert len(res) == 2

    msg0, msg1 = res

    assert msg0.dat == st
    assert msg1.dat == st

    assert msg0.address == at
    assert msg1.address == at

    assert msg0.src == 0x80 | bus
    assert msg1.src == bus

  print("Success")

if __name__ == "__main__":
  main()
