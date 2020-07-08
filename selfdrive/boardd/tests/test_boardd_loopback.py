#!/usr/bin/env python3
import os
import random
import time
from collections import defaultdict

import cereal.messaging as messaging
from cereal import car
from common.basedir import PARAMS
from common.params import Params
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.car import make_can_msg
from selfdrive.test.helpers import with_processes

os.environ['PARAMS_PATH'] = PARAMS
@with_processes(['boardd'])
def test_boardd_loopback():

  # wait for boardd to init
  time.sleep(2)

  # boardd blocks on CarVin and CarParams
  cp = car.CarParams.new_message()
  cp.safetyModel = car.CarParams.SafetyModel.allOutput
  Params().put("CarVin", b"0"*17)
  Params().put("CarParams", cp.to_bytes())

  sendcan = messaging.pub_sock('sendcan')
  can = messaging.sub_sock('can', timeout=1000)

  time.sleep(2)

  for _ in range(20):
    sent_msgs = defaultdict(set)
    for _ in range(random.randrange(5)):
      to_send = []
      for _ in range(random.randrange(10)):
        bus = random.randrange(3)
        addr = random.randrange(1, 0x800)
        dat = bytes([random.getrandbits(8) for _ in range(random.randrange(1, 9))])
        sent_msgs[bus].add((addr, dat))
        to_send.append(make_can_msg(addr, dat, bus))
      sendcan.send(can_list_to_can_capnp(to_send, msgtype='sendcan'))

    time.sleep(2)

    recvd = messaging.drain_sock(can, wait_for_one=True)
    for msg in recvd:
      for m in msg.can:
        if m.src >= 128:
          k = (m.address, m.dat)
          assert k in sent_msgs[m.src-128], k
          sent_msgs[m.src-128].discard(k)

    # each set should be empty
    for bus in range(3):
      assert not len(sent_msgs[bus]), f"bus {bus}: missing {len(sent_msgs[bus])} messages"

