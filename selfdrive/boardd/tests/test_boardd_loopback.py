#!/usr/bin/env python3
import os
import random
import time
import unittest
from collections import defaultdict
from functools import wraps

import cereal.messaging as messaging
from cereal import car
from common.params import Params
from common.spinner import Spinner
from common.timeout import Timeout
from panda import Panda
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.car import make_can_msg
from selfdrive.test.helpers import phone_only, with_processes


def reset_panda(f):
  @wraps(f)
  def wrapper(*args, **kwargs):
    p = Panda()
    for i in [0, 1, 2, 0xFFFF]:
      p.can_clear(i)
    p.reset()
    p.close()
    f(*args, **kwargs)
  return wrapper


class TestBoardd(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    os.environ['STARTED'] = '1'
    os.environ['BOARDD_LOOPBACK'] = '1'
    cls.spinner = Spinner()

  @classmethod
  def tearDownClass(cls):
    cls.spinner.close()

  @phone_only
  @reset_panda
  @with_processes(['pandad'])
  def test_loopback(self):
    # wait for boardd to init
    time.sleep(2)

    with Timeout(60, "boardd didn't start"):
      sm = messaging.SubMaster(['pandaStates'])
      while sm.rcv_frame['pandaStates'] < 1:
        sm.update(1000)

    # boardd blocks on CarVin and CarParams
    cp = car.CarParams.new_message()

    safety_config = car.CarParams.SafetyConfig.new_message()
    safety_config.safetyModel = car.CarParams.SafetyModel.allOutput
    cp.safetyConfigs = [safety_config]

    params = Params()
    params.put("CarVin", b"0"*17)
    params.put_bool("ControlsReady", True)
    params.put("CarParams", cp.to_bytes())

    sendcan = messaging.pub_sock('sendcan')
    can = messaging.sub_sock('can', conflate=False, timeout=100)

    time.sleep(1)

    n = 1000
    for i in range(n):
      self.spinner.update(f"boardd loopback {i}/{n}")

      sent_msgs = defaultdict(set)
      for _ in range(random.randrange(10)):
        to_send = []
        for __ in range(random.randrange(100)):
          bus = random.randrange(3)
          addr = random.randrange(1, 1<<29)
          dat = bytes([random.getrandbits(8) for _ in range(random.randrange(1, 9))])
          sent_msgs[bus].add((addr, dat))
          to_send.append(make_can_msg(addr, dat, bus))
        sendcan.send(can_list_to_can_capnp(to_send, msgtype='sendcan'))

      max_recv = 100
      while max_recv > 0 and any(len(sent_msgs[bus]) for bus in range(3)):
        recvd = messaging.drain_sock(can, wait_for_one=True)
        for msg in recvd:
          for m in msg.can:
            if m.src >= 128:
              k = (m.address, m.dat)
              assert k in sent_msgs[m.src-128]
              sent_msgs[m.src-128].discard(k)
        max_recv -= 1

      # if a set isn't empty, messages got dropped
      for bus in range(3):
        assert not len(sent_msgs[bus]), f"loop {i}: bus {bus} missing {len(sent_msgs[bus])} messages"


if __name__ == "__main__":
  unittest.main()
