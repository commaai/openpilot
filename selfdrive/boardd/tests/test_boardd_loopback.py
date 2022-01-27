#!/usr/bin/env python3
import os
import random
import time
import unittest
from collections import defaultdict

import cereal.messaging as messaging
from cereal import car
from common.params import Params
from common.spinner import Spinner
from common.timeout import Timeout
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.car import make_can_msg
from selfdrive.hardware import TICI
from selfdrive.test.helpers import phone_only, with_processes


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
  @with_processes(['pandad'])
  def test_loopback(self):
    # wait for boardd to init
    time.sleep(2)

    with Timeout(60, "boardd didn't start"):
      sm = messaging.SubMaster(['pandaStates'])
      while sm.rcv_frame['pandaStates'] < 1 and len(sm['pandaStates']) == 0:
        sm.update(1000)

    num_pandas = len(sm['pandaStates'])
    if TICI:
      self.assertGreater(num_pandas, 1, "connect another panda for multipanda tests")

    # boardd blocks on CarVin and CarParams
    cp = car.CarParams.new_message()

    safety_config = car.CarParams.SafetyConfig.new_message()
    safety_config.safetyModel = car.CarParams.SafetyModel.allOutput
    cp.safetyConfigs = [safety_config]*num_pandas

    params = Params()
    params.put("CarVin", b"0"*17)
    params.put_bool("ControlsReady", True)
    params.put("CarParams", cp.to_bytes())

    sendcan = messaging.pub_sock('sendcan')
    can = messaging.sub_sock('can', conflate=False, timeout=100)
    time.sleep(0.2)

    n = 200
    for i in range(n):
      self.spinner.update(f"boardd loopback {i}/{n}")

      sent_msgs = defaultdict(set)
      for _ in range(random.randrange(10)):
        to_send = []
        for __ in range(random.randrange(100)):
          bus = random.choice([b for b in range(3*num_pandas) if b % 4 != 3])
          addr = random.randrange(1, 1<<29)
          dat = bytes(random.getrandbits(8) for _ in range(random.randrange(1, 9)))
          sent_msgs[bus].add((addr, dat))
          to_send.append(make_can_msg(addr, dat, bus))
        sendcan.send(can_list_to_can_capnp(to_send, msgtype='sendcan'))

      for _ in range(100 * 2):
        recvd = messaging.drain_sock(can, wait_for_one=True)
        for msg in recvd:
          for m in msg.can:
            if m.src >= 128:
              key = (m.address, m.dat)
              assert key in sent_msgs[m.src-128], f"got unexpected msg: {m.src=} {m.address=} {m.dat=}"
              sent_msgs[m.src-128].discard(key)

        if all(len(v) == 0 for v in sent_msgs.values()):
          break

      # if a set isn't empty, messages got dropped
      for bus in sent_msgs.keys():
        assert not len(sent_msgs[bus]), f"loop {i}: bus {bus} missing {len(sent_msgs[bus])} messages"


if __name__ == "__main__":
  unittest.main()
