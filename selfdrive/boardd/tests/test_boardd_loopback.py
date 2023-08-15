#!/usr/bin/env python3
import os
import copy
import random
import time
import unittest
from collections import defaultdict
from pprint import pprint

import cereal.messaging as messaging
from cereal import car, log
from common.params import Params
from common.spinner import Spinner
from common.timeout import Timeout
from selfdrive.boardd.boardd import can_list_to_can_capnp
from selfdrive.car import make_can_msg
from system.hardware import TICI
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
    params = Params()
    params.put_bool("IsOnroad", False)

    with Timeout(90, "boardd didn't start"):
      sm = messaging.SubMaster(['pandaStates'])
      while sm.rcv_frame['pandaStates'] < 1 or len(sm['pandaStates']) == 0 or \
          any(ps.pandaType == log.PandaState.PandaType.unknown for ps in sm['pandaStates']):
        sm.update(1000)

    num_pandas = len(sm['pandaStates'])
    expected_pandas = 2 if TICI and "SINGLE_PANDA" not in os.environ else 1
    self.assertEqual(num_pandas, expected_pandas, "connected pandas ({num_pandas}) doesn't match expected panda count ({expected_pandas}). \
                                                   connect another panda for multipanda tests.")

    # boardd safety setting relies on these params
    cp = car.CarParams.new_message()

    safety_config = car.CarParams.SafetyConfig.new_message()
    safety_config.safetyModel = car.CarParams.SafetyModel.allOutput
    cp.safetyConfigs = [safety_config]*num_pandas

    params.put_bool("IsOnroad", True)
    params.put_bool("FirmwareQueryDone", True)
    params.put_bool("ControlsReady", True)
    params.put("CarParams", cp.to_bytes())

    sendcan = messaging.pub_sock('sendcan')
    can = messaging.sub_sock('can', conflate=False, timeout=100)
    sm = messaging.SubMaster(['pandaStates'])
    time.sleep(0.5)

    n = 200
    for i in range(n):
      print(f"boardd loopback {i}/{n}")
      self.spinner.update(f"boardd loopback {i}/{n}")

      sent_msgs = defaultdict(set)
      for _ in range(random.randrange(20, 100)):
        to_send = []
        for __ in range(random.randrange(20)):
          bus = random.choice([b for b in range(3*num_pandas) if b % 4 != 3])
          addr = random.randrange(1, 1<<29)
          dat = bytes(random.getrandbits(8) for _ in range(random.randrange(1, 9)))
          sent_msgs[bus].add((addr, dat))
          to_send.append(make_can_msg(addr, dat, bus))
        sendcan.send(can_list_to_can_capnp(to_send, msgtype='sendcan'))

      sent_loopback = copy.deepcopy(sent_msgs)
      sent_loopback.update({k+128: copy.deepcopy(v) for k, v in sent_msgs.items()})
      sent_total = {k: len(v) for k, v in sent_loopback.items()}
      for _ in range(100 * 5):
        sm.update(0)
        recvd = messaging.drain_sock(can, wait_for_one=True)
        for msg in recvd:
          for m in msg.can:
            key = (m.address, m.dat)
            assert key in sent_loopback[m.src], f"got unexpected msg: {m.src=} {m.address=} {m.dat=}"
            sent_loopback[m.src].discard(key)

        if all(len(v) == 0 for v in sent_loopback.values()):
          break

      # if a set isn't empty, messages got dropped
      pprint(sent_msgs)
      pprint(sent_loopback)
      print({k: len(x) for k, x in sent_loopback.items()})
      print(sum([len(x) for x in sent_loopback.values()]))
      pprint(sm['pandaStates'])  # may drop messages due to RX buffer overflow
      for bus in sent_loopback.keys():
        assert not len(sent_loopback[bus]), f"loop {i}: bus {bus} missing {len(sent_loopback[bus])} out of {sent_total[bus]} messages"


if __name__ == "__main__":
  unittest.main()
