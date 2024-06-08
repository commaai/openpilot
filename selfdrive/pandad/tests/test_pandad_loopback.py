import os
import copy
import random
import time
import pytest
from collections import defaultdict
from pprint import pprint

import cereal.messaging as messaging
from cereal import car, log
from openpilot.common.retry import retry
from openpilot.common.params import Params
from openpilot.common.timeout import Timeout
from openpilot.selfdrive.pandad import can_list_to_can_capnp
from openpilot.selfdrive.car import make_can_msg
from openpilot.system.hardware import TICI
from openpilot.selfdrive.test.helpers import phone_only, with_processes


@retry(attempts=3)
def setup_pandad(num_pandas):
  params = Params()
  params.clear_all()
  params.put_bool("IsOnroad", False)

  sm = messaging.SubMaster(['pandaStates'])
  with Timeout(90, "pandad didn't start"):
    while sm.recv_frame['pandaStates'] < 1 or len(sm['pandaStates']) == 0 or \
        any(ps.pandaType == log.PandaState.PandaType.unknown for ps in sm['pandaStates']):
      sm.update(1000)

  found_pandas = len(sm['pandaStates'])
  assert num_pandas == found_pandas, "connected pandas ({found_pandas}) doesn't match expected panda count ({num_pandas}). \
                                      connect another panda for multipanda tests."

  # pandad safety setting relies on these params
  cp = car.CarParams.new_message()

  safety_config = car.CarParams.SafetyConfig.new_message()
  safety_config.safetyModel = car.CarParams.SafetyModel.allOutput
  cp.safetyConfigs = [safety_config]*num_pandas

  params.put_bool("IsOnroad", True)
  params.put_bool("FirmwareQueryDone", True)
  params.put_bool("ControlsReady", True)
  params.put("CarParams", cp.to_bytes())

  with Timeout(90, "pandad didn't set safety mode"):
    while any(ps.safetyModel != car.CarParams.SafetyModel.allOutput for ps in sm['pandaStates']):
      sm.update(1000)

def send_random_can_messages(sendcan, count, num_pandas=1):
  sent_msgs = defaultdict(set)
  for _ in range(count):
    to_send = []
    for __ in range(random.randrange(20)):
      bus = random.choice([b for b in range(3*num_pandas) if b % 4 != 3])
      addr = random.randrange(1, 1<<29)
      dat = bytes(random.getrandbits(8) for _ in range(random.randrange(1, 9)))
      if (addr, dat) in sent_msgs[bus]:
        continue
      sent_msgs[bus].add((addr, dat))
      to_send.append(make_can_msg(addr, dat, bus))
    sendcan.send(can_list_to_can_capnp(to_send, msgtype='sendcan'))
  return sent_msgs


@pytest.mark.tici
class TestBoarddLoopback:
  @classmethod
  def setup_class(cls):
    os.environ['STARTED'] = '1'
    os.environ['BOARDD_LOOPBACK'] = '1'

  @phone_only
  @with_processes(['pandad'])
  def test_loopback(self):
    num_pandas = 2 if TICI and "SINGLE_PANDA" not in os.environ else 1
    setup_pandad(num_pandas)

    sendcan = messaging.pub_sock('sendcan')
    can = messaging.sub_sock('can', conflate=False, timeout=100)
    sm = messaging.SubMaster(['pandaStates'])
    time.sleep(1)

    n = 200
    for i in range(n):
      print(f"pandad loopback {i}/{n}")

      sent_msgs = send_random_can_messages(sendcan, random.randrange(20, 100), num_pandas)

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
