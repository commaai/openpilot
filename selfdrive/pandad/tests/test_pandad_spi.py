import os
import time
import numpy as np
import pytest
import random

import cereal.messaging as messaging
from cereal.services import SERVICE_LIST
from openpilot.system.hardware import HARDWARE
from openpilot.selfdrive.test.helpers import with_processes
from openpilot.selfdrive.pandad.tests.test_pandad_loopback import setup_pandad, send_random_can_messages

JUNGLE_SPAM = "JUNGLE_SPAM" in os.environ

@pytest.mark.tici
class TestBoarddSpi:
  @classmethod
  def setup_class(cls):
    if HARDWARE.get_device_type() == 'tici':
      pytest.skip("only for spi pandas")
    os.environ['STARTED'] = '1'
    os.environ['SPI_ERR_PROB'] = '0.001'
    if not JUNGLE_SPAM:
      os.environ['BOARDD_LOOPBACK'] = '1'

  @with_processes(['pandad'])
  def test_spi_corruption(self, subtests):
    setup_pandad(1)

    sendcan = messaging.pub_sock('sendcan')
    socks = {s: messaging.sub_sock(s, conflate=False, timeout=100) for s in ('can', 'pandaStates', 'peripheralState')}
    time.sleep(2)
    for s in socks.values():
      messaging.drain_sock_raw(s)

    total_recv_count = 0
    total_sent_count = 0
    sent_msgs = {bus: list() for bus in range(3)}

    st = time.monotonic()
    ts = {s: list() for s in socks.keys()}
    for _ in range(int(os.getenv("TEST_TIME", "20"))):
      # send some CAN messages
      if not JUNGLE_SPAM:
        sent = send_random_can_messages(sendcan, random.randrange(2, 20))
        for k, v in sent.items():
          sent_msgs[k].extend(list(v))
          total_sent_count += len(v)

      for service, sock in socks.items():
        for m in messaging.drain_sock(sock):
          ts[service].append(m.logMonoTime)

          # sanity check for corruption
          assert m.valid or (service == "can")
          if service == "can":
            for msg in m.can:
              if JUNGLE_SPAM:
                # PandaJungle.set_generated_can(True)
                i = msg.address - 0x200
                assert msg.address >= 0x200
                assert msg.src == (i%3)
                assert msg.dat == b"\xff"*(i%8)
                total_recv_count += 1
                continue

              if msg.src > 4:
                continue
              key = (msg.address, msg.dat)
              assert key in sent_msgs[msg.src], f"got unexpected msg: {msg.src=} {msg.address=} {msg.dat=}"
              # TODO: enable this
              #sent_msgs[msg.src].remove(key)
              total_recv_count += 1
          elif service == "pandaStates":
            assert len(m.pandaStates) == 1
            ps = m.pandaStates[0]
            assert ps.uptime < 1000
            assert ps.pandaType == "tres"
            assert ps.ignitionLine
            assert not ps.ignitionCan
            assert 4000 < ps.voltage < 14000
          elif service == "peripheralState":
            ps = m.peripheralState
            assert ps.pandaType == "tres"
            assert 4000 < ps.voltage < 14000
            assert 50 < ps.current < 1000
            assert ps.fanSpeedRpm < 10000

      time.sleep(0.5)
    et = time.monotonic() - st

    print("\n======== timing report ========")
    for service, times in ts.items():
      dts = np.diff(times)/1e6
      print(service.ljust(17), f"{np.mean(dts):7.2f} {np.min(dts):7.2f} {np.max(dts):7.2f}")
      with subtests.test(msg="timing check", service=service):
        edt = 1e3 / SERVICE_LIST[service].frequency
        assert edt*0.9 < np.mean(dts) < edt*1.1
        assert np.max(dts) < edt*8
        assert np.min(dts) < edt
        assert len(dts) >= ((et-0.5)*SERVICE_LIST[service].frequency*0.8)

    with subtests.test(msg="CAN traffic"):
      print(f"Sent {total_sent_count} CAN messages, got {total_recv_count} back. {total_recv_count/(total_sent_count+1e-4):.2%} received")
      assert total_recv_count > 20
