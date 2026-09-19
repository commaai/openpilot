import binascii
import random
from unittest.mock import patch

from panda.tests.hitl.base import PandaTestCase
from panda import Panda
from panda.python.spi import PandaProtocolMismatch, PandaSpiNackResponse


class TestSpi(PandaTestCase):
  def _ping(self, panda):
    # should work with no retries
    with patch.object(panda._handle, '_wait_for_ack', wraps=panda._handle._wait_for_ack) as spy:
      panda.health()
      assert spy.call_count == 2

  def test_protocol_version_check(self):
    p = self.p
    for bootstub in (False, True):
      p.reset(enter_bootstub=bootstub)
      with patch('panda.python.spi.PandaSpiHandle.PROTOCOL_VERSION', return_value="abc"):
        # list should still work with wrong version
        assert p._serial in Panda.list()

        # connect but raise protocol error
        with self.assertRaises(PandaProtocolMismatch):
          Panda(p._serial)

  def test_protocol_version_data(self):
    p = self.p
    for bootstub in (False, True):
      p.reset(enter_bootstub=bootstub)
      v = p._handle.get_protocol_version()

      uid = binascii.hexlify(v[:12]).decode()
      assert uid == p.get_uid()

      hwtype = v[12]
      assert hwtype == ord(p.get_type())

      bstub = v[13]
      assert bstub == (0xEE if bootstub else 0xCC)

  def test_all_comm_types(self):
    p = self.p
    spy = self.enterContext(patch.object(p._handle, '_wait_for_ack', wraps=p._handle._wait_for_ack))

    # controlRead + controlWrite
    p.health()
    p.can_clear(0)
    assert spy.call_count == 2*2

    # bulkRead + bulkWrite
    p.can_recv()
    p.can_send(0x123, b"somedata", 0)
    assert spy.call_count == 2*4

  def test_bad_header(self):
    p = self.p
    with patch('panda.python.spi.SYNC', return_value=0):
      with self.assertRaises(PandaSpiNackResponse):
        p._handle.controlRead(Panda.REQUEST_IN, 0xd2, 0, 0, p.HEALTH_STRUCT.size, timeout=50)
    self._ping(p)

  def test_bad_checksum(self):
    p = self.p
    cnt = p.health()['spi_error_count']
    with patch('panda.python.spi.PandaSpiHandle._calc_checksum', return_value=0):
      with self.assertRaises(PandaSpiNackResponse):
        p._handle.controlRead(Panda.REQUEST_IN, 0xd2, 0, 0, p.HEALTH_STRUCT.size, timeout=50)
    self._ping(p)
    assert (p.health()['spi_error_count'] - cnt) > 0

  def test_non_existent_endpoint(self):
    p = self.p
    for _ in range(10):
      ep = random.randint(4, 20)
      with self.assertRaises(PandaSpiNackResponse):
        p._handle.bulkRead(ep, random.randint(1, 1000), timeout=50)

      self._ping(p)

      with self.assertRaises(PandaSpiNackResponse):
        p._handle.bulkWrite(ep, b"abc", timeout=50)

      self._ping(p)
