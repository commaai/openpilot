import threading
import time
import tempfile
import shutil
import unittest

from common.params import Params, ParamKeyType, UnknownKeyName, put_nonblocking

class TestParams(unittest.TestCase):
  def setUp(self):
    self.tmpdir = tempfile.mkdtemp()
    print("using", self.tmpdir)
    self.params = Params(self.tmpdir)

  def tearDown(self):
    shutil.rmtree(self.tmpdir)

  def test_params_put_and_get(self):
    self.params.put("DongleId", "cb38263377b873ee")
    assert self.params.get("DongleId") == b"cb38263377b873ee"

  def test_params_non_ascii(self):
    st = b"\xe1\x90\xff"
    self.params.put("CarParams", st)
    assert self.params.get("CarParams") == st

  def test_params_get_cleared_panda_disconnect(self):
    self.params.put("CarParams", "test")
    self.params.put("DongleId", "cb38263377b873ee")
    assert self.params.get("CarParams") == b"test"
    self.params.clear_all(ParamKeyType.CLEAR_ON_PANDA_DISCONNECT)
    assert self.params.get("CarParams") is None
    assert self.params.get("DongleId") is not None

  def test_params_get_cleared_manager_start(self):
    self.params.put("CarParams", "test")
    self.params.put("DongleId", "cb38263377b873ee")
    assert self.params.get("CarParams") == b"test"
    self.params.clear_all(ParamKeyType.CLEAR_ON_MANAGER_START)
    assert self.params.get("CarParams") is None
    assert self.params.get("DongleId") is not None

  def test_params_two_things(self):
    self.params.put("DongleId", "bob")
    self.params.put("AthenadPid", "123")
    assert self.params.get("DongleId") == b"bob"
    assert self.params.get("AthenadPid") == b"123"

  def test_params_get_block(self):
    def _delayed_writer():
      time.sleep(0.1)
      self.params.put("CarParams", "test")
    threading.Thread(target=_delayed_writer).start()
    assert self.params.get("CarParams") is None
    assert self.params.get("CarParams", True) == b"test"

  def test_params_unknown_key_fails(self):
    with self.assertRaises(UnknownKeyName):
      self.params.get("swag")

    with self.assertRaises(UnknownKeyName):
      self.params.get_bool("swag")

    with self.assertRaises(UnknownKeyName):
      self.params.put("swag", "abc")

    with self.assertRaises(UnknownKeyName):
      self.params.put_bool("swag", True)

  def test_delete_not_there(self):
    assert self.params.get("CarParams") is None
    self.params.delete("CarParams")
    assert self.params.get("CarParams") is None

  def test_get_bool(self):
    self.params.delete("IsMetric")
    self.assertFalse(self.params.get_bool("IsMetric"))

    self.params.put_bool("IsMetric", True)
    self.assertTrue(self.params.get_bool("IsMetric"))

    self.params.put_bool("IsMetric", False)
    self.assertFalse(self.params.get_bool("IsMetric"))

    self.params.put("IsMetric", "1")
    self.assertTrue(self.params.get_bool("IsMetric"))

    self.params.put("IsMetric", "0")
    self.assertFalse(self.params.get_bool("IsMetric"))

  def test_put_non_blocking_with_get_block(self):
    q = Params(self.tmpdir)
    def _delayed_writer():
      time.sleep(0.1)
      put_nonblocking("CarParams", "test", self.tmpdir)
    threading.Thread(target=_delayed_writer).start()
    assert q.get("CarParams") is None
    assert q.get("CarParams", True) == b"test"


if __name__ == "__main__":
  unittest.main()
