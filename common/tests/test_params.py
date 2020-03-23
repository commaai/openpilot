from common.params import Params, UnknownKeyName
import threading
import time
import tempfile
import shutil
import unittest


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
    self.params.panda_disconnect()
    assert self.params.get("CarParams") is None
    assert self.params.get("DongleId") is not None

  def test_params_get_cleared_manager_start(self):
    self.params.put("CarParams", "test")
    self.params.put("DongleId", "cb38263377b873ee")
    assert self.params.get("CarParams") == b"test"
    self.params.manager_start()
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


if __name__ == "__main__":
  unittest.main()
