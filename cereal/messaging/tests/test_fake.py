import os
import unittest
import multiprocessing
import platform
from parameterized import parameterized_class
from typing import Optional

import cereal.messaging as messaging

WAIT_TIMEOUT = 5


@unittest.skipIf(platform.system() == "Darwin", "Events not supported on macOS")
class TestEvents(unittest.TestCase):

  def test_mutation(self):
    handle = messaging.fake_event_handle("carState")
    event = handle.recv_called_event

    self.assertFalse(event.peek())
    event.set()
    self.assertTrue(event.peek())
    event.clear()
    self.assertFalse(event.peek())

    del event

  def test_wait(self):
    handle = messaging.fake_event_handle("carState")
    event = handle.recv_called_event

    event.set()
    try:
      event.wait(WAIT_TIMEOUT)
      self.assertTrue(event.peek())
    except RuntimeError:
      self.fail("event.wait() timed out")

  def test_wait_multiprocess(self):
    handle = messaging.fake_event_handle("carState")
    event = handle.recv_called_event

    def set_event_run():
      event.set()

    try:
      p = multiprocessing.Process(target=set_event_run)
      p.start()
      event.wait(WAIT_TIMEOUT)
      self.assertTrue(event.peek())
    except RuntimeError:
      self.fail("event.wait() timed out")

    p.kill()

  def test_wait_zero_timeout(self):
    handle = messaging.fake_event_handle("carState")
    event = handle.recv_called_event

    try:
      event.wait(0)
      self.fail("event.wait() did not time out")
    except RuntimeError:
      self.assertFalse(event.peek())


@unittest.skipIf(platform.system() == "Darwin", "FakeSockets not supported on macOS")
@unittest.skipIf("ZMQ" in os.environ, "FakeSockets not supported on ZMQ")
@parameterized_class([{"prefix": None}, {"prefix": "test"}])
class TestFakeSockets(unittest.TestCase):
  prefix: Optional[str] = None

  def setUp(self):
    messaging.toggle_fake_events(True)
    if self.prefix is not None:
      messaging.set_fake_prefix(self.prefix)
    else:
      messaging.delete_fake_prefix()

  def tearDown(self):
    messaging.toggle_fake_events(False)
    messaging.delete_fake_prefix()

  def test_event_handle_init(self):
    handle = messaging.fake_event_handle("controlsState", override=True)

    self.assertFalse(handle.enabled)
    self.assertGreaterEqual(handle.recv_called_event.fd, 0)
    self.assertGreaterEqual(handle.recv_ready_event.fd, 0)

  def test_non_managed_socket_state(self):
    # non managed socket should have zero state
    _ = messaging.pub_sock("ubloxGnss")

    handle = messaging.fake_event_handle("ubloxGnss", override=False)

    self.assertFalse(handle.enabled)
    self.assertEqual(handle.recv_called_event.fd, 0)
    self.assertEqual(handle.recv_ready_event.fd, 0)

  def test_managed_socket_state(self):
    # managed socket should not change anything about the state
    handle = messaging.fake_event_handle("ubloxGnss")
    handle.enabled = True

    expected_enabled = handle.enabled
    expected_recv_called_fd = handle.recv_called_event.fd
    expected_recv_ready_fd = handle.recv_ready_event.fd

    _ = messaging.pub_sock("ubloxGnss")

    self.assertEqual(handle.enabled, expected_enabled)
    self.assertEqual(handle.recv_called_event.fd, expected_recv_called_fd)
    self.assertEqual(handle.recv_ready_event.fd, expected_recv_ready_fd)

  def test_sockets_enable_disable(self):
    carState_handle = messaging.fake_event_handle("ubloxGnss", enable=True)
    recv_called = carState_handle.recv_called_event
    recv_ready = carState_handle.recv_ready_event

    pub_sock = messaging.pub_sock("ubloxGnss")
    sub_sock = messaging.sub_sock("ubloxGnss")

    try:
      carState_handle.enabled = True
      recv_ready.set()
      pub_sock.send(b"test")
      _ = sub_sock.receive()
      self.assertTrue(recv_called.peek())
      recv_called.clear()

      carState_handle.enabled = False
      recv_ready.set()
      pub_sock.send(b"test")
      _ = sub_sock.receive()
      self.assertFalse(recv_called.peek())
    except RuntimeError:
      self.fail("event.wait() timed out")

  def test_synced_pub_sub(self):
    def daemon_repub_process_run():
      pub_sock = messaging.pub_sock("ubloxGnss")
      sub_sock = messaging.sub_sock("carState")

      frame = -1
      while True:
        frame += 1
        msg = sub_sock.receive(non_blocking=True)
        if msg is None:
          print("none received")
          continue

        bts = frame.to_bytes(8, 'little')
        pub_sock.send(bts)

    carState_handle = messaging.fake_event_handle("carState", enable=True)
    recv_called = carState_handle.recv_called_event
    recv_ready = carState_handle.recv_ready_event

    p = multiprocessing.Process(target=daemon_repub_process_run)
    p.start()

    pub_sock = messaging.pub_sock("carState")
    sub_sock = messaging.sub_sock("ubloxGnss")

    try:
      for i in range(10):
        recv_called.wait(WAIT_TIMEOUT)
        recv_called.clear()

        if i == 0:
          sub_sock.receive(non_blocking=True)

        bts = i.to_bytes(8, 'little')
        pub_sock.send(bts)

        recv_ready.set()
        recv_called.wait(WAIT_TIMEOUT)

        msg = sub_sock.receive(non_blocking=True)
        self.assertIsNotNone(msg)
        self.assertEqual(len(msg), 8)

        frame = int.from_bytes(msg, 'little')
        self.assertEqual(frame, i)
    except RuntimeError:
      self.fail("event.wait() timed out")
    finally:
      p.kill()


if __name__ == "__main__":
  unittest.main()
