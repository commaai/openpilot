import pytest
import os
import multiprocessing
import platform
import msgq
from parameterized import parameterized_class
from typing import Optional

WAIT_TIMEOUT = 5


@pytest.mark.skipif(condition=platform.system() == "Darwin", reason="Events not supported on macOS")
class TestEvents:

  def test_mutation(self):
    handle = msgq.fake_event_handle("carState")
    event = handle.recv_called_event

    assert not event.peek()
    event.set()
    assert event.peek()
    event.clear()
    assert not event.peek()

    del event

  def test_wait(self):
    handle = msgq.fake_event_handle("carState")
    event = handle.recv_called_event

    event.set()
    try:
      event.wait(WAIT_TIMEOUT)
      assert event.peek()
    except RuntimeError:
      pytest.fail("event.wait() timed out")

  def test_wait_multiprocess(self):
    handle = msgq.fake_event_handle("carState")
    event = handle.recv_called_event

    def set_event_run():
      event.set()

    try:
      p = multiprocessing.Process(target=set_event_run)
      p.start()
      event.wait(WAIT_TIMEOUT)
      assert event.peek()
    except RuntimeError:
      pytest.fail("event.wait() timed out")

    p.kill()

  def test_wait_zero_timeout(self):
    handle = msgq.fake_event_handle("carState")
    event = handle.recv_called_event

    try:
      event.wait(0)
      pytest.fail("event.wait() did not time out")
    except RuntimeError:
      assert not event.peek()


@pytest.mark.skipif(condition=platform.system() == "Darwin", reason="FakeSockets not supported on macOS")
@pytest.mark.skipif(condition="ZMQ" in os.environ, reason="FakeSockets not supported on ZMQ")
@parameterized_class([{"prefix": None}, {"prefix": "test"}])
class TestFakeSockets:
  prefix: Optional[str] = None

  def setup_method(self):
    msgq.toggle_fake_events(True)
    if self.prefix is not None:
      msgq.set_fake_prefix(self.prefix)
    else:
      msgq.delete_fake_prefix()

  def teardown_method(self):
    msgq.toggle_fake_events(False)
    msgq.delete_fake_prefix()

  def test_event_handle_init(self):
    handle = msgq.fake_event_handle("controlsState", override=True)

    assert not handle.enabled
    assert handle.recv_called_event.fd >= 0
    assert handle.recv_ready_event.fd >= 0

  def test_non_managed_socket_state(self):
    # non managed socket should have zero state
    _ = msgq.pub_sock("ubloxGnss")

    handle = msgq.fake_event_handle("ubloxGnss", override=False)

    assert not handle.enabled
    assert handle.recv_called_event.fd == 0
    assert handle.recv_ready_event.fd == 0

  def test_managed_socket_state(self):
    # managed socket should not change anything about the state
    handle = msgq.fake_event_handle("ubloxGnss")
    handle.enabled = True

    expected_enabled = handle.enabled
    expected_recv_called_fd = handle.recv_called_event.fd
    expected_recv_ready_fd = handle.recv_ready_event.fd

    _ = msgq.pub_sock("ubloxGnss")

    assert handle.enabled == expected_enabled
    assert handle.recv_called_event.fd == expected_recv_called_fd
    assert handle.recv_ready_event.fd == expected_recv_ready_fd

  def test_sockets_enable_disable(self):
    carState_handle = msgq.fake_event_handle("ubloxGnss", enable=True)
    recv_called = carState_handle.recv_called_event
    recv_ready = carState_handle.recv_ready_event

    pub_sock = msgq.pub_sock("ubloxGnss")
    sub_sock = msgq.sub_sock("ubloxGnss")

    try:
      carState_handle.enabled = True
      recv_ready.set()
      pub_sock.send(b"test")
      _ = sub_sock.receive()
      assert recv_called.peek()
      recv_called.clear()

      carState_handle.enabled = False
      recv_ready.set()
      pub_sock.send(b"test")
      _ = sub_sock.receive()
      assert not recv_called.peek()
    except RuntimeError:
      pytest.fail("event.wait() timed out")

  def test_synced_pub_sub(self):
    def daemon_repub_process_run():
      pub_sock = msgq.pub_sock("ubloxGnss")
      sub_sock = msgq.sub_sock("carState")

      frame = -1
      while True:
        frame += 1
        msg = sub_sock.receive(non_blocking=True)
        if msg is None:
          print("none received")
          continue

        bts = frame.to_bytes(8, 'little')
        pub_sock.send(bts)

    carState_handle = msgq.fake_event_handle("carState", enable=True)
    recv_called = carState_handle.recv_called_event
    recv_ready = carState_handle.recv_ready_event

    p = multiprocessing.Process(target=daemon_repub_process_run)
    p.start()

    pub_sock = msgq.pub_sock("carState")
    sub_sock = msgq.sub_sock("ubloxGnss")

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
        assert msg is not None
        assert len(msg) == 8

        frame = int.from_bytes(msg, 'little')
        assert frame == i
    except RuntimeError:
      pytest.fail("event.wait() timed out")
    finally:
      p.kill()
