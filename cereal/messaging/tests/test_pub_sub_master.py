import random
import time
from typing import Sized, cast

import cereal.messaging as messaging
from cereal.messaging.tests.test_messaging import events, random_sock, random_socks, \
                                                  random_bytes, random_carstate, assert_carstate, \
                                                  zmq_sleep


class TestSubMaster:

  def setup_method(self):
    # ZMQ pub socket takes too long to die
    # sleep to prevent multiple publishers error between tests
    zmq_sleep(3)

  def test_init(self):
    sm = messaging.SubMaster(events)
    for p in [sm.updated, sm.recv_time, sm.recv_frame, sm.alive,
              sm.sock, sm.data, sm.logMonoTime, sm.valid]:
      assert len(cast(Sized, p)) == len(events)

  def test_init_state(self):
    socks = random_socks()
    sm = messaging.SubMaster(socks)
    assert sm.frame == -1
    assert not any(sm.updated.values())
    assert not any(sm.alive.values())
    assert all(t == 0. for t in sm.recv_time.values())
    assert all(f == 0 for f in sm.recv_frame.values())
    assert all(t == 0 for t in sm.logMonoTime.values())

    for p in [sm.updated, sm.recv_time, sm.recv_frame, sm.alive,
              sm.sock, sm.data, sm.logMonoTime, sm.valid]:
      assert len(cast(Sized, p)) == len(socks)

  def test_getitem(self):
    sock = "carState"
    pub_sock = messaging.pub_sock(sock)
    sm = messaging.SubMaster([sock,])
    zmq_sleep()

    msg = random_carstate()
    pub_sock.send(msg.to_bytes())
    sm.update(1000)
    assert_carstate(msg.carState, sm[sock])

  # TODO: break this test up to individually test SubMaster.update and SubMaster.update_msgs
  def test_update(self):
    sock = "carState"
    pub_sock = messaging.pub_sock(sock)
    sm = messaging.SubMaster([sock,])
    zmq_sleep()

    for i in range(10):
      msg = messaging.new_message(sock)
      pub_sock.send(msg.to_bytes())
      sm.update(1000)
      assert sm.frame == i
      assert all(sm.updated.values())

  def test_update_timeout(self):
    sock = random_sock()
    sm = messaging.SubMaster([sock,])
    timeout = random.randrange(1000, 3000)
    start_time = time.monotonic()
    sm.update(timeout)
    t = time.monotonic() - start_time
    assert t >= timeout/1000.
    assert t < 3
    assert not any(sm.updated.values())

  def test_avg_frequency_checks(self):
    for poll in (True, False):
      sm = messaging.SubMaster(["modelV2", "carParams", "carState", "cameraOdometry", "liveCalibration"],
                               poll=("modelV2" if poll else None),
                               frequency=(20. if not poll else None))

      checks = {
        "carState": (20, 20),
        "modelV2": (20, 20 if poll else 10),
        "cameraOdometry": (20, 10),
        "liveCalibration": (4, 4),
        "carParams": (None, None),
      }

      for service, (max_freq, min_freq) in checks.items():
        if max_freq is not None:
          assert sm._check_avg_freq(service)
          assert sm.freq_tracker[service].max_freq == max_freq*1.2
          assert sm.freq_tracker[service].min_freq == min_freq*0.8
        else:
          assert not sm._check_avg_freq(service)

  def test_alive(self):
    pass

  def test_ignore_alive(self):
    pass

  def test_valid(self):
    pass

  # SubMaster should always conflate
  def test_conflate(self):
    sock = "carState"
    pub_sock = messaging.pub_sock(sock)
    sm = messaging.SubMaster([sock,])

    n = 10
    for i in range(n+1):
      msg = messaging.new_message(sock)
      msg.carState.vEgo = i
      pub_sock.send(msg.to_bytes())
      time.sleep(0.01)
    sm.update(1000)
    assert sm[sock].vEgo == n


class TestPubMaster:

  def setup_method(self):
    # ZMQ pub socket takes too long to die
    # sleep to prevent multiple publishers error between tests
    zmq_sleep(3)

  def test_init(self):
    messaging.PubMaster(events)

  def test_send(self):
    socks = random_socks()
    pm = messaging.PubMaster(socks)
    sub_socks = {s: messaging.sub_sock(s, conflate=True, timeout=1000) for s in socks}
    zmq_sleep()

    # PubMaster accepts either a capnp msg builder or bytes
    for capnp in [True, False]:
      for i in range(100):
        sock = socks[i % len(socks)]

        if capnp:
          try:
            msg = messaging.new_message(sock)
          except Exception:
            msg = messaging.new_message(sock, random.randrange(50))
        else:
          msg = random_bytes()

        pm.send(sock, msg)
        recvd = sub_socks[sock].receive()

        if capnp:
          msg.clear_write_flag()
          msg = msg.to_bytes()
        assert msg == recvd, i
