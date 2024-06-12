#!/usr/bin/env python3
import os
import capnp
import multiprocessing
import numbers
import random
import threading
import time
import unittest
from parameterized import parameterized

from cereal import log, car
import cereal.messaging as messaging
from cereal.services import SERVICE_LIST

events = [evt for evt in log.Event.schema.union_fields if evt in SERVICE_LIST.keys()]

def random_sock():
  return random.choice(events)

def random_socks(num_socks=10):
  return list({random_sock() for _ in range(num_socks)})

def random_bytes(length=1000):
  return bytes([random.randrange(0xFF) for _ in range(length)])

def zmq_sleep(t=1):
  if "ZMQ" in os.environ:
    time.sleep(t)

def zmq_expected_failure(func):
  if "ZMQ" in os.environ:
    return unittest.expectedFailure(func)
  else:
    return func


# TODO: this should take any capnp struct and returrn a msg with random populated data
def random_carstate():
  fields = ["vEgo", "aEgo", "gas", "steeringAngleDeg"]
  msg = messaging.new_message("carState")
  cs = msg.carState
  for f in fields:
    setattr(cs, f, random.random() * 10)
  return msg

# TODO: this should compare any capnp structs
def assert_carstate(cs1, cs2):
  for f in car.CarState.schema.non_union_fields:
    # TODO: check all types
    val1, val2 = getattr(cs1, f), getattr(cs2, f)
    if isinstance(val1, numbers.Number):
      assert val1 == val2, f"{f}: sent '{val1}' vs recvd '{val2}'"

def delayed_send(delay, sock, dat):
  def send_func():
    sock.send(dat)
  threading.Timer(delay, send_func).start()


class TestMessaging(unittest.TestCase):
  def setUp(self):
    # TODO: ZMQ tests are too slow; all sleeps will need to be
    # replaced with logic to block on the necessary condition
    if "ZMQ" in os.environ:
      raise unittest.SkipTest

    # ZMQ pub socket takes too long to die
    # sleep to prevent multiple publishers error between tests
    zmq_sleep()

  @parameterized.expand(events)
  def test_new_message(self, evt):
    try:
      msg = messaging.new_message(evt)
    except capnp.lib.capnp.KjException:
      msg = messaging.new_message(evt, random.randrange(200))
    self.assertLess(time.monotonic() - msg.logMonoTime, 0.1)
    self.assertFalse(msg.valid)
    self.assertEqual(evt, msg.which())

  @parameterized.expand(events)
  def test_pub_sock(self, evt):
    messaging.pub_sock(evt)

  @parameterized.expand(events)
  def test_sub_sock(self, evt):
    messaging.sub_sock(evt)

  @parameterized.expand([
    (messaging.drain_sock, capnp._DynamicStructReader),
    (messaging.drain_sock_raw, bytes),
  ])
  def test_drain_sock(self, func, expected_type):
    sock = "carState"
    pub_sock = messaging.pub_sock(sock)
    sub_sock = messaging.sub_sock(sock, timeout=1000)
    zmq_sleep()

    # no wait and no msgs in queue
    msgs = func(sub_sock)
    self.assertIsInstance(msgs, list)
    self.assertEqual(len(msgs), 0)

    # no wait but msgs are queued up
    num_msgs = random.randrange(3, 10)
    for _ in range(num_msgs):
      pub_sock.send(messaging.new_message(sock).to_bytes())
    time.sleep(0.1)
    msgs = func(sub_sock)
    self.assertIsInstance(msgs, list)
    self.assertTrue(all(isinstance(msg, expected_type) for msg in msgs))
    self.assertEqual(len(msgs), num_msgs)

  def test_recv_sock(self):
    sock = "carState"
    pub_sock = messaging.pub_sock(sock)
    sub_sock = messaging.sub_sock(sock, timeout=100)
    zmq_sleep()

    # no wait and no msg in queue, socket should timeout
    recvd = messaging.recv_sock(sub_sock)
    self.assertTrue(recvd is None)

    # no wait and one msg in queue
    msg = random_carstate()
    pub_sock.send(msg.to_bytes())
    time.sleep(0.01)
    recvd = messaging.recv_sock(sub_sock)
    self.assertIsInstance(recvd, capnp._DynamicStructReader)
    # https://github.com/python/mypy/issues/13038
    assert_carstate(msg.carState, recvd.carState)

  def test_recv_one(self):
    sock = "carState"
    pub_sock = messaging.pub_sock(sock)
    sub_sock = messaging.sub_sock(sock, timeout=1000)
    zmq_sleep()

    # no msg in queue, socket should timeout
    recvd = messaging.recv_one(sub_sock)
    self.assertTrue(recvd is None)

    # one msg in queue
    msg = random_carstate()
    pub_sock.send(msg.to_bytes())
    recvd = messaging.recv_one(sub_sock)
    self.assertIsInstance(recvd, capnp._DynamicStructReader)
    assert_carstate(msg.carState, recvd.carState)

  @zmq_expected_failure
  def test_recv_one_or_none(self):
    sock = "carState"
    pub_sock = messaging.pub_sock(sock)
    sub_sock = messaging.sub_sock(sock)
    zmq_sleep()

    # no msg in queue, socket shouldn't block
    recvd = messaging.recv_one_or_none(sub_sock)
    self.assertTrue(recvd is None)

    # one msg in queue
    msg = random_carstate()
    pub_sock.send(msg.to_bytes())
    recvd = messaging.recv_one_or_none(sub_sock)
    self.assertIsInstance(recvd, capnp._DynamicStructReader)
    assert_carstate(msg.carState, recvd.carState)

  def test_recv_one_retry(self):
    sock = "carState"
    sock_timeout = 0.1
    pub_sock = messaging.pub_sock(sock)
    sub_sock = messaging.sub_sock(sock, timeout=round(sock_timeout*1000))
    zmq_sleep()

    # this test doesn't work with ZMQ since multiprocessing interrupts it
    if "ZMQ" not in os.environ:
      # wait 15 socket timeouts and make sure it's still retrying
      p = multiprocessing.Process(target=messaging.recv_one_retry, args=(sub_sock,))
      p.start()
      time.sleep(sock_timeout*15)
      self.assertTrue(p.is_alive())
      p.terminate()

    # wait 15 socket timeouts before sending
    msg = random_carstate()
    delayed_send(sock_timeout*15, pub_sock, msg.to_bytes())
    start_time = time.monotonic()
    recvd = messaging.recv_one_retry(sub_sock)
    self.assertGreaterEqual(time.monotonic() - start_time, sock_timeout*15)
    self.assertIsInstance(recvd, capnp._DynamicStructReader)
    assert_carstate(msg.carState, recvd.carState)

if __name__ == "__main__":
  unittest.main()
