import os
import random
import time
import string
import msgq


def random_sock():
  return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

def random_bytes(length=1000):
  return bytes([random.randrange(0xFF) for _ in range(length)])

def zmq_sleep(t=1):
  if "ZMQ" in os.environ:
    time.sleep(t)

class TestPubSubSockets:

  def setup_method(self):
    # ZMQ pub socket takes too long to die
    # sleep to prevent multiple publishers error between tests
    zmq_sleep()

  def test_pub_sub(self):
    sock = random_sock()
    pub_sock = msgq.pub_sock(sock)
    sub_sock = msgq.sub_sock(sock, conflate=False, timeout=None)
    zmq_sleep(3)

    for _ in range(1000):
      msg = random_bytes()
      pub_sock.send(msg)
      recvd = sub_sock.receive()
      assert msg == recvd

  def test_conflate(self):
    sock = random_sock()
    pub_sock = msgq.pub_sock(sock)
    for conflate in [True, False]:
      for _ in range(10):
        num_msgs = random.randint(3, 10)
        sub_sock = msgq.sub_sock(sock, conflate=conflate, timeout=None)
        zmq_sleep()

        sent_msgs = []
        for __ in range(num_msgs):
          msg = random_bytes()
          pub_sock.send(msg)
          sent_msgs.append(msg)
        time.sleep(0.1)
        recvd_msgs = msgq.drain_sock_raw(sub_sock)
        if conflate:
          assert len(recvd_msgs) == 1
        else:
          # TODO: compare actual data
          assert len(recvd_msgs) == len(sent_msgs)

  def test_receive_timeout(self):
    sock = random_sock()
    for _ in range(10):
      timeout = random.randrange(200)
      sub_sock = msgq.sub_sock(sock, timeout=timeout)
      zmq_sleep()

      start_time = time.monotonic()
      recvd = sub_sock.receive()
      assert (time.monotonic() - start_time) < 0.2
      assert recvd is None
