import random
import time
import string
import msgq
import pytest

def random_sock():
  return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

def random_bytes(length=1000):
  return bytes([random.randrange(0xFF) for _ in range(length)])

class TestPubSubSockets:

  def test_pub_sub(self):
    sock = random_sock()
    pub_sock = msgq.pub_sock(sock)
    sub_sock = msgq.sub_sock(sock, conflate=False, timeout=None)

    for _ in range(1000):
      msg = random_bytes()
      pub_sock.send(msg)
      recvd = sub_sock.receive()
      assert msg == recvd

  def test_conflate(self):
    sock = random_sock()
    pub_sock = msgq.pub_sock(sock)
    for conflate in [True, False]:
      num_msgs = random.randint(3, 10)
      sub_sock = msgq.sub_sock(sock, conflate=conflate, timeout=None)

      sent_msgs = []
      for __ in range(num_msgs):
        msg = random_bytes()
        pub_sock.send(msg)
        sent_msgs.append(msg)
      time.sleep(0.1)
      recvd_msgs = msgq.drain_sock_raw(sub_sock)
      if conflate:
        assert len(recvd_msgs) == 1
        assert recvd_msgs[0] == sent_msgs[-1]
      else:
        assert len(recvd_msgs) == len(sent_msgs)
        for rec_msg, sent_msg in zip(recvd_msgs, sent_msgs):
          assert rec_msg == sent_msg

  @pytest.mark.flaky(retries=3, delay=1)
  def test_receive_timeout(self):
    sock = random_sock()
    timeout = random.randrange(200)
    sub_sock = msgq.sub_sock(sock, timeout=timeout)

    start_time = time.monotonic()
    recvd = sub_sock.receive()
    assert (time.monotonic() - start_time) < (timeout + 0.1)
    assert recvd is None
