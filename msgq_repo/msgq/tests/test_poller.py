import pytest
import time
import msgq
import concurrent.futures

SERVICE_NAME = 'myService'

def poller():
  context = msgq.Context()

  p = msgq.Poller()

  sub = msgq.SubSocket()
  sub.connect(context, SERVICE_NAME)
  p.registerSocket(sub)

  socks = p.poll(10000)
  r = [s.receive(non_blocking=True) for s in socks]

  return r


class TestPoller:
  def test_poll_once(self):
    context = msgq.Context()

    pub = msgq.PubSocket()
    pub.connect(context, SERVICE_NAME)

    with concurrent.futures.ThreadPoolExecutor() as e:
      poll = e.submit(poller)

      time.sleep(0.1)  # Slow joiner syndrome

      # Send message
      pub.send(b"a")

      # Wait for poll result
      result = poll.result()

    del pub
    context.term()

    assert result == [b"a"]

  def test_poll_and_create_many_subscribers(self):
    context = msgq.Context()

    pub = msgq.PubSocket()
    pub.connect(context, SERVICE_NAME)

    with concurrent.futures.ThreadPoolExecutor() as e:
      poll = e.submit(poller)

      time.sleep(0.1)  # Slow joiner syndrome
      c = msgq.Context()
      for _ in range(10):
        msgq.SubSocket().connect(c, SERVICE_NAME)

      time.sleep(0.1)

      # Send message
      pub.send(b"a")

      # Wait for poll result
      result = poll.result()

    del pub
    context.term()

    assert result == [b"a"]

  def test_multiple_publishers_exception(self):
    context = msgq.Context()

    with pytest.raises(msgq.MultiplePublishersError):
      pub1 = msgq.PubSocket()
      pub1.connect(context, SERVICE_NAME)

      pub2 = msgq.PubSocket()
      pub2.connect(context, SERVICE_NAME)

      pub1.send(b"a")

    del pub1
    del pub2
    context.term()

  def test_multiple_messages(self):
    context = msgq.Context()

    pub = msgq.PubSocket()
    pub.connect(context, SERVICE_NAME)

    sub = msgq.SubSocket()
    sub.connect(context, SERVICE_NAME)

    time.sleep(0.1)  # Slow joiner

    for i in range(1, 100):
      pub.send(b'a'*i)

    msg_seen = False
    i = 1
    while True:
      r = sub.receive(non_blocking=True)

      if r is not None:
        assert b'a'*i == r

        msg_seen = True
        i += 1

      if r is None and msg_seen:  # ZMQ sometimes receives nothing on the first receive
        break

    del pub
    del sub
    context.term()

  def test_conflate(self):
    context = msgq.Context()

    pub = msgq.PubSocket()
    pub.connect(context, SERVICE_NAME)

    sub = msgq.SubSocket()
    sub.connect(context, SERVICE_NAME, conflate=True)

    time.sleep(0.1)  # Slow joiner
    pub.send(b'a')
    pub.send(b'b')

    assert b'b' == sub.receive()

    del pub
    del sub
    context.term()
