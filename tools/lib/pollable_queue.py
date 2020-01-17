import os
import select
import fcntl
from itertools import count
from collections import deque

Empty = OSError
Full = OSError
ExistentialError = OSError

class PollableQueue(object):
  """A Queue that you can poll().
     Only works with a single producer.
  """
  def __init__(self, maxlen=None):
    with open("/proc/sys/fs/pipe-max-size") as f:
      max_maxlen = int(f.read().rstrip())

    if maxlen is None:
      maxlen = max_maxlen
    else:
      maxlen = min(maxlen, max_maxlen)

    self._maxlen = maxlen
    self._q = deque()
    self._get_fd, self._put_fd = os.pipe()
    fcntl.fcntl(self._get_fd, fcntl.F_SETFL, os.O_NONBLOCK)
    fcntl.fcntl(self._put_fd, fcntl.F_SETFL, os.O_NONBLOCK)

    fcntl.fcntl(self._get_fd, fcntl.F_SETLEASE + 7, self._maxlen)
    fcntl.fcntl(self._put_fd, fcntl.F_SETLEASE + 7, self._maxlen)

    get_poller = select.epoll()
    put_poller = select.epoll()
    get_poller.register(self._get_fd, select.EPOLLIN)
    put_poller.register(self._put_fd, select.EPOLLOUT)

    self._get_poll = get_poller.poll
    self._put_poll = put_poller.poll


  def get_fd(self):
    return self._get_fd

  def put_fd(self):
    return self._put_fd

  def put(self, item, block=True, timeout=None):
    if block:
      while self._put_poll(timeout if timeout is not None else -1):
        try:
          # TODO: This is broken for multiple push threads when the queue is full.
          return self.put_nowait(item)
        except OSError as e:
          if e.errno != 11:
            raise

        raise Full()
    else:
      return self.put_nowait(item)

  def put_nowait(self, item):
    self._q.appendleft(item)
    os.write(self._put_fd, b"\x00")

  def get(self, block=True, timeout=None):
    if block:
      while self._get_poll(timeout if timeout is not None else -1):
        try:
          return self.get_nowait()
        except OSError as e:
          if e.errno != 11:
            raise

      raise Empty()
    else:
      return self.get_nowait()

  def get_nowait(self):
    os.read(self._get_fd, 1)
    return self._q.pop()

  def get_multiple(self, block=True, timeout=None):
    if block:
      if self._get_poll(timeout if timeout is not None else -1):
        return self.get_multiple_nowait()
      else:
        raise Empty()
    else:
      return self.get_multiple_nowait()

  def get_multiple_nowait(self, max_messages=None):
    num_read = len(os.read(self._get_fd, max_messages or self._maxlen))
    return [self._q.pop() for _ in range(num_read)]

  def empty(self):
    return len(self._q) == 0

  def full(self):
    return len(self._q) >= self._maxlen

  def close(self):
    os.close(self._get_fd)
    os.close(self._put_fd)

  def __len__(self):
    return len(self._q)