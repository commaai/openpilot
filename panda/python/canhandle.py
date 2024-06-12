import struct
import signal

from .base import BaseHandle


class CanHandle(BaseHandle):
  def __init__(self, p, bus):
    self.p = p
    self.bus = bus

  def transact(self, dat):
    def _handle_timeout(signum, frame):
      # will happen on reset or can error
      raise TimeoutError

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(1)

    try:
      self.p.isotp_send(1, dat, self.bus, recvaddr=2)
    finally:
      signal.alarm(0)

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(1)
    try:
      ret = self.p.isotp_recv(2, self.bus, sendaddr=1)
    finally:
      signal.alarm(0)

    return ret

  def close(self):
    pass

  def controlWrite(self, request_type, request, value, index, data, timeout=0, expect_disconnect=False):
    # ignore data in reply, panda doesn't use it
    return self.controlRead(request_type, request, value, index, 0, timeout)

  def controlRead(self, request_type, request, value, index, length, timeout=0):
    dat = struct.pack("HHBBHHH", 0, 0, request_type, request, value, index, length)
    return self.transact(dat)

  def bulkWrite(self, endpoint, data, timeout=0):
    if len(data) > 0x10:
      raise ValueError("Data must not be longer than 0x10")
    dat = struct.pack("HH", endpoint, len(data)) + data
    return self.transact(dat)

  def bulkRead(self, endpoint, length, timeout=0):
    dat = struct.pack("HH", endpoint, 0)
    return self.transact(dat)
