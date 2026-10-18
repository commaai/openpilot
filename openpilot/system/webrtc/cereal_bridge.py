import logging
import struct
import threading

from openpilot.cereal import log, messaging

# raw cereal over an ordered, reliable datachannel. the channel is treated as a byte
# stream of length-prefixed events, so events larger than the channel's max message size just span chunks.
HEADER = struct.Struct("<I")
MAX_BUFFERED_BYTES = 8 * 1024 * 1024

logger = logging.getLogger("cereal_bridge")


def pack(dat: bytes) -> bytes:
  return HEADER.pack(len(dat)) + dat


class Unpacker:
  def __init__(self):
    self.buf = bytearray()

  def feed(self, chunk: bytes) -> list[bytes]:
    self.buf += chunk
    events, pos = [], 0
    while pos + HEADER.size <= len(self.buf):
      end = pos + HEADER.size + HEADER.unpack_from(self.buf, pos)[0]
      if end > len(self.buf):
        break
      events.append(bytes(self.buf[pos + HEADER.size:end]))
      pos = end
    del self.buf[:pos]
    return events


class CerealBridge:
  """Forwards services_out from local msgq into the channel and publishes services_in received from it.
  Runs the same way on both ends of the connection."""

  def __init__(self, services_out: list[str], services_in: list[str]):
    self.channel = None
    self.services_out = list(services_out)
    self.pub_socks = {s: messaging.pub_sock(s) for s in services_in}
    self.unpacker = Unpacker()
    # upper bound on the channel's buffered bytes. DataChannel.buffered_amount() segfaults in libdatachannel-py,
    # but send() reports whether it had to buffer and the low callback reports when the buffer drained.
    self.pending = 0
    self.dropped = 0
    self.exit_event = threading.Event()
    self.thread = threading.Thread(target=self._send_thread, name="cereal_bridge", daemon=True)

  def start(self, channel):
    self.channel = channel
    self.channel.set_buffered_amount_low_threshold(0)
    self.channel.on_buffered_amount_low(self._on_drained)
    if self.services_out:
      self.thread.start()

  def _on_drained(self):
    self.pending = 0

  def stop(self):
    self.exit_event.set()
    if self.thread.is_alive():
      self.thread.join()

  def on_message(self, message: bytes | str):
    if not isinstance(message, bytes):
      return
    for dat in self.unpacker.feed(message):
      with log.Event.from_bytes(dat) as evt:
        service = evt.which()
      if service in self.pub_socks:
        self.pub_socks[service].send(dat)

  def _send_thread(self):
    poller = messaging.Poller()
    for s in self.services_out:
      messaging.sub_sock(s, poller)
    max_size = max(1024, min(self.channel.max_message_size(), 256 * 1024))

    while not self.exit_event.is_set():
      batch = bytearray()
      for sock in poller.poll(100):
        for dat in messaging.drain_sock_raw(sock):
          batch += pack(dat)
      if not batch or not self.channel.is_open():
        continue

      # the peer or the link can't keep up, drop instead of buffering without bound
      if self.pending > MAX_BUFFERED_BYTES:
        self.dropped += 1
        if self.dropped % 100 == 1:
          logger.warning("bridge is falling behind, dropped %d batches", self.dropped)
        continue

      try:
        for i in range(0, len(batch), max_size):
          chunk = bytes(batch[i:i + max_size])
          self.pending = 0 if self.channel.send(chunk) else self.pending + len(chunk)
      except Exception:
        logger.exception("bridge send failed")
        return
