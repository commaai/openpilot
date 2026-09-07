import fcntl
import mmap
import os
import struct


class ShmQueue:
  """Best-effort MPSC byte ring. Callers serialize threads sharing an instance.

  Layout shared with shm_queue.h: two native little-endian uint64 positions,
  followed by a byte ring of uint32-length-prefixed records. Leave one byte free
  to distinguish full from empty. Never unlink a queue while clients are alive.
  """
  CAPACITY = 64 * 1024 * 1024
  HEADER_SIZE = 16

  def __init__(self, path: str, capacity: int = CAPACITY, *, blocking: bool = True):
    self.capacity = capacity
    self.fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    try:
      fcntl.flock(self.fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
      size = self.HEADER_SIZE + capacity
      if os.fstat(self.fd).st_size == 0:
        os.ftruncate(self.fd, size)
      if os.fstat(self.fd).st_size != size:
        raise ValueError("shared memory queue size mismatch")
      self.mem = mmap.mmap(self.fd, size)
    except BaseException:
      os.close(self.fd)
      raise
    finally:
      # Closing the descriptor on failure already releases the lock.
      if hasattr(self, 'mem'):
        fcntl.flock(self.fd, fcntl.LOCK_UN)

  def close(self):
    self.mem.close()
    os.close(self.fd)

  def _copy(self, pos: int, data: bytes):
    first = min(len(data), self.capacity - pos)
    self.mem[16 + pos:16 + pos + first] = data[:first]
    self.mem[16:16 + len(data) - first] = data[first:]

  def _read(self, pos: int, size: int) -> bytes:
    first = min(size, self.capacity - pos)
    return self.mem[16 + pos:16 + pos + first] + self.mem[16:16 + size - first]

  def send(self, data: bytes) -> bool:
    if len(data) + 4 >= self.capacity:
      return False
    try:
      fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      return False
    try:
      read, write = struct.unpack_from('<QQ', self.mem)
      if len(data) + 4 > (read - write - 1) % self.capacity:
        return False
      self._copy(write, struct.pack('<I', len(data)))
      self._copy((write + 4) % self.capacity, data)
      # Publish only after the complete record has been copied.
      self.mem[8:16] = struct.pack('<Q', (write + 4 + len(data)) % self.capacity)
      return True
    finally:
      fcntl.flock(self.fd, fcntl.LOCK_UN)

  def receive(self) -> bytes | None:
    fcntl.flock(self.fd, fcntl.LOCK_EX)
    try:
      read, write = struct.unpack_from('<QQ', self.mem)
      if read == write:
        return None
      size, = struct.unpack('<I', self._read(read, 4))
      data = self._read((read + 4) % self.capacity, size)
      self.mem[0:8] = struct.pack('<Q', (read + 4 + size) % self.capacity)
      return data
    finally:
      fcntl.flock(self.fd, fcntl.LOCK_UN)
