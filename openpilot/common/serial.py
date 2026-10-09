import errno
import fcntl
import os
import select
import struct
import termios
import time


# Modem control lines (linux/termios.h); fall back to common x86_64 values.
TIOCMBIS = getattr(termios, "TIOCMBIS", 0x5416)
TIOCMBIC = getattr(termios, "TIOCMBIC", 0x5417)
TIOCM_DTR = getattr(termios, "TIOCM_DTR", 0x002)
TIOCM_RTS = getattr(termios, "TIOCM_RTS", 0x004)
_TIOCM_DTR = struct.pack("I", TIOCM_DTR)
_TIOCM_RTS = struct.pack("I", TIOCM_RTS)


class SerialException(OSError):
  pass


class Serial:
  def __init__(self, port: str, baudrate: int = 9600, timeout: float | None = None, *,
               rtscts: bool = False, dsrdtr: bool = False, exclusive: bool = False):
    self._port = port
    self._baudrate = baudrate
    self._timeout = timeout
    self._rtscts = rtscts
    self._dsrdtr = dsrdtr
    self._exclusive = exclusive
    self._dtr = True
    self._fd = -1
    self.open()

  def __enter__(self):
    return self

  def __exit__(self, *args) -> None:
    self.close()

  @property
  def fd(self) -> int:
    self._ensure_open()
    return self._fd

  @property
  def baudrate(self) -> int:
    return self._baudrate

  @baudrate.setter
  def baudrate(self, value: int) -> None:
    self._baudrate = int(value)
    if self._fd >= 0:
      self._configure()

  @property
  def dtr(self) -> bool:
    return self._dtr

  @dtr.setter
  def dtr(self, value: bool) -> None:
    self._dtr = bool(value)
    if self._fd >= 0:
      self._set_line(TIOCM_DTR, _TIOCM_DTR, self._dtr)

  def open(self) -> None:
    if self._fd >= 0:
      return
    try:
      self._fd = os.open(self._port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    except OSError as e:
      self._fd = -1
      raise SerialException(e.errno, f"could not open port {self._port}: {e}") from e

    try:
      if self._exclusive:
        try:
          fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
          raise SerialException(e.errno, f"could not exclusively lock port {self._port}: {e}") from e

      self._configure()

      # When not using hardware DSR/DTR handshaking, drive lines ourselves.
      if not self._dsrdtr:
        try:
          self._set_line(TIOCM_DTR, _TIOCM_DTR, self._dtr)
          if not self._rtscts:
            self._set_line(TIOCM_RTS, _TIOCM_RTS, True)
        except OSError as e:
          if e.errno not in (errno.EINVAL, errno.ENOTTY):
            raise

      self.reset_input_buffer()
    except BaseException:
      self._close_fd()
      raise

  def close(self) -> None:
    self._close_fd()

  def read(self, size: int = 1) -> bytes:
    self._ensure_open()
    if size <= 0:
      return b""

    buf = bytearray()
    deadline = self._deadline()
    while len(buf) < size:
      remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
      if not self._wait_readable(remaining):
        break
      try:
        chunk = os.read(self._fd, size - len(buf))
      except InterruptedError:
        continue
      except OSError as e:
        if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
          if self._timeout == 0:
            break
          continue
        raise SerialException(e.errno, f"read failed: {e}") from e
      if not chunk:
        break
      buf.extend(chunk)
    return bytes(buf)

  def readline(self) -> bytes:
    self._ensure_open()
    buf = bytearray()
    deadline = self._deadline()
    while True:
      remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
      if deadline is not None and remaining == 0.0 and not buf:
        # match pyserial: timed-out readline returns empty
        if not self._wait_readable(0.0):
          return b""
      elif not self._wait_readable(remaining):
        return bytes(buf)

      try:
        chunk = os.read(self._fd, 1)
      except InterruptedError:
        continue
      except OSError as e:
        if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
          if self._timeout == 0:
            return bytes(buf)
          continue
        raise SerialException(e.errno, f"read failed: {e}") from e
      if not chunk:
        return bytes(buf)
      buf.extend(chunk)
      if chunk == b"\n":
        return bytes(buf)

  def write(self, data: bytes) -> int:
    self._ensure_open()
    if not data:
      return 0
    view = memoryview(data)
    total = 0
    while total < len(data):
      try:
        n = os.write(self._fd, view[total:])
      except InterruptedError:
        continue
      except OSError as e:
        if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
          select.select([], [self._fd], [], None)
          continue
        raise SerialException(e.errno, f"write failed: {e}") from e
      if n == 0:
        raise SerialException("write returned 0")
      total += n
    return total

  def flush(self) -> None:
    self._ensure_open()
    termios.tcdrain(self._fd)

  def reset_input_buffer(self) -> None:
    self._ensure_open()
    termios.tcflush(self._fd, termios.TCIFLUSH)

  def reset_output_buffer(self) -> None:
    self._ensure_open()
    termios.tcflush(self._fd, termios.TCOFLUSH)

  def _close_fd(self) -> None:
    if self._fd >= 0:
      try:
        if self._exclusive:
          fcntl.flock(self._fd, fcntl.LOCK_UN)
      except OSError:
        pass
      try:
        os.close(self._fd)
      except OSError:
        pass
      self._fd = -1

  def _ensure_open(self) -> None:
    if self._fd < 0:
      raise SerialException("port is not open")

  def _deadline(self) -> float | None:
    if self._timeout is None:
      return None
    if self._timeout == 0:
      return time.monotonic()
    return time.monotonic() + self._timeout

  def _wait_readable(self, timeout: float | None) -> bool:
    """Return True if fd is readable. timeout None blocks; 0 polls."""
    if timeout is not None and timeout < 0:
      timeout = 0.0
    try:
      ready, _, _ = select.select([self._fd], [], [], timeout)
    except InterruptedError:
      return False
    return bool(ready)

  def _baud_constant(self, baudrate: int) -> int:
    try:
      return getattr(termios, f"B{baudrate}")
    except AttributeError as e:
      raise ValueError(f"unsupported baud rate: {baudrate}") from e

  def _configure(self) -> None:
    self._ensure_open()
    try:
      attrs = termios.tcgetattr(self._fd)
    except termios.error as e:
      raise SerialException(f"could not get port attributes: {e}") from e

    iflag, oflag, cflag, lflag, _ispeed, _ospeed, cc = attrs

    # raw binary 8N1
    iflag = 0
    oflag = 0
    lflag = 0
    cflag |= termios.CLOCAL | termios.CREAD
    cflag &= ~termios.CSIZE
    cflag |= termios.CS8
    cflag &= ~(termios.PARENB | termios.PARODD | termios.CSTOPB)

    if hasattr(termios, "CRTSCTS"):
      if self._rtscts:
        cflag |= termios.CRTSCTS
      else:
        cflag &= ~termios.CRTSCTS

    speed = self._baud_constant(self._baudrate)
    cc = list(cc)
    # Non-blocking reads are handled via select + O_NONBLOCK; keep VMIN/VTIME at 0.
    cc[termios.VMIN] = 0
    cc[termios.VTIME] = 0

    try:
      termios.tcsetattr(self._fd, termios.TCSANOW, [iflag, oflag, cflag, lflag, speed, speed, cc])
    except termios.error as e:
      raise SerialException(f"could not configure port: {e}") from e

    # Keep the fd non-blocking so timeout=0 and select work consistently.
    flags = fcntl.fcntl(self._fd, fcntl.F_GETFL)
    fcntl.fcntl(self._fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

  def _set_line(self, _bit: int, packed: bytes, enabled: bool) -> None:
    request = TIOCMBIS if enabled else TIOCMBIC
    fcntl.ioctl(self._fd, request, packed)
