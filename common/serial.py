import errno
import fcntl
import os
import select
import struct
import termios
import time

_DTR_BYTES = struct.pack("I", termios.TIOCM_DTR)
_ZERO_BYTES = struct.pack("I", 0)


class SerialException(IOError):
  """Raised on any serial port error."""


class Serial:
  def __init__(self, port: str, baudrate: int = 9600, *,
               rtscts: bool = False, dsrdtr: bool = False,
               timeout: float | None = None,
               exclusive: bool = False) -> None:
    self.port = port
    self._baudrate = baudrate
    self._rtscts = rtscts
    self._dsrdtr = dsrdtr
    self._timeout = timeout
    self._exclusive = exclusive
    self.fd: int | None = None
    self._readline_buf = b""
    self._open()

  # --- lifecycle ----------------------------------------------------

  def _open(self) -> None:
    # O_NONBLOCK on open() avoids blocking on DCD for tty devices.
    try:
      self.fd = os.open(self.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    except OSError as e:
      raise SerialException(f"could not open {self.port}: {e}") from e
    try:
      if self._exclusive:
        try:
          fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
          raise SerialException(f"could not lock {self.port}: {e}") from e
      self._apply_termios()
      # pyserial parity: when not using DTR/DSR flow control, assert DTR on open.
      # Pseudo-terminals don't support modem-control ioctls — ignore those.
      if not self._dsrdtr:
        try:
          self._set_modem_bits(termios.TIOCMBIS, _DTR_BYTES)
        except OSError as e:
          if e.errno not in (errno.EINVAL, errno.ENOTTY):
            raise
    except BaseException:
      os.close(self.fd)
      self.fd = None
      raise

  def close(self) -> None:
    if self.fd is not None:
      try:
        os.close(self.fd)
      finally:
        self.fd = None
        self._readline_buf = b""

  def __enter__(self):
    return self

  def __exit__(self, *_exc) -> None:
    self.close()

  def __del__(self) -> None:
    try:
      self.close()
    except Exception:
      pass

  # --- termios ------------------------------------------------------

  def _apply_termios(self) -> None:
    try:
      iflag, oflag, cflag, lflag, _ispeed, _ospeed, cc = termios.tcgetattr(self.fd)
    except termios.error as e:
      raise SerialException(f"tcgetattr failed on {self.port}: {e}") from e

    # raw mode: 8N1, ignore modem lines, no canonical/echo/signals
    cflag |= termios.CLOCAL | termios.CREAD
    cflag = (cflag & ~termios.CSIZE) | termios.CS8
    cflag &= ~termios.CSTOPB
    cflag &= ~(termios.PARENB | termios.PARODD)
    cflag = (cflag | termios.CRTSCTS) if self._rtscts else (cflag & ~termios.CRTSCTS)

    lflag &= ~(termios.ICANON | termios.ECHO | termios.ECHOE | termios.ECHOK |
               termios.ECHONL | termios.ISIG | termios.IEXTEN)
    oflag &= ~(termios.OPOST | termios.ONLCR | termios.OCRNL)
    iflag &= ~(termios.INLCR | termios.IGNCR | termios.ICRNL | termios.IGNBRK |
               termios.IXON | termios.IXOFF | termios.IXANY)

    speed = getattr(termios, f"B{self._baudrate}")
    cc[termios.VMIN], cc[termios.VTIME] = self._vmin_vtime()
    termios.tcsetattr(self.fd, termios.TCSANOW,
                      [iflag, oflag, cflag, lflag, speed, speed, cc])

  def _vmin_vtime(self) -> tuple[int, int]:
    """Default: kernel returns whatever's ready; timeout is enforced by select."""
    return 0, 0

  # --- I/O ----------------------------------------------------------

  def _read_into(self, out: bytearray, want: int, deadline: float | None) -> None:
    """Block (up to deadline) reading bytes into `out` until len reaches want."""
    while len(out) < want:
      left = None if deadline is None else max(0.0, deadline - time.monotonic())
      try:
        ready, _, _ = select.select([self.fd], [], [], left)
      except InterruptedError:
        continue
      if not ready:
        return
      try:
        chunk = os.read(self.fd, want - len(out))
      except OSError as e:
        if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EINTR):
          continue
        raise SerialException(f"read failed: {e}") from e
      if not chunk:
        raise SerialException(f"{self.port}: device disconnected")
      out.extend(chunk)

  def read(self, size: int = 1) -> bytes:
    self._require_open()
    deadline = None if self._timeout is None else time.monotonic() + self._timeout
    out = bytearray()
    self._read_into(out, size, deadline)
    return bytes(out)

  def readline(self) -> bytes:
    """Read until b'\\n' (included) or self.timeout. Returns whatever was buffered."""
    self._require_open()
    deadline = None if self._timeout is None else time.monotonic() + self._timeout
    while b"\n" not in self._readline_buf:
      left = None if deadline is None else max(0.0, deadline - time.monotonic())
      try:
        ready, _, _ = select.select([self.fd], [], [], left)
      except InterruptedError:
        continue
      if not ready:
        break
      try:
        chunk = os.read(self.fd, 4096)
      except OSError as e:
        if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EINTR):
          continue
        raise SerialException(f"readline failed: {e}") from e
      if not chunk:
        raise SerialException(f"{self.port}: device disconnected")
      self._readline_buf += chunk

    if b"\n" in self._readline_buf:
      line, _, self._readline_buf = self._readline_buf.partition(b"\n")
      return line + b"\n"
    line, self._readline_buf = self._readline_buf, b""
    return line

  def write(self, data: bytes) -> int:
    self._require_open()
    view = memoryview(data)
    written = 0
    while written < len(data):
      try:
        written += os.write(self.fd, view[written:])
      except BlockingIOError:
        select.select([], [self.fd], [])
      except InterruptedError:
        continue
      except OSError as e:
        raise SerialException(f"write failed: {e}") from e
    return written

  def flush(self) -> None:
    self._require_open()
    termios.tcdrain(self.fd)

  def reset_input_buffer(self) -> None:
    self._require_open()
    self._readline_buf = b""
    termios.tcflush(self.fd, termios.TCIFLUSH)

  def reset_output_buffer(self) -> None:
    self._require_open()
    termios.tcflush(self.fd, termios.TCOFLUSH)

  # --- modem control / baudrate -------------------------------------

  def _set_modem_bits(self, op: int, bits: bytes) -> None:
    fcntl.ioctl(self.fd, op, bits)

  @property
  def dtr(self) -> bool:
    self._require_open()
    s = fcntl.ioctl(self.fd, termios.TIOCMGET, _ZERO_BYTES)
    return bool(struct.unpack("I", s)[0] & termios.TIOCM_DTR)

  @dtr.setter
  def dtr(self, asserted: bool) -> None:
    self._require_open()
    self._set_modem_bits(termios.TIOCMBIS if asserted else termios.TIOCMBIC, _DTR_BYTES)

  @property
  def baudrate(self) -> int:
    return self._baudrate

  @baudrate.setter
  def baudrate(self, v: int) -> None:
    self._baudrate = v
    if self.fd is not None:
      self._apply_termios()

  @property
  def timeout(self) -> float | None:
    return self._timeout

  @timeout.setter
  def timeout(self, v: float | None) -> None:
    self._timeout = v
    # VTIMESerial bakes timeout into termios; plain Serial uses select().
    if self.fd is not None and self._vmin_vtime() != (0, 0):
      self._apply_termios()

  # --- helpers ------------------------------------------------------

  def _require_open(self) -> None:
    if self.fd is None:
      raise SerialException(f"{self.port}: port not open")


class VTIMESerial(Serial):
  def _vmin_vtime(self) -> tuple[int, int]:
    if self._timeout is None:
      return 1, 0                                          # block for >= 1 byte
    deciseconds = max(0, min(255, round(self._timeout * 10)))
    return 0, deciseconds                                  # 0..25.5s

  def _apply_termios(self) -> None:
    super()._apply_termios()
    # Blocking reads + VMIN/VTIME require clearing O_NONBLOCK we set at open().
    fcntl.fcntl(self.fd, fcntl.F_SETFL, 0)

  def read(self, size: int = 1) -> bytes:
    self._require_open()
    out = bytearray()
    while len(out) < size:
      try:
        chunk = os.read(self.fd, size - len(out))
      except InterruptedError:
        continue
      if not chunk:
        break  # VTIME elapsed with no data
      out.extend(chunk)
    return bytes(out)
