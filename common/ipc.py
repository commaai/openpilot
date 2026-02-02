import os
import socket


class PushSocket:
  """Non-blocking PUSH socket using Unix datagram sockets."""

  def __init__(self):
    self.sock: socket.socket | None = None
    self.path: str | None = None

  def connect(self, ipc_path: str):
    """Connect to a Unix domain socket.

    Args:
      ipc_path: Socket path in format 'ipc:///path/to/socket' or '/path/to/socket'
    """
    self.path = ipc_path.replace("ipc://", "")
    self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    self.sock.setblocking(False)

  def send(self, data: bytes) -> bool:
    """Send data to the socket (non-blocking).

    Returns True if sent successfully, False if dropped.
    Max message size is ~64KB on Linux (kernel limit for datagrams).
    """
    if self.sock is None or self.path is None:
      return False
    try:
      self.sock.sendto(data, self.path)
      return True
    except (BlockingIOError, ConnectionRefusedError, FileNotFoundError, OSError):
      # Drop message on any send error (matches ZMQ NOBLOCK behavior)
      # OSError includes EMSGSIZE (message too long) for large datagrams
      return False

  def close(self):
    if self.sock:
      self.sock.close()
      self.sock = None


class PullSocket:
  """Blocking/non-blocking PULL socket using Unix datagram sockets."""

  def __init__(self):
    self.sock: socket.socket | None = None
    self.path: str | None = None

  def bind(self, ipc_path: str):
    """Bind to a Unix domain socket.

    Args:
      ipc_path: Socket path in format 'ipc:///path/to/socket' or '/path/to/socket'
    """
    self.path = ipc_path.replace("ipc://", "")
    if os.path.exists(self.path):
      os.unlink(self.path)
    self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    self.sock.bind(self.path)

  def recv(self, flags: int = 0) -> bytes:
    """Receive data from the socket.

    Args:
      flags: If non-zero, use non-blocking mode

    Returns:
      Received bytes

    Raises:
      BlockingIOError: If non-blocking and no data available
    """
    if self.sock is None:
      raise RuntimeError("Socket not bound")
    if flags:  # NOBLOCK
      self.sock.setblocking(False)
    else:
      self.sock.setblocking(True)
    return self.sock.recv(65536)

  def close(self):
    if self.sock:
      self.sock.close()
      self.sock = None
    if self.path and os.path.exists(self.path):
      os.unlink(self.path)
