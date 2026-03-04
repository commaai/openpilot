from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Iterable
from pathlib import Path

import serial

MODEM_STATE_PATH = Path("/dev/shm/modem_state.txt")
MODEM_LOCK_PATH = Path("/tmp/modem_at.lock")
DEFAULT_PORTS = ("/dev/ttyUSB2", "/dev/ttyUSB3", "/dev/ttyUSB0", "/dev/ttyACM0")
DEFAULT_BAUDRATE = 115200
READ_CHUNK_SIZE = 4096

RESPONSE_TERMINATORS = ("OK", "ERROR", "+CME ERROR", "+CMS ERROR")


def read_modem_state(state_path: os.PathLike[str] | str = MODEM_STATE_PATH) -> str | None:
  try:
    state = Path(state_path).read_text().strip()
    return state or None
  except OSError:
    return None


def write_modem_state(state: str, state_path: os.PathLike[str] | str = MODEM_STATE_PATH) -> None:
  path = Path(state_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp_path = path.with_name(f"{path.name}.tmp")
  tmp_path.write_text(f"{state}\n")
  os.replace(tmp_path, path)


class ModemPort:
  def __init__(self, port: str, timeout: float = 1.0, baudrate: int = DEFAULT_BAUDRATE, lock_path: os.PathLike[str] | str = MODEM_LOCK_PATH):
    self.port = port
    self.timeout = timeout
    self.baudrate = baudrate
    self.lock_path = Path(lock_path)
    self._serial: serial.Serial | None = None
    self._lock_file = None

  def __enter__(self):
    self.lock_path.parent.mkdir(parents=True, exist_ok=True)
    self._lock_file = self.lock_path.open("w")
    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
    self._serial = serial.Serial(self.port, baudrate=self.baudrate, timeout=self.timeout)
    self._serial.reset_input_buffer()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    if self._serial is not None:
      self._serial.close()
    if self._lock_file is not None:
      fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
      self._lock_file.close()

  def _read_response(self, timeout: float) -> str | None:
    assert self._serial is not None

    deadline = time.monotonic() + timeout
    response = bytearray()

    while time.monotonic() < deadline:
      waiting = max(self._serial.in_waiting, 1)
      chunk = self._serial.read(min(waiting, READ_CHUNK_SIZE))
      if chunk:
        response.extend(chunk)
        decoded = response.decode("utf-8", errors="ignore")
        if any(term in decoded for term in RESPONSE_TERMINATORS):
          return decoded.strip()
      else:
        time.sleep(0.01)

    if not response:
      return None
    return response.decode("utf-8", errors="ignore").strip()

  def cmd(self, cmd: str, timeout: float = 2.0) -> str | None:
    assert self._serial is not None
    self._serial.reset_input_buffer()
    self._serial.write(f"{cmd}\r\n".encode())
    return self._read_response(timeout=timeout)

  def upload_file(self, local_path: os.PathLike[str] | str, remote_name: str, timeout: float = 60.0) -> bool:
    assert self._serial is not None

    payload = Path(local_path).read_bytes()
    self._serial.reset_input_buffer()
    self._serial.write(f'AT+QFUPL="{remote_name}",{len(payload)},{int(timeout)}\r\n'.encode())

    connect_deadline = time.monotonic() + min(10.0, timeout)
    connect_response = bytearray()
    while time.monotonic() < connect_deadline:
      waiting = max(self._serial.in_waiting, 1)
      chunk = self._serial.read(min(waiting, READ_CHUNK_SIZE))
      if chunk:
        connect_response.extend(chunk)
        if b"CONNECT" in connect_response:
          break
        if b"ERROR" in connect_response or b"+CME ERROR" in connect_response or b"+CMS ERROR" in connect_response:
          return False
      else:
        time.sleep(0.01)
    else:
      return False

    self._serial.write(payload)
    response = self._read_response(timeout=timeout)
    if response is None:
      return False
    return "OK" in response and "ERROR" not in response


def _candidate_ports(port: str | None) -> Iterable[str]:
  if port is not None:
    yield port
    return
  yield from DEFAULT_PORTS


def at_cmd(cmd: str, port: str | None = None, timeout: float = 2.0) -> str | None:
  for candidate in _candidate_ports(port):
    try:
      with ModemPort(candidate, timeout=max(0.2, timeout)) as modem_port:
        return modem_port.cmd(cmd, timeout=timeout)
    except (serial.SerialException, PermissionError, OSError):
      continue
  return None
