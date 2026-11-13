"""Minimal client for wpa_supplicant's control socket (see wpa_supplicant/ctrl_iface.c)."""
import itertools
import os
import re
import socket

CTRL_PATH = "/run/wpa_supplicant/wlan0"

# BSS output fields: id, level, flags, ssid, and a delimiter after each entry
BSS_MASK = 0x1 | 0x80 | 0x800 | 0x1000 | 0x20000

_sock_ids = itertools.count()


class WpaCtrlError(Exception):
  pass


def decode_ssid(text: str) -> bytes:
  """Undo wpa_supplicant's printf_encode() of an SSID."""
  out = bytearray()
  raw = text.encode("utf-8", "surrogateescape")
  i = 0
  while i < len(raw):
    c = raw[i]
    if c == ord("\\") and i + 1 < len(raw):
      n = chr(raw[i + 1])
      if n == "x" and i + 3 < len(raw):
        try:
          out.append(int(raw[i + 2:i + 4], 16))
          i += 4
          continue
        except ValueError:
          pass
      out += {"\\": b"\\", '"': b'"', "e": b"\x1b", "n": b"\n", "r": b"\r", "t": b"\t"}.get(n, b"\\" + n.encode())
      i += 2
    else:
      out.append(c)
      i += 1
  return bytes(out)


def ssid_str(ssid: bytes) -> str:
  return ssid.decode("utf-8", "replace")


def parse_kv(text: str) -> dict[str, str]:
  return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


class WpaCtrl:
  """One connection to wpa_supplicant's control socket, for one thread. It never reconnects by itself: a network id is
  only valid on the wpa_supplicant it came from, which may have restarted since, so after an error it stays closed
  until connect() makes a new connection."""

  def __init__(self, path: str = CTRL_PATH):
    self._path = path
    self._sock: socket.socket | None = None

  @property
  def connected(self) -> bool:
    return self._sock is not None

  def connect(self):
    self.close()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
      sock.bind(f"\0openpilot-wifi-{os.getpid()}-{next(_sock_ids)}")
      sock.connect(self._path)
    except OSError as e:
      sock.close()
      raise WpaCtrlError(f"can't connect to {self._path}: {e}") from e
    self._sock = sock

  def close(self):
    if self._sock is not None:
      self._sock.close()
      self._sock = None

  def request(self, cmd: str, timeout: float = 2.0) -> str:
    sock = self._sock
    if sock is None:
      raise WpaCtrlError("not connected")
    try:
      sock.settimeout(timeout)
      sock.send(cmd.encode())
      while True:
        reply = sock.recv(65536).decode("utf-8", "surrogateescape")
        if not reply.startswith("<"):  # skip events on an attached socket
          return reply
    except OSError as e:
      self.close()  # wpa_supplicant went away, or may have
      raise WpaCtrlError(f"{cmd.split(maxsplit=1)[0]}: {e}") from e

  def ok(self, cmd: str) -> bool:
    return self.request(cmd).strip() == "OK"

  def get_network(self, net_id: int, field: str) -> str | None:
    """A network's field, with a string's quotes removed. None if it isn't set."""
    reply = self.request(f"GET_NETWORK {net_id} {field}").strip()
    if reply.startswith("FAIL"):
      return None
    return reply[1:-1] if len(reply) >= 2 and reply[0] == reply[-1] == '"' else reply

  def attach(self):
    """Connect, to receive events."""
    self.connect()
    if self.request("ATTACH").strip() != "OK":
      self.close()
      raise WpaCtrlError("ATTACH failed")

  def recv_event(self, timeout: float) -> str | None:
    """Next unsolicited event like 'CTRL-EVENT-CONNECTED ...', without its '<N>' level prefix."""
    sock = self._sock
    if sock is None:
      raise WpaCtrlError("not connected")
    sock.settimeout(timeout)
    try:
      msg = sock.recv(65536).decode("utf-8", "surrogateescape")
    except TimeoutError:
      return None
    except OSError as e:
      self.close()
      raise WpaCtrlError(str(e)) from e
    if msg.startswith("<") and ">" in msg:
      return msg.split(">", 1)[1]
    return None

  # helpers

  def status(self) -> dict[str, str]:
    return parse_kv(self.request("STATUS"))

  def list_networks(self) -> list[tuple[int, bytes, str]]:
    """[(id, ssid, flags)]. A reply holds 4 KB, so this asks for the networks after the last one until there are none."""
    nets: list[tuple[int, bytes, str]] = []
    while True:
      last_id = nets[-1][0] if nets else -1
      page = []
      for line in self.request(f"LIST_NETWORKS LAST_ID={last_id}").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 4 and parts[0].isdigit() and int(parts[0]) > last_id:
          page.append((int(parts[0]), decode_ssid(parts[1]), parts[3]))
      if not page:
        return nets
      nets += page

  def bss_list(self) -> list[dict[str, str]]:
    """The scan results. A reply holds 4 KB, so this asks for the entries after the last one until there are none.
    By id, so an entry that expires meanwhile doesn't end the list early."""
    out: list[dict[str, str]] = []
    while True:
      reply = self.request(f"BSS RANGE={int(out[-1]['id']) + 1 if out else 0}- MASK=0x{BSS_MASK:x}")
      page = [bss for entry in re.split(r"^(?:====|####)$", reply, flags=re.M) if "id" in (bss := parse_kv(entry))]
      if not page:
        return out
      out += page
