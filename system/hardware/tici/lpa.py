# SGP.22 v2.3: https://www.gsma.com/solutions-and-impact/technologies/esim/wp-content/uploads/2021/07/SGP.22-v2.3.pdf

import atexit
import base64
import math
import os
import serial
import sys

from collections.abc import Generator

from openpilot.system.hardware.base import LPABase, Profile


DEFAULT_DEVICE = "/dev/ttyUSB2"
DEFAULT_BAUD = 9600
DEFAULT_TIMEOUT = 5.0
# https://euicc-manual.osmocom.org/docs/lpa/applet-id/
ISDR_AID = "A0000005591010FFFFFFFF8900000100"
MM = "org.freedesktop.ModemManager1"
MM_MODEM = MM + ".Modem"
ES10X_MSS = 120
DEBUG = os.environ.get("DEBUG") == "1"

# TLV Tags
TAG_ICCID = 0x5A
TAG_PROFILE_INFO_LIST = 0xBF2D

STATE_LABELS = {0: "disabled", 1: "enabled", 255: "unknown"}
ICON_LABELS = {0: "jpeg", 1: "png", 255: "unknown"}
CLASS_LABELS = {0: "test", 1: "provisioning", 2: "operational", 255: "unknown"}


def b64e(data: bytes) -> str:
  return base64.b64encode(data).decode("ascii")


class AtClient:
  def __init__(self, device: str, baud: int, timeout: float, debug: bool) -> None:
    self.debug = debug
    self.channel: str | None = None
    self._timeout = timeout
    self._serial: serial.Serial | None = None
    try:
      self._serial = serial.Serial(device, baudrate=baud, timeout=timeout)
      self._serial.reset_input_buffer()
    except (serial.SerialException, PermissionError, OSError):
      pass

  def close(self) -> None:
    try:
      if self.channel:
        self.query(f"AT+CCHC={self.channel}")
        self.channel = None
    finally:
      if self._serial:
        self._serial.close()

  def _send(self, cmd: str) -> None:
    if self.debug:
      print(f"SER >> {cmd}", file=sys.stderr)
    self._serial.write((cmd + "\r").encode("ascii"))

  def _expect(self) -> list[str]:
    lines: list[str] = []
    while True:
      raw = self._serial.readline()
      if not raw:
        raise TimeoutError("AT command timed out")
      line = raw.decode(errors="ignore").strip()
      if not line:
        continue
      if self.debug:
        print(f"SER << {line}", file=sys.stderr)
      if line == "OK":
        return lines
      if line == "ERROR" or line.startswith("+CME ERROR"):
        raise RuntimeError(f"AT command failed: {line}")
      lines.append(line)

  def _get_modem(self):
    import dbus
    bus = dbus.SystemBus()
    mm = bus.get_object(MM, '/org/freedesktop/ModemManager1')
    objects = mm.GetManagedObjects(dbus_interface="org.freedesktop.DBus.ObjectManager", timeout=self._timeout)
    modem_path = list(objects.keys())[0]
    return bus.get_object(MM, modem_path)

  def _dbus_query(self, cmd: str) -> list[str]:
    if self.debug:
      print(f"DBUS >> {cmd}", file=sys.stderr)
    try:
      result = str(self._get_modem().Command(cmd, math.ceil(self._timeout), dbus_interface=MM_MODEM, timeout=self._timeout))
    except Exception as e:
      raise RuntimeError(f"AT command failed: {e}") from e
    lines = [line.strip() for line in result.splitlines() if line.strip()]
    if self.debug:
      for line in lines:
        print(f"DBUS << {line}", file=sys.stderr)
    return lines

  def query(self, cmd: str) -> list[str]:
    if self._serial:
      self._send(cmd)
      return self._expect()
    return self._dbus_query(cmd)

  def open_isdr(self) -> None:
    # close any stale logical channel from a previous crashed session
    try:
      self.query("AT+CCHC=1")
    except RuntimeError:
      pass
    for line in self.query(f'AT+CCHO="{ISDR_AID}"'):
      if line.startswith("+CCHO:") and (ch := line.split(":", 1)[1].strip()):
        self.channel = ch
        return
    raise RuntimeError("Failed to open ISD-R application")

  def send_apdu(self, apdu: bytes) -> tuple[bytes, int, int]:
    if not self.channel:
      raise RuntimeError("Logical channel is not open")
    hex_payload = apdu.hex().upper()
    for line in self.query(f'AT+CGLA={self.channel},{len(hex_payload)},"{hex_payload}"'):
      if line.startswith("+CGLA:"):
        parts = line.split(":", 1)[1].split(",", 1)
        if len(parts) == 2:
          data = bytes.fromhex(parts[1].strip().strip('"'))
          if len(data) >= 2:
            return data[:-2], data[-2], data[-1]
    raise RuntimeError("Missing +CGLA response")


# --- TLV utilities ---

def iter_tlv(data: bytes, with_positions: bool = False) -> Generator:
  idx, length = 0, len(data)
  while idx < length:
    start_pos = idx
    tag = data[idx]
    idx += 1
    if tag & 0x1F == 0x1F:  # Multi-byte tag
      tag_value = tag
      while idx < length:
        next_byte = data[idx]
        idx += 1
        tag_value = (tag_value << 8) | next_byte
        if not (next_byte & 0x80):
          break
    else:
      tag_value = tag
    if idx >= length:
      break
    size = data[idx]
    idx += 1
    if size & 0x80:  # Multi-byte length
      num_bytes = size & 0x7F
      if idx + num_bytes > length:
        break
      size = int.from_bytes(data[idx : idx + num_bytes], "big")
      idx += num_bytes
    if idx + size > length:
      break
    value = data[idx : idx + size]
    idx += size
    yield (tag_value, value, start_pos, idx) if with_positions else (tag_value, value)


def find_tag(data: bytes, target: int) -> bytes | None:
  return next((v for t, v in iter_tlv(data) if t == target), None)


def tbcd_to_string(raw: bytes) -> str:
  return "".join(str(n) for b in raw for n in (b & 0x0F, b >> 4) if n <= 9)


# Profile field decoders: TLV tag -> (field_name, decoder)
_PROFILE_FIELDS = {
  TAG_ICCID: ("iccid", tbcd_to_string),
  0x4F: ("isdpAid", lambda v: v.hex().upper()),
  0x9F70: ("profileState", lambda v: STATE_LABELS.get(v[0], "unknown")),
  0x90: ("profileNickname", lambda v: v.decode("utf-8", errors="ignore") or None),
  0x91: ("serviceProviderName", lambda v: v.decode("utf-8", errors="ignore") or None),
  0x92: ("profileName", lambda v: v.decode("utf-8", errors="ignore") or None),
  0x93: ("iconType", lambda v: ICON_LABELS.get(v[0], "unknown")),
  0x94: ("icon", b64e),
  0x95: ("profileClass", lambda v: CLASS_LABELS.get(v[0], "unknown")),
}


def _decode_profile_fields(data: bytes) -> dict:
  """Parse known profile metadata TLV fields into a dict."""
  result = {}
  for tag, value in iter_tlv(data):
    if (field := _PROFILE_FIELDS.get(tag)):
      result[field[0]] = field[1](value)
  return result


# --- ES10x command transport ---

def es10x_command(client: AtClient, data: bytes) -> bytes:
  response = bytearray()
  sequence = 0
  offset = 0
  while offset < len(data):
    chunk = data[offset : offset + ES10X_MSS]
    offset += len(chunk)
    is_last = offset == len(data)
    apdu = bytes([0x80, 0xE2, 0x91 if is_last else 0x11, sequence & 0xFF, len(chunk)]) + chunk
    segment, sw1, sw2 = client.send_apdu(apdu)
    response.extend(segment)
    while True:
      if sw1 == 0x61:  # More data available
        segment, sw1, sw2 = client.send_apdu(bytes([0x80, 0xC0, 0x00, 0x00, sw2 or 0]))
        response.extend(segment)
        continue
      if (sw1 & 0xF0) == 0x90:
        break
      raise RuntimeError(f"APDU failed with SW={sw1:02X}{sw2:02X}")
    sequence += 1
  return bytes(response)


# --- Profile operations ---

def decode_profiles(blob: bytes) -> list[dict]:
  root = find_tag(blob, TAG_PROFILE_INFO_LIST)
  if root is None:
    raise RuntimeError("Missing ProfileInfoList")
  list_ok = find_tag(root, 0xA0)
  if list_ok is None:
    return []
  defaults = {name: None for name, _ in _PROFILE_FIELDS.values()}
  return [{**defaults, **_decode_profile_fields(value)} for tag, value in iter_tlv(list_ok) if tag == 0xE3]


def list_profiles(client: AtClient) -> list[dict]:
  return decode_profiles(es10x_command(client, TAG_PROFILE_INFO_LIST.to_bytes(2, "big") + b"\x00"))


class TiciLPA(LPABase):
  _instance = None

  def __new__(cls):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
    return cls._instance

  def __init__(self):
    if hasattr(self, '_client'):
      return
    self._client = AtClient(DEFAULT_DEVICE, DEFAULT_BAUD, DEFAULT_TIMEOUT, debug=DEBUG)
    self._client.open_isdr()
    atexit.register(self._client.close)

  def list_profiles(self) -> list[Profile]:
    return [
      Profile(
        iccid=p.get("iccid", ""),
        nickname=p.get("profileNickname") or "",
        enabled=p.get("profileState") == "enabled",
        provider=p.get("serviceProviderName") or "",
      )
      for p in list_profiles(self._client)
    ]

  def get_active_profile(self) -> Profile | None:
    return None

  def delete_profile(self, iccid: str) -> None:
    return None

  def download_profile(self, qr: str, nickname: str | None = None) -> None:
    return None

  def nickname_profile(self, iccid: str, nickname: str) -> None:
    return None

  def switch_profile(self, iccid: str) -> None:
    return None
