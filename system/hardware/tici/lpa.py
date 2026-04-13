# SGP.22 v2.3: https://www.gsma.com/solutions-and-impact/technologies/esim/wp-content/uploads/2021/07/SGP.22-v2.3.pdf

import atexit
import base64
import fcntl
import math
import os
import serial
import subprocess
import sys
import termios
import time

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

from openpilot.system.hardware.base import LPABase, LPAError, Profile


DEFAULT_DEVICE = "/dev/modem_at0"
DEFAULT_BAUD = 9600
DEFAULT_TIMEOUT = 5.0
# https://euicc-manual.osmocom.org/docs/lpa/applet-id/
ISDR_AID = "A0000005591010FFFFFFFF8900000100"
MM = "org.freedesktop.ModemManager1"
MM_MODEM = MM + ".Modem"
ES10X_MSS = 120
OPEN_ISDR_RETRIES = 10
OPEN_ISDR_RETRY_DELAY_S = 0.25
OPEN_ISDR_RESET_ATTEMPT = 5
SEND_APDU_RETRIES = 3
LOCK_FILE = '/dev/shm/modem_lpa.lock'
DEBUG = os.environ.get("DEBUG") == "1"


# TLV Tags
TAG_ICCID = 0x5A
TAG_STATUS = 0x80
TAG_PROFILE_INFO_LIST = 0xBF2D
TAG_SET_NICKNAME = 0xBF29
TAG_ENABLE_PROFILE = 0xBF31
TAG_DELETE_PROFILE = 0xBF33
TAG_OK = 0xA0

PROFILE_OK = 0x00
PROFILE_NOT_IN_DISABLED_STATE = 0x02
PROFILE_CAT_BUSY = 0x05

PROFILE_ERROR_CODES = {
  0x01: "iccidOrAidNotFound", PROFILE_NOT_IN_DISABLED_STATE: "profileNotInDisabledState",
  0x03: "disallowedByPolicy", 0x04: "wrongProfileReenabling",
  PROFILE_CAT_BUSY: "catBusy", 0x06: "undefinedError",
}

STATE_LABELS = {0: "disabled", 1: "enabled", 255: "unknown"}
ICON_LABELS = {0: "jpeg", 1: "png", 255: "unknown"}
CLASS_LABELS = {0: "test", 1: "provisioning", 2: "operational", 255: "unknown"}

# TLV tag -> (field_name, decoder)
FieldMap = dict[int, tuple[str, Callable[[bytes], Any]]]


def b64e(data: bytes) -> str:
  return base64.b64encode(data).decode("ascii")


def base64_trim(s: str) -> str:
  return "".join(c for c in s if c not in "\n\r \t")


def b64d(s: str) -> bytes:
  return base64.b64decode(base64_trim(s))


class AtClient:
  def __init__(self, device: str, baud: int, timeout: float) -> None:
    self.channel: str | None = None
    self._device = device
    self._baud = baud
    self._timeout = timeout
    self._serial: serial.Serial | None = None
    self._use_dbus = not os.path.exists(device)

  def send_raw(self, data: bytes) -> None:
    self._ensure_serial()
    self._serial.reset_input_buffer()
    self._serial.write(data)
    self._serial.flush()

  def close(self) -> None:
    try:
      if self.channel:
        try:
          self.query(f"AT+CCHC={self.channel}")
        except (RuntimeError, TimeoutError):
          pass
        self.channel = None
    finally:
      if self._serial:
        self._serial.close()

  def _send(self, cmd: str) -> None:
    if DEBUG:
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
      if DEBUG:
        print(f"SER << {line}", file=sys.stderr)
      if line == "OK":
        return lines
      if line == "ERROR" or line.startswith("+CME ERROR"):
        raise RuntimeError(f"AT command failed: {line}")
      lines.append(line)

  def _ensure_serial(self, reconnect: bool = False) -> None:
    if reconnect:
      self.channel = None
      try:
        if self._serial:
          self._serial.close()
      except Exception:
        pass
      self._serial = None
    if self._serial is None:
      self._serial = serial.Serial(self._device, baudrate=self._baud, timeout=self._timeout)

  def _get_modem(self):
    import dbus
    bus = dbus.SystemBus()
    mm = bus.get_object(MM, '/org/freedesktop/ModemManager1')
    objects = mm.GetManagedObjects(dbus_interface="org.freedesktop.DBus.ObjectManager", timeout=self._timeout)
    modem_path = list(objects.keys())[0]
    return bus.get_object(MM, modem_path)

  def _dbus_query(self, cmd: str) -> list[str]:
    if DEBUG:
      print(f"DBUS >> {cmd}", file=sys.stderr)
    try:
      result = str(self._get_modem().Command(cmd, math.ceil(self._timeout), dbus_interface=MM_MODEM, timeout=self._timeout))
    except Exception as e:
      raise RuntimeError(f"AT command failed: {e}") from e
    lines = [line.strip() for line in result.splitlines() if line.strip()]
    if DEBUG:
      for line in lines:
        print(f"DBUS << {line}", file=sys.stderr)
    return lines

  def query(self, cmd: str) -> list[str]:
    if self._use_dbus:
      return self._dbus_query(cmd)
    self._ensure_serial()
    try:
      self._send(cmd)
      return self._expect()
    except serial.SerialException:
      self._ensure_serial(reconnect=True)
      self._send(cmd)
      return self._expect()

  def _open_isdr_once(self) -> None:
    if self.channel:
      try:
        self.query(f"AT+CCHC={self.channel}")
      except RuntimeError:
        pass
      self.channel = None
    # drain any unsolicited responses before opening
    if self._serial and not self._use_dbus:
      try:
        self._serial.reset_input_buffer()
      except (OSError, serial.SerialException, termios.error):
        self._ensure_serial(reconnect=True)
    for line in self.query(f'AT+CCHO="{ISDR_AID}"'):
      if line.startswith("+CCHO:") and (ch := line.split(":", 1)[1].strip()):
        self.channel = ch
        return
    raise RuntimeError("Failed to open ISD-R application")

  def _reset_modem(self) -> None:
    if self._serial:
      try:
        self._serial.close()
      except Exception:
        pass
      self._serial = None
    subprocess.run(['/usr/comma/lte/lte.sh', 'start'], capture_output=True)

  def open_isdr(self) -> None:
    for attempt in range(OPEN_ISDR_RETRIES):
      try:
        self._open_isdr_once()
        return
      except (RuntimeError, TimeoutError, termios.error, serial.SerialException):
        time.sleep(OPEN_ISDR_RETRY_DELAY_S)
        if attempt == OPEN_ISDR_RESET_ATTEMPT:
          self._reset_modem()
    raise RuntimeError("Failed to open ISD-R after retries")

  def send_apdu(self, apdu: bytes) -> tuple[bytes, int, int]:
    for attempt in range(SEND_APDU_RETRIES):
      try:
        if not self.channel:
          self.open_isdr()
        hex_payload = apdu.hex().upper()
        for line in self.query(f'AT+CGLA={self.channel},{len(hex_payload)},"{hex_payload}"'):
          if line.startswith("+CGLA:"):
            parts = line.split(":", 1)[1].split(",", 1)
            if len(parts) == 2:
              data = bytes.fromhex(parts[1].strip().strip('"'))
              if len(data) >= 2:
                return data[:-2], data[-2], data[-1]
        raise RuntimeError("Missing +CGLA response")
      except (RuntimeError, ValueError):
        self.channel = None
        if attempt == SEND_APDU_RETRIES - 1:
          raise
    raise RuntimeError("send_apdu failed")


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


def require_tag(data: bytes, target: int, label: str = "") -> bytes:
  v = find_tag(data, target)
  if v is None:
    raise RuntimeError(f"Missing {label or f'tag 0x{target:X}'}")
  return v


def tbcd_to_string(raw: bytes) -> str:
  return "".join(str(n) for b in raw for n in (b & 0x0F, b >> 4) if n <= 9)


def string_to_tbcd(s: str) -> bytes:
  digits = [int(c) for c in s if c.isdigit()]
  return bytes(digits[i] | ((digits[i + 1] if i + 1 < len(digits) else 0xF) << 4) for i in range(0, len(digits), 2))


def encode_tlv(tag: int, value: bytes) -> bytes:
  tag_bytes = bytes([(tag >> 8) & 0xFF, tag & 0xFF]) if tag > 255 else bytes([tag])
  vlen = len(value)
  if vlen <= 127:
    return tag_bytes + bytes([vlen]) + value
  length_bytes = vlen.to_bytes((vlen.bit_length() + 7) // 8, "big")
  return tag_bytes + bytes([0x80 | len(length_bytes)]) + length_bytes + value


def int_bytes(n: int) -> bytes:
  """Encode a positive integer as minimal big-endian bytes (at least 1 byte)."""
  return n.to_bytes((n.bit_length() + 7) // 8 or 1, "big")


PROFILE: FieldMap = {
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


def decode_struct(data: bytes, field_map: FieldMap) -> dict[str, Any]:
  """Parse TLV data using a {tag: (field_name, decoder)} map into a dict."""
  result: dict[str, Any] = {name: None for name, _ in field_map.values()}
  for tag, value in iter_tlv(data):
    if (field := field_map.get(tag)):
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
  root = require_tag(blob, TAG_PROFILE_INFO_LIST, "ProfileInfoList")
  list_ok = find_tag(root, TAG_OK)
  if list_ok is None:
    return []
  return [decode_struct(value, PROFILE) for tag, value in iter_tlv(list_ok) if tag == 0xE3]


def list_profiles(client: AtClient) -> list[dict]:
  return decode_profiles(es10x_command(client, TAG_PROFILE_INFO_LIST.to_bytes(2, "big") + b"\x00"))


def set_profile_nickname(client: AtClient, iccid: str, nickname: str) -> None:
  nickname_bytes = nickname.encode("utf-8")
  if len(nickname_bytes) > 64:
    raise ValueError("Profile nickname must be 64 bytes or less")
  content = encode_tlv(TAG_ICCID, string_to_tbcd(iccid)) + encode_tlv(0x90, nickname_bytes)
  response = es10x_command(client, encode_tlv(TAG_SET_NICKNAME, content))
  code = require_tag(require_tag(response, TAG_SET_NICKNAME, "SetNicknameResponse"), TAG_STATUS, "SetNickname status")[0]
  if code == 0x01:
    raise LPAError(f"profile {iccid} not found")
  if code != 0x00:
    raise RuntimeError(f"SetNickname failed with status 0x{code:02X}")


class TiciLPA(LPABase):
  def __init__(self):
    if hasattr(self, '_client'):
      return
    self._client = AtClient(DEFAULT_DEVICE, DEFAULT_BAUD, DEFAULT_TIMEOUT)
    atexit.register(self._client.close)

  @contextmanager
  def _acquire_channel(self):
    fd = os.open(LOCK_FILE, os.O_CREAT | os.O_RDWR)
    try:
      fcntl.flock(fd, fcntl.LOCK_EX)
      self._client.open_isdr()
      yield
    finally:
      if self._client.channel:
        try:
          self._client.query(f"AT+CCHC={self._client.channel}")
        except (RuntimeError, TimeoutError):
          pass
        self._client.channel = None
      fcntl.flock(fd, fcntl.LOCK_UN)
      os.close(fd)

  def list_profiles(self) -> list[Profile]:
    with self._acquire_channel():
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
    if self.is_comma_profile(iccid):
      raise LPAError("refusing to delete a comma profile")
    with self._acquire_channel():
      request = encode_tlv(TAG_DELETE_PROFILE, encode_tlv(TAG_ICCID, string_to_tbcd(iccid)))
      response = es10x_command(self._client, request)
      code = require_tag(require_tag(response, TAG_DELETE_PROFILE, "DeleteProfileResponse"), TAG_STATUS, "DeleteProfile status")[0]
    if code != PROFILE_OK:
      raise LPAError(f"DeleteProfile failed: {PROFILE_ERROR_CODES.get(code, 'unknown')} (0x{code:02X})")

  def download_profile(self, qr: str, nickname: str | None = None) -> None:
    return None

  def nickname_profile(self, iccid: str, nickname: str) -> None:
    with self._acquire_channel():
      set_profile_nickname(self._client, iccid, nickname)

  def _enable_profile(self, iccid: str) -> int:
    inner = encode_tlv(TAG_OK, encode_tlv(TAG_ICCID, string_to_tbcd(iccid)))
    inner += b'\x01\x01\x01'  # refreshFlag=1
    response = es10x_command(self._client, encode_tlv(TAG_ENABLE_PROFILE, inner))
    return require_tag(require_tag(response, TAG_ENABLE_PROFILE, "EnableProfileResponse"), TAG_STATUS, "EnableProfile status")[0]

  def switch_profile(self, iccid: str) -> None:
    with self._acquire_channel():
      code = self._enable_profile(iccid)
      if code == PROFILE_CAT_BUSY:  # stale eUICC transaction, reset and retry
        self._client._reset_modem()
        self._client.open_isdr()
        code = self._enable_profile(iccid)
      if code not in (PROFILE_OK, PROFILE_NOT_IN_DISABLED_STATE):
        raise LPAError(f"EnableProfile failed: {PROFILE_ERROR_CODES.get(code, 'unknown')} (0x{code:02X})")
    from openpilot.system.hardware import HARDWARE
    if HARDWARE.get_device_type() == "mici":
      self._client.send_raw(b'AT+CFUN=0\rAT+CFUN=1\r')  # mici has no SIM presence pin; raw because CFUN=0 drops serial
      self._client._ensure_serial(reconnect=True)
