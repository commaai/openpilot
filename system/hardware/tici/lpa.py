# SGP.22 v2.3: https://www.gsma.com/solutions-and-impact/technologies/esim/wp-content/uploads/2021/07/SGP.22-v2.3.pdf

import atexit
import base64
import os
import requests
import serial
import sys
import time

from collections.abc import Callable, Generator
from typing import Any
from pathlib import Path

from openpilot.common.time_helpers import system_time_valid
from openpilot.common.utils import retry
from openpilot.system.hardware.base import LPABase, LPAError, Profile

GSMA_CI_BUNDLE = str(Path(__file__).parent / 'gsma_ci_bundle.pem')


DEFAULT_DEVICE = "/dev/ttyUSB2"
DEFAULT_BAUD = 9600
DEFAULT_TIMEOUT = 5.0
# https://euicc-manual.osmocom.org/docs/lpa/applet-id/
ISDR_AID = "A0000005591010FFFFFFFF8900000100"
ES10X_MSS = 120
DEBUG = os.environ.get("DEBUG") == "1"

# TLV Tags
TAG_ICCID = 0x5A
TAG_STATUS = 0x80
TAG_PROFILE_INFO_LIST = 0xBF2D
TAG_LIST_NOTIFICATION = 0xBF28
TAG_RETRIEVE_NOTIFICATION = 0xBF2B
TAG_NOTIFICATION_METADATA = 0xBF2F
TAG_NOTIFICATION_SENT = 0xBF30
TAG_ENABLE_PROFILE = 0xBF31
TAG_DELETE_PROFILE = 0xBF33
TAG_PROFILE_INSTALL_RESULT = 0xBF37
TAG_OK = 0xA0

CAT_BUSY = 0x05

PROFILE_ERROR_CODES = {
  0x01: "iccidOrAidNotFound", 0x02: "profileNotInDisabledState",
  0x03: "disallowedByPolicy", 0x04: "wrongProfileReenabling",
  CAT_BUSY: "catBusy", 0x06: "undefinedError",
}

STATE_LABELS = {0: "disabled", 1: "enabled", 255: "unknown"}
ICON_LABELS = {0: "jpeg", 1: "png", 255: "unknown"}
CLASS_LABELS = {0: "test", 1: "provisioning", 2: "operational", 255: "unknown"}

# TLV tag -> (field_name, decoder)
FieldMap = dict[int, tuple[str, Callable[[bytes], Any]]]


def b64e(data: bytes) -> str:
  return base64.b64encode(data).decode("ascii")


class AtClient:
  def __init__(self, device: str, baud: int, timeout: float, debug: bool) -> None:
    self.serial = serial.Serial(device, baudrate=baud, timeout=timeout)
    self.debug = debug
    self.channel: str | None = None
    self.serial.reset_input_buffer()

  def close(self) -> None:
    try:
      if self.channel:
        try:
          self.query(f"AT+CCHC={self.channel}")
        except (RuntimeError, TimeoutError):
          pass
        self.channel = None
    finally:
      self.serial.close()

  def send(self, cmd: str) -> None:
    if self.debug:
      print(f">> {cmd}", file=sys.stderr)
    self.serial.write((cmd + "\r").encode("ascii"))

  def expect(self) -> list[str]:
    lines: list[str] = []
    while True:
      raw = self.serial.readline()
      if not raw:
        raise TimeoutError("AT command timed out")
      line = raw.decode(errors="ignore").strip()
      if not line:
        continue
      if self.debug:
        print(f"<< {line}", file=sys.stderr)
      if line == "OK":
        return lines
      if line == "ERROR" or line.startswith("+CME ERROR"):
        raise RuntimeError(f"AT command failed: {line}")
      lines.append(line)

  def query(self, cmd: str) -> list[str]:
    self.send(cmd)
    return self.expect()

  @retry(attempts=10, delay=2.0)
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

  def send_apdu(self, apdu: bytes, max_retries: int = 3) -> tuple[bytes, int, int]:
    for attempt in range(max_retries):
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
      except RuntimeError:
        self.channel = None
        if attempt == max_retries - 1:
          raise
        time.sleep(1 + attempt)


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

NOTIFICATION: FieldMap = {
  TAG_STATUS: ("seqNumber", lambda v: int.from_bytes(v, "big")),
  0x81: ("profileManagementOperation", lambda v: next((m for m in [0x80, 0x40, 0x20, 0x10] if len(v) >= 2 and v[1] & m), 0xFF)),
  0x0C: ("notificationAddress", lambda v: v.decode("utf-8", errors="ignore")),
  TAG_ICCID: ("iccid", tbcd_to_string),
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


# --- ES9P HTTP ---

def es9p_request(smdp_address: str, endpoint: str, payload: dict, error_prefix: str = "Request") -> dict:
  if not system_time_valid():
    raise RuntimeError("System time is not set; TLS certificate validation requires a valid clock")
  url = f"https://{smdp_address}/gsma/rsp2/es9plus/{endpoint}"
  headers = {"User-Agent": "gsma-rsp-lpad", "X-Admin-Protocol": "gsma/rsp/v2.3.0", "Content-Type": "application/json"}
  resp = requests.post(url, json=payload, headers=headers, timeout=30, verify=GSMA_CI_BUNDLE)
  resp.raise_for_status()
  if not resp.content:
    return {}
  data = resp.json()
  if "header" in data and "functionExecutionStatus" in data["header"]:
    status = data["header"]["functionExecutionStatus"]
    if status.get("status") == "Failed":
      sd = status.get("statusCodeData", {})
      raise RuntimeError(f"{error_prefix} failed: {sd.get('reasonCode', 'unknown')}/{sd.get('subjectCode', 'unknown')} - {sd.get('message', 'unknown')}")
  return data


# --- Notifications ---

def list_notifications(client: AtClient) -> list[dict]:
  response = es10x_command(client, encode_tlv(TAG_LIST_NOTIFICATION, b""))
  root = require_tag(response, TAG_LIST_NOTIFICATION, "ListNotificationResponse")
  metadata_list = find_tag(root, TAG_OK)
  if metadata_list is None:
    return []
  notifications: list[dict] = []
  for tag, value in iter_tlv(metadata_list):
    if tag != TAG_NOTIFICATION_METADATA:
      continue
    notification = decode_struct(value, NOTIFICATION)
    if notification["seqNumber"] is not None and notification["profileManagementOperation"] is not None and notification["notificationAddress"]:
      notifications.append(notification)
  return notifications


def process_notifications(client: AtClient) -> None:
  for notification in list_notifications(client):
    seq_number, smdp_address = notification["seqNumber"], notification["notificationAddress"]
    try:
      # retrieve notification
      request = encode_tlv(TAG_RETRIEVE_NOTIFICATION, encode_tlv(TAG_OK, encode_tlv(TAG_STATUS, int_bytes(seq_number))))
      response = es10x_command(client, request)
      content = require_tag(require_tag(response, TAG_RETRIEVE_NOTIFICATION, "RetrieveNotificationsListResponse"),
                            TAG_OK, "RetrieveNotificationsListResponse")
      pending_notif = next((v for t, v in iter_tlv(content) if t in (TAG_PROFILE_INSTALL_RESULT, 0x30)), None)
      if pending_notif is None:
        raise RuntimeError("Missing PendingNotification")

      # send to SM-DP+
      es9p_request(smdp_address, "handleNotification", {"pendingNotification": b64e(pending_notif)}, "HandleNotification")

      # remove notification
      response = es10x_command(client, encode_tlv(TAG_NOTIFICATION_SENT, encode_tlv(TAG_STATUS, int_bytes(seq_number))))
      root = require_tag(response, TAG_NOTIFICATION_SENT, "NotificationSentResponse")
      if int.from_bytes(require_tag(root, TAG_STATUS, "RemoveNotificationFromList status"), "big") != 0:
        raise RuntimeError("RemoveNotificationFromList failed")
    except Exception:
      pass



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

  def _reboot_modem(self) -> None:
    """Reboot modem and re-open ISD-R."""
    from openpilot.system.hardware import HARDWARE
    HARDWARE.reboot_modem()
    self._client.channel = None
    self._client.serial.close()
    self._reconnect_serial()
    self._client.open_isdr()

  @retry(attempts=3, delay=1.0)
  def _reconnect_serial(self) -> None:
    self._client.serial = serial.Serial(DEFAULT_DEVICE, DEFAULT_BAUD, timeout=DEFAULT_TIMEOUT)
    self._client.serial.reset_input_buffer()

  def _delete_profile(self, iccid: str) -> int:
    request = encode_tlv(TAG_DELETE_PROFILE, encode_tlv(TAG_ICCID, string_to_tbcd(iccid)))
    response = es10x_command(self._client, request)
    root = require_tag(response, TAG_DELETE_PROFILE, "DeleteProfileResponse")
    return require_tag(root, TAG_STATUS, "status in DeleteProfileResponse")[0]

  def delete_profile(self, iccid: str) -> None:
    if self.is_comma_profile(iccid):
      raise LPAError("refusing to delete a comma profile")
    code = self._delete_profile(iccid)
    if code == CAT_BUSY:
      self._reboot_modem()
      code = self._delete_profile(iccid)
    if code != 0x00:
      raise RuntimeError(f"DeleteProfile failed: {PROFILE_ERROR_CODES.get(code, 'unknown')} (0x{code:02X})")
    process_notifications(self._client)

  def download_profile(self, qr: str, nickname: str | None = None) -> None:
    return None

  def nickname_profile(self, iccid: str, nickname: str) -> None:
    return None

  def _enable_profile(self, iccid: str, refresh: bool = True) -> int:
    inner = encode_tlv(TAG_OK, encode_tlv(TAG_ICCID, string_to_tbcd(iccid)))
    inner += b'\x01\x01' + (b'\xFF' if refresh else b'\x00')  # refreshFlag BOOLEAN
    request = encode_tlv(TAG_ENABLE_PROFILE, inner)
    response = es10x_command(self._client, request)
    root = require_tag(response, TAG_ENABLE_PROFILE, "EnableProfileResponse")
    return require_tag(root, TAG_STATUS, "status in EnableProfileResponse")[0]

  def switch_profile(self, iccid: str) -> None:
    code = self._enable_profile(iccid, refresh=False)
    if code == CAT_BUSY:
      self._reboot_modem()
      code = self._enable_profile(iccid, refresh=False)
    if code not in (0x00, 0x02):  # 0x02 = already enabled
      raise RuntimeError(f"EnableProfile failed: {PROFILE_ERROR_CODES.get(code, 'unknown')} (0x{code:02X})")
    process_notifications(self._client)
    if code == 0x00:
      self._reboot_modem()
