# Persistent eSIM profile pin stored in the active profile's SIM SMS storage (EF-SMS).
#
# Modem eSIMs (e.g. Webbing) autonomously switch the enabled profile back to their own
# and cannot be configured out of it (see TS-PA-UP probing notes). modemd reverts such
# switchbacks when it knows which profile is pinned. The pin param in /data is lost on
# factory reset, but the SIM SMS storage is not - so we stash the pin in a draft SMS on
# the SIM itself (payload: "comma-pin:<iccid>").
#
# Note: the pin stored in a profile's SMS storage is only readable while that profile is
# enabled. That's exactly when we need it: a switchback makes the webbing profile active,
# at which point modemd can read the pin back from the SIM (and modemd/esim.py re-seed
# the marker whenever the webbing profile is active and a pin is known).

import fcntl
import os
import time

from openpilot.common.esim.lpa import AtClient, DEFAULT_DEVICE, DEFAULT_BAUD, DEFAULT_TIMEOUT, LOCK_FILE

PIN_SMS_PREFIX = "comma-pin:"
_DEST_ADDRESS = "123"  # dummy recipient for the stored draft; never sent


# --- 7-bit GSM 03.38 packing helpers (ascii-safe subset) ---

def _gsm7_pack(text: str) -> bytes:
  out = bytearray()
  buf = nbits = 0
  for ch in text:
    buf |= ord(ch) << nbits
    nbits += 7
    while nbits >= 8:
      out.append(buf & 0xFF)
      buf >>= 8
      nbits -= 8
  if nbits:
    out.append(buf & 0xFF)
  return bytes(out)


def _gsm7_unpack(data: bytes, num_septets: int) -> str:
  out = []
  buf = nbits = 0
  for b in data:
    buf |= b << nbits
    nbits += 8
    while nbits >= 7:
      out.append(chr(buf & 0x7F))
      buf >>= 7
      nbits -= 7
  return "".join(out[:num_septets])


def _tbcd(digits: str) -> bytes:
  return bytes(int(d) | ((int(digits[i + 1]) if i + 1 < len(digits) else 0xF) << 4) for i, d in enumerate(digits) if i % 2 == 0)


# --- SMS-PDU encode/decode (7-bit text, no VP) ---

def _encode_pdu(text: str) -> tuple[bytes, int]:
  """Returns (full pdu, tpdu length). SMS-SUBMIT, 7-bit GSM, VPF none, DCS 0."""
  ud = _gsm7_pack(text)
  da = _tbcd(_DEST_ADDRESS)
  tpdu = bytes([0x11, 0x00, len(_DEST_ADDRESS), 0x81]) + da + bytes([0x00, 0x00, len(text)]) + ud
  return b"\x00" + tpdu, len(tpdu)  # SCA length 0 = modem default SMSC


def _decode_pdu(pdu: bytes) -> str:
  sca_len = pdu[0] + 1
  tpdu = pdu[sca_len:]
  mti = tpdu[0] & 0x03
  i = 1
  if mti == 0x01:  # SUBMIT: skip TP-MR
    i += 1
  addr_digits = tpdu[i]
  i += 2 + (addr_digits + 1) // 2  # len, toa, tbcd number
  i += 2  # TP-PID, TP-DCS
  if mti == 0x00:  # DELIVER: skip TP-SCTS
    i += 7
  udl = tpdu[i]
  ud = tpdu[i + 1:i + 1 + (udl * 7 + 7) // 8]
  return _gsm7_unpack(ud, udl)


# --- SIM SMS store access over the modem AT port ---

class _Store:
  def __init__(self) -> None:
    self._owns_client = False
    self._client: AtClient | None = None

  def __enter__(self) -> "_Store":
    self._fd = os.open(LOCK_FILE, os.O_CREAT | os.O_RDWR)
    fcntl.flock(self._fd, fcntl.LOCK_EX)
    self._client = AtClient(DEFAULT_DEVICE, DEFAULT_BAUD, DEFAULT_TIMEOUT)
    return self

  def __exit__(self, *args) -> None:
    try:
      if self._client is not None:
        self._client.close()
    finally:
      fcntl.flock(self._fd, fcntl.LOCK_UN)
      os.close(self._fd)

  def _query(self, cmd: str) -> list[str]:
    return self._client.query(cmd)

  def _select_sim_storage(self) -> None:
    try:
      self._query('AT+CPMS="SM","SM","SM"')
    except (RuntimeError, TimeoutError):
      self._query('AT+CPMS="SM"')

  def _gateway_write(self, cmd: str, hc_data: str) -> list[str]:
    """Send a prompt-mode command (e.g. +CMGW) and its ctrl-Z-terminated payload."""
    s = self._client._serial
    self._client._ensure_serial()
    s.reset_input_buffer()
    self._client._send(cmd)
    deadline = time.monotonic() + 10
    buf = b""
    while time.monotonic() < deadline:
      line = s.readline()
      if line:
        buf += line
      if b">" in buf:
        break
      if b"ERROR" in buf:
        raise RuntimeError(f"prompt command failed: {buf!r}")
    else:
      raise TimeoutError(f"no prompt for {cmd}")
    s.write(hc_data.encode("ascii") + b"\x1a")
    s.flush()
    return self._client._expect()

  def read_messages(self) -> list[tuple[int, str]]:
    """Returns [(slot_index, decoded_text), ...] for all SMS currently in SIM storage."""
    self._query("AT+CMGF=0")
    self._select_sim_storage()
    lines = self._query("AT+CMGL=4")
    out: list[tuple[int, str]] = []
    headers = [l for l in lines if l.startswith("+CMGL:")]
    payload_candidates = [l for l in lines if not l.startswith("+CMGL:")]
    # map each header to the following payload line in order of appearance
    payloads = iter(payload_candidates)
    for h in headers:
      try:
        idx = int(h.split(":", 1)[1].split(",", 1)[0].strip())
        pdu_line = next(p for p in payloads if all(c in "0123456789ABCDEF" for c in p.strip()) and len(p.strip()) > 10)
      except (ValueError, StopIteration):
        continue
      try:
        text = _decode_pdu(bytes.fromhex(pdu_line.strip()))
      except Exception:
        continue
      out.append((idx, text))
    return out

  def delete_slot(self, idx: int) -> None:
    self._query(f"AT+CMGD={idx}")

  def write_message(self, text: str) -> int:
    self._query("AT+CMGF=0")
    self._select_sim_storage()
    pdu, tlen = _encode_pdu(text)
    lines = self._gateway_write(f"AT+CMGW={tlen},2", pdu.hex().upper())
    for line in lines:
      if line.startswith("+CMGW:"):
        return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"CMGW failed: {lines}")


def _markers(store: _Store) -> list[tuple[int, str]]:
  return [(i, t[len(PIN_SMS_PREFIX):].strip()) for i, t in store.read_messages() if t.startswith(PIN_SMS_PREFIX)]


def read_pin() -> str:
  """Read the pinned iccid from SIM SMS storage, or '' if absent/unavailable."""
  try:
    with _Store() as store:
      ms = _markers(store)
      return ms[-1][1] if ms else ""
  except Exception:
    return ""


def write_pin(iccid: str) -> bool:
  """Persist the pin marker into the *currently active* profile's SIM SMS storage.

  Deletes any stale markers first. iccid='' only deletes markers. Returns False if the
  write wasn't possible (e.g. storage full) - callers have the /data param as backup.
  """
  try:
    with _Store() as store:
      for idx, _ in _markers(store):
        store.delete_slot(idx)
      if iccid:
        store.write_message(f"{PIN_SMS_PREFIX}{iccid}")
    return True
  except Exception:
    return False
