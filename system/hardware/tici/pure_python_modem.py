from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from serial import Serial, SerialException


MODEM_STATE_PATH = Path("/dev/shm/modem_state.txt")


class ModemState(Enum):
  DISCONNECTED = "disconnected"
  SIM_PENDING = "sim_pending"
  REGISTERING = "registering"
  REGISTERED = "registered"
  ATTACHED = "attached"


@dataclass
class ModemSnapshot:
  state: ModemState
  sim_ready: bool
  registered: bool
  attached: bool
  model: str
  last_response: str


class ModemATError(RuntimeError):
  pass


class QuectelATClient:
  """
  Thin AT-command client for EG25/EG916 style UART ports.
  """

  def __init__(
    self,
    port: str = "/dev/ttyUSB2",
    baudrate: int = 115200,
    timeout: float = 0.8,
    write_timeout: float = 0.8,
  ) -> None:
    self.port = port
    self.baudrate = baudrate
    self.timeout = timeout
    self.write_timeout = write_timeout
    self._serial: Serial | None = None
    self._lock = threading.Lock()

  def open(self) -> None:
    if self._serial is not None:
      return
    self._serial = Serial(
      self.port,
      baudrate=self.baudrate,
      timeout=self.timeout,
      write_timeout=self.write_timeout,
      exclusive=True,
    )
    self._serial.reset_input_buffer()
    self._serial.reset_output_buffer()

  def close(self) -> None:
    if self._serial is None:
      return
    try:
      self._serial.close()
    finally:
      self._serial = None

  def _ensure_open(self) -> Serial:
    if self._serial is None:
      self.open()
    assert self._serial is not None
    return self._serial

  @staticmethod
  def _clean_line(raw: bytes) -> str:
    return raw.decode("utf-8", errors="ignore").strip()

  def send(
    self,
    command: str,
    timeout: float = 2.5,
    ok_tokens: Sequence[str] = ("OK",),
    error_tokens: Sequence[str] = ("ERROR", "+CME ERROR", "+CMS ERROR"),
  ) -> list[str]:
    if not command:
      raise ValueError("command cannot be empty")

    with self._lock:
      ser = self._ensure_open()
      ser.reset_input_buffer()
      ser.write((command + "\r").encode("ascii", errors="ignore"))
      ser.flush()

      lines: list[str] = []
      deadline = time.monotonic() + timeout
      while time.monotonic() < deadline:
        try:
          raw = ser.readline()
        except SerialException as err:
          self.close()
          raise ModemATError(f"serial_read_failed: {err}") from err

        if not raw:
          continue
        line = self._clean_line(raw)
        if not line:
          continue
        lines.append(line)

        if any(tok in line for tok in error_tokens):
          raise ModemATError(f"AT command failed: {command} | {line}")

        if any(line == tok or line.endswith(tok) for tok in ok_tokens):
          return lines

      raise ModemATError(f"AT command timeout: {command} | lines={lines[-4:]}")


class QuectelModemStateMachine:
  """
  Minimal boot-up state machine suitable for flaky edge links:
  SIM -> registration -> packet attach.
  """

  def __init__(
    self,
    client: QuectelATClient,
    apn: str = "",
    registration_timeout: float = 60.0,
    registration_poll_interval: float = 0.1,
    startup_retries: int = 1,
    fast_boot: bool = True,
    state_path: Path = MODEM_STATE_PATH,
  ) -> None:
    self.client = client
    self.apn = apn
    self.registration_timeout = registration_timeout
    self.registration_poll_interval = registration_poll_interval
    self.startup_retries = max(0, startup_retries)
    self.fast_boot = fast_boot
    self.state_path = state_path
    self._last_state: ModemState | None = None

  @staticmethod
  def _contains(lines: Iterable[str], needle: str) -> bool:
    low = needle.lower()
    return any(low in str(line).lower() for line in lines)

  @staticmethod
  def _parse_cereg_status(lines: Sequence[str]) -> int:
    # +CEREG: <n>,<stat> or +CEREG: <stat>
    for line in lines:
      if not line.startswith("+CEREG:"):
        continue
      raw = line.split(":", 1)[-1].strip()
      parts = [x.strip() for x in raw.split(",") if x.strip()]
      if not parts:
        continue
      status = parts[-1]
      if status.startswith(("0x", "0X")):
        return int(status, 16)
      if status.isdigit():
        return int(status)
    return -1

  @staticmethod
  def _external_state_name(state: ModemState) -> str:
    mapping = {
      ModemState.DISCONNECTED: "DISCONNECTED",
      ModemState.SIM_PENDING: "SIM_PENDING",
      ModemState.REGISTERING: "CONNECTING",
      ModemState.REGISTERED: "REGISTERED",
      ModemState.ATTACHED: "CONNECTED",
    }
    return mapping[state]

  def _publish_state(self, state: ModemState) -> None:
    if self._last_state == state:
      return
    self._last_state = state
    try:
      self.state_path.parent.mkdir(parents=True, exist_ok=True)
      self.state_path.write_text(self._external_state_name(state) + "\n", encoding="utf-8")
    except OSError:
      pass

  def is_sim_ready(self) -> tuple[bool, str]:
    lines = self.client.send("AT+CPIN?", timeout=2.0 if self.fast_boot else 4.0)
    return self._contains(lines, "READY"), "\n".join(lines)

  def set_apn(self) -> str:
    apn = self.apn
    cmd = f'AT+CGDCONT=1,"IP","{apn}"'
    lines = self.client.send(cmd, timeout=3.0 if self.fast_boot else 6.0)
    return "\n".join(lines)

  def identify_model(self) -> str:
    lines = self.client.send("ATI", timeout=3.0 if self.fast_boot else 5.0)
    return detect_quectel_generation("\n".join(lines))

  def apply_model_setup(self, model: str, sim_id: str = "") -> None:
    # Keep one state machine for both chips, only setup commands differ.
    cmds: list[str] = []
    if model == "EG25":
      cmds += [
        'AT+QSIMDET=1,0',
        'AT+QSIMSTAT=1',
      ]
      if self.apn is not None:
        cmds.append(f'AT+CGDCONT=1,"IP","{self.apn}"')
    elif model == "EG916":
      cmds += [
        'AT$QCSIMSLEEP=0',
        'AT$QCSIMCFG=SimPowerSave,0',
        'AT$QCPCFG=usbNet,1',
      ]
      if self.apn is not None and len(self.apn) > 0:
        cmds.append(f'AT+CGDCONT=1,"IP","{self.apn}"')
    else:
      if self.apn is not None and len(self.apn) > 0:
        cmds.append(f'AT+CGDCONT=1,"IP","{self.apn}"')

    if sim_id is None:
      sim_id = ""
    if len(sim_id) == 0 and model == "EG916":
      cmds.append("AT$QCSIMCFG=SimPowerSave,0")

    for cmd in cmds:
      self.client.send(cmd, timeout=2.5 if self.fast_boot else 6.0)

  def wait_for_registration(self) -> tuple[bool, str]:
    deadline = time.monotonic() + self.registration_timeout
    last_lines: list[str] = []
    while time.monotonic() < deadline:
      lines = self.client.send("AT+CEREG?", timeout=2.5 if self.fast_boot else 4.0)
      last_lines = lines
      status = self._parse_cereg_status(lines)
      # 1=home, 5=roaming
      if status in (1, 5):
        return True, "\n".join(lines)
      time.sleep(self.registration_poll_interval)
    return False, "\n".join(last_lines)

  def attach_packet_service(self) -> tuple[bool, str]:
    self.client.send("AT+CGATT=1", timeout=5.0 if self.fast_boot else 8.0)
    self.client.send("AT+CGACT=1,1", timeout=6.0 if self.fast_boot else 10.0)
    lines = self.client.send("AT+CGPADDR=1", timeout=2.0 if self.fast_boot else 4.0)
    attached = self._has_routable_cgpaddr(lines)
    return attached, "\n".join(lines)

  @staticmethod
  def _has_routable_cgpaddr(lines: Sequence[str]) -> bool:
    for line in lines:
      if not line.startswith("+CGPADDR:"):
        continue
      payload = line.split(":", 1)[-1]
      fields = [x.strip().strip('"') for x in payload.split(",")]
      for token in fields[1:]:
        if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", token):
          if token != "0.0.0.0":
            return True
        elif ":" in token and token not in {"::", "0:0:0:0:0:0:0:0"}:
          return True
    return False

  def recover_link(self) -> None:
    self.client.send("AT+CFUN=0", timeout=4.0 if self.fast_boot else 8.0)
    time.sleep(0.2 if self.fast_boot else 0.5)
    self.client.send("AT+CFUN=1", timeout=6.0 if self.fast_boot else 10.0)
    time.sleep(0.5 if self.fast_boot else 1.0)

  def _run_startup_once(self, sim_id: str = "") -> ModemSnapshot:
    self._publish_state(ModemState.REGISTERING)
    self.client.send("AT", timeout=2.0 if self.fast_boot else 3.0)
    model = self.identify_model()

    sim_ready, sim_resp = self.is_sim_ready()
    if not sim_ready:
      self._publish_state(ModemState.SIM_PENDING)
      return ModemSnapshot(
        state=ModemState.SIM_PENDING,
        sim_ready=False,
        registered=False,
        attached=False,
        model=model,
        last_response=sim_resp,
      )

    self.apply_model_setup(model=model, sim_id=sim_id)

    registered, reg_resp = self.wait_for_registration()
    if not registered:
      self._publish_state(ModemState.REGISTERING)
      return ModemSnapshot(
        state=ModemState.REGISTERING,
        sim_ready=True,
        registered=False,
        attached=False,
        model=model,
        last_response=reg_resp,
      )

    self._publish_state(ModemState.REGISTERED)
    attached, att_resp = self.attach_packet_service()
    if attached:
      self._publish_state(ModemState.ATTACHED)
      return ModemSnapshot(
        state=ModemState.ATTACHED,
        sim_ready=True,
        registered=True,
        attached=True,
        model=model,
        last_response=att_resp,
      )

    return ModemSnapshot(
      state=ModemState.REGISTERED,
      sim_ready=True,
      registered=True,
      attached=False,
      model=model,
      last_response=att_resp,
    )

  def run_startup_sequence(self, sim_id: str = "") -> ModemSnapshot:
    total_attempts = self.startup_retries + 1
    last_snapshot: ModemSnapshot | None = None

    for attempt in range(total_attempts):
      snapshot = self._run_startup_once(sim_id=sim_id)
      last_snapshot = snapshot
      if snapshot.attached:
        return snapshot
      if snapshot.state == ModemState.SIM_PENDING:
        return snapshot
      if attempt >= (total_attempts - 1):
        break

      self._publish_state(ModemState.DISCONNECTED)
      try:
        self.recover_link()
      except Exception:
        break

    assert last_snapshot is not None
    return last_snapshot


def enforce_wifi_over_lte_priority(
  wifi_ifaces: Sequence[str] = ("wlan0", "wlan1"),
  lte_ifaces: Sequence[str] = ("rmnet_data0", "wwan0", "usb0"),
  wifi_metric: int = 10,
  lte_metric: int = 500,
) -> None:
  # Route preference: WiFi (lower metric) wins over LTE.
  for iface in wifi_ifaces:
    os.system(f"ip route replace default dev {iface} metric {wifi_metric} >/dev/null 2>&1")
  for iface in lte_ifaces:
    os.system(f"ip route replace default dev {iface} metric {lte_metric} >/dev/null 2>&1")


def modem_state_from_shm(path: Path = MODEM_STATE_PATH) -> str:
  try:
    return path.read_text(encoding="utf-8").strip()
  except OSError:
    return ""


def modem_response_contains(response: str | Sequence[str] | None, needle: str) -> bool:
  if response is None:
    return False
  if isinstance(response, str):
    return needle.lower() in response.lower()
  blob = "\n".join([str(x) for x in response])
  return needle.lower() in blob.lower()


def detect_quectel_generation(response: str) -> str:
  txt = response.upper()
  if re.search(r"EG25", txt):
    return "EG25"
  if re.search(r"EG916", txt):
    return "EG916"
  return "UNKNOWN"
