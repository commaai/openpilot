from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "system" / "hardware" / "tici" / "pure_python_modem.py"
SPEC = importlib.util.spec_from_file_location("pure_python_modem_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pure_python_modem = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pure_python_modem
SPEC.loader.exec_module(pure_python_modem)

ModemState = pure_python_modem.ModemState
QuectelATClient = pure_python_modem.QuectelATClient
QuectelModemStateMachine = pure_python_modem.QuectelModemStateMachine
modem_state_from_shm = pure_python_modem.modem_state_from_shm


class FakeATClient:
  def __init__(self, scripted: list[list[str]]) -> None:
    self.scripted = scripted
    self.calls: list[str] = []

  def send(self, command: str, timeout: float = 0.0):
    self.calls.append(command)
    if not self.scripted:
      raise RuntimeError(f"no scripted response left for {command}")
    return self.scripted.pop(0)


class FakeSerial:
  def __init__(self, responses: list[bytes]) -> None:
    self._responses = list(responses)
    self.reset_input_buffer_calls = 0
    self.reset_output_buffer_calls = 0
    self.write_calls: list[bytes] = []
    self.flush_calls = 0

  def reset_input_buffer(self) -> None:
    self.reset_input_buffer_calls += 1

  def reset_output_buffer(self) -> None:
    self.reset_output_buffer_calls += 1

  def write(self, payload: bytes) -> None:
    self.write_calls.append(payload)

  def flush(self) -> None:
    self.flush_calls += 1

  def readline(self) -> bytes:
    if self._responses:
      return self._responses.pop(0)
    return b""


def test_parse_cereg_status_variants():
  lines = ["+CEREG: 2,1", "OK"]
  assert QuectelModemStateMachine._parse_cereg_status(lines) == 1

  lines = ["+CEREG: 5", "OK"]
  assert QuectelModemStateMachine._parse_cereg_status(lines) == 5

  lines = ["+CEREG: 0x5", "OK"]
  assert QuectelModemStateMachine._parse_cereg_status(lines) == 5


def test_state_machine_happy_path(tmp_path):
  fake = FakeATClient(
    scripted=[
      ["OK"],                        # AT
      ["Quectel EG25", "OK"],        # ATI
      ["+CPIN: READY", "OK"],        # AT+CPIN?
      ["OK"],                        # AT+QSIMDET=1,0
      ["OK"],                        # AT+QSIMSTAT=1
      ["OK"],                        # AT+CGDCONT...
      ["+CEREG: 2,2", "OK"],         # first registration probe
      ["+CEREG: 2,1", "OK"],         # registered
      ["OK"],                        # AT+CGATT=1
      ["OK"],                        # AT+CGACT=1,1
      ["+CGPADDR: 1,10.0.0.2", "OK"] # AT+CGPADDR=1
    ]
  )
  state_path = tmp_path / "modem_state.txt"
  sm = QuectelModemStateMachine(
    client=fake,
    apn="internet",
    registration_timeout=1.0,
    state_path=state_path,
  )
  snap = sm.run_startup_sequence(sim_id="1234567890")
  assert snap.state == ModemState.ATTACHED
  assert snap.sim_ready is True
  assert snap.registered is True
  assert snap.attached is True
  assert snap.model == "EG25"
  assert fake.calls[1] == "ATI"
  assert modem_state_from_shm(state_path) == "CONNECTED"


def test_enforce_wifi_over_lte_priority_invokes_route_commands(monkeypatch):
  calls: list[str] = []

  def fake_system(cmd: str) -> int:
    calls.append(cmd)
    return 0

  monkeypatch.setattr(pure_python_modem.os, "system", fake_system)
  pure_python_modem.enforce_wifi_over_lte_priority(
    wifi_ifaces=("wlan0",),
    lte_ifaces=("rmnet_data0",),
    wifi_metric=10,
    lte_metric=500,
  )

  cmds = calls
  assert any("wlan0 metric 10" in c for c in cmds)
  assert any("rmnet_data0 metric 500" in c for c in cmds)


def test_at_client_send_with_mocked_serial_stream(monkeypatch):
  fake_serial = FakeSerial([
    b"+CPIN: READY\r\n",
    b"OK\r\n",
  ])

  monkeypatch.setattr(pure_python_modem, "Serial", lambda *args, **kwargs: fake_serial)
  client = QuectelATClient(port="/dev/ttyFAKE2", timeout=0.1, write_timeout=0.1)
  lines = client.send("AT+CPIN?", timeout=0.5)

  assert lines == ["+CPIN: READY", "OK"]
  assert fake_serial.reset_input_buffer_calls >= 1
  assert fake_serial.reset_output_buffer_calls >= 1
  assert fake_serial.write_calls == [b"AT+CPIN?\r"]
  assert fake_serial.flush_calls == 1


def test_cgpaddr_requires_non_zero_ip():
  assert QuectelModemStateMachine._has_routable_cgpaddr([
    "+CGPADDR: 1,0.0.0.0",
    "OK",
  ]) is False
  assert QuectelModemStateMachine._has_routable_cgpaddr([
    "+CGPADDR: 1,10.0.0.2",
    "OK",
  ]) is True


def test_startup_sequence_retries_after_attach_failure(monkeypatch, tmp_path):
  fake = FakeATClient(
    scripted=[
      ["OK"],                         # first AT
      ["Quectel EG25", "OK"],         # first ATI
      ["+CPIN: READY", "OK"],         # first CPIN
      ["OK"],                         # first QSIMDET
      ["OK"],                         # first QSIMSTAT
      ["OK"],                         # first APN
      ["+CEREG: 2,1", "OK"],          # first registration
      ["OK"],                         # first CGATT
      ["OK"],                         # first CGACT
      ["+CGPADDR: 1,0.0.0.0", "OK"],  # first address (invalid)
      ["OK"],                         # recovery CFUN=0
      ["OK"],                         # recovery CFUN=1
      ["OK"],                         # second AT
      ["Quectel EG25", "OK"],         # second ATI
      ["+CPIN: READY", "OK"],         # second CPIN
      ["OK"],                         # second QSIMDET
      ["OK"],                         # second QSIMSTAT
      ["OK"],                         # second APN
      ["+CEREG: 2,1", "OK"],          # second registration
      ["OK"],                         # second CGATT
      ["OK"],                         # second CGACT
      ["+CGPADDR: 1,10.0.0.2", "OK"], # second address (valid)
    ]
  )
  monkeypatch.setattr(pure_python_modem.time, "sleep", lambda _s: None)
  sm = QuectelModemStateMachine(
    client=fake,
    apn="internet",
    startup_retries=1,
    registration_timeout=1.0,
    state_path=tmp_path / "modem_state.txt",
  )
  snap = sm.run_startup_sequence(sim_id="123")

  assert snap.attached is True
  assert "AT+CFUN=0" in fake.calls
  assert "AT+CFUN=1" in fake.calls
