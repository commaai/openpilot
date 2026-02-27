from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
  from openpilot.system.hardware.tici.pure_python_modem import (
    MODEM_STATE_PATH,
    QuectelATClient,
    QuectelModemStateMachine,
    enforce_wifi_over_lte_priority,
    modem_state_from_shm,
  )
except ModuleNotFoundError:
  module_path = Path(__file__).resolve().parent / "pure_python_modem.py"
  spec = importlib.util.spec_from_file_location("pure_python_modem_local", module_path)
  assert spec is not None and spec.loader is not None
  pure_python_modem = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = pure_python_modem
  spec.loader.exec_module(pure_python_modem)

  MODEM_STATE_PATH = pure_python_modem.MODEM_STATE_PATH
  QuectelATClient = pure_python_modem.QuectelATClient
  QuectelModemStateMachine = pure_python_modem.QuectelModemStateMachine
  enforce_wifi_over_lte_priority = pure_python_modem.enforce_wifi_over_lte_priority
  modem_state_from_shm = pure_python_modem.modem_state_from_shm


class ScriptedATClient:
  def __init__(self, scripted: list[list[str]]) -> None:
    self.scripted = scripted
    self.calls: list[dict[str, Any]] = []

  def send(self, command: str, timeout: float = 0.0) -> list[str]:
    ts = time.monotonic()
    if not self.scripted:
      raise RuntimeError(f"no scripted response left for command={command}")
    response = self.scripted.pop(0)
    self.calls.append({
      "ts": ts,
      "command": command,
      "timeout": timeout,
      "response": response,
    })
    return response


def _build_simulated_script(transient_fail: bool) -> list[list[str]]:
  first = [
    ["OK"],                         # AT
    ["Quectel EG25", "OK"],         # ATI
    ["+CPIN: READY", "OK"],         # AT+CPIN?
    ["OK"],                         # QSIMDET
    ["OK"],                         # QSIMSTAT
    ["OK"],                         # APN
    ["+CEREG: 2,1", "OK"],          # registered
    ["OK"],                         # CGATT
    ["OK"],                         # CGACT
  ]
  if transient_fail:
    first += [
      ["+CGPADDR: 1,0.0.0.0", "OK"],  # attach fail (triggers recovery)
      ["OK"],                         # CFUN=0
      ["OK"],                         # CFUN=1
      ["OK"],                         # AT
      ["Quectel EG25", "OK"],         # ATI
      ["+CPIN: READY", "OK"],         # CPIN
      ["OK"],                         # QSIMDET
      ["OK"],                         # QSIMSTAT
      ["OK"],                         # APN
      ["+CEREG: 2,1", "OK"],          # registered
      ["OK"],                         # CGATT
      ["OK"],                         # CGACT
      ["+CGPADDR: 1,10.0.0.2", "OK"], # attached
    ]
    return first

  first += [
    ["+CGPADDR: 1,10.0.0.2", "OK"],   # attached
  ]
  return first


def _run_state_machine(
  client: QuectelATClient | ScriptedATClient,
  apn: str,
  registration_timeout: float,
  startup_retries: int,
  state_path: Path,
) -> dict[str, Any]:
  sm = QuectelModemStateMachine(
    client=client,
    apn=apn,
    registration_timeout=registration_timeout,
    registration_poll_interval=0.1,
    startup_retries=startup_retries,
    fast_boot=True,
    state_path=state_path,
  )
  started = time.monotonic()
  snapshot = sm.run_startup_sequence(sim_id="")
  elapsed = time.monotonic() - started
  return {
    "elapsed_sec": round(elapsed, 3),
    "state": snapshot.state.name,
    "sim_ready": snapshot.sim_ready,
    "registered": snapshot.registered,
    "attached": snapshot.attached,
    "model": snapshot.model,
    "last_response": snapshot.last_response,
    "published_state": modem_state_from_shm(state_path),
  }


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Dry-run diagnostic for pure Python Quectel modem state machine."
  )
  parser.add_argument("--simulate", action="store_true", help="Run with scripted AT responses.")
  parser.add_argument(
    "--simulate-transient-fail",
    action="store_true",
    help="In simulate mode, fail first attach then auto-recover on retry.",
  )
  parser.add_argument("--port", default="/dev/ttyUSB2", help="AT serial port for real mode.")
  parser.add_argument("--apn", default="", help="APN value.")
  parser.add_argument("--registration-timeout", type=float, default=45.0, help="Registration timeout seconds.")
  parser.add_argument("--startup-retries", type=int, default=1, help="State machine startup retries.")
  parser.add_argument(
    "--state-path",
    default=str(MODEM_STATE_PATH),
    help="Path to modem state marker file (default: /dev/shm/modem_state.txt).",
  )
  parser.add_argument("--enforce-route", action="store_true", help="Apply WiFi > LTE route metrics.")
  parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
  args = parser.parse_args()

  state_path = Path(args.state_path)
  if args.simulate:
    scripted = _build_simulated_script(transient_fail=args.simulate_transient_fail)
    client: QuectelATClient | ScriptedATClient = ScriptedATClient(scripted)
  else:
    client = QuectelATClient(port=args.port, timeout=0.8, write_timeout=0.8)

  result = _run_state_machine(
    client=client,
    apn=args.apn,
    registration_timeout=args.registration_timeout,
    startup_retries=max(0, args.startup_retries),
    state_path=state_path,
  )

  if args.enforce_route:
    enforce_wifi_over_lte_priority()

  payload: dict[str, Any] = {
    "mode": "simulate" if args.simulate else "real",
    "result": result,
  }
  if args.simulate and isinstance(client, ScriptedATClient):
    payload["at_trace"] = client.calls

  if args.json:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
  else:
    print(f"mode={payload['mode']}")
    print(f"state={result['state']} model={result['model']} attached={result['attached']} elapsed_sec={result['elapsed_sec']}")
    print(f"published_state={result['published_state']}")
    if args.simulate and isinstance(client, ScriptedATClient):
      print(f"at_trace_count={len(client.calls)}")
      for idx, item in enumerate(client.calls, start=1):
        cmd = item["command"]
        resp = " | ".join(item["response"])
        print(f"{idx:02d}. {cmd} -> {resp}")

  return 0 if result["attached"] else 2


if __name__ == "__main__":
  raise SystemExit(main())
