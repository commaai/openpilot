#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from openpilot.common.swaglog import cloudlog
from openpilot.common.utils import atomic_write
from openpilot.selfdrive.car.toyota_tss3_oracle_kit import oracle_kit_compatibility
from openpilot.selfdrive.car.toyota_tss3_oracle_status import OracleStatus

TOOL_PATH = Path(os.getenv("TSS3_ORACLE_TOOL", "/data/tss3-oracle/tss3-request-signer"))
RUN_ROOT = Path(os.getenv("TSS3_ORACLE_RUN_ROOT", "/data/tss3-oracle-runs"))
STATUS_PATH = Path(os.getenv("TSS3_ORACLE_AUTO_STATUS", "/data/tss3-oracle-auto-status.json"))
STATUS_SCHEMA = "tss3-oracle-auto-arm-status-v1"
NATIVE_CATCH_PATH = Path(os.getenv("TSS3_ORACLE_NATIVE_CATCH", "/tmp/tss3-oracle-native-catch.json"))
NATIVE_NOTIFY_PATH = Path(os.getenv("TSS3_ORACLE_NATIVE_NOTIFY", "/tmp/tss3-oracle-native-catch.sock"))
WARM_WORKER_PATH = Path(os.getenv("TSS3_ORACLE_WARM_WORKER", "/tmp/tss3-oracle-warm-worker.sock"))

_exit_requested = False
_child: subprocess.Popen[str] | None = None
_child_lock = threading.Lock()
_listener: socket.socket | None = None
_warm_worker: subprocess.Popen[str] | None = None


def _warm_worker_ready() -> bool:
  return _warm_worker is not None and _warm_worker.poll() is None and WARM_WORKER_PATH.is_socket()


def _write_status(state: str, detail: str, **extra: Any) -> None:
  status: dict[str, Any] = {
    "schema": STATUS_SCHEMA,
    "state": state,
    "detail": detail,
    "monotonic_ns": time.monotonic_ns(),
    **extra,
  }
  STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
  with atomic_write(str(STATUS_PATH), "w", overwrite=True) as f:
    json.dump(status, f, indent=2, sort_keys=True)
    f.write("\n")



def _terminate_child() -> None:
  with _child_lock:
    proc = _child
  if proc is None or proc.poll() is not None:
    return
  try:
    os.killpg(proc.pid, signal.SIGTERM)
  except ProcessLookupError:
    pass


def _stop_warm_worker() -> None:
  global _warm_worker
  proc = _warm_worker
  _warm_worker = None
  if proc is not None and proc.poll() is None:
    try:
      os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
      pass
    try:
      proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
      try:
        os.killpg(proc.pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
  WARM_WORKER_PATH.unlink(missing_ok=True)


def _start_warm_worker(*, publish_ready_status: bool = True) -> bool:
  global _warm_worker
  _stop_warm_worker()
  compatible, detail = oracle_kit_compatibility(TOOL_PATH)
  if not compatible:
    _write_status("error", detail)
    return False

  cmd = [
    str(TOOL_PATH), "ui-worker", str(WARM_WORKER_PATH), str(os.getpid()),
  ]
  try:
    proc = subprocess.Popen(
      cmd,
      cwd=str(TOOL_PATH.parent),
      stdin=subprocess.DEVNULL,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      text=True,
      start_new_session=True,
    )
  except OSError as exc:
    _write_status("error", f"failed to start warm oracle worker: {type(exc).__name__}: {exc}")
    return False
  _warm_worker = proc

  deadline = time.monotonic() + 15.0
  while time.monotonic() < deadline and not _exit_requested:
    if proc.poll() is not None:
      _write_status("error", f"warm oracle worker exited during startup: {proc.returncode}")
      _warm_worker = None
      return False
    if WARM_WORKER_PATH.is_socket():
      if publish_ready_status:
        _write_status("armed", "Automatic TSS3 oracle uploader is warm and waiting for native vehicle wake/start detection.")
      return True
    time.sleep(0.02)

  _stop_warm_worker()
  _write_status("error", "warm oracle worker did not become ready")
  return False


def _signal_handler(signum, _frame) -> None:
  global _exit_requested
  cloudlog.info(f"tss3oracled caught signal {signum}")
  _exit_requested = True
  _terminate_child()
  _stop_warm_worker()
  if _listener is not None:
    _listener.close()


def _claim_native_catch(path: Path = NATIVE_CATCH_PATH) -> tuple[Path, dict[str, Any]] | None:
  """Claim pandad's catch marker. The rename is the single-claimant guard.
  The native catcher writes this file from one code path, after sending
  PROGRAMMING on a car it already verified is TSS3, so beyond the schema
  identity nothing here needs re-validating."""
  try:
    marker = json.loads(path.read_text(encoding="utf-8"))
  except (FileNotFoundError, json.JSONDecodeError, OSError):
    return None
  if not isinstance(marker, dict) or marker.get("schema") != "tss3-oracle-native-catch-v1":
    return None
  if not isinstance(marker.get("pandad_wrapper_pid"), int):
    return None

  claimed = path.with_name(f"{path.name}.claimed-{os.getpid()}")
  try:
    os.replace(path, claimed)
  except FileNotFoundError:
    return None
  return claimed, marker


def _record_trigger_timing(run_dir: Path, trigger_fallback: Path, *, native_catch: dict[str, Any], catch_received_ns: int,
                           backend_launch_ns: int, returncode: int) -> dict[str, Any]:
  record: dict[str, Any] = {
    "schema": "tss3-oracle-auto-arm-trigger-v1",
    "native_catch": native_catch,
    "native_catch_daemon_received_monotonic_ns": catch_received_ns,
    "backend_launch_monotonic_ns": backend_launch_ns,
    "daemon_to_backend_launch_ms": (backend_launch_ns - catch_received_ns) / 1e6,
    "returncode": returncode,
  }

  programming_ns = native_catch.get("programming_tx_monotonic_ns")
  if isinstance(programming_ns, int):
    record["programming_to_daemon_ms"] = (catch_received_ns - programming_ns) / 1e6

  wake_ns = native_catch.get("wake_trigger_monotonic_ns")
  ignition_ns = native_catch.get("ignition_monotonic_ns")
  if isinstance(wake_ns, int) and wake_ns > 0:
    for key, out_key in (
      ("first_extended_tx_monotonic_ns", "wake_to_first_10_03_ms"),
      ("ignition_monotonic_ns", "wake_to_ignition_ms"),
      ("positive_extended_monotonic_ns", "wake_to_50_03_ms"),
      ("programming_tx_monotonic_ns", "wake_to_10_02_ms"),
    ):
      value = native_catch.get(key)
      if isinstance(value, int) and value > 0:
        record[out_key] = (value - wake_ns) / 1e6

  if isinstance(ignition_ns, int) and ignition_ns > 0:
    for key, out_key in (
      ("first_extended_tx_monotonic_ns", "ignition_to_first_10_03_ms"),
      ("power_wake_complete_monotonic_ns", "ignition_to_power_wake_complete_ms"),
      ("positive_extended_monotonic_ns", "ignition_to_50_03_ms"),
      ("programming_tx_monotonic_ns", "ignition_to_10_02_ms"),
    ):
      value = native_catch.get(key)
      if isinstance(value, int):
        record[out_key] = (value - ignition_ns) / 1e6

  trigger_path = (run_dir / "auto-trigger.json") if run_dir.is_dir() else trigger_fallback
  with atomic_write(str(trigger_path), "w", overwrite=True) as f:
    json.dump(record, f, indent=2, sort_keys=True)
    f.write("\n")
  return record


def _run_bringup(native_catch_path: Path, native_catch: dict[str, Any], *, catch_received_ns: int) -> bool:
  global _child
  stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
  RUN_ROOT.mkdir(parents=True, exist_ok=True)
  # Paths only; the backend owns creating its output directory.
  name = f"auto-{stamp}-{catch_received_ns}"
  run_dir = RUN_ROOT / name
  log_path = RUN_ROOT / f"{name}.auto-daemon.log"
  trigger_fallback = RUN_ROOT / f"{name}.auto-trigger.json"
  if not _warm_worker_ready():
    _write_status("error", "native PROGRAMMING was caught but the warm oracle uploader is unavailable")
    return False
  cmd = [
    str(TOOL_PATH), "ui-resume-warm",
    str(WARM_WORKER_PATH), str(native_catch_path), str(run_dir), str(native_catch["pandad_wrapper_pid"]),
  ]
  launch_ns = time.monotonic_ns()

  _write_status(
    "triggered",
    "Native pandad caught PROGRAMMING; resuming TSS3 oracle bringup.",
    run_dir=str(run_dir),
    auto_daemon_log=str(log_path),
    native_catch_daemon_received_monotonic_ns=catch_received_ns,
    backend_launch_monotonic_ns=launch_ns,
  )
  cloudlog.warning(f"tss3oracled triggering startup bringup: {run_dir}")

  try:
    proc = subprocess.Popen(
      cmd,
      cwd=str(TOOL_PATH.parent),
      stdin=subprocess.DEVNULL,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      encoding="utf-8",
      errors="replace",
      bufsize=1,
      start_new_session=True,
    )
  except OSError as exc:
    _write_status("error", f"failed to launch oracle bringup: {type(exc).__name__}: {exc}", run_dir=str(run_dir))
    return False

  with _child_lock:
    _child = proc

  report = OracleStatus()
  try:
    assert proc.stdout is not None
    with log_path.open("w", encoding="utf-8") as log:
      for raw in proc.stdout:
        log.write(raw)
        log.flush()
        row = report.feed(raw)
        if row is not None:
          _write_status(
            "running",
            row["detail"],
            run_dir=str(run_dir),
            backend_stage=row["stage"],
            backend_progress=row["progress"],
            backend_done=row["done"],
            backend_error=row["error"],
          )

    returncode = proc.wait()
  finally:
    with _child_lock:
      _child = None

  timing = _record_trigger_timing(
    run_dir, trigger_fallback,
    native_catch=native_catch,
    catch_received_ns=catch_received_ns,
    backend_launch_ns=launch_ns,
    returncode=returncode,
  )

  final_status = report.finish(returncode)
  success = not final_status["error"]

  if success:
    _write_status(
      "complete",
      "Automatic TSS3 oracle bringup passed.",
      run_dir=str(run_dir),
      trigger_timing=timing,
    )
  else:
    _write_status(
      "error",
      final_status["detail"],
      run_dir=str(run_dir),
      returncode=returncode,
      trigger_timing=timing,
    )
  return success


def main() -> None:
  global _exit_requested, _listener
  signal.signal(signal.SIGINT, _signal_handler)
  signal.signal(signal.SIGTERM, _signal_handler)

  listener = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
  _listener = listener
  NATIVE_NOTIFY_PATH.unlink(missing_ok=True)
  listener.bind(str(NATIVE_NOTIFY_PATH))
  # Do not block forever here.  The warm uploader is deliberately a separate
  # process and can die while the vehicle remains OFF; without a bounded wakeup
  # the daemon would stay alive with a stale "armed" status, then lose the next
  # already-caught PROGRAMMING transition because no worker exists to take it.
  listener.settimeout(0.25)

  try:
    if not _start_warm_worker():
      return
    while not _exit_requested:
      if not _warm_worker_ready():
        cloudlog.warning("tss3oracled warm uploader disappeared while armed; restarting it")
        if not _start_warm_worker():
          break

      claimed = _claim_native_catch()
      if claimed is None:
        try:
          listener.recv(1)
        except TimeoutError:
          continue
        except OSError:
          if _exit_requested:
            break
          raise
        continue
      claimed_path, marker = claimed
      catch_received_ns = time.monotonic_ns()
      try:
        _run_bringup(
          claimed_path, marker,
          catch_received_ns=catch_received_ns,
        )
      finally:
        claimed_path.unlink(missing_ok=True)
      # Re-arming must not immediately erase the previous run's result.
      if not _exit_requested and not _start_warm_worker(publish_ready_status=False):
        break
  finally:
    _terminate_child()
    _stop_warm_worker()
    listener.close()
    _listener = None
    NATIVE_NOTIFY_PATH.unlink(missing_ok=True)
    if _exit_requested:
      _write_status("stopped", "Automatic TSS3 oracle uploader stopped.")


if __name__ == "__main__":
  main()
