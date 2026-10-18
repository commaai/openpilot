from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from openpilot.selfdrive.car.toyota_tss3_oracle_kit import oracle_kit_compatibility
from openpilot.selfdrive.car.toyota_tss3_oracle_status import OracleStatus

TOOL_PATH = Path(os.getenv("TSS3_ORACLE_TOOL", "/data/tss3-oracle/tss3-request-signer"))
RUN_ROOT = Path(os.getenv("TSS3_ORACLE_RUN_ROOT", "/data/tss3-oracle-runs"))
CANCEL_FILENAME = "cancel-requested"

STAGE_LABELS = {
  "arming": "starting",
  "armed": "waiting for POWER",
  "programming": "installing RAM oracle",
  "waiting_ready": "waiting for READY / Park",
  "verifying": "checking peer health + signer",
  "finishing": "waiting for backend exit",
  "done": "complete",
  "error": "failed",
}


def tool_available(tool_path: Path = TOOL_PATH) -> bool:
  return tool_path.is_file() and os.access(tool_path, os.X_OK)


def request_cooperative_cancel(run_dir: Path) -> None:
  """Ask the backend to stop only while PROGRAMMING is still avoidable.

  Never kill the bringup process group here. If 10 02 has already crossed the
  wire, the backend deliberately ignores this late request and completes the RAM
  handoff rather than abandoning the EPS between application and bootloader.
  """
  try:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / CANCEL_FILENAME).write_text("cancel\n", encoding="utf-8")
  except OSError:
    # Closing the UI must remain best-effort; the running backend is safer than
    # escalating a cancellation failure into a force-kill.
    pass


class Tss3OracleRunner:
  """Own one bring-up backend process and its validated status stream."""

  def __init__(self, tool_path: Path = TOOL_PATH, run_root: Path = RUN_ROOT, *, start: bool = True):
    self._tool_path = tool_path
    self._run_root = run_root
    self._lock = threading.Lock()
    self._status: dict[str, Any] = {
      "stage": "arming",
      "title": "Arming oracle bringup",
      "detail": "Preparing the startup catcher.",
      "progress": 0,
      "done": False,
      "error": False,
    }
    self._proc: subprocess.Popen[str] | None = None
    self._run_dir: Path | None = None
    if start:
      self.start()

  def snapshot(self) -> dict[str, Any]:
    with self._lock:
      return dict(self._status)

  def _set_status(self, status: dict[str, Any]) -> None:
    with self._lock:
      self._status = dict(status)

  def start(self) -> None:
    if self._proc is not None or self._run_dir is not None:
      raise RuntimeError("TSS3 oracle runner already started")

    compatible, detail = oracle_kit_compatibility(self._tool_path)
    if not compatible:
      title = "Oracle kit invalid" if detail.startswith("oracle kit metadata invalid:") else "Oracle tool unavailable"
      self._set_status(
        {
          "stage": "error",
          "title": title,
          "detail": detail,
          "progress": 0,
          "done": False,
          "error": True,
        }
      )
      return

    stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
    self._run_dir = self._run_root / f"{stamp}-{os.getpid()}-{time.monotonic_ns()}"
    cmd = [str(self._tool_path), "ui-bringup", str(self._run_dir)]

    try:
      self._proc = subprocess.Popen(
        cmd,
        cwd=str(self._tool_path.parent),
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
      self._set_status(
        {
          "stage": "error",
          "title": "Could not start oracle bringup",
          "detail": f"{type(exc).__name__}: {exc}",
          "progress": 0,
          "done": False,
          "error": True,
        }
      )
      return

    threading.Thread(target=self._reader, name="tss3_oracle_ui", daemon=True).start()

  def request_cancel(self) -> None:
    status = self.snapshot()
    proc = self._proc
    if proc is not None and proc.poll() is None and not status.get("done") and not status.get("error") and self._run_dir is not None:
      request_cooperative_cancel(self._run_dir)

  def _reader(self) -> None:
    assert self._proc is not None and self._proc.stdout is not None
    report = OracleStatus()
    for raw in self._proc.stdout:
      status = report.feed(raw)
      if status is None:
        continue
      if status["done"]:
        self._set_status(
          {
            **status,
            "stage": "finishing",
            "title": "Finalizing bringup",
            "detail": "Waiting for the backend to finish.",
            "done": False,
          }
        )
      else:
        self._set_status(status)

    self._set_status(report.finish(self._proc.wait()))
