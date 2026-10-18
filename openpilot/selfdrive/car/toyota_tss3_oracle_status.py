"""Host-only parsing and display of bring-up results; no vehicle or process control."""
from __future__ import annotations

import json
from typing import Any

STATUS_SCHEMA = "tss3-request-signer-ui-status-v1"


def parse_status(line: str) -> dict[str, Any] | None:
  try:
    row = json.loads(line)
  except json.JSONDecodeError:
    return None
  if not isinstance(row, dict) or row.get("schema") != STATUS_SCHEMA:
    return None
  if not all(isinstance(row.get(key), str) for key in ("stage", "title", "detail")):
    return None
  if type(row.get("progress")) is not int or not 0 <= row["progress"] <= 100:
    return None
  if type(row.get("done")) is not bool or type(row.get("error")) is not bool:
    return None
  if row["done"] != (row["stage"] == "done") or row["error"] != (row["stage"] == "error"):
    return None
  return row


class OracleStatus:
  """Accumulate backend output; the first error wins and completion requires a clean exit."""

  def __init__(self):
    self._status: dict[str, Any] | None = None
    self._last_output = ""

  def feed(self, line: str) -> dict[str, Any] | None:
    """Return an accepted status update, or None when there is nothing to publish."""
    line = line.strip()
    if not line:
      return None
    status = parse_status(line)
    if status is None:
      self._last_output = line
      return None
    if self._status is not None and self._status["error"]:
      return None
    self._status = status
    return status

  def finish(self, returncode: int) -> dict[str, Any]:
    if self._status is not None and (self._status["error"] or (returncode == 0 and self._status["done"])):
      return self._status
    detail = f"Backend exited with status {returncode}." if returncode != 0 else "Backend exited without a completion status."
    if self._last_output:
      detail += f" Last output: {self._last_output}"
    return {
      "schema": STATUS_SCHEMA, "stage": "error", "title": "Request-signer bringup stopped",
      "detail": detail, "progress": 0, "done": False, "error": True,
    }
