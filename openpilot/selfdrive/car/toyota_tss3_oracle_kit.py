from __future__ import annotations

import json
import os
from pathlib import Path

from opendbc.car.toyota.values import CAR, ToyotaFlags

DEFAULT_ORACLE_TOOL = Path(__file__).resolve().with_name("toyota_tss3_oracle") / "tss3-request-signer"


def tss3_fingerprints() -> frozenset[str]:
  """Every Toyota platform OpenDBC currently marks as TSS3."""
  return frozenset(platform.value for platform in CAR if platform.config.flags & ToyotaFlags.TSS3)


def oracle_ui_state(car_fingerprint: str | None, *, is_release: bool, tool_available: bool) -> tuple[bool, bool, str]:
  """Return visibility, vehicle eligibility, and the manual bringup label.

  A bundled development kit remains visible as a disabled diagnostic when a
  harness fault makes fingerprinting fall back to MOCK. Never make the action
  eligible until the live CarParams identifies a TSS3 platform.
  """
  tss3 = car_fingerprint in tss3_fingerprints()
  visible = not is_release and (tss3 or tool_available)
  label = "ARM" if tss3 and tool_available else "KIT MISSING" if tss3 else "CHECK HARNESS"
  return visible, tss3, label


def oracle_kit_compatibility(tool_path: Path) -> tuple[bool, str]:
  """The kit is a per-vehicle signed bundle, so accept any structurally valid
  kit; the vehicle it targets is the operator's choice, not a fixed platform."""
  if not tool_path.is_file() or not os.access(tool_path, os.X_OK):
    return False, f"oracle tool unavailable: {tool_path}"

  meta_path = tool_path.parent / "bundle" / "request_signer.json"
  try:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    target = meta["target"]["name"]
  except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
    return False, f"oracle kit metadata invalid: {meta_path}"

  if not isinstance(target, str) or not target:
    return False, f"oracle kit metadata invalid: {meta_path}"
  return True, ""
